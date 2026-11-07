// LFM2-MoE support for Strata (local, opt-in). Hyperparameters and tensor binding.
#pragma once

#include "lfm2_gguf.hpp"

namespace lfm2 {

struct Hparams {
    int n_layer = 0, n_embd = 0, n_ff = 0, n_ctx_train = 0;
    int n_head = 0, head_dim = 0, n_head_kv = 0;
    int n_expert = 0, n_expert_used = 0, n_ff_exp = 0, n_dense = 0;
    int n_vocab = 0, l_cache = 0;
    float rope_base = 0, eps = 0;
    int bos = -1, eos = -1, pad = -1;
    std::vector<int> kv_per_layer;
};

struct Layer {
    bool is_attn = false, is_moe = false;
    int attn_index = -1;
    const TensorInfo *attn_norm = nullptr, *ffn_norm = nullptr;
    const TensorInfo *wq = nullptr, *wk = nullptr, *wv = nullptr, *wo = nullptr, *q_norm = nullptr, *k_norm = nullptr;
    const TensorInfo *conv = nullptr, *in_proj = nullptr, *out_proj = nullptr;
    const TensorInfo *gate = nullptr, *up = nullptr, *down = nullptr;
    const TensorInfo *router = nullptr, *exp_bias = nullptr, *gate_e = nullptr, *up_e = nullptr, *down_e = nullptr;
};

struct Model {
    Gguf g;
    Hparams hp;
    const TensorInfo *tok_embd = nullptr, *out_norm = nullptr;
    std::vector<Layer> layers;
    int n_attn_layers = 0;

    void load(const std::string& path) {
        g.open(path);
        const std::string arch = g.get_str("general.architecture");
        if (arch != "lfm2moe") throw std::runtime_error("architecture is '" + arch + "', expected lfm2moe");
        auto I = [&](const char* k) {
            int64_t v = g.get_int(std::string("lfm2moe.") + k, -1);
            if (v < 0) throw std::runtime_error(std::string("missing key lfm2moe.") + k);
            return (int)v;
        };
        hp.n_layer = I("block_count");
        hp.n_embd = I("embedding_length");
        hp.n_ff = I("feed_forward_length");
        hp.n_ctx_train = I("context_length");
        hp.n_head = I("attention.head_count");
        hp.n_expert = I("expert_count");
        hp.n_expert_used = I("expert_used_count");
        hp.n_ff_exp = I("expert_feed_forward_length");
        hp.n_dense = I("leading_dense_block_count");
        hp.l_cache = I("shortconv.l_cache");
        hp.eps = (float)g.get_float("lfm2moe.attention.layer_norm_rms_epsilon", 1e-5);
        hp.rope_base = (float)g.get_float("lfm2moe.rope.freq_base", 1e6);
        hp.head_dim = hp.n_embd / hp.n_head;
        hp.bos = (int)g.get_int("tokenizer.ggml.bos_token_id", -1);
        hp.eos = (int)g.get_int("tokenizer.ggml.eos_token_id", -1);
        hp.pad = (int)g.get_int("tokenizer.ggml.padding_token_id", -1);
        if (hp.l_cache != 3) throw std::runtime_error("only shortconv l_cache=3 is supported");

        auto it = g.int_arrays.find("lfm2moe.attention.head_count_kv");
        if (it != g.int_arrays.end()) {
            for (auto v : it->second) hp.kv_per_layer.push_back((int)v);
        } else {
            int64_t v = g.get_int("lfm2moe.attention.head_count_kv", -1);
            if (v < 0) throw std::runtime_error("missing head_count_kv");
            hp.kv_per_layer.assign(hp.n_layer, (int)v);
        }
        if ((int)hp.kv_per_layer.size() != hp.n_layer) throw std::runtime_error("head_count_kv size mismatch");
        for (int v : hp.kv_per_layer) {
            if (v == 0) continue;
            if (hp.n_head_kv == 0) hp.n_head_kv = v;
            else if (hp.n_head_kv != v) throw std::runtime_error("variable head_count_kv is not supported");
        }
        if (hp.n_head_kv == 0 || hp.n_head % hp.n_head_kv) throw std::runtime_error("bad head_count_kv");

        tok_embd = &g.need("token_embd.weight");
        out_norm = &g.need("token_embd_norm.weight");
        hp.n_vocab = (int)tok_embd->ne[1];
        if ((int)tok_embd->ne[0] != hp.n_embd) throw std::runtime_error("token_embd shape mismatch");

        layers.resize(hp.n_layer);
        for (int il = 0; il < hp.n_layer; ++il) {
            Layer& L = layers[il];
            auto N = [&](const char* s) { return &g.need("blk." + std::to_string(il) + "." + s); };
            L.attn_norm = N("attn_norm.weight");
            L.ffn_norm = N("ffn_norm.weight");
            L.is_attn = hp.kv_per_layer[il] > 0;
            if (L.is_attn) {
                L.attn_index = n_attn_layers++;
                L.wq = N("attn_q.weight");
                L.wk = N("attn_k.weight");
                L.wv = N("attn_v.weight");
                L.wo = N("attn_output.weight");
                L.q_norm = N("attn_q_norm.weight");
                L.k_norm = N("attn_k_norm.weight");
            } else {
                L.conv = N("shortconv.conv.weight");
                L.in_proj = N("shortconv.in_proj.weight");
                L.out_proj = N("shortconv.out_proj.weight");
                if (L.conv->type != GT_F32 || (int)L.conv->ne[0] != hp.l_cache)
                    throw std::runtime_error("unexpected shortconv.conv layout");
            }
            L.is_moe = il >= hp.n_dense;
            if (L.is_moe) {
                L.router = N("ffn_gate_inp.weight");
                L.exp_bias = N("exp_probs_b.bias");
                L.gate_e = N("ffn_gate_exps.weight");
                L.up_e = N("ffn_up_exps.weight");
                L.down_e = N("ffn_down_exps.weight");
                if (L.router->type != GT_F32 || L.exp_bias->type != GT_F32)
                    throw std::runtime_error("router/bias must be F32");
            } else {
                L.gate = N("ffn_gate.weight");
                L.up = N("ffn_up.weight");
                L.down = N("ffn_down.weight");
            }
        }
    }
};

}  // namespace lfm2
