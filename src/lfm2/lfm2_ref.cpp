// Referência em CPU do LFM2-MoE (lfm2moe), só para validar o motor de GPU.
// Uso: lfm2_ref <gguf> <n_gen> <id>...
#include "lfm2_model.hpp"
#include "lfm2_vocab.hpp"
#include "lfm2_cli.hpp"
#include <immintrin.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace lfm2;

static inline float f16(const uint8_t* p) { uint16_t h; std::memcpy(&h, p, 2); return _cvtsh_ss(h); }

static float dot_q4_0(const uint8_t* row, const float* x, int n) {
    float s = 0.f;
    for (int b = 0; b < n / 32; ++b) {
        const uint8_t* blk = row + b * 18;
        const float d = f16(blk);
        const uint8_t* qs = blk + 2;
        const float* xb = x + b * 32;
        float bs = 0.f;
        for (int j = 0; j < 16; ++j)
            bs += float((qs[j] & 15) - 8) * xb[j] + float((qs[j] >> 4) - 8) * xb[j + 16];
        s += d * bs;
    }
    return s;
}

static void deq_q6_k(const uint8_t* blk, float* y) {
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    const float d = f16(blk + 208);
    for (int n = 0; n < 256; n += 128) {
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            int q1 = int((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            int q2 = int((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            int q3 = int((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            int q4 = int((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[l] = d * sc[is] * q1;
            y[l + 32] = d * sc[is + 2] * q2;
            y[l + 64] = d * sc[is + 4] * q3;
            y[l + 96] = d * sc[is + 6] * q4;
        }
        y += 128; ql += 64; qh += 32; sc += 8;
    }
}

static float dot_q6_k(const uint8_t* row, const float* x, int n) {
    float s = 0.f, tmp[256];
    for (int b = 0; b < n / 256; ++b) {
        deq_q6_k(row + b * 210, tmp);
        const float* xb = x + b * 256;
        float bs = 0.f;
        for (int i = 0; i < 256; ++i) bs += tmp[i] * xb[i];
        s += bs;
    }
    return s;
}

static float dot_row(uint32_t type, const uint8_t* row, const float* x, int n) {
    switch (type) {
    case GT_Q4_0: return dot_q4_0(row, x, n);
    case GT_Q6_K: return dot_q6_k(row, x, n);
    case GT_F32: { const float* w = (const float*)row; float s = 0; for (int i = 0; i < n; ++i) s += w[i] * x[i]; return s; }
    case GT_F16: { float s = 0; for (int i = 0; i < n; ++i) s += f16(row + 2 * i) * x[i]; return s; }
    }
    throw std::runtime_error("tipo de tensor sem suporte: " + std::to_string(type));
}

// y[rows] = W[rows x n_in] * x ; base aponta para a primeira linha.
static void matvec(const TensorInfo* t, const uint8_t* base, int rows, int n_in, const float* x, float* y) {
    const uint64_t rb = row_bytes(t->type, n_in);
    const uint32_t ty = t->type;
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) y[r] = dot_row(ty, base + rb * r, x, n_in);
}
static void matvec(const TensorInfo* t, const float* x, float* y) {
    matvec(t, t->data, (int)t->ne[1], (int)t->ne[0], x, y);
}

static void rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
    double ss = 0; for (int i = 0; i < n; ++i) ss += double(x[i]) * x[i];
    const float r = 1.f / std::sqrt(float(ss / n) + eps);
    for (int i = 0; i < n; ++i) y[i] = x[i] * r * w[i];
}

static inline float silu(float v) { return v / (1.f + std::exp(-v)); }

struct State {
    std::vector<std::vector<float>> s0, s1;     // estado do shortconv por camada
    std::vector<std::vector<float>> kc, vc;     // KV cache por camada de atenção
    int pos = 0;
};

struct Runner {
    Model& m;
    const Hparams& hp;
    State st;
    std::vector<float> x, xn, h, tmp, bcx, q, k, v, att, g, u, emb;
    Runner(Model& mm) : m(mm), hp(mm.hp) {
        const int E = hp.n_embd;
        st.s0.assign(hp.n_layer, std::vector<float>(E, 0.f));
        st.s1.assign(hp.n_layer, std::vector<float>(E, 0.f));
        st.kc.resize(m.n_attn_layers); st.vc.resize(m.n_attn_layers);
        x.resize(E); xn.resize(E); h.resize(E); tmp.resize(std::max(E, hp.n_ff)); bcx.resize(3 * E);
        q.resize(hp.n_head * hp.head_dim); k.resize(hp.n_head_kv * hp.head_dim); v.resize(k.size());
        att.resize(q.size()); g.resize(std::max(hp.n_ff, hp.n_ff_exp)); u.resize(g.size()); emb.resize(E);
    }

    void embed(int id) {
        const TensorInfo* t = m.tok_embd;
        const int E = hp.n_embd;
        const uint8_t* row = t->data + row_bytes(t->type, E) * (uint64_t)id;
        if (t->type == GT_Q6_K) { for (int b = 0; b < E / 256; ++b) deq_q6_k(row + b * 210, x.data() + b * 256); }
        else if (t->type == GT_F32) std::memcpy(x.data(), row, E * 4);
        else if (t->type == GT_F16) { for (int i = 0; i < E; ++i) x[i] = f16(row + 2 * i); }
        else if (t->type == GT_Q4_0) {
            for (int b = 0; b < E / 32; ++b) { const uint8_t* blk = row + b * 18; float d = f16(blk);
                for (int j = 0; j < 16; ++j) { x[b*32+j] = ((blk[2+j] & 15) - 8) * d; x[b*32+j+16] = ((blk[2+j] >> 4) - 8) * d; } }
        } else throw std::runtime_error("tipo do embedding sem suporte");
    }

    void shortconv(int il, const Layer& L, const float* in, float* out) {
        const int E = hp.n_embd;
        matvec(L.in_proj, in, bcx.data());
        const float* B = bcx.data(); const float* C = B + E; const float* X = C + E;
        const float* w = (const float*)L.conv->data;
        auto& s0 = st.s0[il]; auto& s1 = st.s1[il];
        for (int c = 0; c < E; ++c) {
            float bx = B[c] * X[c];
            float o = w[c * 3 + 0] * s0[c] + w[c * 3 + 1] * s1[c] + w[c * 3 + 2] * bx;
            s0[c] = s1[c]; s1[c] = bx;
            tmp[c] = C[c] * o;
        }
        matvec(L.out_proj, tmp.data(), out);
    }

    void rope(float* vec, int pos) {
        const int D = hp.head_dim, half = D / 2;
        for (int i = 0; i < half; ++i) {
            float theta = float(pos) * std::pow(hp.rope_base, -2.f * i / D);
            float c = std::cos(theta), s = std::sin(theta);
            float a = vec[i], b = vec[i + half];
            vec[i] = a * c - b * s; vec[i + half] = a * s + b * c;
        }
    }

    void attention(const Layer& L, const float* in, float* out) {
        const int D = hp.head_dim, NH = hp.n_head, NKV = hp.n_head_kv, pos = st.pos;
        matvec(L.wq, in, q.data()); matvec(L.wk, in, k.data()); matvec(L.wv, in, v.data());
        const float* qn = (const float*)L.q_norm->data; const float* kn = (const float*)L.k_norm->data;
        float t[256];
        for (int hh = 0; hh < NH; ++hh) { rmsnorm(&q[hh * D], qn, t, D, hp.eps); std::memcpy(&q[hh * D], t, D * 4); rope(&q[hh * D], pos); }
        for (int hh = 0; hh < NKV; ++hh) { rmsnorm(&k[hh * D], kn, t, D, hp.eps); std::memcpy(&k[hh * D], t, D * 4); rope(&k[hh * D], pos); }
        auto& kc = st.kc[L.attn_index]; auto& vc = st.vc[L.attn_index];
        kc.insert(kc.end(), k.begin(), k.end()); vc.insert(vc.end(), v.begin(), v.end());
        const int T = pos + 1, KVW = NKV * D, grp = NH / NKV;
        const float scale = 1.f / std::sqrt(float(D));
#pragma omp parallel for schedule(static)
        for (int hh = 0; hh < NH; ++hh) {
            const int kh = hh / grp;
            std::vector<float> sc(T);
            float mx = -1e30f;
            for (int j = 0; j < T; ++j) {
                const float* kj = &kc[(size_t)j * KVW + kh * D];
                float s = 0; for (int d = 0; d < D; ++d) s += q[hh * D + d] * kj[d];
                sc[j] = s * scale; mx = std::max(mx, sc[j]);
            }
            float sum = 0; for (int j = 0; j < T; ++j) { sc[j] = std::exp(sc[j] - mx); sum += sc[j]; }
            float* o = &att[hh * D];
            for (int d = 0; d < D; ++d) o[d] = 0;
            for (int j = 0; j < T; ++j) { const float* vj = &vc[(size_t)j * KVW + kh * D]; float p = sc[j] / sum;
                for (int d = 0; d < D; ++d) o[d] += p * vj[d]; }
        }
        matvec(L.wo, att.data(), out);
    }

    void dense_ffn(const Layer& L, const float* in, float* out) {
        const int F = (int)L.gate->ne[1];
        matvec(L.gate, in, g.data()); matvec(L.up, in, u.data());
        for (int i = 0; i < F; ++i) g[i] = silu(g[i]) * u[i];
        matvec(L.down, g.data(), out);
    }

    void moe(const Layer& L, const float* in, float* out) {
        const int E = hp.n_embd, NE = hp.n_expert, K = hp.n_expert_used, F = hp.n_ff_exp;
        std::vector<float> logit(NE), p(NE), sel(NE);
        matvec(L.router, in, logit.data());
        const float* bias = L.exp_bias ? (const float*)L.exp_bias->data : nullptr;
        for (int e = 0; e < NE; ++e) { p[e] = 1.f / (1.f + std::exp(-logit[e])); sel[e] = p[e] + (bias ? bias[e] : 0.f); }
        std::vector<int> idx(NE); for (int e = 0; e < NE; ++e) idx[e] = e;
        std::partial_sort(idx.begin(), idx.begin() + K, idx.end(), [&](int a, int b) { return sel[a] > sel[b]; });
        float wsum = 0; for (int i = 0; i < K; ++i) wsum += p[idx[i]];
        wsum = std::max(wsum, 6.103515625e-5f);
        for (int i = 0; i < E; ++i) out[i] = 0;
        std::vector<float> y(E);
        const uint64_t rb_in = row_bytes(L.gate_e->type, E), rb_ff = row_bytes(L.down_e->type, F);
        for (int i = 0; i < K; ++i) {
            const int e = idx[i]; const float w = p[e] / wsum;
            matvec(L.gate_e, L.gate_e->data + (uint64_t)e * F * rb_in, F, E, in, g.data());
            matvec(L.up_e, L.up_e->data + (uint64_t)e * F * rb_in, F, E, in, u.data());
            for (int j = 0; j < F; ++j) g[j] = silu(g[j]) * u[j];
            matvec(L.down_e, L.down_e->data + (uint64_t)e * E * rb_ff, E, F, g.data(), y.data());
            for (int j = 0; j < E; ++j) out[j] += w * y[j];
        }
    }

    // Processa um token; se logits != null, calcula a cabeça.
    void step(int id, std::vector<float>* logits) {
        const int E = hp.n_embd;
        embed(id);
        for (int il = 0; il < hp.n_layer; ++il) {
            const Layer& L = m.layers[il];
            rmsnorm(x.data(), (const float*)L.attn_norm->data, xn.data(), E, hp.eps);
            if (L.is_attn) attention(L, xn.data(), h.data()); else shortconv(il, L, xn.data(), h.data());
            for (int i = 0; i < E; ++i) h[i] += x[i];
            rmsnorm(h.data(), (const float*)L.ffn_norm->data, xn.data(), E, hp.eps);
            if (L.is_moe) moe(L, xn.data(), x.data()); else dense_ffn(L, xn.data(), x.data());
            for (int i = 0; i < E; ++i) x[i] += h[i];
        }
        st.pos++;
        if (logits) {
            rmsnorm(x.data(), (const float*)m.out_norm->data, xn.data(), E, hp.eps);
            logits->resize(hp.n_vocab);
            matvec(m.tok_embd, m.tok_embd->data, hp.n_vocab, E, xn.data(), logits->data());
        }
    }
};

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { std::fputs(usage(), stdout); return 0; }
        const Options o = options(argc, argv);
        if (o.tok) return tokenize_only(o);
        Model m; m.load(o.path);
        Tokenizer tokenizer; tokenizer.init(m.g);
        const auto ids = prompt_ids(o, tokenizer);
        Detok d; d.init(m.g.str_arrays.at("tokenizer.ggml.tokens"), &m.g.int_arrays.at("tokenizer.ggml.token_type"));
        std::fprintf(stderr, "camadas=%d atencao=%d vocab=%d threads=%d\n", m.hp.n_layer, m.n_attn_layers, m.hp.n_vocab, omp_get_max_threads());
        Runner r(m);
        std::vector<float> logits;
        auto t0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < ids.size(); ++i) r.step(ids[i], i + 1 == ids.size() ? &logits : nullptr);
        auto t1 = std::chrono::steady_clock::now();
        const double tp = std::chrono::duration<double>(t1 - t0).count();
        std::fprintf(stderr, "prefill %zu tok: %.3f s (%.2f tok/s)\n", ids.size(), tp, ids.size() / tp);
        std::string text;
        TextStream stream;
        std::mt19937 rng(std::random_device{}());
        int emitted = 0, decode_steps = 0, stop_id = -1;
        for (int s = 0; s < o.n_gen; ++s) {
            const int next = sample(logits, o, rng);
            if (o.prompt) {
                if (tokenizer.is_end(next)) { stop_id = next; break; }
                if (!d.is_special(next)) stream.push(d.piece(next));
            } else {
                auto top = top_tokens(logits, 5);
                std::printf("passo %d top5:", s);
                for (int id : top) std::printf(" %d(%.3f)", id, logits[id]);
                text += d.piece(next);
                std::printf(" | %s\n", text.c_str()); std::fflush(stdout);
                if (tokenizer.is_end(next)) { stop_id = next; break; }
            }
            ++emitted;
            if (s + 1 < o.n_gen) { r.step(next, &logits); ++decode_steps; }
        }
        if (o.prompt) stream.finish();
        auto t2 = std::chrono::steady_clock::now();
        const double td = std::chrono::duration<double>(t2 - t1).count();
        std::fprintf(stderr, "decode %d passos, %d tokens emitidos: %.3f s (%.2f tok/s)\n",
                     decode_steps, emitted, td, td > 0 ? decode_steps / td : 0);
        if (stop_id >= 0) std::fprintf(stderr, "parada: token %d (%s)\n", stop_id, tokenizer.toks[stop_id].c_str());
        else std::fprintf(stderr, "parada: limite de %d tokens\n", o.n_gen);
    } catch (const std::exception& e) { std::fprintf(stderr, "erro: %s\n", e.what()); return 1; }
    return 0;
}
