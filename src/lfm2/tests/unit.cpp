#include "lfm2_cli.hpp"
#include "lfm2_vocab.hpp"
#include <functional>
#include <iostream>

using namespace lfm2;
static void check(bool ok) { if (!ok) throw std::runtime_error("unit check failed"); }
static void rejects(const std::function<void()>& f) {
    try { f(); } catch (const std::exception&) { return; }
    throw std::runtime_error("expected rejection");
}
static Options args(std::initializer_list<const char*> a) {
    std::vector<char*> v;
    for (auto s : a) v.push_back(const_cast<char*>(s));
    return options((int)v.size(), v.data());
}

int main() {
    try {
        Gguf g;
        auto& tokens = g.str_arrays["tokenizer.ggml.tokens"];
        auto& types = g.int_arrays["tokenizer.ggml.token_type"];
        for (auto cp : byte_to_cp()) {
            std::string s; utf8_append(s, cp); tokens.push_back(s); types.push_back(1);
        }
        for (const auto* s : {"<|startoftext|>", "<|im_end|>", "<|endoftext|>", "<|im_start|>"}) {
            tokens.emplace_back(s); types.push_back(3);
        }
        g.str_arrays["tokenizer.ggml.merges"] = {};
        g.strs["tokenizer.ggml.pre"] = "lfm2";
        g.ints["tokenizer.ggml.bos_token_id"] = 256;
        g.ints["tokenizer.ggml.eos_token_id"] = 257;
        Tokenizer t; t.init(g);
        check(t.add_bos && t.bos == 256 && t.eos == 257);
        check(t.encode("a") == std::vector<int>({256, 97}));
        check(t.encode("a", false) == std::vector<int>({97}));
        check(t.encode("", false).empty());
        check(t.encode("") == std::vector<int>({256}));
        check(t.encode("<|im_end|>", false) == std::vector<int>({257}));
        check(t.is_end(257) && t.is_end(258) && !t.is_end(-1) && !t.is_end(256));
        Detok dt; dt.init(tokens, &types);
        std::string text = "Olá, ação! 中 🙂\n";
        std::string decoded;
        for (int id : t.encode(text, false)) decoded += dt.piece(id);
        check(decoded == text);
        g.ints["tokenizer.ggml.add_bos_token"] = 0;
        t.init(g); check(!t.add_bos && t.encode("a") == std::vector<int>({97}));
        check(t.specials.size() == 4);  // init must not retain old state
        g.ints["tokenizer.ggml.add_bos_token"] = 1;
        t.init(g); check(t.encode("a")[0] == 256);
        g.ints["tokenizer.ggml.bos_token_id"] = -1;
        rejects([&] { t.init(g); });
        g.ints["tokenizer.ggml.add_bos_token"] = 0;
        t.init(g); rejects([&] { t.encode("a", true); });
        g.strs["tokenizer.ggml.pre"] = "unsupported";
        rejects([&] { t.init(g); });
        g.strs["tokenizer.ggml.pre"] = "lfm2";
        g.ints["tokenizer.ggml.bos_token_id"] = 256;
        g.ints["tokenizer.ggml.add_bos_token"] = 1;
        t.init(g);

        auto o = args({"bin", "model.gguf", "-p", "oi", "--chat", "-n", "8", "--temp", ".8", "--top-k", "4"});
        check(o.chat && o.prompt && o.n_gen == 8 && o.top_k == 4 && o.temp == .8f);
        check(prompt_ids(o, t) == t.encode("<|im_start|>user\noi<|im_end|>\n<|im_start|>assistant\n"));
        auto old = args({"bin", "model.gguf", "3", "256", "97"});
        check(!old.prompt && prompt_ids(old, t) == std::vector<int>({256, 97}));
        auto tok = args({"bin", "--tok", "model.gguf", "", "--no-bos"});
        check(tok.tok && tok.no_bos);
        for (auto bad : {"-1", "12x", "2147483648"}) rejects([&] { integer(bad); });
        rejects([&] { args({"bin", "m", "-p"}); });
        rejects([&] { args({"bin", "m", "--chat"}); });
        rejects([&] { args({"bin", "m", "-p", "x", "--temp", "nan"}); });
        rejects([&] { args({"bin", "m", "-p", "x", "--temp", "-1"}); });
        old.ids = {999}; rejects([&] { prompt_ids(old, t); });
        old.ids.clear(); rejects([&] { prompt_ids(old, t); });
        o.ctx = 1; rejects([&] { prompt_ids(o, t); });
        std::mt19937 rng(42);
        Options greedy;
        check(sample({1, 3, 3, 0}, greedy, rng) == 1);
        greedy.temp = .5; greedy.top_k = 1;
        for (int i = 0; i < 100; ++i) check(sample({1, 5, 3}, greedy, rng) == 1);
        greedy.top_k = 2;
        for (int i = 0; i < 100; ++i) check(sample({-1000, 5, 3}, greedy, rng) != 0);
        rejects([&] { sample({NAN}, greedy, rng); });
        rejects([&] { sample({}, greedy, rng); });
        TextStream stream;
        stream.push("\xc3"); check(stream.pending == "\xc3");
        stream.push("\xa9"); check(stream.pending.empty());
        stream.push("\xf0\x9f"); check(stream.pending.size() == 2);
        stream.push("\x99\x82"); check(stream.pending.empty());
        stream.push("\xff"); check(stream.pending.empty());
        stream.push("\xc3"); check(stream.pending.size() == 1);
        stream.finish();
        std::cout << "LFM2 unit tests passed\n";
    } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
