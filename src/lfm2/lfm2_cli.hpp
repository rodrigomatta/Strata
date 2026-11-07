// Shared command line, sampling and UTF-8 streaming for the opt-in LFM2 runners.
#pragma once
#include "lfm2_tokenizer.hpp"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <numeric>
#include <random>

namespace lfm2 {
struct Options {
    std::string path, text;
    std::vector<int> ids;
    int n_gen = 32, top_k = 0, ctx = 4096;
    float temp = 0;
    bool tok = false, chat = false, prompt = false, no_bos = false, worker = false;
};

inline int integer(const std::string& s) {
    size_t used = 0;
    long long n = std::stoll(s, &used);
    if (used != s.size() || n < 0 || n > std::numeric_limits<int>::max())
        throw std::runtime_error("invalid nonnegative integer: " + s);
    return (int)n;
}

inline const char* usage() {
    return "uso: <bin> <gguf> <n_gen> <ids...>\n"
           "     <bin> <gguf> -p \"prompt\" [--chat] [-n 32] [--temp 0] [--top-k 0]\n"
           "     <bin> --tok <gguf> \"texto\" [--no-bos]\n"
           "     <bin> <gguf> --worker  (IPC privado da GUI)\n"
           "--chat: um turno ChatML; greedy por padrao; LFM2_CTX define o contexto.\n";
}

inline Options options(int argc, char** argv) {
    Options o;
    if (argc < 2) throw std::runtime_error(usage());
    if (const char* ctx = std::getenv("LFM2_CTX")) o.ctx = integer(ctx);
    if (o.ctx == 0) throw std::runtime_error("LFM2_CTX must be positive");
    if (std::string(argv[1]) == "--tok") {
        if (argc != 4 && !(argc == 5 && std::string(argv[4]) == "--no-bos"))
            throw std::runtime_error(usage());
        o.tok = true; o.path = argv[2]; o.text = argv[3]; o.no_bos = argc == 5;
        return o;
    }
    if (std::string(argv[1]) == "--help") throw std::runtime_error(usage());
    o.path = argv[1];
    if (argc == 3 && std::string(argv[2]) == "--worker") { o.worker = true; return o; }
    if (argc >= 4 && argv[2][0] != '-') {
        o.n_gen = integer(argv[2]);
        for (int i = 3; i < argc; ++i) o.ids.push_back(integer(argv[i]));
        return o;
    }
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--chat") { o.chat = true; continue; }
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        std::string v = argv[++i];
        if (a == "-p") { o.text = v; o.prompt = true; }
        else if (a == "-n") o.n_gen = integer(v);
        else if (a == "--top-k") o.top_k = integer(v);
        else if (a == "--temp") {
            size_t used = 0; o.temp = std::stof(v, &used);
            if (used != v.size() || !std::isfinite(o.temp) || o.temp < 0)
                throw std::runtime_error("temperature must be finite and nonnegative");
        } else throw std::runtime_error("unknown option: " + a);
    }
    if (!o.prompt) throw std::runtime_error("-p is required for text/chat mode");
    return o;
}

inline int tokenize_only(const Options& o) {
    Gguf g; g.open(o.path);
    Tokenizer t; t.init(g);
    std::fprintf(stderr, "pre=%s bos=%d eos=%d im_end=%d endoftext=%d add_bos=%d\n",
                 t.pre.c_str(), t.bos, t.eos, t.im_end, t.endoftext, t.add_bos);
    auto ids = t.encode(o.text, t.add_bos && !o.no_bos);
    for (size_t i = 0; i < ids.size(); ++i) std::printf("%s%d", i ? " " : "", ids[i]);
    std::puts("");
    return 0;
}

inline std::vector<int> prompt_ids(const Options& o, const Tokenizer& t) {
    auto ids = o.ids;
    if (o.prompt) {
        if (o.chat) {
            if (!t.tok2id.count("<|im_start|>") || t.im_end < 0)
                throw std::runtime_error("ChatML tokens missing");
            ids = t.encode("<|im_start|>user\n" + o.text +
                           "<|im_end|>\n<|im_start|>assistant\n");
        } else ids = t.encode(o.text);
    }
    if (ids.empty()) throw std::runtime_error("empty prompt: supply text or token IDs");
    for (int id : ids)
        if (id < 0 || id >= (int)t.toks.size()) throw std::runtime_error("token ID outside vocabulary");
    if (ids.size() + (size_t)o.n_gen > (size_t)o.ctx)
        throw std::runtime_error("contexto pequeno demais");
    return ids;
}

inline std::vector<int> top_tokens(const std::vector<float>& logits, int k) {
    std::vector<int> ids(logits.size());
    std::iota(ids.begin(), ids.end(), 0);
    k = std::min(k, (int)ids.size());
    std::partial_sort(ids.begin(), ids.begin() + k, ids.end(), [&](int a, int b) {
        return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
    });
    ids.resize(k);
    return ids;
}

inline int sample(const std::vector<float>& logits, const Options& o, std::mt19937& rng) {
    if (logits.empty()) throw std::runtime_error("empty logits");
    for (float v : logits) if (!std::isfinite(v)) throw std::runtime_error("non-finite logits");
    if (o.temp == 0) return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    auto ids = top_tokens(logits, o.top_k ? o.top_k : (int)logits.size());
    std::vector<double> weights;
    for (int id : ids) weights.push_back(std::exp(((double)logits[id] - logits[ids[0]]) / o.temp));
    std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
    return ids[dist(rng)];
}

struct TextStream {
    std::string pending;
    void push(const std::string& bytes) {
        pending += bytes;
        std::string ready;
        size_t i = 0;
        while (i < pending.size()) {
            unsigned char c = pending[i];
            size_t n = c < 0x80 ? 1 : c >= 0xc2 && c <= 0xdf ? 2 :
                       c >= 0xe0 && c <= 0xef ? 3 : c >= 0xf0 && c <= 0xf4 ? 4 : 0;
            bool valid = n != 0;
            uint32_t cp = n == 1 ? c : c & (n == 2 ? 0x1f : n == 3 ? 0xf : 7);
            for (size_t k = 1; valid && k < n && i + k < pending.size(); ++k) {
                unsigned char b = pending[i + k];
                valid = (b & 0xc0) == 0x80;
                cp = (cp << 6) | (b & 0x3f);
            }
            if (valid && i + n > pending.size()) break;
            valid = valid && !(n == 2 && cp < 0x80) && !(n == 3 && cp < 0x800) &&
                    !(n == 4 && cp < 0x10000) && cp <= 0x10ffff && !(cp >= 0xd800 && cp <= 0xdfff);
            if (!valid) { ready += "\xef\xbf\xbd"; ++i; continue; }
            ready.append(pending, i, n);
            i += n;
        }
        std::fwrite(ready.data(), 1, ready.size(), stdout);
        pending.erase(0, i);
        std::fflush(stdout);
    }
    void finish() {
        // A generation limit may cut a character between tokens.
        if (!pending.empty()) std::fputs("\xef\xbf\xbd", stdout);
        pending.clear(); std::puts("");
    }
};
} // namespace lfm2
