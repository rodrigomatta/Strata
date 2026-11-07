// LFM2 BPE tokenizer (GPT-2 ByteLevel + llama3-style pre-split), written for Strata.
// Behaviour follows llama.cpp (MIT); the Unicode category table is taken from it.
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "lfm2_gguf.hpp"

namespace lfm2 {

#include "lfm2_unicode_data.inc"

enum : uint16_t {
    CF_UNDEF = 0x1, CF_NUM = 0x2, CF_LETTER = 0x4, CF_SEP = 0x8, CF_ACCENT = 0x10,
    CF_PUNCT = 0x20, CF_SYMBOL = 0x40, CF_CONTROL = 0x80, CF_WS = 0x100,
};
static const uint32_t kOutOfRange = 0xFFFFFFFFu;

inline const std::vector<uint16_t>& cpt_flag_table() {
    static std::vector<uint16_t> t = [] {
        std::vector<uint16_t> v(0x110000, 0);
        const size_t n = sizeof(kLfm2CptRanges) / sizeof(kLfm2CptRanges[0]);
        for (size_t i = 0; i + 1 < n; ++i)
            for (uint32_t c = kLfm2CptRanges[i][0]; c < kLfm2CptRanges[i + 1][0] && c < 0x110000; ++c)
                v[c] = (uint16_t)kLfm2CptRanges[i][1];
        for (uint32_t c : kLfm2Whitespace) v[c] |= CF_WS;
        return v;
    }();
    return t;
}

inline std::vector<uint32_t> utf8_to_cpts(const std::string& s) {
    std::vector<uint32_t> out;
    out.reserve(s.size());
    const unsigned char* p = (const unsigned char*)s.data();
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned c = p[i];
        int len = 0; uint32_t cp = 0;
        if (c < 0x80) { cp = c; len = 1; }
        else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; len = 2; }
        else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; len = 3; }
        else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; len = 4; }
        bool ok = len > 0 && i + len <= n;
        for (int k = 1; ok && k < len; ++k) {
            if ((p[i + k] & 0xC0) != 0x80) ok = false;
            else cp = (cp << 6) | (p[i + k] & 0x3F);
        }
        if (!ok) { out.push_back(0xFFFD); i += 1; continue; }
        out.push_back(cp); i += len;
    }
    return out;
}

inline void append_utf8(std::string& o, uint32_t cp) {
    if (cp < 0x80) o += (char)cp;
    else if (cp < 0x800) { o += (char)(0xC0 | (cp >> 6)); o += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        o += (char)(0xE0 | (cp >> 12)); o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F));
    } else {
        o += (char)(0xF0 | (cp >> 18)); o += (char)(0x80 | ((cp >> 12) & 0x3F));
        o += (char)(0x80 | ((cp >> 6) & 0x3F)); o += (char)(0x80 | (cp & 0x3F));
    }
}

// llama3 pre-tokenizer split over cpts[0..n). Returns lengths of the pieces.
inline std::vector<size_t> split_llama3(const std::vector<uint32_t>& cp) {
    const auto& T = cpt_flag_table();
    const size_t end = cp.size();
    auto cpt = [&](size_t p) -> uint32_t { return p < end ? cp[p] : kOutOfRange; };
    auto fl = [&](size_t p) -> uint16_t { return p < end ? (cp[p] < 0x110000 ? T[cp[p]] : 0) : 0; };
    auto low = [](uint32_t c) -> uint32_t { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
    std::vector<size_t> out;
    size_t prev = 0;
    auto emit = [&](size_t x) { if (x > prev) out.push_back(x - prev); prev = x; };
    for (size_t pos = 0; pos < end;) {
        const uint32_t c = cpt(pos);
        const uint16_t f = fl(pos);
        if (c == '\'' && pos + 1 < end) {
            uint32_t n1 = low(cpt(pos + 1));
            if (n1 == 's' || n1 == 't' || n1 == 'm' || n1 == 'd') { pos += 2; emit(pos); continue; }
            if (pos + 2 < end) {
                uint32_t n2 = low(cpt(pos + 2));
                if ((n1 == 'r' && n2 == 'e') || (n1 == 'v' && n2 == 'e') || (n1 == 'l' && n2 == 'l')) {
                    pos += 3; emit(pos); continue;
                }
            }
        }
        if (!(c == '\r' || c == '\n' || (f & CF_NUM))) {
            if ((f & CF_LETTER) || (fl(pos + 1) & CF_LETTER)) {
                pos++;
                while (fl(pos) & CF_LETTER) pos++;
                emit(pos); continue;
            }
        }
        if (f & CF_NUM) {
            size_t ini = pos;
            while (fl(pos) & CF_NUM) {
                if (++pos - ini >= 3) { emit(pos); ini = pos; }
            }
            emit(pos); continue;
        }
        uint16_t f2 = (c == ' ') ? fl(pos + 1) : f;
        if (!(f2 & (CF_WS | CF_LETTER | CF_NUM)) && f) {
            pos += (c == ' ');
            while (!(f2 & (CF_WS | CF_LETTER | CF_NUM)) && f2) f2 = fl(++pos);
            uint32_t c2 = cpt(pos);
            while (c2 == '\r' || c2 == '\n') c2 = cpt(++pos);
            emit(pos); continue;
        }
        size_t nws = 0, last_rn = 0;
        while (fl(pos + nws) & CF_WS) {
            uint32_t c2 = cpt(pos + nws);
            if (c2 == '\r' || c2 == '\n') last_rn = pos + nws + 1;
            nws++;
        }
        if (last_rn > 0) { pos = last_rn; emit(pos); continue; }
        if (nws > 1 && cpt(pos + nws) != kOutOfRange) { pos += nws - 1; emit(pos); continue; }
        if (nws > 0) { pos += nws; emit(pos); continue; }
        emit(++pos);
    }
    return out;
}

struct Tokenizer {
    std::vector<std::string> toks;
    std::vector<int64_t> tt;
    std::unordered_map<std::string, int> tok2id;
    std::unordered_map<std::string, int> merge_rank;
    std::vector<int> specials;  // sorted by text length, longest first
    std::string byte2u[256];
    int bos = -1, eos = -1, im_end = -1, endoftext = -1;
    bool add_bos = false;
    std::string pre;

    void init(const Gguf& g) {
        tok2id.clear(); merge_rank.clear(); specials.clear();
        toks = g.str_arrays.at("tokenizer.ggml.tokens");
        tt = g.int_arrays.at("tokenizer.ggml.token_type");
        if (tt.size() != toks.size()) throw std::runtime_error("token types/vocabulary size mismatch");
        pre = g.get_str("tokenizer.ggml.pre");
        if (pre != "llama3" && pre != "lfm2")
            throw std::runtime_error("unsupported tokenizer pre: " + pre);
        const auto& merges = g.str_arrays.at("tokenizer.ggml.merges");
        bos = (int)g.get_int("tokenizer.ggml.bos_token_id", -1);
        eos = (int)g.get_int("tokenizer.ggml.eos_token_id", -1);
        // Both supported pre-tokenizers default to BOS in llama.cpp. Honor an
        // explicit GGUF false as well as true; do not hard-code the token ID.
        add_bos = g.get_int("tokenizer.ggml.add_bos_token", bos >= 0 ? 1 : 0) != 0;
        for (int id : {bos, eos})
            if (id < -1 || id >= (int)toks.size()) throw std::runtime_error("invalid BOS/EOS ID");
        if (add_bos && bos < 0) throw std::runtime_error("add_bos requires a BOS token");
        tok2id.reserve(toks.size() * 2);
        for (size_t i = 0; i < toks.size(); ++i) tok2id.emplace(toks[i], (int)i);
        auto end = tok2id.find("<|im_end|>");
        im_end = end == tok2id.end() ? -1 : end->second;
        end = tok2id.find("<|endoftext|>");
        endoftext = end == tok2id.end() ? -1 : end->second;
        merge_rank.reserve(merges.size() * 2);
        for (size_t i = 0; i < merges.size(); ++i) merge_rank.emplace(merges[i], (int)i);
        for (size_t i = 0; i < toks.size(); ++i) {
            int64_t t = i < tt.size() ? tt[i] : 1;
            if (t == 2 || t == 3 || t == 4) specials.push_back((int)i);
        }
        std::stable_sort(specials.begin(), specials.end(),
                         [&](int a, int b) { return toks[a].size() > toks[b].size(); });
        // GPT-2 bytes_to_unicode
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            bool direct = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
            uint32_t cp = direct ? (uint32_t)b : (uint32_t)(256 + n++);
            byte2u[b].clear();
            append_utf8(byte2u[b], cp);
        }
    }

    void bpe_word(const std::string& w, std::vector<int>& out) const {
        auto hit = tok2id.find(w);
        if (hit != tok2id.end()) { out.push_back(hit->second); return; }  // ignore_merges
        std::vector<std::string> sym;
        for (size_t i = 0; i < w.size();) {
            unsigned char c = (unsigned char)w[i];
            size_t len = c < 0x80 ? 1 : (c & 0xE0) == 0xC0 ? 2 : (c & 0xF0) == 0xE0 ? 3 : 4;
            len = std::min(len, w.size() - i);
            sym.push_back(w.substr(i, len));
            i += len;
        }
        while (sym.size() > 1) {
            int best = -1, best_rank = 0x7fffffff;
            for (size_t i = 0; i + 1 < sym.size(); ++i) {
                auto m = merge_rank.find(sym[i] + " " + sym[i + 1]);
                if (m != merge_rank.end() && m->second < best_rank) { best_rank = m->second; best = (int)i; }
            }
            if (best < 0) break;
            sym[best] += sym[best + 1];
            sym.erase(sym.begin() + best + 1);
        }
        for (const auto& s : sym) {
            auto t = tok2id.find(s);
            if (t != tok2id.end()) { out.push_back(t->second); continue; }
            throw std::runtime_error("BPE symbol missing from vocabulary");
        }
    }

    void encode_text(const std::string& text, std::vector<int>& out) const {
        std::vector<uint32_t> cp = utf8_to_cpts(text);
        std::vector<size_t> lens = split_llama3(cp);
        size_t off = 0;
        std::string utf8, bl;
        for (size_t L : lens) {
            utf8.clear(); bl.clear();
            for (size_t k = 0; k < L; ++k) append_utf8(utf8, cp[off + k]);
            off += L;
            for (unsigned char b : utf8) bl += byte2u[b];
            bpe_word(bl, out);
        }
    }

    std::vector<int> encode(const std::string& text, bool add_bos, bool parse_special = true) const {
        // fragments: id >= 0 is a special token, id < 0 is raw text [a,b)
        struct Frag { int id; size_t a, b; };
        std::vector<Frag> fr{{-1, 0, text.size()}};
        for (int sid : specials) {
            int64_t t = tt[sid];
            if (!parse_special && t != 4) continue;
            const std::string& s = toks[sid];
            if (s.empty()) continue;
            std::vector<Frag> nf;
            nf.reserve(fr.size());
            for (const Frag& f : fr) {
                if (f.id >= 0) { nf.push_back(f); continue; }
                size_t a = f.a;
                while (a < f.b) {
                    size_t p = text.find(s, a);
                    if (p == std::string::npos || p + s.size() > f.b) break;
                    if (p > a) nf.push_back({-1, a, p});
                    nf.push_back({sid, p, p + s.size()});
                    a = p + s.size();
                }
                if (a < f.b) nf.push_back({-1, a, f.b});
            }
            fr.swap(nf);
        }
        std::vector<int> out;
        if (add_bos) {
            if (bos < 0) throw std::runtime_error("BOS requested but absent");
            out.push_back(bos);
        }
        for (const Frag& f : fr) {
            if (f.id >= 0) out.push_back(f.id);
            else encode_text(text.substr(f.a, f.b - f.a), out);
        }
        return out;
    }
    std::vector<int> encode(const std::string& text) const { return encode(text, add_bos); }
    bool is_end(int id) const { return id >= 0 && (id == eos || id == im_end || id == endoftext); }
};

}  // namespace lfm2
