// LFM2-MoE support for Strata (local, opt-in). GPT-2 byte-level detokenizer.
#pragma once

#include <array>
#include <string>
#include <unordered_map>
#include <vector>

namespace lfm2 {

// GPT-2 byte -> unicode codepoint table.
inline std::array<uint32_t, 256> byte_to_cp() {
    std::array<uint32_t, 256> t{};
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        bool keep = (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255);
        t[b] = keep ? (uint32_t)b : (uint32_t)(256 + n++);
    }
    return t;
}

inline void utf8_append(std::string& s, uint32_t cp) {
    if (cp < 0x80) s += (char)cp;
    else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) {
        s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F));
    } else {
        s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 0x3F));
        s += (char)(0x80 | ((cp >> 6) & 0x3F)); s += (char)(0x80 | (cp & 0x3F));
    }
}

// Decode one codepoint at s[i]; advances i. Invalid bytes are returned as-is.
inline uint32_t utf8_next(const std::string& s, size_t& i) {
    unsigned char c = (unsigned char)s[i];
    int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
    if (i + len > s.size()) len = 1;
    uint32_t cp = len == 1 ? c : len == 2 ? (c & 0x1F) : len == 3 ? (c & 0x0F) : (c & 0x07);
    for (int k = 1; k < len; ++k) cp = (cp << 6) | ((unsigned char)s[i + k] & 0x3F);
    i += len;
    return cp;
}

struct Detok {
    const std::vector<std::string>* tokens = nullptr;
    const std::vector<int64_t>* types = nullptr;  // tokenizer.ggml.token_type (may be null)
    std::unordered_map<uint32_t, uint8_t> inv;

    void init(const std::vector<std::string>& toks, const std::vector<int64_t>* tt) {
        tokens = &toks;
        types = tt;
        auto t = byte_to_cp();
        for (int b = 0; b < 256; ++b) inv[t[b]] = (uint8_t)b;
    }
    bool is_special(int id) const {
        return types && id >= 0 && (size_t)id < types->size() && (*types)[id] == 3;  // CONTROL
    }
    std::string piece(int id) const {
        if (!tokens || id < 0 || (size_t)id >= tokens->size()) return "";
        const std::string& s = (*tokens)[id];
        if (is_special(id)) return s;
        std::string out;
        for (size_t i = 0; i < s.size();) {
            uint32_t cp = utf8_next(s, i);
            auto it = inv.find(cp);
            if (it != inv.end()) out += (char)it->second;
            else utf8_append(out, cp);
        }
        return out;
    }
};

}  // namespace lfm2
