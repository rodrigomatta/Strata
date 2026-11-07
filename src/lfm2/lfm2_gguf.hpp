// LFM2-MoE support for Strata (local, opt-in). Minimal mmap GGUF reader.
// Reads keys, arrays and tensor infos; tensor data stays in the mmap.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace lfm2 {

enum GgmlType : uint32_t { GT_F32 = 0, GT_F16 = 1, GT_Q4_0 = 2, GT_Q6_K = 14 };

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> ne;  // ne[0] = row length (input dim)
    uint32_t type = 0;
    uint64_t offset = 0;       // relative to data start
    const uint8_t* data = nullptr;
    uint64_t nbytes = 0;
    uint64_t nelem() const { uint64_t n = 1; for (auto d : ne) n *= d; return n; }
};

inline uint64_t row_bytes(uint32_t type, uint64_t n) {
    switch (type) {
        case GT_F32: return n * 4;
        case GT_F16: return n * 2;
        case GT_Q4_0: return n / 32 * 18;
        case GT_Q6_K: return n / 256 * 210;
    }
    throw std::runtime_error("unsupported ggml type " + std::to_string(type));
}

struct Gguf {
    std::map<std::string, int64_t> ints;
    std::map<std::string, double> floats;
    std::map<std::string, std::string> strs;
    std::map<std::string, std::vector<int64_t>> int_arrays;
    std::map<std::string, std::vector<double>> float_arrays;
    std::map<std::string, std::vector<std::string>> str_arrays;
    std::map<std::string, TensorInfo> tensors;
    std::vector<std::string> tensor_order;

    const uint8_t* base = nullptr;
    size_t size = 0;
    int fd = -1;

    Gguf() = default;
    Gguf(const Gguf&) = delete;
    Gguf& operator=(const Gguf&) = delete;
    ~Gguf() {
        if (base) munmap(const_cast<uint8_t*>(base), size);
        if (fd >= 0) close(fd);
    }

    void open(const std::string& path) {
        fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path);
        struct stat st{};
        fstat(fd, &st);
        size = (size_t)st.st_size;
        void* p = mmap(nullptr, size, PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        base = (const uint8_t*)p;
        parse();
    }

    int64_t get_int(const std::string& k, int64_t def = -1) const {
        auto it = ints.find(k);
        return it == ints.end() ? def : it->second;
    }
    double get_float(const std::string& k, double def = 0) const {
        auto it = floats.find(k);
        return it == floats.end() ? def : it->second;
    }
    std::string get_str(const std::string& k) const {
        auto it = strs.find(k);
        return it == strs.end() ? std::string() : it->second;
    }
    const TensorInfo* tensor(const std::string& n) const {
        auto it = tensors.find(n);
        return it == tensors.end() ? nullptr : &it->second;
    }
    const TensorInfo& need(const std::string& n) const {
        auto t = tensor(n);
        if (!t) throw std::runtime_error("missing tensor " + n);
        return *t;
    }

private:
    size_t pos = 0;
    template <class T> T rd() {
        if (pos + sizeof(T) > size) throw std::runtime_error("gguf truncated");
        T v; std::memcpy(&v, base + pos, sizeof(T)); pos += sizeof(T); return v;
    }
    std::string rd_str() {
        uint64_t n = rd<uint64_t>();
        if (pos + n > size) throw std::runtime_error("gguf truncated string");
        std::string s((const char*)base + pos, n); pos += n; return s;
    }
    // scalar of type t -> classified
    void rd_scalar(uint32_t t, int64_t& iv, double& fv, std::string& sv, int& kind) {
        switch (t) {
            case 0: iv = rd<uint8_t>(); kind = 0; break;
            case 1: iv = rd<int8_t>(); kind = 0; break;
            case 2: iv = rd<uint16_t>(); kind = 0; break;
            case 3: iv = rd<int16_t>(); kind = 0; break;
            case 4: iv = rd<uint32_t>(); kind = 0; break;
            case 5: iv = rd<int32_t>(); kind = 0; break;
            case 6: fv = rd<float>(); kind = 1; break;
            case 7: iv = rd<uint8_t>(); kind = 0; break;
            case 8: sv = rd_str(); kind = 2; break;
            case 10: iv = (int64_t)rd<uint64_t>(); kind = 0; break;
            case 11: iv = rd<int64_t>(); kind = 0; break;
            case 12: fv = rd<double>(); kind = 1; break;
            default: throw std::runtime_error("bad gguf value type " + std::to_string(t));
        }
    }
    void parse() {
        if (size < 24 || std::memcmp(base, "GGUF", 4) != 0) throw std::runtime_error("not a GGUF file");
        pos = 4;
        uint32_t version = rd<uint32_t>();
        if (version < 2) throw std::runtime_error("GGUF v1 unsupported");
        uint64_t n_tensors = rd<uint64_t>();
        uint64_t n_kv = rd<uint64_t>();
        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = rd_str();
            uint32_t t = rd<uint32_t>();
            int64_t iv = 0; double fv = 0; std::string sv; int kind = 0;
            if (t == 9) {
                uint32_t at = rd<uint32_t>();
                uint64_t n = rd<uint64_t>();
                std::vector<int64_t> ia; std::vector<double> fa; std::vector<std::string> sa;
                for (uint64_t j = 0; j < n; ++j) {
                    rd_scalar(at, iv, fv, sv, kind);
                    if (kind == 0) ia.push_back(iv);
                    else if (kind == 1) fa.push_back(fv);
                    else sa.push_back(std::move(sv));
                }
                if (at == 8) str_arrays[key] = std::move(sa);
                else if (at == 6 || at == 12) float_arrays[key] = std::move(fa);
                else int_arrays[key] = std::move(ia);
            } else {
                rd_scalar(t, iv, fv, sv, kind);
                if (kind == 0) ints[key] = iv;
                else if (kind == 1) floats[key] = fv;
                else strs[key] = sv;
            }
        }
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo ti;
            ti.name = rd_str();
            uint32_t nd = rd<uint32_t>();
            for (uint32_t d = 0; d < nd; ++d) ti.ne.push_back(rd<uint64_t>());
            ti.type = rd<uint32_t>();
            ti.offset = rd<uint64_t>();
            tensor_order.push_back(ti.name);
            tensors[ti.name] = std::move(ti);
        }
        uint64_t align = (uint64_t)get_int("general.alignment", 32);
        size_t data_start = (pos + align - 1) / align * align;
        for (auto& [n, ti] : tensors) {
            uint64_t rows = ti.nelem() / ti.ne[0];
            ti.nbytes = row_bytes(ti.type, ti.ne[0]) * rows;
            if (data_start + ti.offset + ti.nbytes > size) throw std::runtime_error("tensor out of file: " + n);
            ti.data = base + data_start + ti.offset;
        }
    }
};

}  // namespace lfm2
