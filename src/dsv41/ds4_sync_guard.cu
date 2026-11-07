// The dsv41 launch loops' sync interposer — compiled only by STRATA_DS4_SYNC_GUARD=ON builds (CMake wires this into
// strata-dsv41 with -Wl,--wrap on the symbols below, and the guard's test builds it with the same wraps).  Every
// sync-CAPABLE runtime call the loops can make is wrapped at link time, so the engine's own code is untouched: the
// wrapper tells sync_guard::hit() and passes through to the real call (__real_*, the linker's spelling).  What
// counts is a hit INSIDE a Window — the load's own allocations and copies outside one are its business, and the
// runtime's internal syncs showing up here is the point: those are the implicit stalls nothing else shows.
//
// STRATA_DS4_FORBID_SYNC says what a hit in a window does: unset or "log" one line a kind and a summary at the
// window's end, "0" count silently, "abort" (or any other non-zero) stop right there for a debugger.
#include "ds4_sync_guard.hpp"

#if defined(STRATA_DS4_SYNC_GUARD)

#include <cuda_runtime.h>
#ifdef __HIP__   // (hip_extra maps the calls and the kind values; the kind's TYPE spelling is the runtime's own,
using cudaMemcpyKind = hipMemcpyKind;   // and the wrappers below have to name it)
#endif
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::ds4::sync_guard {

namespace {

// one counter a kind, registered once before main (the wrappers below are the only names); the counts are atomic
// because the prompt path runs a thread a GPU
struct Kind {
    const char* what;
    std::atomic<uint64_t> hits{0};
    std::atomic<bool> logged{false};
};
Kind g_kinds[16];
int g_nkinds = 0;
Kind* find(const char* what) {
    for (int i = 0; i < g_nkinds; ++i)
        if (std::strcmp(g_kinds[i].what, what) == 0) return &g_kinds[i];
    if (g_nkinds < (int) (sizeof g_kinds / sizeof g_kinds[0])) {
        g_kinds[g_nkinds].hits.store(0, std::memory_order_relaxed);
        g_kinds[g_nkinds].logged.store(false, std::memory_order_relaxed);
        g_kinds[g_nkinds].what = what;
        return &g_kinds[g_nkinds++];
    }
    return nullptr;   // (more kinds than slots: the wrappers fit, this is a guard against typos)
}
struct Register {   // the wrappers' names, before any thread can hit
    Register() {
        for (const char* w : {"cudaMemcpy", "cudaMalloc", "cudaFree", "cudaHostAlloc", "cudaFreeHost",
                              "cudaHostRegister", "cudaStreamSynchronize", "cudaDeviceSynchronize",
                              "cudaEventSynchronize", "cudaMemGetInfo"})
            find(w);
    }
};
const Register g_register;

// the guard's own window state: per-thread depth, its hit count for the summary line
thread_local int t_depth = 0;
thread_local uint64_t t_window_hits = 0;

// 0 = count silently, 1 = log, 2 = abort (unset logs: a guard build wants to say what it saw)
int g_mode = [] {
    const char* v = std::getenv("STRATA_DS4_FORBID_SYNC");
    if (v == nullptr || std::strcmp(v, "log") == 0) return 1;
    if (std::strcmp(v, "0") == 0) return 0;
    return 2;
}();

}  // namespace

void enter() { ++t_depth; }

void leave() {
    if (t_depth > 0) --t_depth;
    if (t_depth == 0 && t_window_hits > 0 && g_mode >= 1) {
        std::fprintf(stderr, "ds4 sync guard: the launch loop made %llu sync-capable calls (",
                     (unsigned long long) t_window_hits);
        bool first = true;
        for (int i = 0; i < g_nkinds; ++i) {
            const uint64_t h = g_kinds[i].hits.load(std::memory_order_relaxed);
            if (h == 0) continue;
            std::fprintf(stderr, "%s%s %llu", first ? "" : ", ", g_kinds[i].what, (unsigned long long) h);
            first = false;
        }
        std::fprintf(stderr, ") - each one can stall the pipeline; STRATA_DS4_FORBID_SYNC=abort stops on the first\n");
        t_window_hits = 0;
    }
}

void hit(const char* what) {
    if (t_depth == 0) return;   // outside a window: the load's business
    ++t_window_hits;
    Kind* k = find(what);
    if (k == nullptr) return;
    k->hits.fetch_add(1, std::memory_order_relaxed);
    if (g_mode >= 1 && !k->logged.exchange(true))
        std::fprintf(stderr, "ds4 sync guard: %s inside the launch loop (each one can stall the pipeline)\n", what);
    if (g_mode == 2) std::abort();
}

uint64_t count(const char* what) {
    for (int i = 0; i < g_nkinds; ++i)
        if (std::strcmp(g_kinds[i].what, what) == 0) return g_kinds[i].hits.load(std::memory_order_relaxed);
    return 0;
}

void reset() {
    for (int i = 0; i < g_nkinds; ++i) {
        g_kinds[i].hits.store(0, std::memory_order_relaxed);
        g_kinds[i].logged.store(false, std::memory_order_relaxed);
    }
    t_window_hits = 0;
}

}  // namespace strata::ds4::sync_guard

// the wrappers, one a sync-capable call the launch loops can make.  The real/wrap names are PASTED from the call's
// own name after it expands - on a HIP build hip_extra.h has already mapped cuda* to hip*, so the wrappers and the
// -Wl,--wrap flags the CMake file passes name the same runtime symbols (a plain __real_cudaMemcpy spelling would
// stay cuda* on both: #define maps whole tokens, not substrings).
#define SG_CAT2(a, b) a##b
#define SG_CAT(a, b) SG_CAT2(a, b)
#define SG_REAL(name) SG_CAT(__real_, name)
#define SG_WRAPFN(name) SG_CAT(__wrap_, name)

extern "C" cudaError_t SG_REAL(cudaMemcpy)(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind);
extern "C" cudaError_t SG_WRAPFN(cudaMemcpy)(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    strata::ds4::sync_guard::hit("cudaMemcpy");
    return SG_REAL(cudaMemcpy)(dst, src, bytes, kind);
}
extern "C" cudaError_t SG_REAL(cudaMalloc)(void** p, size_t bytes);
extern "C" cudaError_t SG_WRAPFN(cudaMalloc)(void** p, size_t bytes) {
    strata::ds4::sync_guard::hit("cudaMalloc");
    return SG_REAL(cudaMalloc)(p, bytes);
}
extern "C" cudaError_t SG_REAL(cudaFree)(void* p);
extern "C" cudaError_t SG_WRAPFN(cudaFree)(void* p) {
    strata::ds4::sync_guard::hit("cudaFree");
    return SG_REAL(cudaFree)(p);
}
extern "C" cudaError_t SG_REAL(cudaHostAlloc)(void** p, size_t bytes, unsigned flags);
extern "C" cudaError_t SG_WRAPFN(cudaHostAlloc)(void** p, size_t bytes, unsigned flags) {
    strata::ds4::sync_guard::hit("cudaHostAlloc");
    return SG_REAL(cudaHostAlloc)(p, bytes, flags);
}
extern "C" cudaError_t SG_REAL(cudaFreeHost)(void* p);
extern "C" cudaError_t SG_WRAPFN(cudaFreeHost)(void* p) {
    strata::ds4::sync_guard::hit("cudaFreeHost");
    return SG_REAL(cudaFreeHost)(p);
}
extern "C" cudaError_t SG_REAL(cudaHostRegister)(void* p, size_t bytes, unsigned flags);
extern "C" cudaError_t SG_WRAPFN(cudaHostRegister)(void* p, size_t bytes, unsigned flags) {
    strata::ds4::sync_guard::hit("cudaHostRegister");
    return SG_REAL(cudaHostRegister)(p, bytes, flags);
}
extern "C" cudaError_t SG_REAL(cudaStreamSynchronize)(cudaStream_t s);
extern "C" cudaError_t SG_WRAPFN(cudaStreamSynchronize)(cudaStream_t s) {
    strata::ds4::sync_guard::hit("cudaStreamSynchronize");
    return SG_REAL(cudaStreamSynchronize)(s);
}
extern "C" cudaError_t SG_REAL(cudaDeviceSynchronize)();
extern "C" cudaError_t SG_WRAPFN(cudaDeviceSynchronize)() {
    strata::ds4::sync_guard::hit("cudaDeviceSynchronize");
    return SG_REAL(cudaDeviceSynchronize)();
}
extern "C" cudaError_t SG_REAL(cudaEventSynchronize)(cudaEvent_t e);
extern "C" cudaError_t SG_WRAPFN(cudaEventSynchronize)(cudaEvent_t e) {
    strata::ds4::sync_guard::hit("cudaEventSynchronize");
    return SG_REAL(cudaEventSynchronize)(e);
}
extern "C" cudaError_t SG_REAL(cudaMemGetInfo)(size_t* free, size_t* total);
extern "C" cudaError_t SG_WRAPFN(cudaMemGetInfo)(size_t* free, size_t* total) {
    strata::ds4::sync_guard::hit("cudaMemGetInfo");
    return SG_REAL(cudaMemGetInfo)(free, total);
}

#endif  // STRATA_DS4_SYNC_GUARD
