// The dsv41 launch loops' sync guard (src/dsv41/ds4_sync_guard.{hpp,cu}, STRATA_DS4_SYNC_GUARD builds): the window
// logic on the CPU — a hit outside a window passes silently, inside one it counts, nesting keeps counting — and,
// with a GPU (no model), the link-time interposition itself: a wrapped allocation inside a Window is seen, an
// async copy inside one is not, and a wrapped call outside one passes through untouched.
//   ds4_sync_guard_test     exit 0: all passed (without a GPU the interposition half skips, said so)
#include "ds4_sync_guard.hpp"

#include <cuda_runtime.h>
#include <cstdio>

#if defined(STRATA_DS4_SYNC_GUARD)

namespace sg = strata::ds4::sync_guard;

static int fails = 0;
#define CHECK(c)                                                           \
    do {                                                                   \
        if (!(c)) {                                                        \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);               \
            ++fails;                                                       \
        }                                                                  \
    } while (0)

int main() {
    // the window logic, straight through the guard's own API
    sg::reset();
    sg::hit("cudaMalloc");   // outside any window: the load's business, never counted
    CHECK(sg::count("cudaMalloc") == 0);
    {
        sg::Window w;
        sg::hit("cudaMalloc");
        sg::hit("cudaMalloc");
        sg::hit("cudaMemcpy");
        CHECK(sg::count("cudaMalloc") == 2 && sg::count("cudaMemcpy") == 1);
        {
            sg::Window inner;                 // nesting: the inner loop of one engine run
            sg::hit("cudaFree");
        }
        CHECK(sg::count("cudaFree") == 1);    // (the outer window still owns it)
        sg::hit("cudaFree");                  // ...and after the inner one left
        CHECK(sg::count("cudaFree") == 2);
    }
    sg::hit("cudaMalloc");                    // the window closed: silent again
    CHECK(sg::count("cudaMalloc") == 2);

    // the interposition, with a GPU
    int dev = -1;
    cudaError_t e = cudaGetDeviceCount(&dev);
    const bool gpu = e == cudaSuccess && dev > 0;
    if (!gpu) {
        std::printf("no GPU here: the interposition half skipped (the window logic above stands)\n");
        if (fails == 0) std::printf("ds4_sync_guard_test: all passed\n");
        else std::printf("ds4_sync_guard_test: %d FAILED\n", fails);
        return fails == 0 ? 0 : 1;
    }
    cudaSetDevice(0);
    sg::reset();
    {
        sg::Window w;
        void* p = nullptr;
        e = cudaMalloc(&p, 4096);   // wrapped: the wrapper saw it, then the real call ran
        CHECK(e == cudaSuccess && p != nullptr);
        CHECK(sg::count("cudaMalloc") == 1);
        int host = 7;
        e = cudaMemcpyAsync(p, &host, sizeof host, cudaMemcpyHostToDevice, nullptr);   // async: never wrapped
        CHECK(e == cudaSuccess);
        CHECK(sg::count("cudaMemcpy") == 0);
        if (p != nullptr) cudaFree(p);   // inside the window: wrapped and counted, the guard's own doing
        CHECK(sg::count("cudaFree") == 1);
    }
    void* q = nullptr;
    e = cudaMalloc(&q, 4096);   // outside the window: the wrapper passes it through uncounted
    CHECK(e == cudaSuccess);
    CHECK(sg::count("cudaMalloc") == 1);
    if (q != nullptr) {
        cudaFree(q);   // (and this one too: the load's business again)
        CHECK(sg::count("cudaFree") == 1);
    }
    if (fails == 0) std::printf("ds4_sync_guard_test: all passed\n");
    else std::printf("ds4_sync_guard_test: %d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}

#else
int main() { return 99; }   // (a build without the guard has nothing to test; CMake never builds this)
#endif
