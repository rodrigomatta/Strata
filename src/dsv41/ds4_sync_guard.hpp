#pragma once

// The dsv41 launch loops' CPU-GPU sync guard, DeepInfer's forbid_cuda_sync idea reimplemented for this engine
// (behaviour parity, no code taken): an implicit sync in the decode or prefill launch loop — a blocking read-back, a
// stray cudaDeviceSynchronize, an allocation mid-run — stalls the pipeline the loop overlaps, and nothing else shows
// it.  STRATA_DS4_SYNC_GUARD (CMake, OFF by default — a debug/test build option, never a release one) compiles the
// interposer: the sync-CAPABLE calls the loops can make are wrapped at link time (-Wl,--wrap, ds4_sync_guard.cu), so
// the loops' own code is not touched, and a Window around the two loops (feed_windows, forward_chunks — anywhere the
// engine launches work for a prompt) tells the wrappers whether they are in one.  STRATA_DS4_FORBID_SYNC then says
// what a hit does: `log` (the default) one line a kind and a count per window, `abort` stop right there for a
// debugger.  Outside a Window every call passes through untouched, and a build without the option has none of this:
// Window is an empty type and the loops compile exactly as before.

#if defined(STRATA_DS4_SYNC_GUARD)

#include <cstdint>

namespace strata::ds4::sync_guard {

/// The launch loop begins / ends (RAII: Window).  Nesting is allowed — the inner loops of one engine run.
void enter();
void leave();

/// A sync-capable call ran (the wrappers call this with their own name).  Counted — and logged, or a stop in the
/// `abort` mode — only inside a Window; outside one it is the load's business and passes silently.
void hit(const char* what);

/// The window's hit count of `what` so far (the test's eyes; 0 when never hit).
uint64_t count(const char* what);
/// Every counter back to zero.
void reset();

struct Window {
    Window() { enter(); }
    ~Window() { leave(); }
};

}  // namespace strata::ds4::sync_guard

#else

namespace strata::ds4::sync_guard {
struct Window {};   // (nothing: a build without the guard compiles the loops exactly as before)
}  // namespace strata::ds4::sync_guard

#endif
