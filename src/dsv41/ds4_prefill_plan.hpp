#pragma once

// The dsv41 prompt chunk, priced against the host — the #2824 plan of the main engine (include/strata/program/
// prefill_plan.hpp) adapted to this engine's prompt path.  Opt-in (STRATA_DS4_PREFILL_PLAN=1, read where this is
// called): the load's chunk is what VRAM can buffer, but a prompt reads the CPU lane's file set once per chunk
// through it, and when that set cannot stay in the RAM the OS caches, the least-time chunk can be smaller.  Pure
// model, no engine and no GPU — the same shape that made the main engine's plan testable on the CPU.
//
// Why the main engine's model transfers to this path:
//   - a (chunk, layer) streams the experts its tokens route to plus the likely ones (the flow, ds4_model.cu): with
//     k_exp = 8 over n_exp = 256, a chunk of 256+ tokens routes to ~every expert of every layer
//     (1 - (1 - 8/256)^256 = 0.9997), so each chunk reads the CPU lane's whole file set — the same per-chunk set the
//     main engine streams past stream_all_min_tokens(), and the same file_reads = n * f(W / page_cache) * W with
//     f = the two-rig fit reused here (pp::miss_share).
//   - the set the chunk reads DEPENDS on the chunk, as it does in the main engine (there the loan displaces
//     RAM-backed cache, #2824: a 15.1 GiB set against a 14.5 GiB cache at 8192, 11.6 GiB at 4096 — 28-45 GB read
//     from the drive against ~1): a smaller chunk's buffers free VRAM the experts take back, and the caller
//     measures that slope for real (one probe lay of the buffers, the byte accounting the engine already keeps) and
//     passes it as `freed_per_tok`.  A set deep beyond the cached RAM never fits at any chunk, and then the fewest
//     passes — the biggest chunk — wins, exactly as #2824 concluded; the win lives at the cache's edge, where the
//     VRAM a smaller chunk hands back drops the set under the fit cliff while n only doubles.
//   - layer-major (one device, two or more chunks): the layer cache keeps a layer's read experts for its later
//     chunks (each read once a run, lm_begin), so the drive term loses its per-chunk factor — predicted by running
//     lm_begin's own test ((n - 1) * file_b > moved, and the cache fitting ram_avail() less kRamMargin) per
//     candidate.  The moved bytes are 2 * rows * (hand bytes / chunk) * layers — hand bytes are per chunk, so the
//     test's right side does not depend on the candidate.
//   - the RAM store (--ssd-cache) holds its GB of the CPU lane's experts: they cross PCIe every chunk but never the
//     drive, so they leave the file set and stay in the stream term.
//
// The plan only ever SHRINKS the load's chunk (buffers for less always fit; the main engine's grid is at or below
// its VRAM pick for the same reason), on the 256 grid, and keeps it unless the pick saves >= 5% and >= 2 s —
// pp::choose's guards: a set that fits the cached RAM is never touched, and near-ties keep the established chunk.
// The compute term is common to every option (no ring to trade against the chunk here), so compute_tok_s only
// scales the seconds in the log, never the pick.  The inputs are the load's own estimates — the same room and need
// the read-ahead computes next to the call site, and the layers split the set evenly for the layer cache's share
// (this engine's trunk layers are alike) — estimates like the read-ahead's, not measurements.  Tuning, as in the
// main engine: STRATA_DS4_PLAN_H2D_GBPS, STRATA_DS4_PLAN_SSD_GBPS (rotational ~0.15, SATA ~0.45, NVMe ~2.5),
// STRATA_DS4_PLAN_TOKS.

#include "strata/program/prefill_plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <sstream>
#include <string>

namespace strata::ds4::prefill_plan {

namespace pp = strata::program::prefill_plan;

struct In {
    int64_t tokens = 0;         ///< the prompt the plan is for (the context the buffers were sized for)
    int64_t chunk = 0;          ///< the chunk the load picked — the baseline; the plan only shrinks it
    uint64_t experts_all = 0;   ///< every routed expert's bytes (the pack's, before VRAM takes its share)
    uint64_t experts_room = 0;  ///< bytes of experts the VRAM left over takes at the baseline chunk
    double freed_per_tok = 0.0; ///< VRAM a smaller chunk's buffers hand the experts, bytes per token (the probe)
    uint64_t ram_tier = 0;      ///< the store's bytes (--ssd-cache): RAM, crosses PCIe, never the drive
    uint64_t ram_avail = 0;     ///< ram_avail(); ~0 = unknown, and then nothing is treated as cached
    uint64_t ram_margin = 0;    ///< RAM left to the rest (kRamMargin: the read-ahead's own budget line)
    bool one_device = false;    ///< layer-major needs exactly one device on the flow (lm_begin)
    int64_t layers = 0;         ///< the routed layers (the layer cache's share of a run's chunk states)
    int64_t hand_bytes = 0;     ///< one chunk's hand-off state (chunk_hand_bytes() of the baseline chunk)
    double h2d_gbps = 12.0;     ///< host -> device, GB/s (the main engine's default; STRATA_DS4_PLAN_H2D_GBPS)
    double ssd_gbps = 0.45;     ///< the file tier's reads, GB/s (STRATA_DS4_PLAN_SSD_GBPS)
    double compute_tok_s = 1200.0;  ///< prompt compute without streaming (STRATA_DS4_PLAN_TOKS; scales the log)
};

struct Cost {
    double seconds = 0.0, read_bytes = 0.0, file_set = 0.0, cpu_bytes = 0.0;
    int64_t chunks = 1;
    bool lm = false, fits = false;
};

struct Plan {
    int64_t chunk = 0;          ///< the chunk to run (== In::chunk when nothing beats it)
    bool changed = false;
    bool fits = false;          ///< the pick's set sits in the cached RAM with room to spare
    bool lm = false;            ///< the pick predicts layer-major (the set read once a run, not once a chunk)
    double file_set = 0.0, page_cache = 0.0, read_gb = 0.0, base_s = 0.0, pick_s = 0.0;
    std::string line;           ///< one log line, the house format
};

/// The CPU lane's bytes when the chunk is `c`: the VRAM the smaller chunk's buffers hand back takes its share
/// first (the probe-measured slope; the estimate only ever overstates a smaller chunk's set when it errs, which
/// only ever keeps the plan at the baseline).
inline uint64_t cpu_bytes(const In& in, int64_t c) {
    const double room = (double) in.experts_room + in.freed_per_tok * (double) (in.chunk - c);
    const double cpu = (double) in.experts_all - std::max(0.0, room);
    return cpu > 0.0 ? (uint64_t) cpu : 0;
}

/// The seconds of a `tokens`-token prompt read in chunks of `c`: compute + a pass of the CPU lane over PCIe per
/// chunk + the drive bytes the cached RAM does not hold (once a run when layer-major takes the set).
inline Cost price(const In& in, double page_cache, int64_t c) {
    Cost k;
    k.chunks = c > 0 ? (in.tokens + c - 1) / c : 1;
    k.cpu_bytes = (double) cpu_bytes(in, c);
    k.file_set = k.cpu_bytes > (double) in.ram_tier ? k.cpu_bytes - (double) in.ram_tier : 0.0;
    const double r = page_cache > 0 ? k.file_set / page_cache : 1e9;   // unknown RAM: nothing is cached
    k.fits = page_cache > 0 && r <= 0.85;   // pp::price's own line between "fits" and "climbs"
    const double per_tok = in.chunk > 0 ? (double) in.hand_bytes / (double) in.chunk : 0.0;
    const double moved = 2.0 * (double) in.tokens * per_tok * (double) in.layers;
    // lm_begin's test, on this engine's even split of the set over the layers (one layer's experts at a time in
    // the cache, plus the run's chunk states against the same RAM budget the read-ahead uses)
    const double one_layer = in.layers > 0 ? k.file_set / (double) in.layers : k.file_set;
    const double lc_room = in.ram_avail != ~uint64_t{0}
                               ? (double) std::max<uint64_t>(0, in.ram_avail > in.ram_margin
                                                                       ? in.ram_avail - in.ram_margin : 0)
                               : 1e18;
    k.lm = in.one_device && k.chunks >= 2 && (double) (k.chunks - 1) * k.file_set > moved &&
           one_layer + (double) in.tokens * per_tok <= lc_room;
    const double miss = pp::miss_share(r);
    k.read_bytes = (k.lm ? 1.0 : (double) k.chunks) * k.file_set * miss;
    k.seconds = (double) in.tokens / in.compute_tok_s +
                (double) k.chunks * k.cpu_bytes / (in.h2d_gbps * 1e9) + k.read_bytes / (in.ssd_gbps * 1e9);
    return k;
}

/// One number as the house log format keeps it: one decimal, no exponent.
inline std::string num(double v) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(1) << v;
    return os.str();
}

/// The plan for `in`: the baseline chunk against every 256-grid chunk below it (down to 512, chunk_min's default —
/// shorter runs take the windows), kept unless a pick saves 5% and 2 s (pp::choose's guards; a tie takes the bigger
/// chunk).  `line` says what was chosen and why, in the one-line format of the main engine's plan.
inline Plan plan_chunk(In in) {
    if (const char* v = std::getenv("STRATA_DS4_PLAN_H2D_GBPS"); v != nullptr && std::atof(v) > 0)
        in.h2d_gbps = std::atof(v);
    if (const char* v = std::getenv("STRATA_DS4_PLAN_SSD_GBPS"); v != nullptr && std::atof(v) > 0)
        in.ssd_gbps = std::atof(v);
    if (const char* v = std::getenv("STRATA_DS4_PLAN_TOKS"); v != nullptr && std::atof(v) > 0)
        in.compute_tok_s = std::atof(v);
    Plan p;
    p.chunk = in.chunk;
    p.page_cache = in.ram_avail != ~uint64_t{0}
                       ? (double) std::max<uint64_t>(0, in.ram_avail > in.ram_margin ? in.ram_avail - in.ram_margin : 0)
                       : 0.0;
    const Cost base = price(in, p.page_cache, in.chunk);
    p.base_s = base.seconds;
    p.pick_s = base.seconds;
    p.file_set = base.file_set;
    p.fits = base.fits;
    p.lm = base.lm;
    p.read_gb = base.read_bytes / 1e9;
    Cost best_c = base;
    int64_t best = in.chunk;
    for (int64_t c = in.chunk - 256; c >= 512; c -= 256) {
        const Cost k = price(in, p.page_cache, c);
        if (k.seconds < best_c.seconds - 1e-9 ||
            (std::abs(k.seconds - best_c.seconds) <= 1e-9 && c > best)) {
            best_c = k;
            best = c;
        }
    }
    p.changed = best != in.chunk;
    if (p.changed && !(p.base_s - best_c.seconds >= 0.05 * p.base_s && p.base_s - best_c.seconds >= 2.0)) {
        best = in.chunk;
        best_c = base;
        p.changed = false;
    }
    p.chunk = best;
    p.pick_s = best_c.seconds;
    p.file_set = best_c.file_set;
    p.fits = best_c.fits;
    p.lm = best_c.lm;
    p.read_gb = best_c.read_bytes / 1e9;
    std::ostringstream os;
    if (!p.changed) {
        os << "ds4: prefill plan: chunk " << p.chunk << " kept: file set " << num(p.file_set / pp::kGiB) << " GiB "
           << (p.fits ? "fits" : "does not fit") << " the " << num(p.page_cache / pp::kGiB)
           << " GiB the RAM can cache (about " << num(p.read_gb) << " GB read from the drive per " << in.tokens
           << "-token prompt)" << (p.lm ? "; layer-major reads it once a run" : "");
    } else {
        os << "ds4: prefill plan: chunk " << in.chunk << " -> " << p.chunk << ": file set "
           << num(base.file_set / pp::kGiB) << " GiB " << (base.fits ? "fits" : "does not fit") << " the "
           << num(p.page_cache / pp::kGiB) << " GiB the RAM can cache (about " << num(p.read_gb)
           << " GB read from the drive per " << in.tokens << "-token prompt, " << num(p.base_s) << " s -> "
           << num(p.pick_s) << " s)" << (p.lm ? "; layer-major reads it once a run" : "")
           << "; --prefill-chunk N keeps your own chunk";
    }
    p.line = os.str();
    return p;
}

}  // namespace strata::ds4::prefill_plan
