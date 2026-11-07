// The prompt chunk's host-aware plan without a GPU or a model (src/dsv41/ds4_prefill_plan.hpp, the #2824 plan of
// the main engine adapted to this engine's prompt path): the chunk the load picked is kept unless a smaller one
// reads less from the drive by enough (5% and 2 s, pp::choose's guards) — the boundary rig where the set the
// chunk reads sits at the cached RAM's edge, layer-major reading it once a run, the RAM store taking its share off
// the drive's, and the guards keeping everything else exactly as the load chose it.
//   ds4_prefill_plan_test     exit 0: all passed
// (No STRATA_DS4_PLAN_* env overrides set: their tuning paths are the main engine's, tested there.)
#include "ds4_prefill_plan.hpp"

#include <cstdio>
#include <cstdint>

namespace t = strata::ds4::prefill_plan;

static int fails = 0;
#define CHECK(c)                                                           \
    do {                                                                   \
        if (!(c)) {                                                        \
            std::printf("FAIL line %d: %s\n", __LINE__, #c);               \
            ++fails;                                                       \
        }                                                                  \
    } while (0)

static constexpr uint64_t GiB = 1073741824ull;

// a rig like the plan's comments: a 29K prompt in 8192-token chunks, the set at the cached RAM's edge
static t::In edge_rig() {
    t::In in;
    in.tokens = 29696;
    in.chunk = 8192;
    in.experts_all = 26 * GiB;
    in.experts_room = (uint64_t) (14.8 * GiB);   // the CPU lane: 11.2 GiB
    in.freed_per_tok = 65536;                    // 64 KiB of VRAM a smaller chunk's buffers hand the experts
    in.ram_avail = 17 * GiB;
    in.ram_margin = 4 * GiB;                     // 13 GiB the OS can cache: r = 0.86, the miss curve's foot
    in.one_device = false;
    in.layers = 60;
    in.hand_bytes = 8192 * 1000;
    in.ssd_gbps = 0.15;                          // the win needs the drive slow (rotational): reads must weigh
    return in;
}

int main() {
    // the buffer slope: a smaller chunk's set is smaller by the VRAM its buffers hand back
    {
        t::In in = edge_rig();
        const uint64_t base = t::cpu_bytes(in, in.chunk);
        CHECK(base == (uint64_t) (11.2 * GiB) + 1);   // 26 - 14.8 GiB (a hair over: the double cast)
        CHECK(t::cpu_bytes(in, 4096) < base);
        CHECK(t::cpu_bytes(in, in.chunk + 256) > base);   // (the slope's linear reach: a bigger chunk's buffers
                                                           //  eat room - the plan never goes there, the model stays
                                                           //  linear through the baseline it was measured at)
        in.freed_per_tok = 0;
        CHECK(t::cpu_bytes(in, 512) == base);              // no slope: the set is the chunk's own business
        in.experts_room = in.experts_all + GiB;
        CHECK(t::cpu_bytes(in, in.chunk) == 0);            // VRAM holds every expert: no CPU lane at all
    }
    // a set that fits: never touched, whatever the drive
    {
        t::In in = edge_rig();
        in.experts_room = 15 * GiB;   // 5 GiB against the 13 GiB cache
        const t::Plan p = t::plan_chunk(in);
        CHECK(!p.changed && p.chunk == in.chunk && p.fits);
        CHECK(!p.line.empty());
    }
    // the boundary rig: the set just past the fit cliff, a slow drive - a smaller chunk drops it under and reads
    // nothing; the biggest such chunk wins (fewest passes over PCIe)
    {
        t::In in = edge_rig();
        const t::Plan p = t::plan_chunk(in);
        CHECK(p.changed);
        CHECK(p.chunk < in.chunk && p.chunk >= 512 && p.chunk % 256 == 0);
        CHECK(p.pick_s < p.base_s - 2.0);   // the guards' own bar, and then some
        CHECK(p.fits);                      // the pick drops under the cliff; it is the baseline that did not fit
        CHECK(p.line.find("->") != std::string::npos);
    }
    // the same rig on an NVMe: the drive term never weighs, the extra PCIe passes do - the chunk stays
    {
        t::In in = edge_rig();
        in.ssd_gbps = 2.5;
        const t::Plan p = t::plan_chunk(in);
        CHECK(!p.changed && p.chunk == in.chunk);
    }
    // a saving below the guards (5% and 2 s): kept, exactly as pp::choose keeps the main engine's
    {
        t::In in = edge_rig();
        in.ssd_gbps = 1.0;
        in.h2d_gbps = 1000;   // the stream term off, so only the reads differ: a sub-2 s saving
        in.experts_room = (uint64_t) (14.85 * GiB);
        in.freed_per_tok = 32768;
        const t::Plan p = t::plan_chunk(in);
        CHECK(!p.changed && p.chunk == in.chunk);
    }
    // layer-major (one device, the cache fitting a layer's share): the set is read once a run whatever the chunk,
    // so only the PCIe passes weigh - the biggest chunk, the load's own, stays
    {
        t::In in = edge_rig();
        in.one_device = true;
        in.experts_room = 1 * GiB;   // a 25 GiB CPU lane
        in.ram_avail = 30 * GiB;     // r = 25/26: it climbs, but lm_begin's test passes
        in.ssd_gbps = 0.15;
        const t::Plan p = t::plan_chunk(in);
        CHECK(!p.changed && p.chunk == in.chunk && p.lm);
        CHECK(p.line.find("layer-major") != std::string::npos);
    }
    // the RAM store (--ssd-cache) holds its share: it crosses PCIe, never the drive
    {
        t::In in = edge_rig();
        in.ram_tier = 12 * GiB;   // the set's whole remainder
        const t::Plan p = t::plan_chunk(in);
        CHECK(!p.changed && p.fits && p.read_gb < 0.1);
    }
    // unknown RAM (ram_avail failed): nothing is cached, every chunk pays the whole set - the fewest passes win
    {
        t::In in = edge_rig();
        in.ram_avail = ~uint64_t{0};
        in.experts_room = 2 * GiB;   // a 24 GiB set, deep beyond any cache
        const t::Plan p = t::plan_chunk(in);
        CHECK(!p.changed && p.chunk == in.chunk && !p.fits);
        CHECK(!p.line.empty());
    }
    // VRAM holds every expert: nothing to read, nothing to stream, nothing to plan
    {
        t::In in = edge_rig();
        in.experts_room = in.experts_all;
        const t::Plan p = t::plan_chunk(in);
        CHECK(!p.changed && p.chunk == in.chunk && p.fits);
    }
    if (fails == 0) std::printf("ds4_prefill_plan_test: all passed\n");
    else std::printf("ds4_prefill_plan_test: %d FAILED\n", fails);
    return fails == 0 ? 0 : 1;
}
