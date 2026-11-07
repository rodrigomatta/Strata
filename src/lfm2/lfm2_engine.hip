// Strata LFM2-MoE engine (HIP, gfx1100). Local opt-in target. Greedy decode, one token per step.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <unordered_map>
#include <algorithm>
#include "lfm2_model.hpp"
#include "lfm2_vocab.hpp"
#include "lfm2_cli.hpp"

using namespace lfm2;

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  fprintf(stderr, "HIP error %s at %s:%d: %s\n", hipGetErrorString(e_), __FILE__, __LINE__, #x); exit(1); } } while (0)

enum { T_F32 = 0, T_F16 = 1, T_Q4_0 = 2, T_Q6_K = 14 };

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  return v;
}
__device__ __forceinline__ float h2f(const uint8_t* p) {
  uint16_t u = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
  _Float16 h; __builtin_memcpy(&h, &u, 2);
  return (float)h;
}

template <int T>
__device__ __forceinline__ float dot_warp(const uint8_t* row, const float* x, int n, int lane) {
  float acc = 0.f;
  if constexpr (T == T_F32) {
    const float* r = (const float*)row;
    for (int i = lane; i < n; i += 32) acc += r[i] * x[i];
  } else if constexpr (T == T_F16) {
    const _Float16* r = (const _Float16*)row;
    for (int i = lane; i < n; i += 32) acc += (float)r[i] * x[i];
  } else if constexpr (T == T_Q4_0) {
    const int nb = n / 32, part = lane & 3;
    for (int b = lane >> 2; b < nb; b += 8) {
      const uint8_t* q = row + (size_t)b * 18;
      const float d = h2f(q);
      const uint8_t* qs = q + 2 + part * 4;
      const float* xb = x + b * 32 + part * 4;
      float s = 0.f;
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const uint8_t v = qs[j];
        s += (float)((int)(v & 15) - 8) * xb[j] + (float)((int)(v >> 4) - 8) * xb[j + 16];
      }
      acc += d * s;
    }
  } else if constexpr (T == T_Q6_K) {
    const int nh = n / 128;
    const int l = lane, is = l / 16;
    for (int hh = 0; hh < nh; ++hh) {
      const int b = hh >> 1, h = hh & 1;
      const uint8_t* q = row + (size_t)b * 210;
      const uint8_t* ql = q + h * 64;
      const uint8_t* qh = q + 128 + h * 32;
      const int8_t* sc = (const int8_t*)(q + 192 + h * 8);
      const float d = h2f(q + 208);
      const uint8_t hb = qh[l];
      const int q1 = (int)((ql[l] & 15) | (((hb >> 0) & 3) << 4)) - 32;
      const int q2 = (int)((ql[l + 32] & 15) | (((hb >> 2) & 3) << 4)) - 32;
      const int q3 = (int)((ql[l] >> 4) | (((hb >> 4) & 3) << 4)) - 32;
      const int q4 = (int)((ql[l + 32] >> 4) | (((hb >> 6) & 3) << 4)) - 32;
      const float* xb = x + b * 256 + h * 128;
      acc += d * ((float)sc[is] * q1 * xb[l] + (float)sc[is + 2] * q2 * xb[l + 32] +
                  (float)sc[is + 4] * q3 * xb[l + 64] + (float)sc[is + 6] * q4 * xb[l + 96]);
    }
  }
  return warp_sum(acc);
}

template <int T>
__global__ void __launch_bounds__(256) k_matvec(const uint8_t* W, size_t rb, const float* x, float* y,
                                                int rows, int n, int acc) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r = blockIdx.x * 8 + w;
  if (r >= rows) return;
  const float s = dot_warp<T>(W + (size_t)r * rb, x, n, lane);
  if (lane == 0) y[r] = acc ? y[r] + s : s;
}

template <int T>
__global__ void __launch_bounds__(256) k_gate_up(const uint8_t* G, const uint8_t* U, size_t rb, size_t estride,
                                                 const int* ids, const float* x, float* h, int F, int n, int nslots) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r = blockIdx.x * 8 + w, k = blockIdx.y;
  if (r >= F || k >= nslots) return;
  const size_t e = ids ? (size_t)ids[k] : 0;
  const float g = dot_warp<T>(G + e * estride + (size_t)r * rb, x, n, lane);
  const float u = dot_warp<T>(U + e * estride + (size_t)r * rb, x, n, lane);
  if (lane == 0) h[k * F + r] = g / (1.f + expf(-g)) * u;
}

template <int T>
__global__ void __launch_bounds__(256) k_down(const uint8_t* D, size_t rb, size_t estride, const int* ids,
                                              const float* wts, const float* h, float* y, int rows, int F, int nslots) {
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int r = blockIdx.x * 8 + w;
  if (r >= rows) return;
  float tot = 0.f;
  for (int k = 0; k < nslots; ++k) {
    const size_t e = ids ? (size_t)ids[k] : 0;
    const float s = dot_warp<T>(D + e * estride + (size_t)r * rb, h + (size_t)k * F, F, lane);
    tot += (wts ? wts[k] : 1.f) * s;
  }
  if (lane == 0) y[r] += tot;
}

__device__ float block_sum(float v, float* red) {
  v = warp_sum(v);
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float t = 0.f;
  for (int i = 0; i < nw; ++i) t += red[i];
  __syncthreads();
  return t;
}
__device__ float block_max(float v, float* red) {
  v = warp_max(v);
  const int w = threadIdx.x >> 5, lane = threadIdx.x & 31, nw = blockDim.x >> 5;
  __syncthreads();
  if (lane == 0) red[w] = v;
  __syncthreads();
  float t = -INFINITY;
  for (int i = 0; i < nw; ++i) t = fmaxf(t, red[i]);
  __syncthreads();
  return t;
}

__global__ void k_rmsnorm(const float* x, const float* w, float* y, int n, float eps) {
  __shared__ float red[32];
  float s = 0.f;
  for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
  s = block_sum(s, red);
  const float r = rsqrtf(s / n + eps);
  for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * r * w[i];
}

__global__ void k_embed_q6k(const uint8_t* E, size_t rb, const int* tok, float* y, int n) {
  const uint8_t* row = E + (size_t)tok[0] * rb;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const int b = i / 256, wi = i % 256, h = wi / 128, w = wi % 128, l = w % 32, qq = w / 32, is = l / 16;
    const uint8_t* q = row + (size_t)b * 210;
    const uint8_t* ql = q + h * 64;
    const uint8_t* qh = q + 128 + h * 32;
    const int8_t* sc = (const int8_t*)(q + 192 + h * 8);
    const float d = h2f(q + 208);
    const uint8_t lo = (qq & 1) ? ql[l + 32] : ql[l];
    const int nib = (qq & 2) ? (lo >> 4) : (lo & 15);
    const int v = (int)(nib | (((qh[l] >> (2 * qq)) & 3) << 4)) - 32;
    y[i] = d * (float)sc[is + 2 * qq] * (float)v;
  }
}

__global__ void k_shortconv(const float* bcx, const float* w, float* s0, float* s1, float* y, int D) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= D) return;
  const float bx = bcx[c] * bcx[2 * D + c];
  const float o = w[c * 3 + 0] * s0[c] + w[c * 3 + 1] * s1[c] + w[c * 3 + 2] * bx;
  s0[c] = s1[c];
  s1[c] = bx;
  y[c] = bcx[D + c] * o;
}

__global__ void k_qk_norm_rope(float* q, const float* k, const float* v, const float* qn, const float* kn,
                               float* kc, float* vc, int pos, int NQ, int NKV, float eps, float base) {
  __shared__ float s[64];
  __shared__ float r;
  const int b = blockIdx.x, t = threadIdx.x;
  const bool isq = b < NQ;
  const int hh = isq ? b : b - NQ;
  const float* src = isq ? q + hh * 64 : k + hh * 64;
  const float* w = isq ? qn : kn;
  const float val = src[t];
  s[t] = val * val;
  __syncthreads();
  if (t == 0) {
    float sum = 0.f;
    for (int i = 0; i < 64; ++i) sum += s[i];
    r = rsqrtf(sum / 64.f + eps);
  }
  __syncthreads();
  const float nv = val * r * w[t];
  __syncthreads();
  s[t] = nv;
  __syncthreads();
  float* dst = isq ? q + hh * 64 : kc + ((size_t)pos * NKV + hh) * 64;
  if (t < 32) {
    const float freq = powf(base, -2.f * (float)t / 64.f);
    const float th = (float)pos * freq;
    const float c = cosf(th), sn = sinf(th);
    const float x0 = s[t], x1 = s[t + 32];
    dst[t] = x0 * c - x1 * sn;
    dst[t + 32] = x0 * sn + x1 * c;
  }
  if (!isq) vc[((size_t)pos * NKV + hh) * 64 + t] = v[hh * 64 + t];
}

__global__ void __launch_bounds__(256) k_attn(const float* q, const float* kc, const float* vc, float* out,
                                              float* scores, int pos, int NQ, int NKV, int max_ctx) {
  __shared__ float qs[64];
  __shared__ float red[32];
  __shared__ float part[4][64];
  const int h = blockIdx.x, tid = threadIdx.x;
  const int kh = h / (NQ / NKV);
  if (tid < 64) qs[tid] = q[h * 64 + tid] * 0.125f;
  __syncthreads();
  float* sc = scores + (size_t)h * max_ctx;
  const int n = pos + 1;
  float mx = -INFINITY;
  for (int t = tid; t < n; t += blockDim.x) {
    const float* kk = kc + ((size_t)t * NKV + kh) * 64;
    float s = 0.f;
    for (int d = 0; d < 64; ++d) s += qs[d] * kk[d];
    sc[t] = s;
    mx = fmaxf(mx, s);
  }
  mx = block_max(mx, red);
  float sum = 0.f;
  for (int t = tid; t < n; t += blockDim.x) {
    const float e = expf(sc[t] - mx);
    sc[t] = e;
    sum += e;
  }
  sum = block_sum(sum, red);
  const int g = tid >> 6, d = tid & 63;
  float a = 0.f;
  for (int t = g; t < n; t += 4) a += sc[t] * vc[((size_t)t * NKV + kh) * 64 + d];
  part[g][d] = a;
  __syncthreads();
  if (tid < 64) out[h * 64 + tid] = (part[0][tid] + part[1][tid] + part[2][tid] + part[3][tid]) / sum;
}

__global__ void k_router(const float* rl, const float* bias, int* ids, float* wts, int E, int K) {
  if (threadIdx.x != 0) return;
  float p[256], sel[256];
  for (int e = 0; e < E; ++e) {
    p[e] = 1.f / (1.f + expf(-rl[e]));
    sel[e] = p[e] + (bias ? bias[e] : 0.f);
  }
  float tot = 0.f;
  for (int k = 0; k < K; ++k) {
    int best = 0;
    float bv = -INFINITY;
    for (int e = 0; e < E; ++e) if (sel[e] > bv) { bv = sel[e]; best = e; }
    sel[best] = -INFINITY;
    ids[k] = best;
    wts[k] = p[best];
    tot += p[best];
  }
  tot = fmaxf(tot, 6.1035e-5f);
  for (int k = 0; k < K; ++k) wts[k] /= tot;
}

// ---------------------------------------------------------------- host side
struct DT {
  uint8_t* d = nullptr;
  int type = 0;
  int64_t ne0 = 0, ne1 = 1, ne2 = 1;
  size_t rb = 0;
  float* f() const { return (float*)d; }
};

static inline const TensorInfo* tp(const TensorInfo* t) { return t; }
static inline const TensorInfo* tp(const TensorInfo& t) { return &t; }

#define DISPATCH(TYPE, KER, GRID, BLOCK, ...) do { switch (TYPE) { \
  case T_F32:  KER<T_F32><<<GRID, BLOCK>>>(__VA_ARGS__); break; \
  case T_F16:  KER<T_F16><<<GRID, BLOCK>>>(__VA_ARGS__); break; \
  case T_Q4_0: KER<T_Q4_0><<<GRID, BLOCK>>>(__VA_ARGS__); break; \
  case T_Q6_K: KER<T_Q6_K><<<GRID, BLOCK>>>(__VA_ARGS__); break; \
  default: fprintf(stderr, "unsupported tensor type %d\n", (int)(TYPE)); exit(1); } \
  CK(hipGetLastError()); } while (0)

struct LayerState {
  float* s0 = nullptr; float* s1 = nullptr;
  float* kc = nullptr; float* vc = nullptr;
};

struct Engine {
  Model& m;
  int D, NQ, NKV, V, E, K, F, max_ctx;
  float eps, base;
  std::unordered_map<const TensorInfo*, DT> cache;
  size_t vram_w = 0;
  float *x, *xn, *buf, *big, *hq, *hk, *hv, *ho, *rl, *wts, *logits, *scores;
  int *ids, *tok;
  std::vector<LayerState> st;

  explicit Engine(Model& mm, int ctx) : m(mm), max_ctx(ctx) {
    const auto& hp = m.hp;
    D = (int)hp.n_embd; NQ = (int)hp.n_head; NKV = (int)hp.n_head_kv; V = (int)hp.n_vocab;
    E = (int)hp.n_expert; K = (int)hp.n_expert_used; F = (int)hp.n_ff_exp;
    eps = (float)hp.eps; base = (float)hp.rope_base;
    if ((int)hp.head_dim != 64) { fprintf(stderr, "head_dim %d != 64\n", (int)hp.head_dim); exit(1); }
  }

  DT up(const TensorInfo* t) {
    if (!t) { fprintf(stderr, "missing tensor\n"); exit(1); }
    auto it = cache.find(t);
    if (it != cache.end()) return it->second;
    DT r;
    r.type = (int)t->type;
    r.ne0 = t->ne.at(0);
    r.ne1 = t->ne.size() > 1 ? t->ne[1] : 1;
    r.ne2 = t->ne.size() > 2 ? t->ne[2] : 1;
    r.rb = row_bytes(t->type, t->ne[0]);
    CK(hipMalloc(&r.d, t->nbytes));
    CK(hipMemcpy(r.d, t->data, t->nbytes, hipMemcpyHostToDevice));
    vram_w += t->nbytes;
    cache[t] = r;
    return r;
  }
  DT upo(const TensorInfo* t) { return t ? up(t) : DT{}; }

  template <class P> float* fbuf(P*& p, size_t n) { CK(hipMalloc(&p, n * sizeof(P))); CK(hipMemset(p, 0, n * sizeof(P))); return (float*)p; }

  void load() {
    up(tp(m.tok_embd)); up(tp(m.out_norm));
    size_t bigN = (size_t)3 * D;
    for (auto& L : m.layers) {
      up(tp(L.attn_norm)); up(tp(L.ffn_norm));
      if (L.is_attn) { up(tp(L.wq)); up(tp(L.wk)); up(tp(L.wv)); up(tp(L.wo)); up(tp(L.q_norm)); up(tp(L.k_norm)); }
      else { up(tp(L.conv)); up(tp(L.in_proj)); up(tp(L.out_proj)); }
      if (L.is_moe) { up(tp(L.router)); upo(tp(L.exp_bias)); up(tp(L.gate_e)); up(tp(L.up_e)); up(tp(L.down_e)); }
      else { DT g = up(tp(L.gate)); up(tp(L.up)); up(tp(L.down)); bigN = std::max(bigN, (size_t)g.ne1); }
    }
    bigN = std::max(bigN, (size_t)K * F);
    fbuf(x, D); fbuf(xn, D); fbuf(buf, D); fbuf(big, bigN);
    fbuf(hq, NQ * 64); fbuf(hk, NKV * 64); fbuf(hv, NKV * 64); fbuf(ho, NQ * 64);
    fbuf(rl, std::max(E, 1)); fbuf(wts, std::max(K, 1)); fbuf(logits, V);
    fbuf(scores, (size_t)NQ * max_ctx);
    fbuf(ids, std::max(K, 1)); fbuf(tok, 1);
    st.resize(m.layers.size());
    for (size_t i = 0; i < m.layers.size(); ++i) {
      if (m.layers[i].is_attn) {
        fbuf(st[i].kc, (size_t)max_ctx * NKV * 64); fbuf(st[i].vc, (size_t)max_ctx * NKV * 64);
      } else {
        fbuf(st[i].s0, D); fbuf(st[i].s1, D);
      }
    }
  }

  void reset() {
    // Attention reads only positions <= the current position, overwritten by prefill.
    // The convolution history must explicitly start empty for every request.
    for (auto& s : st) if (s.s0) {
      CK(hipMemset(s.s0, 0, D * sizeof(float)));
      CK(hipMemset(s.s1, 0, D * sizeof(float)));
    }
  }

  void mv(const DT& w, const float* in, float* out, int acc) {
    const int rows = (int)w.ne1, n = (int)w.ne0;
    DISPATCH(w.type, k_matvec, dim3((rows + 7) / 8), dim3(256), w.d, w.rb, in, out, rows, n, acc);
  }
  void rms(const float* in, const DT& w, float* out, int n) {
    k_rmsnorm<<<1, 1024>>>(in, w.f(), out, n, eps);
    CK(hipGetLastError());
  }
  void ffn(const DT& g, const DT& u, const DT& d, const int* sid, const float* sw, int nslots) {
    const int Fl = (int)g.ne1, n = (int)g.ne0;
    if (g.type != u.type) { fprintf(stderr, "gate/up type mismatch\n"); exit(1); }
    const size_t es_g = g.rb * (size_t)g.ne1, es_d = d.rb * (size_t)d.ne1;
    DISPATCH(g.type, k_gate_up, dim3((Fl + 7) / 8, nslots), dim3(256), g.d, u.d, g.rb, es_g, sid, xn, big, Fl, n, nslots);
    const int rows = (int)d.ne1;
    DISPATCH(d.type, k_down, dim3((rows + 7) / 8), dim3(256), d.d, d.rb, es_d, sid, sw, big, x, rows, Fl, nslots);
  }

  void step(int token, int pos) {
    CK(hipMemcpy(tok, &token, sizeof(int), hipMemcpyHostToDevice));
    const DT emb = up(tp(m.tok_embd));
    if (emb.type != T_Q6_K) { fprintf(stderr, "token_embd must be Q6_K\n"); exit(1); }
    k_embed_q6k<<<(D + 255) / 256, 256>>>(emb.d, emb.rb, tok, x, D);
    CK(hipGetLastError());
    for (size_t li = 0; li < m.layers.size(); ++li) {
      auto& L = m.layers[li];
      rms(x, up(tp(L.attn_norm)), xn, D);
      if (L.is_attn) {
        mv(up(tp(L.wq)), xn, hq, 0); mv(up(tp(L.wk)), xn, hk, 0); mv(up(tp(L.wv)), xn, hv, 0);
        k_qk_norm_rope<<<NQ + NKV, 64>>>(hq, hk, hv, up(tp(L.q_norm)).f(), up(tp(L.k_norm)).f(),
                                         st[li].kc, st[li].vc, pos, NQ, NKV, eps, base);
        CK(hipGetLastError());
        k_attn<<<NQ, 256>>>(hq, st[li].kc, st[li].vc, ho, scores, pos, NQ, NKV, max_ctx);
        CK(hipGetLastError());
        mv(up(tp(L.wo)), ho, x, 1);
      } else {
        mv(up(tp(L.in_proj)), xn, big, 0);
        k_shortconv<<<(D + 255) / 256, 256>>>(big, up(tp(L.conv)).f(), st[li].s0, st[li].s1, buf, D);
        CK(hipGetLastError());
        mv(up(tp(L.out_proj)), buf, x, 1);
      }
      rms(x, up(tp(L.ffn_norm)), xn, D);
      if (L.is_moe) {
        mv(up(tp(L.router)), xn, rl, 0);
        const DT b = upo(tp(L.exp_bias));
        k_router<<<1, 32>>>(rl, b.d ? b.f() : nullptr, ids, wts, E, K);
        CK(hipGetLastError());
        ffn(up(tp(L.gate_e)), up(tp(L.up_e)), up(tp(L.down_e)), ids, wts, K);
      } else {
        ffn(up(tp(L.gate)), up(tp(L.up)), up(tp(L.down)), nullptr, nullptr, 1);
      }
    }
    rms(x, up(tp(m.out_norm)), xn, D);
    DT head = emb; head.ne1 = V; head.ne0 = D;
    mv(head, xn, logits, 0);
  }
};

#include "lfm2_worker.hpp"

int main(int argc, char** argv) {
 try {
  if (argc == 2 && std::string(argv[1]) == "--help") { fputs(usage(), stdout); return 0; }
  const Options o = options(argc, argv);
  // No HIP call or model weights are needed for --tok.
  if (o.tok) return tokenize_only(o);
  auto t0 = std::chrono::steady_clock::now();
  Model m; m.load(o.path);
  Tokenizer tokenizer; tokenizer.init(m.g);
  const auto prompt = o.worker ? std::vector<int>{} : prompt_ids(o, tokenizer);
  Detok dt;
  dt.init(m.g.str_arrays.at("tokenizer.ggml.tokens"), &m.g.int_arrays.at("tokenizer.ggml.token_type"));
  hipDeviceProp_t prop;
  CK(hipGetDeviceProperties(&prop, 0));
  fprintf(stderr, "GPU: %s (%s), VRAM %.2f GB\n", prop.name, prop.gcnArchName, prop.totalGlobalMem / 1e9);
  Engine en(m, o.ctx);
  if (o.worker) {
    size_t free_bytes = 0, total_bytes = 0, needed = 64 * 1024 * 1024;
    CK(hipMemGetInfo(&free_bytes, &total_bytes));
    for (const auto& t : m.g.tensors) needed += t.second.nbytes;
    needed += (size_t)m.n_attn_layers * 2 * o.ctx * en.NKV * 64 * sizeof(float);
    if (needed > free_bytes) throw std::runtime_error("VRAM insuficiente para LFM2: descarregue outro modelo ou reduza LFM2_CTX.");
  }
  en.load();
  CK(hipDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  fprintf(stderr, "carga: %.2f s, pesos na VRAM %.2f GB\n", std::chrono::duration<double>(t1 - t0).count(), en.vram_w / 1e9);
  if (o.worker) return run_worker(en, tokenizer, dt, o);
  std::vector<float> lg(en.V);
  int pos = 0;
  for (int t : prompt) en.step(t, pos++);
  CK(hipDeviceSynchronize());
  auto t2 = std::chrono::steady_clock::now();
  std::string text;
  TextStream stream;
  std::mt19937 rng(std::random_device{}());
  int emitted = 0, decode_steps = 0, stop_id = -1;
  for (int s = 0; s < o.n_gen; ++s) {
    CK(hipMemcpy(lg.data(), en.logits, lg.size() * sizeof(float), hipMemcpyDeviceToHost));
    const int next = sample(lg, o, rng);
    if (o.prompt) {
      if (tokenizer.is_end(next)) { stop_id = next; break; }
      if (!dt.is_special(next)) stream.push(dt.piece(next));
    } else {
      // Retain the diagnostic top-five output of the original ID interface.
      auto idx = top_tokens(lg, 5);
      if (!dt.is_special(next)) text += dt.piece(next);
      printf("passo %d top5:", s);
      for (int id : idx) printf(" %d(%.3f)", id, lg[id]);
      printf(" | %s\n", text.c_str()); fflush(stdout);
      if (tokenizer.is_end(next)) { stop_id = next; break; }
    }
    ++emitted;
    if (s + 1 < o.n_gen) { en.step(next, pos++); ++decode_steps; }
  }
  if (o.prompt) stream.finish();
  CK(hipDeviceSynchronize());
  auto t3 = std::chrono::steady_clock::now();
  const double tp = std::chrono::duration<double>(t2 - t1).count();
  const double td = std::chrono::duration<double>(t3 - t2).count();
  fprintf(stderr, "prefill %zu tok: %.3f s (%.2f tok/s)\n", prompt.size(), tp, prompt.size() / tp);
  fprintf(stderr, "decode %d passos, %d tokens emitidos: %.3f s (%.2f tok/s)\n",
          decode_steps, emitted, td, td > 0 ? decode_steps / td : 0);
  if (stop_id >= 0) fprintf(stderr, "parada: token %d (%s)\n", stop_id, tokenizer.toks[stop_id].c_str());
  else fprintf(stderr, "parada: limite de %d tokens\n", o.n_gen);
  return 0;
 } catch (const std::exception& e) { fprintf(stderr, "erro: %s\n", e.what()); return 1; }
}
