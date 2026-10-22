// fp8wo_tc.cu: weight-only block-FP8 GEMM for decode-sized M on SM100/SM103 (tcgen05), bf16 activations.
//
//   out[m, col(n)] = sum_k x[m, k] * (q[n, k] * s[n / 128, k / 128])      (q e4m3, s fp32, x bf16, fp32 accumulate)
//
// One CTA owns 128 weight rows (the MMA M) and a contiguous range of 128-wide K blocks (split-K over gridDim.y).
// Warp roles (8 warps):
//   warp 0      TMA producer: per K block, the 128x128 e4m3 weight tile (16 KB, SWIZZLE_128B) and the x tile
//               [BM rows x 128] bf16 as two 64-wide SWIZZLE_128B boxes, into a STAGES-deep ring.
//   warp 1      TMEM allocator + MMA issuer: 8 tcgen05.mma kind::f16 per K block, A from tensor memory, B (x) from
//               shared memory, fp32 accumulator [128 lanes x BM columns] in tensor memory.
//   warps 4-7   widen: each thread owns one weight row (TMEM lane), reads its 128 e4m3 bytes, converts with
//               cvt.rn.bf16x2.e4m3x2 and scales with mul.rn.bf16x2, and stores 64 columns into one of two A
//               buffers in tensor memory (tcgen05.st 32x32b.x64); then the epilogue (tcgen05.ld of the
//               accumulator, store or split-K partial + last-CTA reduction with a self-resetting counter).
// The weight is never written back to shared memory in bf16: shared memory sees 1 byte per weight byte (TMA in,
// one read by the widen warps).
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

namespace fp8wo {

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(bar), "r"(count) : "memory");
}

__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n\t.reg .pred P1;\n\t"
      "WAIT_%=:\n\t"
      "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P1, [%0], %1;\n\t"
      "@P1 bra.uni DONE_%=;\n\t"
      "bra.uni WAIT_%=;\n\t"
      "DONE_%=:\n\t}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}

__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];" ::"r"(bar) : "memory");
}

__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;" ::"r"(bar), "r"(bytes)
               : "memory");
}

__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* map, int c0, int c1, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];" ::"r"(
          dst),
      "l"(map), "r"(c0), "r"(c1), "r"(bar)
      : "memory");
}

__device__ __forceinline__ void bulk_load(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}

__device__ __forceinline__ void tc_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" ::"r"(bar)
               : "memory");
}

// K-major SWIZZLE_128B shared-memory matrix descriptor (SBO = 1024 B between 8-row groups, descriptor version 1).
__device__ __forceinline__ uint64_t sw128_desc(uint32_t addr) {
  return (uint64_t)((addr & 0x3FFFF) >> 4) | ((uint64_t)(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
}

__device__ __forceinline__ void mma_bf16_tmemA(uint32_t d_tmem, uint32_t a_tmem, uint64_t b_desc, uint32_t idesc,
                                               uint32_t accumulate) {
  asm volatile(
      "{\n\t.reg .pred p;\n\t"
      "setp.ne.b32 p, %4, 0;\n\t"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, p;\n\t}\n" ::"r"(d_tmem),
      "r"(a_tmem), "l"(b_desc), "r"(idesc), "r"(accumulate)
      : "memory");
}

// two e4m3 (low byte = lower K) -> bf16x2 of (value * scale). ptxas 13.0 has no direct e4m3x2 -> bf16x2 cvt, so go
// through f16x2 (exact for e4m3) and f32 (scale multiply), then one round to bf16.
__device__ __forceinline__ uint32_t widen2(uint16_t two_e4m3, float s) {
  uint32_t h2;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h2) : "h"(two_e4m3));
  __half2 h = *reinterpret_cast<__half2*>(&h2);
  const float lo = __low2float(h) * s;
  const float hi = __high2float(h) * s;
  __nv_bfloat162 b = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&b);
}

template <int N>
__device__ __forceinline__ void tmem_ld_32x32b(uint32_t taddr, uint32_t* v);

#define FP8WO_TMEM_LD(NCOL, ...)                                                                    \
  template <>                                                                                       \
  __device__ __forceinline__ void tmem_ld_32x32b<NCOL>(uint32_t taddr, uint32_t * v) {             \
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x" #NCOL ".b32 " __VA_ARGS__ ";" : FP8WO_OUTS##NCOL \
                 : "r"(taddr));                                                                     \
  }

#define FP8WO_OUTS16                                                                                         \
  "=r"(v[0]), "=r"(v[1]), "=r"(v[2]), "=r"(v[3]), "=r"(v[4]), "=r"(v[5]), "=r"(v[6]), "=r"(v[7]), "=r"(v[8]), \
      "=r"(v[9]), "=r"(v[10]), "=r"(v[11]), "=r"(v[12]), "=r"(v[13]), "=r"(v[14]), "=r"(v[15])
FP8WO_TMEM_LD(16, "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16]")

template <int NCOLS>
__device__ __forceinline__ void tmem_ld_cols(uint32_t taddr, uint32_t* v) {
#pragma unroll
  for (int c = 0; c < NCOLS; c += 16) tmem_ld_32x32b<16>(taddr + c, v + c);
}

__device__ __forceinline__ void tmem_st_x64(uint32_t taddr, const uint32_t* v) {
  asm volatile(
      "tcgen05.st.sync.aligned.32x32b.x64.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,%17,"
      "%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,"
      "%45,%46,%47,%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64};" ::"r"(taddr),
      "r"(v[0]), "r"(v[1]), "r"(v[2]), "r"(v[3]), "r"(v[4]), "r"(v[5]), "r"(v[6]), "r"(v[7]), "r"(v[8]), "r"(v[9]),
      "r"(v[10]), "r"(v[11]), "r"(v[12]), "r"(v[13]), "r"(v[14]), "r"(v[15]), "r"(v[16]), "r"(v[17]), "r"(v[18]),
      "r"(v[19]), "r"(v[20]), "r"(v[21]), "r"(v[22]), "r"(v[23]), "r"(v[24]), "r"(v[25]), "r"(v[26]), "r"(v[27]),
      "r"(v[28]), "r"(v[29]), "r"(v[30]), "r"(v[31]), "r"(v[32]), "r"(v[33]), "r"(v[34]), "r"(v[35]), "r"(v[36]),
      "r"(v[37]), "r"(v[38]), "r"(v[39]), "r"(v[40]), "r"(v[41]), "r"(v[42]), "r"(v[43]), "r"(v[44]), "r"(v[45]),
      "r"(v[46]), "r"(v[47]), "r"(v[48]), "r"(v[49]), "r"(v[50]), "r"(v[51]), "r"(v[52]), "r"(v[53]), "r"(v[54]),
      "r"(v[55]), "r"(v[56]), "r"(v[57]), "r"(v[58]), "r"(v[59]), "r"(v[60]), "r"(v[61]), "r"(v[62]), "r"(v[63])
      : "memory");
}

#ifndef FP8WO_DBG
#define FP8WO_DBG 0
#endif
// FP8WO_DBG: 1 = skip widen (store raw words), 2 = skip MMA, 3 = skip tcgen05.st, 4 = skip LDS
struct Params {
  const uint8_t* wt;    // tiled weight [n_tiles][nkb][8 chunks][128 rows][16 B] (TILED=1)
  const float* s;       // [ceil(N/128), K/128] fp32
  int s_stride;         // row stride of s (elements)
  void* out;            // [M, ldo] bf16 or fp32
  int ldo;
  int M, N, nkb;        // nkb = K / 128
  int split_k;
  float* ws;            // split-K partials
  int* cnt;             // per N-tile counters (self-resetting)
  int gap_at, gap_len;  // output column remap: col = n + (n >= gap_at ? gap_len : 0)
  int cluster_reduce;   // split-K CTAs of one N tile form a cluster (1, split_k); reduce through DSMEM
  int pdl;              // launched with programmatic stream serialization: prefetch weights before the wait
};

template <int BM, int STAGES, bool OUT_F32, bool TILED>
__global__ void __launch_bounds__(256, 2)
    fp8wo_tc_kernel(const __grid_constant__ CUtensorMap w_map, const __grid_constant__ CUtensorMap x_map,
                    const Params p) {
  constexpr int W_BYTES = 128 * 128;           // e4m3 weight tile
  constexpr int XH_BYTES = BM * 128;            // one 64-wide bf16 half of the x tile
  constexpr int STAGE_BYTES = W_BYTES + 2 * XH_BYTES;
  constexpr int ACC_COLS = BM < 32 ? 32 : BM;
  constexpr int TMEM_COLS = (ACC_COLS + 128) <= 256 ? 256 : 512;
  constexpr uint32_t IDESC = (1u << 4) | (1u << 7) | (1u << 10) | ((uint32_t)(BM >> 3) << 17) | ((128u >> 4) << 24);

  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint8_t* ring = smem;                                       // STAGES x STAGE_BYTES (1024-aligned)
  uint64_t* bars = reinterpret_cast<uint64_t*>(ring + STAGES * STAGE_BYTES);
  // bars: full[STAGES], empty[STAGES], afull[2], aempty[2], accfull[1]
  uint32_t* tmem_slot = reinterpret_cast<uint32_t*>(bars + 2 * STAGES + 5);
  int* last_flag = reinterpret_cast<int*>(tmem_slot + 1);

  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n_tile = blockIdx.x;
  const int n0 = n_tile * 128;
  const int kb_begin = (int)(((long long)blockIdx.y * p.nkb) / p.split_k);
  const int kb_end = (int)(((long long)(blockIdx.y + 1) * p.nkb) / p.split_k);
  const int nk = FP8WO_DBG == 7 ? 0 : kb_end - kb_begin;

  const uint32_t bar_full = smem_u32(bars);
  const uint32_t bar_empty = bar_full + 8 * STAGES;
  const uint32_t bar_afull = bar_empty + 8 * STAGES;
  const uint32_t bar_aempty = bar_afull + 16;
  const uint32_t bar_acc = bar_aempty + 16;

  if (threadIdx.x == 0) {
    for (int i = 0; i < STAGES; i++) {
      mbar_init(bar_full + 8 * i, 1);
      mbar_init(bar_empty + 8 * i, 128 + 1);
    }
    for (int i = 0; i < 2; i++) {
      mbar_init(bar_afull + 8 * i, 128);
      mbar_init(bar_aempty + 8 * i, 1);
    }
    mbar_init(bar_acc, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&w_map) : "memory");
    asm volatile("prefetch.tensormap [%0];" ::"l"(&x_map) : "memory");
  }
  if (warp == 1) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(smem_u32(tmem_slot)),
                 "r"(TMEM_COLS)
                 : "memory");
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
  const uint32_t tmem = *reinterpret_cast<volatile uint32_t*>(tmem_slot);
  const uint32_t tmem_acc = tmem;                 // columns [0, ACC_COLS)
  const uint32_t tmem_a = tmem + ACC_COLS;        // two A buffers of 64 columns

  if (warp == 0) {
    if (lane == 0 && nk > 0) {
      // The weights do not depend on the previous kernel: issue the first ring of weight loads before waiting on it
      // (PDL), then the activation loads.
      const int pre = nk < STAGES ? nk : STAGES;
      for (int i = 0; i < pre; i++) {
        const uint32_t base = smem_u32(ring + i * STAGE_BYTES);
        mbar_arrive_expect_tx(bar_full + 8 * i, FP8WO_DBG == 6 ? W_BYTES : STAGE_BYTES);
        if (TILED) bulk_load(base, p.wt + ((size_t)n_tile * p.nkb + kb_begin + i) * W_BYTES, W_BYTES, bar_full + 8 * i);
        else tma_load_2d(base, &w_map, (kb_begin + i) * 128, n0, bar_full + 8 * i);
      }
      if (p.pdl) asm volatile("griddepcontrol.wait;" ::: "memory");
      for (int i = 0; i < nk; i++) {
        const int st = i % STAGES;
        const uint32_t base = smem_u32(ring + st * STAGE_BYTES);
        const int k = (kb_begin + i) * 128;
        if (i >= pre) {
          const uint32_t ph = ((i / STAGES) & 1) ^ 1;
          mbar_wait(bar_empty + 8 * st, ph);
          mbar_arrive_expect_tx(bar_full + 8 * st, FP8WO_DBG == 6 ? W_BYTES : STAGE_BYTES);
          if (TILED) bulk_load(base, p.wt + ((size_t)n_tile * p.nkb + kb_begin + i) * W_BYTES, W_BYTES, bar_full + 8 * st);
          else tma_load_2d(base, &w_map, k, n0, bar_full + 8 * st);
        }
        if (FP8WO_DBG != 6) {
          tma_load_2d(base + W_BYTES, &x_map, k, 0, bar_full + 8 * st);
          tma_load_2d(base + W_BYTES + XH_BYTES, &x_map, k + 64, 0, bar_full + 8 * st);
        }
      }
      if (p.pdl) asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
    }
  } else if (warp == 1) {
    if (lane == 0 && nk > 0) {
      for (int i = 0; i < nk; i++) {
        const int st = i % STAGES;
        const int ab = i & 1;
        mbar_wait(bar_full + 8 * st, (i / STAGES) & 1);
        if (FP8WO_DBG == 5 || FP8WO_DBG == 6) { mbar_arrive(bar_empty + 8 * st); continue; }
        mbar_wait(bar_afull + 8 * ab, (i >> 1) & 1);
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        const uint32_t xb = smem_u32(ring + st * STAGE_BYTES + W_BYTES);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const uint64_t bd = sw128_desc(xb + (j >> 2) * XH_BYTES + (j & 3) * 32);
          if (FP8WO_DBG != 2 || (i == 0 && j == 0))
            mma_bf16_tmemA(tmem_acc, tmem_a + ab * 64 + j * 8, bd, IDESC, (i > 0 || j > 0) ? 1u : 0u);
        }
        tc_commit(bar_aempty + 8 * ab);
        tc_commit(bar_empty + 8 * st);
      }
      if (FP8WO_DBG == 5 || FP8WO_DBG == 6) mbar_arrive(bar_acc); else tc_commit(bar_acc);
    }
  } else if (warp >= 4) {
    const int row = ((warp & 3) << 5) + lane;       // weight row in the tile == TMEM lane
    const uint32_t lane_off = (uint32_t)((warp & 3) << 5) << 16;
    const float* srow = p.s + (size_t)(n0 >> 7) * p.s_stride;
    for (int i = 0; i < nk; i++) {
      const int st = i % STAGES;
      const int ab = i & 1;
      const float sv = __ldg(srow + kb_begin + i);
      mbar_wait(bar_full + 8 * st, (i / STAGES) & 1);
      if (FP8WO_DBG == 5 || FP8WO_DBG == 6) { mbar_arrive(bar_empty + 8 * st); continue; }
      const uint8_t* wt = ring + st * STAGE_BYTES + row * 128;
      uint4 q[8];
#pragma unroll
      for (int c = 0; c < 8; c++) {
        if (FP8WO_DBG == 4) q[c] = make_uint4(row, c, i, 0);
        else if (TILED) q[c] = *reinterpret_cast<const uint4*>(ring + st * STAGE_BYTES + (c * 128 + row) * 16);
        else q[c] = *reinterpret_cast<const uint4*>(wt + ((c ^ (row & 7)) << 4));
      }
      mbar_arrive(bar_empty + 8 * st);
      uint32_t a[64];
#pragma unroll
      for (int c = 0; c < 8; c++) {
        const uint32_t w4[4] = {q[c].x, q[c].y, q[c].z, q[c].w};
#pragma unroll
        for (int e = 0; e < 4; e++) {
          if (FP8WO_DBG == 1) {
            a[c * 8 + e * 2 + 0] = w4[e];
            a[c * 8 + e * 2 + 1] = w4[e] ^ 0x5555;
          } else {
            a[c * 8 + e * 2 + 0] = widen2((uint16_t)(w4[e] & 0xFFFF), sv);
            a[c * 8 + e * 2 + 1] = widen2((uint16_t)(w4[e] >> 16), sv);
          }
        }
      }
      mbar_wait(bar_aempty + 8 * ab, ((i >> 1) & 1) ^ 1);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      if (FP8WO_DBG != 3) tmem_st_x64(tmem_a + ab * 64 + lane_off, a);
      else if (a[i & 63] == 0x12345678u) tmem_st_x64(tmem_a + ab * 64 + lane_off, a);
      asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
      asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
      mbar_arrive(bar_afull + 8 * ab);
    }
    // ---- epilogue ----
    const int n = n0 + row;
    const int col = n + (n >= p.gap_at ? p.gap_len : 0);
    uint32_t v[ACC_COLS];
    if (nk > 0) {
      mbar_wait(bar_acc, 0);
      asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
      tmem_ld_cols<ACC_COLS>(tmem_acc + lane_off, v);
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    } else {
#pragma unroll
      for (int m = 0; m < ACC_COLS; m++) v[m] = 0u;
    }
    if (p.split_k == 1) {
      if (n < p.N) {
#pragma unroll
        for (int m = 0; m < BM; m++) {
          if (m < p.M) {
            const float f = __uint_as_float(v[m]);
            if constexpr (OUT_F32) {
              reinterpret_cast<float*>(p.out)[(size_t)m * p.ldo + col] = f;
            } else {
              reinterpret_cast<__nv_bfloat16*>(p.out)[(size_t)m * p.ldo + col] = __float2bfloat16_rn(f);
            }
          }
        }
      }
    } else if (p.cluster_reduce) {
      float* red = reinterpret_cast<float*>(ring) + row * BM;   // ring is free: all loads consumed, all MMAs done
#pragma unroll
      for (int m = 0; m < BM; m += 4)
        *reinterpret_cast<float4*>(red + m) = make_float4(__uint_as_float(v[m]), __uint_as_float(v[m + 1]),
                                                          __uint_as_float(v[m + 2]), __uint_as_float(v[m + 3]));
    } else {
      float* part = p.ws + ((size_t)(blockIdx.y * gridDim.x + n_tile) * 128 + row) * BM;
#pragma unroll
      for (int m = 0; m < BM; m += 4)
        __stcg(reinterpret_cast<float4*>(part + m),
               make_float4(__uint_as_float(v[m]), __uint_as_float(v[m + 1]), __uint_as_float(v[m + 2]),
                           __uint_as_float(v[m + 3])));
      __threadfence();
      asm volatile("bar.sync 1, 128;" ::: "memory");
      if (warp == 4 && lane == 0) {
        const int prev = atomicAdd(p.cnt + n_tile, 1);
        *last_flag = (prev == p.split_k - 1) ? 1 : 0;
      }
      asm volatile("bar.sync 1, 128;" ::: "memory");
      if (*reinterpret_cast<volatile int*>(last_flag)) {
        __threadfence();
        float acc[BM];
#pragma unroll
        for (int m = 0; m < BM; m++) acc[m] = 0.f;
        for (int z = 0; z < p.split_k; z++) {
          const float* pz = p.ws + ((size_t)(z * gridDim.x + n_tile) * 128 + row) * BM;
#pragma unroll
          for (int m = 0; m < BM; m += 4) {
            const float4 t = __ldcg(reinterpret_cast<const float4*>(pz + m));
            acc[m] += t.x;
            acc[m + 1] += t.y;
            acc[m + 2] += t.z;
            acc[m + 3] += t.w;
          }
        }
        if (n < p.N) {
#pragma unroll
          for (int m = 0; m < BM; m++) {
            if (m < p.M) {
              if constexpr (OUT_F32) {
                reinterpret_cast<float*>(p.out)[(size_t)m * p.ldo + col] = acc[m];
              } else {
                reinterpret_cast<__nv_bfloat16*>(p.out)[(size_t)m * p.ldo + col] = __float2bfloat16_rn(acc[m]);
              }
            }
          }
        }
        if (warp == 4 && lane == 0) p.cnt[n_tile] = 0;
      }
    }
  }
  if (p.split_k > 1 && p.cluster_reduce) {
    // every split of this N tile is one CTA of the cluster; rank r reduces the rows with row % split == r
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    if (warp >= 4) {
      uint32_t rank;
      asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
      const int row = ((warp & 3) << 5) + lane;
      const int n = n0 + row;
      if ((uint32_t)(row % p.split_k) == rank && n < p.N) {
        const int col = n + (n >= p.gap_at ? p.gap_len : 0);
        float acc[BM];
#pragma unroll
        for (int m = 0; m < BM; m++) acc[m] = 0.f;
        const uint32_t local = smem_u32(reinterpret_cast<float*>(ring) + row * BM);
        for (int z = 0; z < p.split_k; z++) {
          uint32_t remote;
          asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(z));
#pragma unroll
          for (int m = 0; m < BM; m += 4) {
            float a0, a1, a2, a3;
            asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];"
                         : "=f"(a0), "=f"(a1), "=f"(a2), "=f"(a3)
                         : "r"(remote + m * 4));
            acc[m] += a0;
            acc[m + 1] += a1;
            acc[m + 2] += a2;
            acc[m + 3] += a3;
          }
        }
#pragma unroll
        for (int m = 0; m < BM; m++) {
          if (m < p.M) {
            if constexpr (OUT_F32) {
              reinterpret_cast<float*>(p.out)[(size_t)m * p.ldo + col] = acc[m];
            } else {
              reinterpret_cast<__nv_bfloat16*>(p.out)[(size_t)m * p.ldo + col] = __float2bfloat16_rn(acc[m]);
            }
          }
        }
      }
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
  }
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
  __syncthreads();
  if (warp == 1) {
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(tmem), "r"(TMEM_COLS) : "memory");
  }
}

template <int BM, int STAGES>
constexpr int smem_bytes() {
  return 1024 + STAGES * (128 * 128 + 2 * BM * 128) + 8 * (2 * STAGES + 5) + 16;
}

}  // namespace fp8wo

// ------------------------------------------------------------------------------------------------------------------
// host side
// ------------------------------------------------------------------------------------------------------------------
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <unordered_map>
#include <mutex>

namespace fp8wo {

typedef CUresult (*EncodeTiledFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
                                  const cuuint64_t*, const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave,
                                  CUtensorMapSwizzle, CUtensorMapL2promotion, CUtensorMapFloatOOBfill);

static EncodeTiledFn get_encode() {
  static EncodeTiledFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    void* ptr = nullptr;
    TORCH_CHECK(cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &ptr, cudaEnableDefault, &q) == cudaSuccess &&
                    ptr != nullptr,
                "cuTensorMapEncodeTiled not available");
    fn = reinterpret_cast<EncodeTiledFn>(ptr);
  }
  return fn;
}

static CUtensorMap make_map(void* base, CUtensorMapDataType dt, int elem_bytes, int64_t inner, int64_t outer,
                            int64_t row_stride_elems, int box_inner, int box_outer) {
  CUtensorMap m;
  cuuint64_t dims[2] = {(cuuint64_t)inner, (cuuint64_t)outer};
  cuuint64_t strides[1] = {(cuuint64_t)(row_stride_elems * elem_bytes)};
  cuuint32_t box[2] = {(cuuint32_t)box_inner, (cuuint32_t)box_outer};
  cuuint32_t es[2] = {1, 1};
  CUresult r = get_encode()(&m, dt, 2, base, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                            CU_TENSOR_MAP_SWIZZLE_128B, CU_TENSOR_MAP_L2_PROMOTION_L2_256B,
                            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", (int)r);
  return m;
}

struct WKey {
  uintptr_t p;
  int64_t n, k;
  bool operator==(const WKey& o) const { return p == o.p && n == o.n && k == o.k; }
};
struct WKeyHash {
  size_t operator()(const WKey& k) const { return std::hash<uintptr_t>()(k.p) ^ (std::hash<int64_t>()(k.n) << 1); }
};
static std::unordered_map<WKey, CUtensorMap, WKeyHash> g_wmaps;
static std::mutex g_mu;

template <int BM, int STAGES, bool F32, bool TL>
static void launch(const CUtensorMap& wm, const CUtensorMap& xm, const Params& p, int grid_n, int split_k,
                   cudaStream_t st) {
  constexpr int SMEM = smem_bytes<BM, STAGES>();
  auto kern = fp8wo_tc_kernel<BM, STAGES, F32, TL>;
  static bool attr = false;
  if (!attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM);
    cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    attr = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid_n, split_k, 1);
  cfg.blockDim = dim3(256, 1, 1);
  cfg.dynamicSmemBytes = SMEM;
  cfg.stream = st;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (p.cluster_reduce && split_k > 1) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = 1;
    at[na].val.clusterDim.y = split_k;
    at[na].val.clusterDim.z = 1;
    na++;
  }
  if (p.pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    na++;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  C10_CUDA_CHECK(cudaLaunchKernelEx(&cfg, kern, wm, xm, p));
}

template <int BM, bool F32, bool TL>
static void launch_stages(int stages, const CUtensorMap& wm, const CUtensorMap& xm, const Params& p, int gn, int sk,
                          cudaStream_t st) {
  switch (stages) {
    case 2: launch<BM, 2, F32, TL>(wm, xm, p, gn, sk, st); break;
    case 3: launch<BM, 3, F32, TL>(wm, xm, p, gn, sk, st); break;
    case 4: launch<BM, 4, F32, TL>(wm, xm, p, gn, sk, st); break;
    case 6: launch<BM, 6, F32, TL>(wm, xm, p, gn, sk, st); break;
    case 8: launch<BM, 8, F32, TL>(wm, xm, p, gn, sk, st); break;
    default: TORCH_CHECK(false, "stages must be 2, 3, 4 or 6");
  }
}

void run(torch::Tensor x, torch::Tensor w, torch::Tensor s, torch::Tensor out, torch::Tensor ws, torch::Tensor cnt,
         int64_t split_k, int64_t gap_at, int64_t gap_len, int64_t stages, int64_t n_rows, int64_t cluster_reduce,
         int64_t pdl) {
  const bool tiled = w.dim() == 3;
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16 && x.dim() == 2 && x.stride(1) == 1);
  TORCH_CHECK(w.is_contiguous() && (tiled ? (w.scalar_type() == at::kByte && w.size(2) == 128 * 128)
                                          : (w.scalar_type() == at::kFloat8_e4m3fn && w.dim() == 2)));
  TORCH_CHECK(s.scalar_type() == at::kFloat && s.stride(1) == 1);
  TORCH_CHECK(out.stride(1) == 1 && (out.scalar_type() == at::kFloat || out.scalar_type() == at::kBFloat16));
  const int64_t M = x.size(0), K = x.size(1), N = tiled ? n_rows : w.size(0);
  TORCH_CHECK(K % 128 == 0 && M >= 1 && M <= 128);
  TORCH_CHECK(tiled ? (w.size(0) == (N + 127) / 128 && w.size(1) == K / 128) : w.size(1) == K);
  TORCH_CHECK((x.stride(0) * 2) % 16 == 0, "x row stride must be a multiple of 16 bytes");
  const int grid_n = (int)((N + 127) / 128);
  const int sk = (int)split_k;
  const int BM = M <= 16 ? 16 : (M <= 32 ? 32 : (M <= 64 ? 64 : 128));
  if (sk > 1 && !cluster_reduce) {
    TORCH_CHECK(ws.numel() >= (int64_t)sk * grid_n * 128 * BM && cnt.numel() >= grid_n);
  }
  TORCH_CHECK(!cluster_reduce || sk <= 16, "cluster split-K supports at most 16 splits");
  const c10::cuda::CUDAGuard guard(x.device());
  CUtensorMap wm;
  if (tiled) {
    memset(&wm, 0, sizeof(wm));
  } else {
    std::lock_guard<std::mutex> lk(g_mu);
    WKey key{(uintptr_t)w.data_ptr(), N, K};
    auto it = g_wmaps.find(key);
    if (it == g_wmaps.end()) {
      it = g_wmaps.emplace(key, make_map(w.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1, K, N, K, 128, 128)).first;
    }
    wm = it->second;
  }
  CUtensorMap xm = make_map(x.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, K, M, x.stride(0), 64, BM);
  Params p;
  p.wt = tiled ? reinterpret_cast<const uint8_t*>(w.data_ptr()) : nullptr;
  p.s = s.data_ptr<float>();
  p.s_stride = (int)s.stride(0);
  p.out = out.data_ptr();
  p.ldo = (int)out.stride(0);
  p.M = (int)M;
  p.N = (int)N;
  p.nkb = (int)(K / 128);
  p.split_k = sk;
  p.ws = sk > 1 ? ws.data_ptr<float>() : nullptr;
  p.cnt = sk > 1 ? cnt.data_ptr<int>() : nullptr;
  p.gap_at = (int)gap_at;
  p.gap_len = (int)gap_len;
  p.cluster_reduce = (int)cluster_reduce;
  p.pdl = (int)pdl;
  auto st = at::cuda::getCurrentCUDAStream();
  const bool f32 = out.scalar_type() == at::kFloat;
  const int stg = (int)stages;
#define FP8WO_DISPATCH(BMV)                                                                  \
  if (tiled) {                                                                               \
    if (f32) launch_stages<BMV, true, true>(stg, wm, xm, p, grid_n, sk, st);                  \
    else launch_stages<BMV, false, true>(stg, wm, xm, p, grid_n, sk, st);                     \
  } else {                                                                                   \
    if (f32) launch_stages<BMV, true, false>(stg, wm, xm, p, grid_n, sk, st);                 \
    else launch_stages<BMV, false, false>(stg, wm, xm, p, grid_n, sk, st);                    \
  }
  if (BM == 16) { FP8WO_DISPATCH(16) }
  else if (BM == 32) { FP8WO_DISPATCH(32) }
  else if (BM == 64) { FP8WO_DISPATCH(64) }
  else { FP8WO_DISPATCH(128) }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void forget_weight(int64_t ptr) {
  std::lock_guard<std::mutex> lk(g_mu);
  for (auto it = g_wmaps.begin(); it != g_wmaps.end();) {
    if (it->first.p == (uintptr_t)ptr) it = g_wmaps.erase(it); else ++it;
  }
}

}  // namespace fp8wo

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &fp8wo::run, "fp8 weight-only GEMM (tcgen05)");
  m.def("forget_weight", &fp8wo::forget_weight, "drop a cached weight tensor map");
}
