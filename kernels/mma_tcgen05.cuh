// tcgen05 (UMMA) synthetic tensor-core kernels for Thor (sm_110a).
//
// Blackwell's tcgen05 instruction family runs one MMA per SM from a single
// issuing lane against SMEM descriptors, replacing warp-level mma.sync for
// peak throughput measurement. On Thor tcgen05 has a much higher peak
// throughput than the warp-level kernels in mma_nvidia.cuh.
//
// This header is self-contained inline PTX (no CUTLASS dependency) for the
// plain atom kinds only. Block-scaled kinds (fp4/fp6) need scale factors
// staged into TMEM and are not implemented here.
//
// Descriptor and instruction-encoding constants were extracted from CUTLASS
// 4.x CuTe atoms (SM100_MMA_*_SS) running on Thor: 128x128 accumulator,
// K-major SW128 operands in 128-byte K tiles (4 atom steps per tile).
//
// Operands are SMEM garbage; only instruction issue rate is measured. One
// final TMEM read keeps the accumulator observable to the caller.

#ifndef MMA_TCGEN05_CUH
#define MMA_TCGEN05_CUH

#if !defined(__HIP_PLATFORM_AMD__) && defined(__CUDA_ARCH__) &&                \
    (__CUDA_ARCH__ == 1100)
#define ENABLE_TCGEN05

// Tile iterations per kernel launch. Must match the constant in
// benchmarks/mma.cpp used for the work calculation.
#ifndef UMMA_REPEAT
#define UMMA_REPEAT 20000
#endif

namespace tcgen05 {

constexpr unsigned long long kDescCfg =
    0x4000404000010000ULL;                  // SW128, LBO/SBO/version per atom
constexpr unsigned long long kDescStep = 2; // +32B per atom step (16B units)

// Instruction descriptors per kind+format (from CuTe MMA_Traits::idesc_).
constexpr unsigned kIdescF16 = 0x08200010;
constexpr unsigned kIdescBf16 = 0x08200490;
constexpr unsigned kIdescTf32 = 0x08200910;
constexpr unsigned kIdescS8 = 0x082004a0;
constexpr unsigned kIdescE4m3 = 0x08200010;
constexpr unsigned kIdescE5m2 = 0x08200490;

__device__ __forceinline__ unsigned smem_u32(const void *p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ bool elect_one() {
  unsigned pred = 0;
  asm volatile("{\n\t"
               ".reg .pred P;\n\t"
               "elect.sync _|P, 0xffffffff;\n\t"
               "selp.b32 %0, 1, 0, P;\n\t"
               "}"
               : "=r"(pred));
  return pred != 0;
}

// One tcgen05.mma issue per kind; %4 selects zero-vs-accumulate of C.
struct IssuerF16 {
  __device__ static void fma(unsigned tmem, unsigned long long da,
                             unsigned long long db, unsigned idesc,
                             unsigned scale_c) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, {%5, "
                 "%6, %7, %8}, p;\n\t"
                 "}"
                 :
                 : "r"(tmem), "l"(da), "l"(db), "r"(idesc), "r"(scale_c),
                   "r"(0u), "r"(0u), "r"(0u), "r"(0u));
  }
};
struct IssuerTf32 {
  __device__ static void fma(unsigned tmem, unsigned long long da,
                             unsigned long long db, unsigned idesc,
                             unsigned scale_c) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, {%5, "
                 "%6, %7, %8}, p;\n\t"
                 "}"
                 :
                 : "r"(tmem), "l"(da), "l"(db), "r"(idesc), "r"(scale_c),
                   "r"(0u), "r"(0u), "r"(0u), "r"(0u));
  }
};
struct IssuerS8 {
  __device__ static void fma(unsigned tmem, unsigned long long da,
                             unsigned long long db, unsigned idesc,
                             unsigned scale_c) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, {%5, %6, "
                 "%7, %8}, p;\n\t"
                 "}"
                 :
                 : "r"(tmem), "l"(da), "l"(db), "r"(idesc), "r"(scale_c),
                   "r"(0u), "r"(0u), "r"(0u), "r"(0u));
  }
};
struct IssuerF8F6F4 {
  __device__ static void fma(unsigned tmem, unsigned long long da,
                             unsigned long long db, unsigned idesc,
                             unsigned scale_c) {
    asm volatile("{\n\t"
                 ".reg .pred p;\n\t"
                 "setp.ne.b32 p, %4, 0;\n\t"
                 "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, {%5, "
                 "%6, %7, %8}, p;\n\t"
                 "}"
                 :
                 : "r"(tmem), "l"(da), "l"(db), "r"(idesc), "r"(scale_c),
                   "r"(0u), "r"(0u), "r"(0u), "r"(0u));
  }
};

// One synthetic UMMA benchmark kernel: 128x128 accumulator, one K tile
// (128B wide, 4 atom steps) reused UMMA_REPEAT times. 4 warps; warp 0
// allocates TMEM and issues, all warps read the accumulator at the end.
template <class ISSUER, unsigned IDESC>
__device__ __forceinline__ void run(void *data) {
  __shared__ __align__(16) unsigned char smem_a[128 * 128];
  __shared__ __align__(16) unsigned char smem_b[128 * 128];
  __shared__ unsigned long long mbar;
  __shared__ unsigned tmem_holder;

  const unsigned tid = threadIdx.y * blockDim.x + threadIdx.x;
  const unsigned warp = tid >> 5;

  if (tid == 0) {
    asm volatile(
        "mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(smem_u32(&mbar)));
  }
  if (warp == 0) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" ::
            "r"(smem_u32(&tmem_holder)));
  }
  __syncthreads();
  const unsigned tmem = tmem_holder;

  if (warp == 0) {
    const unsigned long long da = kDescCfg | (smem_u32(smem_a) >> 4);
    const unsigned long long db = kDescCfg | (smem_u32(smem_b) >> 4);
    unsigned scale_c = 0;
    if (elect_one()) {
      for (int r = 0; r < UMMA_REPEAT; ++r) {
#pragma unroll
        for (int kb = 0; kb < 4; ++kb) {
          ISSUER::fma(tmem, da + kDescStep * kb, db + kDescStep * kb, IDESC,
                      scale_c);
          scale_c = 1;
        }
      }
      // Commit: the mbarrier flips when all MMAs issued here have completed.
      asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::"
                   "cluster.b64 "
                   "[%0];" ::"r"(smem_u32(&mbar))
                   : "memory");
    }
  }

  {
    // All warps wait for MMA completion via the mbarrier phase.
    asm volatile("{\n\t"
                 ".reg .pred P1;\n\t"
                 "LAB_WAIT:\n\t"
                 "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n\t"
                 "@P1 bra DONE;\n\t"
                 "bra LAB_WAIT;\n\t"
                 "DONE:\n\t"
                 "}"
                 :
                 : "r"(smem_u32(&mbar)), "r"(0u));
  }
  unsigned acc = 0;
  {
    // One TMEM word per warp; garbage values are fine for a synthetic probe.
    const unsigned taddr = tmem + ((warp << 5) << 16);
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];"
                 : "=r"(acc)
                 : "r"(taddr));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
  }
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" ::"r"(tmem));
  }
  if (acc == 0xdeadbeefu) {
    *static_cast<volatile unsigned *>(data) = acc;
  }
}

} // namespace tcgen05

__global__ void tcgen05_f16_128_128_64(void *data) {
  tcgen05::run<tcgen05::IssuerF16, tcgen05::kIdescF16>(data);
}
__global__ void tcgen05_bf16_128_128_64(void *data) {
  tcgen05::run<tcgen05::IssuerF16, tcgen05::kIdescBf16>(data);
}
__global__ void tcgen05_tf32_128_128_32(void *data) {
  tcgen05::run<tcgen05::IssuerTf32, tcgen05::kIdescTf32>(data);
}
__global__ void tcgen05_s8_128_128_128(void *data) {
  tcgen05::run<tcgen05::IssuerS8, tcgen05::kIdescS8>(data);
}
__global__ void tcgen05_e4m3_128_128_128(void *data) {
  tcgen05::run<tcgen05::IssuerF8F6F4, tcgen05::kIdescE4m3>(data);
}
__global__ void tcgen05_e5m2_128_128_128(void *data) {
  tcgen05::run<tcgen05::IssuerF8F6F4, tcgen05::kIdescE5m2>(data);
}

#endif // ENABLE_TCGEN05
#endif // MMA_TCGEN05_CUH
