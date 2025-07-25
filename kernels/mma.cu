#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>

// rocWMMA does not support RDNA2
#if not defined(__gfx1030__) && not defined(__gfx1031__) &&                    \
    not defined(__gfx1032__)
#include <rocwmma/rocwmma-version.hpp>
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
#endif

#include <hip/hip_version.h>
#if HIP_VERSION >= 60300000
#include <hip/hip_fp8.h>
#else
struct hip_fp8_e4m3;
struct hip_fp8_e5m2;
#endif

#else
#include <cuda_fp8.h>
#include <mma.h>

using namespace nvcuda::wmma;
#endif

#define REPEAT_COUNT 32768

inline __device__ unsigned laneid() {
  unsigned laneid;

  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}

#if defined(__HIP_PLATFORM_AMD__)

#if defined(ROCWMMA_VERSION_MAJOR)
#define START                                                                  \
  fragment<accumulator, M, N, K, Tout> sum;                                    \
  fragment<matrix_a, M, N, K, Tmma, row_major> aFrag;                          \
  fragment<matrix_b, M, N, K, Tmma, col_major> bFrag;                          \
  fill_fragment(sum, static_cast<Tout>(0));                                    \
  fill_fragment(aFrag, static_cast<Tin>(0));                                   \
  fill_fragment(bFrag, static_cast<Tin>(0));                                   \
  for (unsigned k = 0; k < REPEAT_COUNT; k++) {
#define END                                                                    \
  }                                                                            \
  Tout *ptr = &data[threadIdx.y * M * N];                                      \
  store_matrix_sync(ptr, sum, N, mem_row_major);

template <typename Tin, typename Tmma, typename Tout, unsigned M, unsigned N,
          unsigned K>
__device__ void mma_kernel(Tout *data) {
  START
  mma_sync(sum, aFrag, bFrag, sum);
  END
}

#include "mma_m16n16k32_fp32bf8bf8fp32.hiph"
#include "mma_m16n16k32_fp32fp8fp8fp32.hiph"
#endif

template <typename Tin, typename Tmma, typename Tout, unsigned M, unsigned N,
          unsigned K>
__device__ void mma_kernel_llvm(Tout *data) {
#if defined(ROCWMMA_VERSION_MAJOR)
  START
  mma_sync_llvm(sum, aFrag, bFrag, sum);
  END
#endif
}

__global__ void mma_fp8_16_16_32(void *data) {
#if defined(ROCWMMA_VERSION_MAJOR)
  mma_kernel_llvm<char, precision::fp8, float, 16, 16, 32>((float *)data);
#endif
}

__global__ void mma_bf8_16_16_32(void *data) {
#if defined(ROCWMMA_VERSION_MAJOR)
  mma_kernel_llvm<char, precision::bf8, float, 16, 16, 32>((float *)data);
#endif
}

__global__ void mma_s8_16_16_32(void *data) {
#if defined(ROCWMMA_VERSION_MAJOR)
  mma_kernel<signed char, signed char, int, 16, 16, 32>((int *)data);
#endif
}

__global__ void mma_f16_16_16_16(void *data) {
#if defined(ROCWMMA_VERSION_MAJOR)
  mma_kernel<half, half, float, 16, 16, 16>((float *)data);
#endif
}

__global__ void mma_bf16_16_16_16(void *data) {
#if defined(ROCWMMA_VERSION_MAJOR)
  mma_kernel<hip_bfloat16, hip_bfloat16, float, 16, 16, 16>((float *)data);
#endif
}

__global__ void mma_f32_16_16_16(void *data) {
#if defined(ROCWMMA_VERSION_MAJOR)
  mma_kernel<float, float, float, 16, 16, 16>((float *)data);
#endif
}

__global__ void mma_f64_16_16_16(void *data) {
#if defined(ROCWMMA_VERSION_MAJOR)
  mma_kernel<double, double, double, 16, 16, 16>((double *)data);
#endif
}

__global__ void mma_xf32_16_16_8(void *data) {
#if defined(ROCWMMA_VERSION_MAJOR)
  mma_kernel<rocwmma::xfloat32_t, rocwmma::xfloat32_t, float, 16, 16, 8>(
      (float *)data);
#endif
}

#else

#define START                                                                  \
  fragment<accumulator, M, N, K, Tout> sum;                                    \
  fragment<matrix_a, M, N, K, Tin, row_major> aFrag;                           \
  fragment<matrix_b, M, N, K, Tin, col_major> bFrag;                           \
  fill_fragment(sum, 0);                                                       \
  fill_fragment(aFrag, 0);                                                     \
  fill_fragment(bFrag, 0);                                                     \
  for (unsigned k = 0; k < REPEAT_COUNT; k++) {
#define END                                                                    \
  __syncwarp();                                                                \
  }                                                                            \
  Tout *ptr = &data[threadIdx.y * M * N];                                      \
  store_matrix_sync(ptr, sum, N, mem_row_major);

template <typename Tin, typename Tout, unsigned M, unsigned N, unsigned K>
__device__ void mma_kernel(Tout *data) {
  START
  mma_sync(sum, aFrag, bFrag, sum);
  END
}

// Turing only supports int1, xor, m8n8k128
#if __CUDA_ARCH__ >= 750
#define ENABLE_INT1
#endif

#if __CUDA_ARCH__ >= 800
#include "mma_m16n8k256_s32b1b1s32.cuh"
#endif

#if __CUDA_ARCH__ >= 750
#define ENABLE_INT4
#define ENABLE_INT8
#include "mma_m8n8k32_s32s4s4s32.cuh"
#endif

#if __CUDA_ARCH__ >= 800
#define ENABLE_TF32
#define ENABLE_BF16
#endif

#if __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1200
#define ENABLE_FP8
#include "mma_m16n8k32_f32f8f8f32.cuh"
#endif

template <typename Tin, typename Tout, unsigned M, unsigned N, unsigned K>
__device__ void mma_kernel_ptx(Tout *data) {
  START
  mma_sync_ptx(sum, aFrag, bFrag, sum);
  END
}

#if defined(ENABLE_INT1)
template <typename Tin, typename Tout, unsigned M, unsigned N, unsigned K,
          experimental::bmmaBitOp bitOp>
__device__ void bmma_kernel(Tout *data) {
  START
  bmma_sync(sum, aFrag, bFrag, sum, bitOp);
  END
}
#endif

__global__ void bmma_b1_8_8_128_xor(void *data) {
#if defined(ENABLE_INT1)
  bmma_kernel<experimental::precision::b1, int, 8, 8, 128,
              experimental::bmmaBitOpXOR>((int *)data);
#endif
}

__global__ void bmma_b1_16_8_256_xor(void *data) {
#if defined(ENABLE_INT1) && (__CUDA_ARCH__ >= 800)
  bmma_kernel<experimental::precision::b1, int, 16, 8, 256,
              experimental::bmmaBitOpXOR>((int *)data);
#endif
}

__global__ void bmma_b1_8_8_128_and(void *data) {
#if defined(ENABLE_INT1) && (__CUDA_ARCH__ >= 800)
  bmma_kernel<experimental::precision::b1, int, 8, 8, 128,
              experimental::bmmaBitOpAND>((int *)data);
#endif
}

__global__ void bmma_b1_16_8_256_and(void *data) {
#if defined(ENABLE_INT1) && (__CUDA_ARCH__ >= 800)
  bmma_kernel<experimental::precision::b1, int, 16, 8, 256,
              experimental::bmmaBitOpAND>((int *)data);
#endif
}

__global__ void mma_s4_8_8_32(void *data) {
#if defined(ENABLE_INT4)
  mma_kernel_ptx<experimental::precision::s4, int, 8, 8, 32>((int *)data);
#endif
}

__global__ void mma_s8_16_16_16(void *data) {
#if defined(ENABLE_INT8)
  mma_kernel<signed char, int, 16, 16, 16>((int *)data);
#endif
}

__global__ void mma_f16_16_16_16(void *data) {
  mma_kernel<half, float, 16, 16, 16>((float *)data);
}

__global__ void mma_bf16_16_16_16(void *data) {
#if defined(ENABLE_BF16)
  mma_kernel<__nv_bfloat16, float, 16, 16, 16>((float *)data);
#endif
}

__global__ void mma_tf32_16_16_8(void *data) {
#if defined(ENABLE_TF32)
  mma_kernel<precision::tf32, float, 16, 16, 8>((float *)data);
#endif
}

__global__ void mma_e4m3_16_8_32(void *data) {
#if defined(ENABLE_FP8)
  mma_kernel_ptx<__nv_fp8_e4m3, float, 16, 8, 32>((float *)data);
#endif
}

__global__ void mma_e5m2_16_8_32(void *data) {
#if defined(ENABLE_FP8)
  mma_kernel_ptx<__nv_fp8_e5m2, float, 16, 8, 32>((float *)data);
#endif
}

#endif