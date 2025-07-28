#include <mma.h>

using namespace nvcuda;

#define START                                                                  \
  wmma::fragment<wmma::accumulator, M, N, K, Tout> sum;                        \
  wmma::fragment<wmma::matrix_a, M, N, K, Ta, wmma::row_major> aFrag;          \
  wmma::fragment<wmma::matrix_b, M, N, K, Tb, wmma::col_major> bFrag;          \
  fill_fragment(sum, 0);                                                       \
  fill_fragment(aFrag, 0);                                                     \
  fill_fragment(bFrag, 0);                                                     \
  for (unsigned k = 0; k < REPEAT_COUNT; k++) {
#define END                                                                    \
  __syncwarp();                                                                \
  }                                                                            \
  Tout *ptr = &data[threadIdx.y * M * N];                                      \
  store_matrix_sync(ptr, sum, N, wmma::mem_row_major);

template <typename Ta, typename Tb, typename Tout, unsigned M, unsigned N,
          unsigned K>
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

template <>
class wmma::fragment<wmma::accumulator, 16, 8, 32, float>
    : public __frag_base<float, 4> {};

#if __CUDA_ARCH__ >= 1000
#define ENABLE_FP4
#include "mma_m16n8k64_f32f4f4f32.cuh"
#include <cuda_fp4.h>
#define ENABLE_FP6
#include "mma_m16n8k32_f32f6f6f32.cuh"
#include <cuda_fp6.h>
#endif

#if __CUDA_ARCH__ >= 800
#define ENABLE_TF32
#define ENABLE_BF16
#endif

#if __CUDA_ARCH__ == 890 || __CUDA_ARCH__ == 900 || __CUDA_ARCH__ == 1200
#define ENABLE_FP8
#include "mma_m16n8k32_f32f8f8f32.cuh"
#include <cuda_fp8.h>
#endif

template <typename Ta, typename Tb, typename Tout, unsigned M, unsigned N,
          unsigned K>
__device__ void mma_kernel_ptx(Tout *data) {
  START
  mma_sync_ptx(sum, aFrag, bFrag, sum);
  END
}

#if defined(ENABLE_INT1)
template <typename Ta, typename Tb, typename Tout, unsigned M, unsigned N,
          unsigned K, experimental::bmmaBitOp bitOp>
__device__ void bmma_kernel(Tout *data) {
  START
  bmma_sync(sum, aFrag, bFrag, sum, bitOp);
  END
}
#endif

__global__ void bmma_b1_8_8_128_xor(void *data) {
#if defined(ENABLE_INT1)
  bmma_kernel<experimental::precision::b1, experimental::precision::b1, int, 8,
              8, 128, experimental::bmmaBitOpXOR>((int *)data);
#endif
}

__global__ void bmma_b1_16_8_256_xor(void *data) {
#if defined(ENABLE_INT1) && (__CUDA_ARCH__ >= 800)
  bmma_kernel<experimental::precision::b1, experimental::precision::b1, int, 16,
              8, 256, experimental::bmmaBitOpXOR>((int *)data);
#endif
}

__global__ void bmma_b1_8_8_128_and(void *data) {
#if defined(ENABLE_INT1) && (__CUDA_ARCH__ >= 800)
  bmma_kernel<experimental::precision::b1, experimental::precision::b1, int, 8,
              8, 128, experimental::bmmaBitOpAND>((int *)data);
#endif
}

__global__ void bmma_b1_16_8_256_and(void *data) {
#if defined(ENABLE_INT1) && (__CUDA_ARCH__ >= 800)
  bmma_kernel<experimental::precision::b1, experimental::precision::b1, int, 16,
              8, 256, experimental::bmmaBitOpAND>((int *)data);
#endif
}

__global__ void mma_s4_8_8_32(void *data) {
#if defined(ENABLE_INT4)
  mma_kernel_ptx<experimental::precision::s4, experimental::precision::s4, int,
                 8, 8, 32>((int *)data);
#endif
}

__global__ void mma_s8_16_16_16(void *data) {
#if defined(ENABLE_INT8)
  mma_kernel<signed char, signed char, int, 16, 16, 16>((int *)data);
#endif
}

__global__ void mma_f16_16_16_16(void *data) {
  mma_kernel<half, half, float, 16, 16, 16>((float *)data);
}

__global__ void mma_bf16_16_16_16(void *data) {
#if defined(ENABLE_BF16)
  mma_kernel<__nv_bfloat16, __nv_bfloat16, float, 16, 16, 16>((float *)data);
#endif
}

__global__ void mma_tf32_16_16_8(void *data) {
#if defined(ENABLE_TF32)
  mma_kernel<precision::tf32, precision::tf32, float, 16, 16, 8>((float *)data);
#endif
}

__global__ void mma_e4m3_16_8_32(void *data) {
#if defined(ENABLE_FP8)
  mma_kernel_ptx<__nv_fp8_e4m3, __nv_fp8_e4m3, float, 16, 8, 32>((float *)data);
#endif
}

__global__ void mma_e5m2_16_8_32(void *data) {
#if defined(ENABLE_FP8)
  mma_kernel_ptx<__nv_fp8_e5m2, __nv_fp8_e5m2, float, 16, 8, 32>((float *)data);
#endif
}

__global__ void mma_e2m1_16_8_64(void *data) {
#if defined(ENABLE_FP4)
  mma_kernel_ptx<__nv_fp4_e2m1, __nv_fp4_e2m1, float, 16, 8, 64>((float *)data);
#endif
}

__global__ void mma_e3m2_16_8_32(void *data) {
#if defined(ENABLE_FP6)
  mma_kernel_ptx<__nv_fp6_e3m2, __nv_fp6_e2m3, float, 16, 8, 32>((float *)data);
#endif
}
