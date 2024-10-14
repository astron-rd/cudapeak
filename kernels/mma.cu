#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include "precision.h"

using namespace rocwmma;
#else
#include <cuda.h>
#include <mma.h>

using namespace nvcuda::wmma;
#endif

#define REPEAT_COUNT 32768

#if defined(__HIP_PLATFORM_AMD__)

#define START                                         \
  fragment<accumulator, M, N, K, Tout> sum;           \
  fragment<matrix_a, M, N, K, Tmma, row_major> aFrag; \
  fragment<matrix_b, M, N, K, Tmma, col_major> bFrag; \
  fill_fragment(sum, static_cast<Tout>(0));           \
  fill_fragment(aFrag, static_cast<Tin>(0));          \
  fill_fragment(bFrag, static_cast<Tin>(0));          \
  for (unsigned k = 0; k < REPEAT_COUNT; k++) {
#define END                               \
  }                                       \
  Tout* ptr = &data[threadIdx.y * M * N]; \
  store_matrix_sync(ptr, sum, N, mem_row_major);

template <typename Tin, typename Tmma, typename Tout, unsigned M, unsigned N,
          unsigned K>
__device__ void mma_kernel(Tout* data) {
  START
  mma_sync(sum, aFrag, bFrag, sum);
  END
}

#include "mma_m16n16k32_fp32fp8fp8fp32.hiph"
#include "mma_m16n16k32_fp32bf8bf8fp32.hiph"

template <typename Tin, typename Tmma, typename Tout, unsigned M, unsigned N,
          unsigned K>
__device__ void mma_kernel_llvm(Tout* data) {
  START
  mma_sync_llvm(sum, aFrag, bFrag, sum);
  END
}

__global__ void mma_fp8_16_16_32(void* data) {
  mma_kernel_llvm<char, precision::fp8, float, 16, 16, 32>((float*)data);
}

__global__ void mma_bf8_16_16_32(void* data) {
  mma_kernel_llvm<char, precision::bf8, float, 16, 16, 32>((float*)data);
}

__global__ void mma_s8_16_16_32(void* data) {
  mma_kernel<signed char, signed char, int, 16, 16, 32>((int*)data);
}

__global__ void mma_f16_16_16_16(void* data) {
  mma_kernel<half, half, float, 16, 16, 16>((float*)data);
}

__global__ void mma_bf16_16_16_16(void* data) {
  mma_kernel<hip_bfloat16, hip_bfloat16, float, 16, 16, 16>((float*)data);
}

__global__ void mma_f32_16_16_16(void* data) {
  mma_kernel<float, float, float, 16, 16, 16>((float*)data);
}

__global__ void mma_f64_16_16_16(void* data) {
  mma_kernel<double, double, double, 16, 16, 16>((double*)data);
}

#else

#define START                                        \
  fragment<accumulator, M, N, K, Tout> sum;          \
  fragment<matrix_a, M, N, K, Tin, row_major> aFrag; \
  fragment<matrix_b, M, N, K, Tin, col_major> bFrag; \
  fill_fragment(sum, 0);                             \
  fill_fragment(aFrag, 0);                           \
  fill_fragment(bFrag, 0);                           \
  for (unsigned k = 0; k < REPEAT_COUNT; k++) {
#define END                               \
  __syncwarp();                           \
  }                                       \
  Tout* ptr = &data[threadIdx.y * M * N]; \
  store_matrix_sync(ptr, sum, N, mem_row_major);

template <typename Tin, typename Tout, unsigned M, unsigned N, unsigned K>
__device__ void mma_kernel(Tout* data) {
  START
  mma_sync(sum, aFrag, bFrag, sum);
  END
}

#include "mma_m8n8k32_s32s4s4s32.cuh"
#include "mma_m16n8k256_s32b1b1s32.cuh"

template <typename Tin, typename Tout, unsigned M, unsigned N, unsigned K>
__device__ void mma_kernel_ptx(Tout* data) {
  START
  mma_sync_ptx(sum, aFrag, bFrag, sum);
  END
}

template <typename Tin, typename Tout, unsigned M, unsigned N, unsigned K,
          experimental::bmmaBitOp bitOp>
__device__ void bmma_kernel(Tout* data) {
  START
  bmma_sync(sum, aFrag, bFrag, sum, bitOp);
  END
}

__global__ void bmma_b1_8_8_128_xor(void* data) {
  bmma_kernel<experimental::precision::b1, int, 8, 8, 128,
              experimental::bmmaBitOpXOR>((int*)data);
}

__global__ void bmma_b1_16_8_256_xor(void* data) {
  bmma_kernel<experimental::precision::b1, int, 16, 8, 256,
              experimental::bmmaBitOpXOR>((int*)data);
}

__global__ void bmma_b1_8_8_128_and(void* data) {
  bmma_kernel<experimental::precision::b1, int, 8, 8, 128,
              experimental::bmmaBitOpAND>((int*)data);
}

__global__ void bmma_b1_16_8_256_and(void* data) {
  bmma_kernel<experimental::precision::b1, int, 16, 8, 256,
              experimental::bmmaBitOpAND>((int*)data);
}

__global__ void mma_s4_8_8_32(void* data) {
  mma_kernel_ptx<experimental::precision::s4, int, 8, 8, 32>((int*)data);
}

__global__ void mma_s8_16_16_16(void* data) {
  mma_kernel<signed char, int, 16, 16, 16>((int*)data);
}

__global__ void mma_f16_16_16_16(void* data) {
  mma_kernel<half, float, 16, 16, 16>((float*)data);
}

__global__ void mma_bf16_16_16_16(void* data) {
  mma_kernel<__nv_bfloat16, float, 16, 16, 16>((float*)data);
}

__global__ void mma_tf32_16_16_8(void* data) {
  mma_kernel<precision::tf32, float, 16, 16, 8>((float*)data);
}
#endif