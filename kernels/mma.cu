#include <cuda.h>
#include <mma.h>

using namespace nvcuda::wmma;

#define REPEAT_COUNT 32768

#define START                                        \
  fragment<accumulator, M, N, K, Tout> sum;          \
  fragment<matrix_a, M, N, K, Tin, row_major> aFrag; \
  fragment<matrix_b, M, N, K, Tin, col_major> bFrag; \
  fill_fragment(sum, 0);                             \
  fill_fragment(aFrag, 0);                           \
  fill_fragment(bFrag, 0);                           \
  for (unsigned k = 0; k < REPEAT_COUNT; k++) {
#define END                               \
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

template <typename Tin, typename Tout, unsigned M, unsigned N, unsigned K>
__device__ void mma_kernel_ptx(Tout* data) {
  START
  mma_sync_ptx(sum, aFrag, bFrag, sum);
  END
}

template <typename Tin, typename Tout, unsigned M, unsigned N, unsigned K>
__device__ void bmma_kernel(Tout* data) {
  START
  bmma_sync(sum, aFrag, bFrag, sum);
  END
}

__global__ void mma_b1(void* data) {
  bmma_kernel<experimental::precision::b1, int, 8, 8, 128>((int*)data);
}

__global__ void mma_s4(void* data) {
  mma_kernel_ptx<experimental::precision::s4, int, 8, 8, 32>((int*)data);
}

__global__ void mma_s8(void* data) {
  mma_kernel<signed char, int, 16, 16, 16>((int*)data);
}

__global__ void mma_f16(void* data) {
  mma_kernel<half, float, 16, 16, 16>((float*)data);
}

__global__ void mma_bf16(void* data) {
  mma_kernel<__nv_bfloat16, float, 16, 16, 16>((float*)data);
}

__global__ void mma_tf32(void* data) {
  mma_kernel<precision::tf32, float, 16, 16, 8>((float*)data);
}