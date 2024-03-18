#include <cuda.h>
#include <mma.h>

using namespace nvcuda::wmma;

#define M 16
#define N 16
#define REPEAT_COUNT 32768

template <typename Tin, typename Tout, unsigned K>
__device__ void mma_kernel(Tout* data) {
  fragment<accumulator, M, N, K, Tout> sum;
  fragment<matrix_a, M, N, K, Tin, row_major> aFrag;
  fragment<matrix_b, M, N, K, Tin, col_major> bFrag;

  fill_fragment(sum, 0);
  fill_fragment(aFrag, 0);
  fill_fragment(bFrag, 0);

  for (unsigned k = 0; k < REPEAT_COUNT; k++) {
    mma_sync(sum, aFrag, bFrag, sum);
  }

  Tout* ptr = &data[threadIdx.y * M * N];
  store_matrix_sync(ptr, sum, N, mem_row_major);
}

__global__ void mma8_kernel(void* data) {
  mma_kernel<signed char, int, 16>((int*)data);
}

__global__ void mma16_kernel(void* data) {
  mma_kernel<half, float, 16>((float*)data);
}

__global__ void mma32_kernel(void* data) {
  mma_kernel<precision::tf32, float, 8>((float*)data);
}