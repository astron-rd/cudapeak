#include <cuda.h>
#include <mma.h>

using namespace nvcuda::wmma;

#define M 8
#define N 8
#define REPEAT_COUNT 32768

template <typename Tin, typename Tout, unsigned K>
__device__ void bmma_kernel(Tout* data) {
  fragment<accumulator, M, N, K, Tout> sum;
  fragment<matrix_a, M, N, K, Tin, row_major> aFrag;
  fragment<matrix_b, M, N, K, Tin, col_major> bFrag;

  fill_fragment(sum, 0);
  fill_fragment(aFrag, 0);
  fill_fragment(bFrag, 0);

  for (unsigned k = 0; k < REPEAT_COUNT; k++) {
    bmma_sync(sum, aFrag, bFrag, sum);
  }

  Tout* ptr = &data[threadIdx.y * M * N];
  store_matrix_sync(ptr, sum, N, mem_row_major);
}

__global__ void bmma_kernel(void* data) {
  bmma_kernel<experimental::precision::b1, int, 128>((int*)data);
}