#include "cuda.h"

#define FETCH_PER_BLOCK 16

__global__ void dmem_kernel(float* ptr) {
  int id = (blockIdx.x * blockDim.x * FETCH_PER_BLOCK) + threadIdx.x;

  asm(".reg .f32 t;");
  for (int i = 0; i < FETCH_PER_BLOCK; i++) {
    asm("ld.global.f32 t, [%0];" : : "l"(ptr + id));
  }

  ptr[(blockIdx.x * blockDim.x) + threadIdx.x] = (float)id;
}
