#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

#define FETCH_PER_BLOCK 16

__global__ void dmem_kernel(float* ptr) {
  int id = (blockIdx.x * blockDim.x * FETCH_PER_BLOCK) + threadIdx.x;

  volatile float t;
  for (int i = 0; i < FETCH_PER_BLOCK; i++) {
    t = ptr[id];
  }

  ptr[(blockIdx.x * blockDim.x) + threadIdx.x] = (float)id;
}
