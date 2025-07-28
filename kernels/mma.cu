#define REPEAT_COUNT 32768

inline __device__ unsigned laneid() {
  unsigned laneid;

  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}

#if defined(__HIP_PLATFORM_AMD__)
#include "mma_amd.hiph"
#else
#include "mma_nvidia.cuh"
#endif
