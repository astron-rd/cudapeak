#define REPEAT_COUNT 32768

inline __device__ unsigned laneid() {
  unsigned laneid;

  asm("mov.u32 %0, %%laneid;" : "=r"(laneid));
  return laneid;
}

// Pseudo-random input data. The kernels used to multiply matrices of zeros,
// which keep most of the datapath of a tensor core from switching and may let
// the GPU run at a higher clock than it can sustain on real data. Every thread,
// matrix (tag) and element gets different bits, identical on every run.
inline __device__ unsigned random_bits(unsigned tag, unsigned index) {
  unsigned x = (threadIdx.y * blockDim.x + threadIdx.x) * 0x9E3779B9u +
               tag * 0x85EBCA6Bu + index * 0xC2B2AE35u + 0x165667B1u;
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

// Uniform in [-1, 1), so that the accumulators stay finite
inline __device__ float random_float(unsigned tag, unsigned index) {
  return static_cast<int>(random_bits(tag, index)) * (1.0f / 2147483648.0f);
}

#if defined(__HIP_PLATFORM_AMD__)
#include "mma_amd.hiph"
#else
#include "mma_nvidia.cuh"
#include "mma_tcgen05.cuh"
#endif
