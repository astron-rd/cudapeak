#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

template <int nr_fp32>
__device__ void fp32_8(float2& a, float2& b, float2& c) {
// Perform nr_fp32 * 4 fma
#if defined(__HIP_PLATFORM_AMD__)
#pragma unroll nr_fp32
  for (int i = 0; i < nr_fp32; i++) {
    asm("v_fma_f32 %0, %1, %2, %3;" : "=v"(a.x) : "v"(b.x), "v"(c.x), "v"(a.x));
    asm("v_fma_f32 %0, %1, %2, %3;"
        : "=v"(a.x)
        : "v"(-b.y), "v"(c.y), "v"(a.x));
    asm("v_fma_f32 %0, %1, %2, %3;" : "=v"(a.y) : "v"(b.x), "v"(c.y), "v"(a.y));
    asm("v_fma_f32 %0, %1, %2, %3;" : "=v"(a.y) : "v"(b.y), "v"(c.x), "v"(a.y));
  }
#else
#pragma unroll nr_fp32
  for (int i = 0; i < nr_fp32; i++) {
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(a.x)
        : "f"(b.x), "f"(c.x), "f"(a.x));
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(a.x)
        : "f"(-b.y), "f"(c.y), "f"(a.x));
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(a.y)
        : "f"(b.x), "f"(c.y), "f"(a.y));
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(a.y)
        : "f"(b.y), "f"(c.x), "f"(a.y));
  }
#endif
}

__global__ void fp32_kernel(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < 2048; i++) {
    fp32_8<4096>(a, b, c);
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;
}
