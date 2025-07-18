#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

#define nr_outer 4096
#define nr_inner 1024

template <int nr_fp32> __device__ void fp32_8(float2 &a, float2 &b, float2 &c) {
// Perform nr_fp32 * 4 fma
#if defined(__HIP_PLATFORM_AMD__)
  for (int i = 0; i < nr_fp32; i++) {
    a.x += b.x * c.x;
    a.x += -b.y * c.y;
    a.y += b.x * c.y;
    a.y += b.y * c.x;
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

__global__ void fp32_kernel(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    fp32_8<nr_inner>(a, b, c);
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;
}
