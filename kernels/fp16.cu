#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

#include "fp16.h"

#define nr_outer 4096
#define nr_inner 1024

template <int nr_fp16>
__device__ void fp16_8(half2& a, half2& b, half2& c) {
// Perform nr_fp16 * 4 fma
#if defined(__HIP_PLATFORM_AMD__)
  for (int i = 0; i < nr_fp16; i++) {
    a.x += b.x * c.x;
    a.x += -b.y * c.y;
    a.y += b.x * c.y;
    a.y += b.y * c.x;
  }
#else
  short a_x = __half_as_short(a.x);
  short b_x = __half_as_short(b.x);
  short c_x = __half_as_short(c.x);
  short a_y = __half_as_short(a.y);
  short b_y = __half_as_short(b.y);
  short b_yn = __half_as_short(-b.y);
  short c_y = __half_as_short(c.y);

#pragma unroll nr_fp16
  for (int i = 0; i < nr_fp16; i++) {
    asm("fma.rn.f16 %0, %1, %2, %3;"
        : "=h"(a_x)
        : "h"(b_x), "h"(c_x), "h"(a_x));
    asm("fma.rn.f16 %0, %1, %2, %3;"
        : "=h"(a_x)
        : "h"(b_yn), "h"(c_y), "h"(a_x));
    asm("fma.rn.f16 %0, %1, %2, %3;"
        : "=h"(a_y)
        : "h"(b_x), "h"(c_y), "h"(a_y));
    asm("fma.rn.f16 %0, %1, %2, %3;"
        : "=h"(a_y)
        : "h"(b_y), "h"(c_x), "h"(a_y));
  }

  a.x = __short_as_half(a_x);
  a.y = __short_as_half(a_y);
#endif
}

#if !defined(__HIP_PLATFORM_AMD__)
template <int nr_fp16>
__device__ void fp16x2_8(half2& a, half2& b, half2& c) {
  __half2 a_xy = __halves2half2(a.x, a.y);
  __half2 b_xx = __halves2half2(b.x, b.x);
  __half2 c_xy = __halves2half2(c.x, c.y);
  __half2 b_yny = __halves2half2(-b.y, b.y);
  __half2 c_yx = __halves2half2(c.y, c.x);

#pragma unroll nr_fp16
  for (int i = 0; i < nr_fp16; i++) {
    unsigned int a_xy_ui = *reinterpret_cast<unsigned int*>(&a_xy);
    unsigned int b_xx_ui = *reinterpret_cast<unsigned int*>(&b_xx);
    unsigned int c_xy_ui = *reinterpret_cast<unsigned int*>(&c_xy);
    unsigned int b_yny_ui = *reinterpret_cast<unsigned int*>(&b_yny);
    unsigned int c_yx_ui = *reinterpret_cast<unsigned int*>(&c_yx);

    asm("fma.rn.f16x2 %0, %1, %2, %3;"
        : "=r"(a_xy_ui)
        : "r"(b_xx_ui), "r"(c_xy_ui), "r"(a_xy_ui));

    asm("fma.rn.f16x2 %0, %1, %2, %3;"
        : "=r"(a_xy_ui)
        : "r"(b_yny_ui), "r"(c_yx_ui), "r"(a_xy_ui));

    a_xy = *reinterpret_cast<__half2*>(&a_xy_ui);
  }

  a.x = __low2half(a_xy);
  a.y = __high2half(a_xy);
}
#endif

__global__ void fp16_kernel(half* ptr) {
  half2 a = make_half2(threadIdx.x, threadIdx.x + 1);
  half2 b = make_half2(1, 2);
  half2 c = make_half2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    fp16_8<nr_inner>(a, b, c);
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;
}

#if !defined(__HIP_PLATFORM_AMD__)
__global__ void fp16x2_kernel(half* ptr) {
  half2 a = make_half2(threadIdx.x, threadIdx.x + 1);
  half2 b = make_half2(1, 2);
  half2 c = make_half2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    fp16x2_8<nr_inner>(a, b, c);
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;
}
#endif