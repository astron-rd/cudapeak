#include "opaque.h"

#define nr_outer 4096
#define nr_inner 1024

// 24-bit integer multiply-add, a = a * b + c with the multiplicands taken from
// the low 24 bits (signed). AMD GPUs have a dedicated instruction for this,
// that is faster than a full 32-bit multiplication (see the int32 benchmark).
// It is only available with HIP.
template <int nr_int24> __device__ void int24_8(int2 &a, int2 &b, int2 &c) {
  // Perform nr_int24 * 4 imad24
  b.x = opaque(b.x);
  b.y = opaque(b.y);
  c.x = opaque(c.x);
  c.y = opaque(c.y);
#pragma unroll 4
  for (int i = 0; i < nr_int24; i++) {
    asm("v_mad_i32_i24 %0, %1, %2, %3"
        : "=v"(a.x)
        : "v"(a.x), "v"(b.x), "v"(c.x));
    asm("v_mad_i32_i24 %0, %1, %2, %3"
        : "=v"(a.x)
        : "v"(a.x), "v"(b.y), "v"(c.y));
    asm("v_mad_i32_i24 %0, %1, %2, %3"
        : "=v"(a.y)
        : "v"(a.y), "v"(b.x), "v"(c.y));
    asm("v_mad_i32_i24 %0, %1, %2, %3"
        : "=v"(a.y)
        : "v"(a.y), "v"(b.y), "v"(c.x));
  }
}

__global__ void int24_kernel(int *ptr) {
  int2 a = make_int2(threadIdx.x, threadIdx.x + 1);
  int2 b = make_int2(1, 2);
  int2 c = make_int2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    int24_8<nr_inner>(a, b, c);
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;
}
