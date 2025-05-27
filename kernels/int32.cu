#define nr_outer 4096
#define nr_inner 1024

template <int nr_int32> __device__ void int32_8(int2 &a, int2 &b, int2 &c) {
// Perform nr_int32 * 4 imad
#pragma unroll nr_int32
  for (int i = 0; i < nr_int32; i++) {
    asm("mad.lo.s32 %0, %1, %2, %3;"
        : "=r"(a.x)
        : "r"(b.x), "r"(c.x), "r"(a.x));
    asm("mad.lo.s32 %0, %1, %2, %3;"
        : "=r"(a.x)
        : "r"(-b.y), "r"(c.y), "r"(a.x));
    asm("mad.lo.s32 %0, %1, %2, %3;"
        : "=r"(a.y)
        : "r"(b.x), "r"(c.y), "r"(a.y));
    asm("mad.lo.s32 %0, %1, %2, %3;"
        : "=r"(a.y)
        : "r"(b.y), "r"(c.x), "r"(a.y));
  }
}

__global__ void int32_kernel(int *ptr) {
  int2 a = make_int2(threadIdx.x, 0);
  int2 b = make_int2(threadIdx.x, 1);
  int2 c = make_int2(threadIdx.x, 2);

  for (int i = 0; i < nr_outer; i++) {
    int32_8<nr_inner>(a, b, c);
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;
}
