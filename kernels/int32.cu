#define nr_outer 4096
#define nr_inner 1024

template <int nr_int32> __device__ void int32_8(int2 &a, int2 &b, int2 &c) {
// Perform nr_int32 * 4 imad
#if defined(__HIP_PLATFORM_AMD__)
  for (int i = 0; i < nr_int32; i++) {
    __asm__ __volatile__("v_mad_i32_i24 v0, v1, v2, v0\n" ::: "v0", "v1", "v2");
  }
#else
  // Each mad multiplies the accumulator, so the chain cannot be rewritten into
  // adds of loop-invariant products (ptxas does this to a += b * c, which made
  // the previous kernel collapse into a few adds).
#pragma unroll nr_int32
  for (int i = 0; i < nr_int32; i++) {
    asm("mad.lo.s32 %0, %1, %2, %3;"
        : "=r"(a.x)
        : "r"(a.x), "r"(b.x), "r"(c.x));
    asm("mad.lo.s32 %0, %1, %2, %3;"
        : "=r"(a.x)
        : "r"(a.x), "r"(b.y), "r"(c.y));
    asm("mad.lo.s32 %0, %1, %2, %3;"
        : "=r"(a.y)
        : "r"(a.y), "r"(b.x), "r"(c.y));
    asm("mad.lo.s32 %0, %1, %2, %3;"
        : "=r"(a.y)
        : "r"(a.y), "r"(b.y), "r"(c.x));
  }
#endif
}

__global__ void int32_kernel(int *ptr) {
  int2 a = make_int2(threadIdx.x, threadIdx.x + 1);
  // The operands must not be compile-time constants, ptxas folds those. The
  // block size is not known to ptxas, and is always below 65536, so zero is
  // always 0 but cannot be proven to be.
  const int zero = blockDim.x >> 16;
  int2 b = make_int2(1 + zero, 2 + zero);
  int2 c = make_int2(3 + zero, 4 + zero);

  for (int i = 0; i < nr_outer; i++) {
    int32_8<nr_inner>(a, b, c);
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;
}
