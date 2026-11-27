#define nr_outer 4096
#define nr_inner 1024

// Accurate sine and cosine. HIP kernels are compiled with -ffast-math, which
// makes sinf and cosf resolve to the fast __sinf and __cosf (the HIP headers
// pick the implementation with __FAST_OR_SLOW), so call the library functions
// directly.
#if defined(__HIP_PLATFORM_AMD__)
#define ACCURATE_SINF __ocml_sin_f32
#define ACCURATE_COSF __ocml_cos_f32
#else
#define ACCURATE_SINF sinf
#define ACCURATE_COSF cosf
#endif

// The kernels start with c.x = 1/1024. The accumulator a.x is the argument of
// the next sincos and grows with c.x: with c.x = 3 the argument of the 1:8
// kernel drifts to millions (a host replica of the kernel measured 1.2e7),
// where the accurate sinf takes its slow range reduction path and the lookup
// table index of the lu kernel overflows. With 1/1024 it stays below about
// 5000.
template <int nr_fma, int nr_sincos>
__device__ void __fp32_sincos_fpu_1_1(float2 &a, float2 &b, float2 &c) {
  for (int i = 0; i < nr_fma; i++) {
#if defined(__HIP_PLATFORM_AMD__)
    a.x = __builtin_fmaf(b.x, c.x, a.x);
#else
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(a.x)
        : "f"(b.x), "f"(c.x), "f"(a.x));
#endif
  }
  // Every sincos depends on the result of the previous one, and the results
  // feed the next fma, so none of it can be folded or removed. This costs one
  // fp32 add per sincos.
  float x = a.x;
  for (int i = 0; i < nr_sincos; i++) {
    b.x = ACCURATE_SINF(x);
    b.y = ACCURATE_COSF(x);
    x = b.x + b.y;
  }
  a.y += b.y;
}

__global__ void fp32_sincos_fpu_1_8(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(1.f / 1024, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner; j++) {
      __fp32_sincos_fpu_1_1<1, 8>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_1_4(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(1.f / 1024, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner; j++) {
      __fp32_sincos_fpu_1_1<1, 4>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_1_2(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(1.f / 1024, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner; j++) {
      __fp32_sincos_fpu_1_1<1, 2>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_1_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(1.f / 1024, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner; j++) {
      __fp32_sincos_fpu_1_1<1, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_2_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(1.f / 1024, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 2; j++) {
      __fp32_sincos_fpu_1_1<2, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_4_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(1.f / 1024, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 4; j++) {
      __fp32_sincos_fpu_1_1<4, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_8_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(1.f / 1024, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 8; j++) {
      __fp32_sincos_fpu_1_1<8, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_16_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(1.f / 1024, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 16; j++) {
      __fp32_sincos_fpu_1_1<16, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_32_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(1.f / 1024, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 32; j++) {
      __fp32_sincos_fpu_1_1<32, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_64_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(1.f / 1024, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 64; j++) {
      __fp32_sincos_fpu_1_1<64, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_128_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(1.f / 1024, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 128; j++) {
      __fp32_sincos_fpu_1_1<128, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}
