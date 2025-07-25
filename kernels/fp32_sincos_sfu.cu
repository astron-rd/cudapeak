#define nr_outer 4096
#define nr_inner 1024

template <int nr_fma, int nr_sincos>
__device__ void __fp32_sincos_sfu_1_1(float2 &a, float2 &b, float2 &c) {
  for (int i = 0; i < nr_fma; i++) {
#if defined(__HIP_PLATFORM_AMD__)
    a.x += b.x * c.x;
    a.x += -b.y * c.y;
    a.y += b.x * c.y;
    a.y += b.y * c.x;
#else
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(a.x)
        : "f"(b.x), "f"(c.x), "f"(a.x));
#endif
  }
  for (int i = 0; i < nr_sincos; i++) {
    b.x = __sinf(a.x);
    b.y = __cosf(a.x);
  }
}

__global__ void fp32_sincos_sfu_1_8(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner; j++) {
      __fp32_sincos_sfu_1_1<1, 8>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_sfu_1_4(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner; j++) {
      __fp32_sincos_sfu_1_1<1, 4>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_sfu_1_2(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner; j++) {
      __fp32_sincos_sfu_1_1<1, 2>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_sfu_1_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner; j++) {
      __fp32_sincos_sfu_1_1<1, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_sfu_2_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 2; j++) {
      __fp32_sincos_sfu_1_1<2, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_sfu_4_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 4; j++) {
      __fp32_sincos_sfu_1_1<4, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_sfu_8_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 8; j++) {
      __fp32_sincos_sfu_1_1<8, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_sfu_16_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 16; j++) {
      __fp32_sincos_sfu_1_1<16, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_sfu_32_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 32; j++) {
      __fp32_sincos_sfu_1_1<32, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_sfu_64_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 64; j++) {
      __fp32_sincos_sfu_1_1<64, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_sfu_128_1(float *ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < nr_outer; i++) {
    for (int j = 0; j < nr_inner / 128; j++) {
      __fp32_sincos_sfu_1_1<128, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}
