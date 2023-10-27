#define NR_REPETITIONS 512
#define NR_ITERATIONS 8192

template <int nr_fma, int nr_sincos>
__device__ void fp32_sincos_fpu_1_1(float2& a, float2& b, float2& c) {
  for (int i = 0; i < nr_fma; i++) {
    asm("fma.rn.f32 %0, %1, %2, %3;"
        : "=f"(a.x)
        : "f"(b.x), "f"(c.x), "f"(a.x));
  }
  for (int i = 0; i < nr_sincos; i++) {
    b.x = sinf(a.x);
    b.y = cosf(a.x);
  }
}

__global__ void fp32_sincos_fpu_1_8(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < NR_REPETITIONS; i++) {
    for (int j = 0; j < NR_ITERATIONS; j++) {
      fp32_sincos_fpu_1_1<1, 8>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_1_4(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < NR_REPETITIONS; i++) {
    for (int j = 0; j < NR_ITERATIONS; j++) {
      fp32_sincos_fpu_1_1<1, 4>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_1_2(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < NR_REPETITIONS; i++) {
    for (int j = 0; j < NR_ITERATIONS; j++) {
      fp32_sincos_fpu_1_1<1, 2>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_1_1(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < NR_REPETITIONS; i++) {
    for (int j = 0; j < NR_ITERATIONS; j++) {
      fp32_sincos_fpu_1_1<1, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_2_1(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < NR_REPETITIONS; i++) {
    for (int j = 0; j < NR_ITERATIONS / 2; j++) {
      fp32_sincos_fpu_1_1<2, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_4_1(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < NR_REPETITIONS; i++) {
    for (int j = 0; j < NR_ITERATIONS / 4; j++) {
      fp32_sincos_fpu_1_1<4, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_8_1(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < NR_REPETITIONS; i++) {
    for (int j = 0; j < NR_ITERATIONS / 8; j++) {
      fp32_sincos_fpu_1_1<8, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_16_1(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < NR_REPETITIONS; i++) {
    for (int j = 0; j < NR_ITERATIONS / 16; j++) {
      fp32_sincos_fpu_1_1<16, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_32_1(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < NR_REPETITIONS; i++) {
    for (int j = 0; j < NR_ITERATIONS / 32; j++) {
      fp32_sincos_fpu_1_1<32, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_64_1(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < NR_REPETITIONS; i++) {
    for (int j = 0; j < NR_ITERATIONS / 64; j++) {
      fp32_sincos_fpu_1_1<64, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_fpu_128_1(float* ptr) {
  float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
  float2 b = make_float2(1, 2);
  float2 c = make_float2(3, 4);

  for (int i = 0; i < NR_REPETITIONS; i++) {
    for (int j = 0; j < NR_ITERATIONS / 128; j++) {
      fp32_sincos_fpu_1_1<128, 1>(a, b, c);
    }
  }

  ptr[blockIdx.x * blockDim.x + threadIdx.x] =
      a.x + a.y + b.x + b.y + c.x + c.y;
}
