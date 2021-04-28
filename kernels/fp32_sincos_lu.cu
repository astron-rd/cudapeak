#define NR_ITERATIONS 512

#include "cosisin.cu"

template<int nr_fma, int nr_sincos>
__device__ void fp32_sincos_lu_1_1(float2& a, float2& b, float2& c)
{
    // Perform nr_fma * 2 fma
    #pragma unroll nr_fma
    for (int i = 0; i < nr_fma; i++) {
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(b.x),  "f"(c.x), "f"(a.x));
    }
    // Perform nr_sincos * 1 sincos
    #pragma unroll nr_sincos
    for (int i = 0; i < nr_sincos; i++) {
        cosisin(a.x, &b.x, &b.y);
    }
}

template<int nr_fma, int nr_sincos>
__device__ void fp32_sincos_lu_2_1(float2& a, float2& b, float2& c)
{
    // Perofrm nr_fma * 2 fma
    #pragma unroll nr_fma
    for (int i = 0; i < nr_fma; i++) {
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(b.x),  "f"(c.x), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(-b.y), "f"(c.y), "f"(a.x));
    }
    // Perform nr_sincos * 1 sincos
    #pragma unroll nr_sincos
    for (int i = 0; i < nr_sincos; i++) {
        cosisin(a.x, &b.x, &b.y);
    }
}

template<int nr_fma, int nr_sincos>
__device__ void fp32_sincos_lu_4_1(float2& a, float2& b, float2& c)
{
    // Perofrm nr_fma * 4 fma
    #pragma unroll nr_fma
    for (int i = 0; i < nr_fma; i++) {
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(b.x),  "f"(c.x), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(-b.y), "f"(c.y), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.x),  "f"(c.y), "f"(a.y));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.y),  "f"(c.x), "f"(a.y));
    }
    // Perform nr_sincos * 1 sincos
    #pragma unroll nr_sincos
    for (int i = 0; i < nr_sincos; i++) {
        cosisin(a.x, &b.x, &b.y);
    }
}

__global__ void fp32_sincos_lu_1_8(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        for (int j = 0; j < 8192; j++) {
            fp32_sincos_lu_1_1<1, 8>(a, b, c);
        }
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_lu_1_4(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        for (int j = 0; j < 8192; j++) {
            fp32_sincos_lu_1_1<1, 4>(a, b, c);
        }
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_lu_1_2(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        for (int j = 0; j < 8192; j++) {
            fp32_sincos_lu_1_1<1, 2>(a, b, c);
        }
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_lu_1_1(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        for (int j = 0; j < 8192; j++) {
            fp32_sincos_lu_1_1<1, 1>(a, b, c);
        }
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_lu_2_1(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        for (int j = 0; j < 4096; j++) {
            fp32_sincos_lu_2_1<1, 1>(a, b, c);
        }
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_lu_4_1(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        for (int j = 0; j < 2048; j++) {
            fp32_sincos_lu_4_1<1, 1>(a, b, c);
        }
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_lu_8_1(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        for (int j = 0; j < 1024; j++) {
            fp32_sincos_lu_4_1<2, 1>(a, b, c);
        }
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_lu_16_1(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        for (int j = 0; j < 512; j++) {
            fp32_sincos_lu_4_1<4, 1>(a, b, c);
        }
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_lu_32_1(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        for (int j = 0; j < 256; j++) {
            fp32_sincos_lu_4_1<8, 1>(a, b, c);
        }
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_lu_64_1(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        for (int j = 0; j < 128; j++) {
            fp32_sincos_lu_4_1<16, 1>(a, b, c);
        }
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void fp32_sincos_lu_128_1(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        for (int j = 0; j < 64; j++) {
            fp32_sincos_lu_4_1<32, 1>(a, b, c);
        }
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}
