#include "cuda.h"

#define FMA_1(x, y)  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x) : "f"(x), "f"(y), "f"(x)); \
                     asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(y), "f"(x), "f"(y));
#define FMA_4(x, y)    FMA_1(x, y)   FMA_1(x, y)   FMA_1(x, y)   FMA_1(x,y)
#define FMA_16(x, y)   FMA_4(x, y)   FMA_4(x, y)   FMA_4(x, y)   FMA_4(x, y)
#define FMA_64(x, y)   FMA_16(x, y)  FMA_16(x, y)  FMA_16(x, y)  FMA_16(x, y)
#define FMA_256(x, y)  FMA_64(x, y)  FMA_64(x, y)  FMA_64(x, y)  FMA_64(x, y)
#define FMA_1024(x, y) FMA_256(x, y) FMA_256(x, y) FMA_256(x, y) FMA_256(x, y)

#define SINCOS_1(x, y) asm("sin.approx.f32  %0, %1;" : "=f"(x) : "f"(x));\
                       asm("cos.approx.f32  %0, %1;" : "=f"(y) : "f"(y));
#define SINCOS_2(x, y)   SINCOS_1(x, y)   SINCOS_1(x, y)
#define SINCOS_4(x, y)   SINCOS_2(x, y)   SINCOS_2(x, y)
#define SINCOS_8(x, y)   SINCOS_4(x, y)   SINCOS_4(x, y)
#define SINCOS_16(x, y)  SINCOS_8(x, y)   SINCOS_8(x, y)
#define SINCOS_32(x, y)  SINCOS_16(x, y)  SINCOS_16(x, y)
#define SINCOS_64(x, y)  SINCOS_32(x, y)  SINCOS_32(x, y)
#define SINCOS_128(x, y) SINCOS_64(x, y)  SINCOS_64(x, y)
#define SINCOS_256(x, y) SINCOS_128(x, y) SINCOS_128(x, y)

__global__ void compute_sp_sincos_b0(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}

__global__ void compute_sp_sincos_b1(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);
        SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);
        SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);
        SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);
        SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);
        SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);
        SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);
        SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);   SINCOS_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}

__global__ void compute_sp_sincos_v00(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);

        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); SINCOS_256(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v01(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);

        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
        SINCOS_256(x, y); FMA_64(x, y); SINCOS_256(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v02(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);

        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
        SINCOS_128(x, y); FMA_64(x, y); SINCOS_128(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v03(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);

        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v04(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);

        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
        SINCOS_64(x, y); FMA_64(x, y); SINCOS_64(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v05(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);

        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
        SINCOS_32(x, y); FMA_64(x, y); SINCOS_32(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v06(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);

        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
        SINCOS_16(x, y); FMA_64(x, y); SINCOS_16(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v07(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);

        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
        SINCOS_8(x, y); FMA_64(x, y); SINCOS_8(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v08(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);

        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
        SINCOS_4(x, y); FMA_64(x, y); SINCOS_4(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v09(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);

        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
        SINCOS_2(x, y); FMA_64(x, y); SINCOS_2(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v10(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);

        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); SINCOS_1(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}


__global__ void compute_sp_sincos_v11(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);

        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
        SINCOS_1(x, y); FMA_64(x, y); FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v12(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_1(x, y); FMA_256(x, y);
        SINCOS_1(x, y); FMA_256(x, y);
        SINCOS_1(x, y); FMA_256(x, y);
        SINCOS_1(x, y); FMA_256(x, y);
        SINCOS_1(x, y); FMA_256(x, y);
        SINCOS_1(x, y); FMA_256(x, y);
        SINCOS_1(x, y); FMA_256(x, y);
        SINCOS_1(x, y); FMA_256(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v13(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_1(x, y); FMA_256(x, y); FMA_256(x, y);
        SINCOS_1(x, y); FMA_256(x, y); FMA_256(x, y);
        SINCOS_1(x, y); FMA_256(x, y); FMA_256(x, y);
        SINCOS_1(x, y); FMA_256(x, y); FMA_256(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v14(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_1(x, y); FMA_1024(x, y);
        SINCOS_1(x, y); FMA_1024(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}

__global__ void compute_sp_sincos_v15(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        SINCOS_1(x, y); FMA_1024(x, y); FMA_1024(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y;
}
