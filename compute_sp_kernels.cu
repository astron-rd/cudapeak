#include "cuda.h"

#define FMA_1(x, y)  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x) : "f"(x), "f"(y), "f"(x)); \
                     asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(y), "f"(x), "f"(y));
#define FMA_4(x, y)  FMA_1(x, y)  FMA_1(x, y)  FMA_1(x, y)  FMA_1(x,y)
#define FMA_16(x, y) FMA_4(x, y)  FMA_4(x, y)  FMA_4(x, y)  FMA_4(x, y)
#define FMA_64(x, y) FMA_16(x, y) FMA_16(x, y) FMA_16(x, y) FMA_16(x, y)

__global__ void compute_sp_v1(float *ptr)
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


__global__ void compute_sp_v2(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    __shared__ float2 data[1024];

    for (int i = 0; i < 1024; i++) {
        x += data[i].x * data[i].y;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}

__global__ void compute_sp_v3(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    __shared__ float2 data[1024];

    for (int i = 0; i < 1024; i++) {
        x += data[i].x * data[i].x;
        x += data[i].y * data[i].y;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}

__global__ void compute_sp_v4(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    __shared__ float2 data[1024];

    for (int i = 0; i < 1024; i++) {
        x += data[i].x * data[i].x;
        x += data[i].y * data[i].y;
        x += data[i].x * data[i].y;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}

__global__ void compute_sp_v5(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    __shared__ float2 data[1024];

    for (int i = 0; i < 1024; i++) {
        float2 d = data[i];

        float t1 = d.x + d.x * d.x;
        float t2 = d.y + d.y * d.y;
        float t3 = d.x + d.x * d.y;

        x += t1 * t2;
        x += t2 * t3;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}

__global__ void compute_sp_v6(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    __shared__ float2 data[1024];

    for (int i = 0; i < 1024; i++) {
        float2 d = data[i];

        float t1 = d.x + d.x * d.x;
        float t2 = d.x + d.y * d.y;
        float t3 = d.x + d.x * d.y;
        float t4 = d.y + d.x * d.x;
        float t5 = d.y + d.y * d.y;
        float t6 = d.y + d.x * d.y;

        x += t1 * t2;
        x += t2 * t3;
        x += t4 * t5;
        x += t5 * t6;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}

__global__ void compute_sp_v7(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    __shared__ float2 data[1024];

    for (int i = 0; i < 1024; i++) {
        float2 d = data[i];

        float t1 = d.x + d.x * d.x;
        float t2 = d.x + d.y * d.y;
        float t3 = d.x + d.x * d.y;
        float t4 = d.y + d.x * d.x;
        float t5 = d.y + d.y * d.y;
        float t6 = d.y + d.x * d.y;

        x += t1 * t2;
        x += t1 * t3;
        x += t1 * t4;
        x += t1 * t5;
        x += t1 * t6;

        x += t3 * t4;
        x += t3 * t5;
        x += t3 * t6;

        x += t4 * t5;
        x += t4 * t6;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}

__global__ void compute_sp_v8(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    __shared__ float2 data[1024];

    for (int i = 0; i < 1024; i++) {
        float2 d = data[i];

        float t1 = d.x + d.x * d.x;
        float t2 = d.x + d.y * d.y;
        float t3 = d.x + d.x * d.y;
        float t4 = d.y + d.x * d.x;
        float t5 = d.y + d.y * d.y;
        float t6 = d.y + d.x * d.y;

        x += t1 * t2;
        x += t1 * t3;
        x += t1 * t4;
        x += t1 * t5;
        x += t1 * t6;

        x += t2 * t3;
        x += t2 * t4;
        x += t2 * t5;
        x += t2 * t6;

        x += t3 * t4;
        x += t3 * t5;
        x += t3 * t6;

        x += t4 * t5;
        x += t4 * t6;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}
