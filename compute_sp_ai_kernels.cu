#include "cuda.h"


__global__ void compute_sp_ai_v1(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        x += data[i].x * data[i].y;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void compute_sp_ai_v2(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        x += data[i].x * data[i].x;
        x += data[i].y * data[i].y;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void compute_sp_ai_v3(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float2 d = data[i];

        float t1 = d.x + d.x * d.x;
        float t2 = d.y + d.y * d.y;

        x += t1 * t1;
        x += t2 * t2;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void compute_sp_ai_v4(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float2 d = data[i];

        float t1 = d.x + d.x * d.x;
        float t2 = d.y + d.y * d.y;
        float t3 = d.x + d.x * d.y;

        x += t1 * t2;
        x += t2 * t3;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void compute_sp_ai_v5(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
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

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void compute_sp_ai_v6(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
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

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void compute_sp_ai_v7(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
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

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
