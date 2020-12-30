#include "cuda.h"

__global__ void fp32_smem_01(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int i = x; i < 512; i += blockDim.x) {
        data[i].x = ptr[i];
        data[i].y = ptr[i] + 1;
    }

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float a = data[i].x;
        float b = data[i].y;
        x += a * b;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void fp32_smem_02(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int i = x; i < 512; i += blockDim.x) {
        data[i].x = ptr[i];
        data[i].y = ptr[i] + 1;
    }

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float a = data[i].x;
        float b = data[i].y;
        x += a * b;
        x += a * b;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void fp32_smem_04(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int i = x; i < 512; i += blockDim.x) {
        data[i].x = ptr[i];
        data[i].y = ptr[i] + 1;
    }

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float a = data[i].x;
        float b = data[i].y;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void fp32_smem_08(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int i = x; i < 512; i += blockDim.x) {
        data[i].x = ptr[i];
        data[i].y = ptr[i] + 1;
    }

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float a = data[i].x;
        float b = data[i].y;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void fp32_smem_16(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int i = x; i < 512; i += blockDim.x) {
        data[i].x = ptr[i];
        data[i].y = ptr[i] + 1;
    }

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float a = data[i].x;
        float b = data[i].y;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;

        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}

__global__ void fp32_smem_32(float *ptr)
{
    float x = threadIdx.x;

    __shared__ float2 data[512];

    for (int i = x; i < 512; i += blockDim.x) {
        data[i].x = ptr[i];
        data[i].y = ptr[i] + 1;
    }

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float a = data[i].x;
        float b = data[i].y;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;

        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;

        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;

        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
