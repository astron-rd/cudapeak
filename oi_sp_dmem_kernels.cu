#include "cuda.h"


__global__ void oi_sp_dmem_01(float2 *ptr)
{
    float x = threadIdx.x;
    float2 *data = &ptr[blockIdx.x * blockDim.x];

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float a = data[i].x;
        float b = data[i].y;
        x += a * b;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = make_float2(x, -x);
}

__global__ void oi_sp_dmem_02(float2 *ptr)
{
    float x = threadIdx.x;
    float2 *data = &ptr[blockIdx.x * blockDim.x];

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float a = data[i].x;
        float b = data[i].y;
        x += a * b;
        x += a * b;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = make_float2(x, -x);
}

__global__ void oi_sp_dmem_04(float2 *ptr)
{
    float x = threadIdx.x;
    float2 *data = &ptr[blockIdx.x * blockDim.x];

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float a = data[i].x;
        float b = data[i].y;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = make_float2(x, -x);
}

__global__ void oi_sp_dmem_08(float2 *ptr)
{
    float x = threadIdx.x;
    float2 *data = &ptr[blockIdx.x * blockDim.x];

    for (int r = 0; r < 128; r++)
    for (int i = 0; i < 512; i++) {
        float a = data[i].x;
        float b = data[i].y;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
        x += a * b; x += a * b;
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = make_float2(x, -x);
}

__global__ void oi_sp_dmem_16(float2 *ptr)
{
    float x = threadIdx.x;
    float2 *data = &ptr[blockIdx.x * blockDim.x];

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

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = make_float2(x, -x);
}

__global__ void oi_sp_dmem_32(float2 *ptr)
{
    float x = threadIdx.x;
    float2 *data = &ptr[blockIdx.x * blockDim.x];

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

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = make_float2(x, -x);
}
