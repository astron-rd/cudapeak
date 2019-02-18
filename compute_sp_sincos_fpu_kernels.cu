#include "flops.h"
#include "sincos_fpu.h"

#define NR_ITERATIONS 512
__global__ void compute_sp_sincos_fpu_01(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        flops_8192(a, b, c); flops_8192(a, b, c);
        flops_8192(a, b, c); flops_8192(a, b, c);
        sincos_fpu_8192(a, b, c); sincos_fpu_8192(a, b, c);
        sincos_fpu_8192(a, b, c); sincos_fpu_8192(a, b, c);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void compute_sp_sincos_fpu_02(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        flops_8192(a, b, c); flops_8192(a, b, c);
        flops_8192(a, b, c); flops_8192(a, b, c);
        sincos_fpu_8192(a, b, c); sincos_fpu_8192(a, b, c);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void compute_sp_sincos_fpu_04(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        flops_8192(a, b, c); flops_8192(a, b, c);
        flops_8192(a, b, c); flops_8192(a, b, c);
        sincos_fpu_8192(a, b, c);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void compute_sp_sincos_fpu_08(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        flops_8192(a, b, c); flops_8192(a, b, c);
        flops_8192(a, b, c); flops_8192(a, b, c);
        sincos_fpu_2048(a, b, c); sincos_fpu_2048(a, b, c);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void compute_sp_sincos_fpu_16(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        flops_8192(a, b, c); flops_8192(a, b, c);
        flops_8192(a, b, c); flops_8192(a, b, c);
        sincos_fpu_2048(a, b, c);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void compute_sp_sincos_fpu_32(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        flops_8192(a, b, c); flops_8192(a, b, c);
        flops_8192(a, b, c); flops_8192(a, b, c);
        sincos_fpu_0512(a, b, c); sincos_fpu_0512(a, b, c);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}

__global__ void compute_sp_sincos_fpu_64(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < NR_ITERATIONS; i++) {
        flops_8192(a, b, c); flops_8192(a, b, c);
        flops_8192(a, b, c); flops_8192(a, b, c);
        sincos_fpu_0512(a, b, c);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y + b.x + b.y + c.x + c.y;
}
