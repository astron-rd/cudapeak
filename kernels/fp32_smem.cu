#include "cuda.h"

template<int nr_fp32>
__device__ void smem_fp32_4_8(
    float2& a, float4* data)
{
    // Load 2 complex numbers
    float4 x = *data;
    float2 b = make_float2(x.x, x.y);
    float2 c = make_float2(x.z, x.w);

    // Perform nr_fp32 * 4 fma
    #pragma unroll nr_fp32
    for (int i = 0; i < nr_fp32; i++) {
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(b.x),  "f"(c.x), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(-b.y), "f"(c.y), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.x),  "f"(c.y), "f"(a.y));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.y),  "f"(c.x), "f"(a.y));
    }
}

#define NR_ITERATIONS 1024
#define NR_ELEMENTS   512

__shared__ float4 data[NR_ELEMENTS];

#define INIT                                                     \
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);        \
    for (int i = blockIdx.x; i < NR_ELEMENTS; i += blockDim.x) { \
        data[i].x = ptr[i];                                      \
        data[i].y = ptr[i] + 1;                                  \
    }

#define FINISH \
    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;

__global__ void fp32_smem_01(float *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        smem_fp32_4_8<1>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_02(float *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/2; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        smem_fp32_4_8<2>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_04(float *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/4; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        smem_fp32_4_8<4>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_08(float *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/8; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        smem_fp32_4_8<8>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_16(float *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/16; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        smem_fp32_4_8<16>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_32(float *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/32; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        smem_fp32_4_8<32>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_64(float *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/64; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        smem_fp32_4_8<64>(a, &data[i]);
    }

    FINISH
}
