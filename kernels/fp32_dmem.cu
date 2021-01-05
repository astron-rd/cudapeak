#include "cuda.h"

template<int nr_fp32>
__device__ void dmem_fp32_16_8(
    float2& a, float4* data)
{
    // Load 2 complex numbers (16 bytes)
    float4 x = *data;
    float2 b = make_float2(x.x, x.y);
    float2 c = make_float2(x.z, x.w);

    // Perform nr_fp32 * 4 fma (8 flops)
    #if 1
    #pragma unroll nr_fp32
    for (int i = 0; i < nr_fp32; i++) {
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(b.x),  "f"(c.x), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(-b.y), "f"(c.y), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.x),  "f"(c.y), "f"(a.y));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.y),  "f"(c.x), "f"(a.y));
    }
    #endif
}

#define FETCH_PER_BLOCK 16

#define INIT \
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1); \
    int id = (blockIdx.x * blockDim.x * FETCH_PER_BLOCK) + threadIdx.x;

#define FINISH \
    ptr[blockIdx.x * blockDim.x + threadIdx.x] = make_float4(a.x, a.y, 0, 0);

__global__ void fp32_dmem_01(float4 *ptr)
{
    INIT

    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        dmem_fp32_16_8<1>(a, &ptr[id]);
    }

    FINISH
}

__global__ void fp32_dmem_02(float4 *ptr)
{
    INIT

    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        dmem_fp32_16_8<2>(a, &ptr[id]);
    }

    FINISH
}

__global__ void fp32_dmem_04(float4 *ptr)
{
    INIT

    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        dmem_fp32_16_8<4>(a, &ptr[id]);
    }

    FINISH
}

__global__ void fp32_dmem_08(float4 *ptr)
{
    INIT

    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        dmem_fp32_16_8<8>(a, &ptr[id]);
    }

    FINISH
}

__global__ void fp32_dmem_16(float4 *ptr)
{
    INIT

    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        dmem_fp32_16_8<16>(a, &ptr[id]);
    }

    FINISH
}

__global__ void fp32_dmem_32(float4 *ptr)
{
    INIT

    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        dmem_fp32_16_8<32>(a, &ptr[id]);
    }

    FINISH
}

__global__ void fp32_dmem_64(float4 *ptr)
{
    INIT

    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        dmem_fp32_16_8<64>(a, &ptr[id]);
    }

    FINISH
}

__global__ void fp32_dmem_128(float4 *ptr)
{
    INIT

    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        dmem_fp32_16_8<128>(a, &ptr[id]);
    }

    FINISH
}

__global__ void fp32_dmem_256(float4 *ptr)
{
    INIT

    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        dmem_fp32_16_8<256>(a, &ptr[id]);
    }

    FINISH
}
