#include "cuda.h"

template<int nr_fp32>
__device__ void dmem_fp32_4_8(
    float2& a, float4* data)
{
    // Load 2 complex numbers
    #if 0
    float4 x = *data;
    float2 b = make_float2(x.x, x.y);
    float2 c = make_float2(x.z, x.w);
    #else
    float2 b, c;
    asm("ld.global.v4.f32 {%0, %1, %2, %3}, [%4];" : "=f"(b.x), "=f"(b.y), "=f"(c.x), "=f"(c.y) : "l"(data));
    #endif

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

#define INIT \
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1); \
    ptr = ptr + NR_ELEMENTS * (blockIdx.y * blockDim.x + blockIdx.x);

#define FINISH \
    ptr[blockIdx.x * blockDim.x + threadIdx.x] = make_float4(a.x, a.y, 0, 0);

__global__ void fp32_dmem_01(float4 *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        dmem_fp32_4_8<1>(a, &ptr[i]);
    }

    FINISH
}

__global__ void fp32_dmem_02(float4 *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/2; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        dmem_fp32_4_8<2>(a, &ptr[i]);
    }

    FINISH
}

__global__ void fp32_dmem_04(float4 *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/4; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        dmem_fp32_4_8<4>(a, &ptr[i]);
    }

    FINISH
}

__global__ void fp32_dmem_08(float4 *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/8; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        dmem_fp32_4_8<8>(a, &ptr[i]);
    }

    FINISH
}

__global__ void fp32_dmem_16(float4 *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/16; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        dmem_fp32_4_8<16>(a, &ptr[i]);
    }

    FINISH
}

__global__ void fp32_dmem_32(float4 *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/32; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        dmem_fp32_4_8<32>(a, &ptr[i]);
    }

    FINISH
}

__global__ void fp32_dmem_64(float4 *ptr)
{
    INIT

    for (int r = 0; r < NR_ITERATIONS/64; r++)
    for (int i = 0; i < NR_ELEMENTS; i++) {
        dmem_fp32_4_8<64>(a, &ptr[i]);
    }

    FINISH
}
