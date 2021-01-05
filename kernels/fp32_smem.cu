#include "cuda.h"

template<int nr_fp32>
__device__ void smem_fp32_16_8(
    float2& a, float4* data)
{
    // Load 2 complex numbers (16 bytes)
    float4 x = *data;
    float2 b = make_float2(x.x, x.y);
    float2 c = make_float2(x.z, x.w);

    // Perform nr_fp32 * 4 fma (8 flops)
    #pragma unroll nr_fp32
    for (int i = 0; i < nr_fp32; i++) {
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(b.x),  "f"(c.x), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(-b.y), "f"(c.y), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.x),  "f"(c.y), "f"(a.y));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.y),  "f"(c.x), "f"(a.y));
    }
}

#define NR_REPETITIONS  1024
#define FETCH_PER_BLOCK 512

__shared__ float4 data[FETCH_PER_BLOCK];

#define INIT                                                     \
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);        \
    for (int i = blockIdx.x; i < FETCH_PER_BLOCK; i += blockDim.x) { \
        data[i].x = ptr[i];                                      \
        data[i].y = ptr[i] + 1;                                  \
    }

#define FINISH \
    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;

__global__ void fp32_smem_01(float *ptr)
{
    INIT

    for (int r = 0; r < NR_REPETITIONS; r++)
    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        smem_fp32_16_8<1>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_02(float *ptr)
{
    INIT

    for (int r = 0; r < NR_REPETITIONS/2; r++)
    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        smem_fp32_16_8<2>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_04(float *ptr)
{
    INIT

    for (int r = 0; r < NR_REPETITIONS/4; r++)
    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        smem_fp32_16_8<4>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_08(float *ptr)
{
    INIT

    for (int r = 0; r < NR_REPETITIONS/8; r++)
    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        smem_fp32_16_8<8>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_16(float *ptr)
{
    INIT

    for (int r = 0; r < NR_REPETITIONS/16; r++)
    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        smem_fp32_16_8<16>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_32(float *ptr)
{
    INIT

    for (int r = 0; r < NR_REPETITIONS/32; r++)
    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        smem_fp32_16_8<32>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_64(float *ptr)
{
    INIT

    for (int r = 0; r < NR_REPETITIONS/64; r++)
    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        smem_fp32_16_8<64>(a, &data[i]);
    }

    FINISH
}

__global__ void fp32_smem_128(float *ptr)
{
    INIT

    for (int r = 0; r < NR_REPETITIONS/128; r++)
    for (int i = 0; i < FETCH_PER_BLOCK; i++) {
        smem_fp32_16_8<128>(a, &data[i]);
    }

    FINISH
}
