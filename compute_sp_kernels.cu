#include "cuda.h"

// complex multiplication and addition: 8 flops
inline __device__ void flops_0008(float2& a, float2 b, float2 c)
{
#if 1
    a.x += b.x * c.x;
    a.x -= b.y * c.y;
    a.y += b.x * c.y;
    a.y += b.y * c.x;
#else
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(b.x),  "f"(c.x), "f"(a.x));
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(-b.y), "f"(c.y), "f"(a.x));
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.x),  "f"(c.y), "f"(a.y));
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.y),  "f"(c.x), "f"(a.y));
#endif
}

inline __device__ void flops_0032(float2& a, float2 b, float2 c)
{
    flops_0008(a, b, c); flops_0008(a, b, c); flops_0008(a, b, c); flops_0008(a, b, c);
}

inline __device__ void flops_0128(float2& a, float2 b, float2 c)
{
    flops_0032(a, b, c); flops_0032(a, b, c); flops_0032(a, b, c); flops_0032(a, b, c);
}

inline __device__ void flops_0512(float2& a, float2 b, float2 c)
{
    flops_0128(a, b, c); flops_0128(a, b, c); flops_0128(a, b, c); flops_0128(a, b, c);
}

inline __device__ void flops_2048(float2& a, float2 b, float2 c)
{
    flops_0512(a, b, c); flops_0512(a, b, c); flops_0512(a, b, c); flops_0512(a, b, c);
}

inline __device__ void flops_8192(float2& a, float2 b, float2 c)
{
    flops_2048(a, b, c); flops_2048(a, b, c); flops_2048(a, b, c); flops_2048(a, b, c);
}

__global__ void compute_sp_v1(float *ptr)
{
    float2 a = make_float2(threadIdx.x, threadIdx.x + 1);
    float2 b = make_float2(1, 2);
    float2 c = make_float2(3, 4);

    for (int i = 0; i < 2048; i++) {
        flops_8192(a, b, c); flops_8192(a, b, c);
        flops_8192(a, b, c); flops_8192(a, b, c);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;
}
