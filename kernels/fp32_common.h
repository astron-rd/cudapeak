// complex multiplication and addition: 8 flops
inline __device__ void fp32_0008(float2& a, float2 b, float2 c)
{
#if 0
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

inline __device__ void fp32_0032(float2& a, float2 b, float2 c)
{
    fp32_0008(a, b, c); fp32_0008(a, b, c); fp32_0008(a, b, c); fp32_0008(a, b, c);
}

inline __device__ void fp32_0128(float2& a, float2 b, float2 c)
{
    fp32_0032(a, b, c); fp32_0032(a, b, c); fp32_0032(a, b, c); fp32_0032(a, b, c);
}

inline __device__ void fp32_0512(float2& a, float2 b, float2 c)
{
    fp32_0128(a, b, c); fp32_0128(a, b, c); fp32_0128(a, b, c); fp32_0128(a, b, c);
}

inline __device__ void fp32_2048(float2& a, float2 b, float2 c)
{
    fp32_0512(a, b, c); fp32_0512(a, b, c); fp32_0512(a, b, c); fp32_0512(a, b, c);
}

inline __device__ void fp32_8192(float2& a, float2 b, float2 c)
{
    fp32_2048(a, b, c); fp32_2048(a, b, c); fp32_2048(a, b, c); fp32_2048(a, b, c);
}
