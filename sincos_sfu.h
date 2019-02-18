// 2x sincos
__device__ void sincos_sfu_0002(float2 a, float2& b, float2& c)
{
    asm("sin.approx.f32  %0, %1;" : "=f"(b.x) : "f"(a.x));
    asm("cos.approx.f32  %0, %1;" : "=f"(b.y) : "f"(a.x));
    asm("sin.approx.f32  %0, %1;" : "=f"(c.x) : "f"(a.x));
    asm("cos.approx.f32  %0, %1;" : "=f"(c.y) : "f"(a.x));
}

__device__ void sincos_sfu_0008(float2 a, float2& b, float2& c)
{
    sincos_sfu_0002(a, b, c); sincos_sfu_0002(a, b, c); sincos_sfu_0002(a, b, c); sincos_sfu_0002(a, b, c);
}

__device__ void sincos_sfu_0032(float2 a, float2& b, float2& c)
{
    sincos_sfu_0008(a, b, c); sincos_sfu_0008(a, b, c); sincos_sfu_0008(a, b, c); sincos_sfu_0008(a, b, c);
}

__device__ void sincos_sfu_0128(float2 a, float2& b, float2& c)
{
    sincos_sfu_0032(a, b, c); sincos_sfu_0032(a, b, c); sincos_sfu_0032(a, b, c); sincos_sfu_0032(a, b, c);
}

__device__ void sincos_sfu_0512(float2 a, float2& b, float2& c)
{
    sincos_sfu_0128(a, b, c); sincos_sfu_0128(a, b, c); sincos_sfu_0128(a, b, c); sincos_sfu_0128(a, b, c);
}

__device__ void sincos_sfu_2048(float2 a, float2& b, float2& c)
{
    sincos_sfu_0512(a, b, c); sincos_sfu_0512(a, b, c); sincos_sfu_0512(a, b, c); sincos_sfu_0512(a, b, c);
}

__device__ void sincos_sfu_8192(float2 a, float2& b, float2& c)
{
    sincos_sfu_2048(a, b, c); sincos_sfu_2048(a, b, c); sincos_sfu_2048(a, b, c); sincos_sfu_2048(a, b, c);
}
