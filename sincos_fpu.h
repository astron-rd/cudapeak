// 2x sincos
__device__ void sincos_fpu_0002(float2 a, float2& b, float2& c)
{
    b.x = cos(a.x);
    b.y = sin(a.x);
    c.x = cos(a.y);
    c.y = sin(a.y);
}

__device__ void sincos_fpu_0008(float2 a, float2& b, float2& c)
{
    sincos_fpu_0002(a, b, c); sincos_fpu_0002(a, b, c); sincos_fpu_0002(a, b, c); sincos_fpu_0002(a, b, c);
}

__device__ void sincos_fpu_0032(float2 a, float2& b, float2& c)
{
    sincos_fpu_0008(a, b, c); sincos_fpu_0008(a, b, c); sincos_fpu_0008(a, b, c); sincos_fpu_0008(a, b, c);
}

__device__ void sincos_fpu_0128(float2 a, float2& b, float2& c)
{
    sincos_fpu_0032(a, b, c); sincos_fpu_0032(a, b, c); sincos_fpu_0032(a, b, c); sincos_fpu_0032(a, b, c);
}

__device__ void sincos_fpu_0512(float2 a, float2& b, float2& c)
{
    sincos_fpu_0128(a, b, c); sincos_fpu_0128(a, b, c); sincos_fpu_0128(a, b, c); sincos_fpu_0128(a, b, c);
}

__device__ void sincos_fpu_2048(float2 a, float2& b, float2& c)
{
    sincos_fpu_0512(a, b, c); sincos_fpu_0512(a, b, c); sincos_fpu_0512(a, b, c); sincos_fpu_0512(a, b, c);
}

__device__ void sincos_fpu_8192(float2 a, float2& b, float2& c)
{
    sincos_fpu_2048(a, b, c); sincos_fpu_2048(a, b, c); sincos_fpu_2048(a, b, c); sincos_fpu_2048(a, b, c);
}
