// complex multiplication and addition: 8 ops
inline __device__ void int32_0008(int2& a, int2 b, int2 c)
{
#if 0
    a.x += b.x * c.x;
    a.x -= b.y * c.y;
    a.y += b.x * c.y;
    a.y += b.y * c.x;
#else
    asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(a.x) : "r"(b.x), "r"(c.x), "r"(a.x));
    asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(a.x) : "r"(-b.y), "r"(c.y), "r"(a.x));
    asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(a.y) : "r"(b.x), "r"(c.y), "r"(a.y));
    asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(a.y) : "r"(b.y), "r"(c.x), "r"(a.y));
#endif
}

inline __device__ void int32_0032(int2& a, int2 b, int2 c)
{
    int32_0008(a, b, c); int32_0008(a, b, c); int32_0008(a, b, c); int32_0008(a, b, c);
}

inline __device__ void int32_0128(int2& a, int2 b, int2 c)
{
    int32_0032(a, b, c); int32_0032(a, b, c); int32_0032(a, b, c); int32_0032(a, b, c);
}

inline __device__ void int32_0512(int2& a, int2 b, int2 c)
{
    int32_0128(a, b, c); int32_0128(a, b, c); int32_0128(a, b, c); int32_0128(a, b, c);
}

inline __device__ void int32_2048(int2& a, int2 b, int2 c)
{
    int32_0512(a, b, c); int32_0512(a, b, c); int32_0512(a, b, c); int32_0512(a, b, c);
}

inline __device__ void int32_8192(int2& a, int2 b, int2 c)
{
    int32_2048(a, b, c); int32_2048(a, b, c); int32_2048(a, b, c); int32_2048(a, b, c);
}
