#define NR_ITERATIONS 2048

template<int nr_fp32, int nr_int32>
__device__ void fp32_int32_8_8(
    float2& a, float2& b, float2& c,
    int2& d, int2& e, int2& f)
{
    // Perform nr_fp32 * 4 fma
    #pragma unroll nr_fp32
    for (int i = 0; i < nr_fp32; i++) {
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(b.x),  "f"(c.x), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.x) : "f"(-b.y), "f"(c.y), "f"(a.x));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.x),  "f"(c.y), "f"(a.y));
        asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a.y) : "f"(b.y),  "f"(c.x), "f"(a.y));
    }
    // Perform nr_int32 * 4 imad
    #pragma unroll nr_int32
    for (int i = 0; i < nr_int32; i++) {
        asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(d.x) : "r"(e.x), "r"(f.x), "r"(d.x));
        asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(d.x) : "r"(-e.y), "r"(f.y), "r"(d.x));
        asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(d.y) : "r"(e.x), "r"(f.y), "r"(d.y));
        asm("mad.lo.s32 %0, %1, %2, %3;" : "=r"(d.y) : "r"(e.y), "r"(f.x), "r"(d.y));
    }
}

#define INIT \
    float2 a = make_float2(threadIdx.x, 0); \
    float2 b = make_float2(threadIdx.x, 1); \
    float2 c = make_float2(threadIdx.x, 2); \
    int2 d = make_int2(threadIdx.x, 0); \
    int2 e = make_int2(threadIdx.x, 1); \
    int2 f = make_int2(threadIdx.x, 2);


#define FINISH \
    ptr[blockIdx.x * blockDim.x + threadIdx.x] = \
        a.x + a.y + \
        d.x + d.y;

__global__ void fp32_int32_1_64(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096/64, 4096>(a, b, c, d, e, f);
    }

    FINISH
}

__global__ void fp32_int32_1_32(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096/32, 4096>(a, b, c, d, e, f);
    }

    FINISH
}

__global__ void fp32_int32_1_16(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096/16, 4096>(a, b, c, d, e, f);
    }

	FINISH
}

__global__ void fp32_int32_1_8(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096/8, 4096>(a, b, c, d, e, f);
    }

	FINISH
}

__global__ void fp32_int32_1_4(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096/4, 4096>(a, b, c, d, e, f);
    }

	FINISH
}

__global__ void fp32_int32_1_2(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096/2, 4096>(a, b, c, d, e, f);
    }

	FINISH
}

__global__ void fp32_int32_1_1(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096, 4096>(a, b, c, d, e, f);
    }

	FINISH
}

__global__ void fp32_int32_2_1(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096, 4096/2>(a, b, c, d, e, f);
    }

	FINISH
}

__global__ void fp32_int32_4_1(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096, 4096/4>(a, b, c, d, e, f);
    }

	FINISH
}

__global__ void fp32_int32_8_1(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096, 4096/8>(a, b, c, d, e, f);
    }

	FINISH
}

__global__ void fp32_int32_16_1(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096, 4096/16>(a, b, c, d, e, f);
    }

	FINISH
}

__global__ void fp32_int32_32_1(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096, 4096/32>(a, b, c, d, e, f);
    }

	FINISH
}

__global__ void fp32_int32_64_1(float *ptr)
{
    INIT

    for (int i = 0; i < NR_ITERATIONS; i++) {
        fp32_int32_8_8<4096, 4096/64>(a, b, c, d, e, f);
    }

	FINISH
}