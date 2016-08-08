#include "cuda.h"

#define CMUL_4(x, y)    x = x * y;     y = y * x;     x = x * y;     y = y * x;
#define CMUL_16(x, y)   CMUL_4(x, y);  CMUL_4(x, y);  CMUL_4(x, y);  CMUL_4(x, y);
#define CMUL_64(x, y)   CMUL_16(x, y); CMUL_16(x, y); CMUL_16(x, y); CMUL_16(x, y);

inline __device__ float2 operator*(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y,
                       a.x * b.y + a.y * b.x);
}

__global__ void compute_sp_v1(float *ptr, float a)
{
    float2 x = make_float2(a, 0);
    float2 y = make_float2(threadIdx.x, 0);

    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);

    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);
    CMUL_64(x, y);   CMUL_64(x, y);

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = y.x + y.y;
}
