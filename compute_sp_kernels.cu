#include "cuda.h"

#define FMA_1(x, y)  asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(x) : "f"(x), "f"(y), "f"(x)); \
                     asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(y) : "f"(y), "f"(x), "f"(y));
#define FMA_4(x, y)  FMA_1(x, y)  FMA_1(x, y)  FMA_1(x, y)  FMA_1(x,y)
#define FMA_16(x, y) FMA_4(x, y)  FMA_4(x, y)  FMA_4(x, y)  FMA_4(x, y)
#define FMA_64(x, y) FMA_16(x, y) FMA_16(x, y) FMA_16(x, y) FMA_16(x, y)

__global__ void compute_sp_v1(float *ptr)
{
    float x = threadIdx.x;
    float y = 0;

    for (int i = 0; i < 1024; i++) {
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);

        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
        FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);   FMA_64(x, y);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = x + y;
}
