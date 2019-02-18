#include "flops.h"

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
