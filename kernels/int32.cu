#include "int32_common.h"

__global__ void int32_kernel(int *ptr)
{
    int2 a = make_int2(threadIdx.x, 0);
    int2 b = make_int2(threadIdx.x, 1);
    int2 c = make_int2(threadIdx.x, 2);

    for (int i = 0; i < 2048; i++) {
        int32_8192(a, b, c); int32_8192(a, b, c);
        int32_8192(a, b, c); int32_8192(a, b, c);
    }

    ptr[blockIdx.x * blockDim.x + threadIdx.x] = a.x + a.y;
}
