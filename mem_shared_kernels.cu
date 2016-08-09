#include "cuda.h"

#ifdef FETCH_PER_THREAD
#undef FETCH_PER_THREAD
#endif
#define FETCH_PER_THREAD 1024

__global__ void mem_shared_v1(float *ptr) {
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    float sum = 0;
    __shared__ float data[FETCH_PER_THREAD];

    for (int i = 0; i < FETCH_PER_THREAD; i++) {
        sum += data[i];
    }

    ptr[id] = sum;
}
