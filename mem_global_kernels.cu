#include "cuda.h"

#define FETCH_2(sum, id, ptr, jump) sum += ptr[id]; id += jump; sum += ptr[id]; id += jump;
#define FETCH_4(sum, id, ptr, jump) FETCH_2(sum, id, ptr, jump);    FETCH_2(sum, id, ptr, jump)
#define FETCH_8(sum, id, ptr, jump) FETCH_4(sum, id, ptr, jump);    FETCH_4(sum, id, ptr, jump)

#define FETCH_PER_BLOCK 16

__global__ void mem_global_v1(float *ptr) {
    int id = (blockIdx.x * blockDim.x * FETCH_PER_BLOCK) + threadIdx.x;
    float sum = 0;

    FETCH_8(sum, id, ptr, blockDim.x);
    FETCH_8(sum, id, ptr, blockDim.x);

    ptr[(blockIdx.x * blockDim.x) + threadIdx.x] =  sum;
}
