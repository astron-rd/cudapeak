#include "common.h"

__global__ void dmem_kernel(float *ptr);

void run(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties)
{
    // Parameters
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    // Amount of work performed
    unsigned fetchPerBlock = 16;
    int maxItems = deviceProperties.totalGlobalMem / sizeof(float) / 2;
    int numItems = roundToPowOf2(maxItems);
    double gbytes = (float) (numItems / fetchPerBlock) * sizeof(float) / 1e9;
    double gflops = 0;

    // Kernel dimensions
    dim3 gridDim(numItems / (fetchPerBlock * maxThreadsPerBlock));
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, numItems * sizeof(float));

    // Run kernel
    double milliseconds;
    milliseconds = run_kernel(stream, deviceProperties, (void *) &dmem_kernel, ptr, gridDim, blockDim);
    report("dmem", milliseconds, gflops, gbytes);

    // Free memory
    cudaFree(ptr);
}
