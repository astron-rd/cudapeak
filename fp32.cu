#include "common.h"

__global__ void fp32_kernel(float *ptr);

void run(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties)
{
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    // Amount of work performed
    int nr_iterations = 2048;
    double gflops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) * (1ULL * nr_iterations * 8 * 4096);
    double gbytes = 0;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernel
    double milliseconds;
    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_kernel, ptr, gridDim, blockDim);
    report("fp32", milliseconds, gflops, gbytes);

    // Free memory
    cudaFree(ptr);
}
