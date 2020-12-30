#include "common.h"

__global__ void int32_kernel(int *ptr);

void run(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties)
{
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    // Amount of work performed
    int nr_iterations = 2048;
    double gflops = 0;
    double gbytes = 0;
    double gops   = (1e-9 * multiProcessorCount * maxThreadsPerBlock) * (1ULL * nr_iterations * 4 * 8192);

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    int *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(int));

    // Run kernel
    double milliseconds;
    milliseconds = run_kernel(stream, deviceProperties, (void *) &int32_kernel, ptr, gridDim, blockDim);
    report("int32", milliseconds, gflops, gbytes, gops);

    // Free memory
    cudaFree(ptr);
}
