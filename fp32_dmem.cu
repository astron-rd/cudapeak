#include "common.h"

__global__ void fp32_dmem_01(float4 *ptr);
__global__ void fp32_dmem_02(float4 *ptr);
__global__ void fp32_dmem_04(float4 *ptr);
__global__ void fp32_dmem_08(float4 *ptr);
__global__ void fp32_dmem_16(float4 *ptr);
__global__ void fp32_dmem_32(float4 *ptr);
__global__ void fp32_dmem_64(float4 *ptr);

void run(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties)
{
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
    int maxBlocksPerSM = 16;
    maxBlocksPerSM = 1;

    // Amount of work performed
    unsigned nr_iterations = 1024;
    unsigned nr_elements = 512;
    unsigned workPerBlock = nr_iterations * nr_elements * 8;
    unsigned globalBlocks = multiProcessorCount * maxBlocksPerSM * maxThreadsPerBlock;
    double gflops = (1e-9 * globalBlocks * workPerBlock);
    double gbytes = (1e-9 * globalBlocks * workPerBlock) * 2;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount, maxBlocksPerSM);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * nr_elements * 4 * sizeof(float));

    // Run kernels
    double milliseconds;
    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_01, ptr, gridDim, blockDim);
    report("flop:byte ->  1:1", milliseconds, gflops, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_02, ptr, gridDim, blockDim);
    report("flop:byte ->  2:1", milliseconds, gflops, gbytes/2);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_04, ptr, gridDim, blockDim);
    report("flop:byte ->  4:1", milliseconds, gflops, gbytes/4);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_08, ptr, gridDim, blockDim);
    report("flop:byte ->  8:1", milliseconds, gflops, gbytes/8);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_16, ptr, gridDim, blockDim);
    report("flop:byte -> 16:1", milliseconds, gflops, gbytes/16);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_32, ptr, gridDim, blockDim);
    report("flop:byte -> 32:1", milliseconds, gflops, gbytes/32);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_64, ptr, gridDim, blockDim);
    report("flop:byte -> 64:1", milliseconds, gflops, gbytes/64);

    // Free memory
    cudaFree(ptr);
}
