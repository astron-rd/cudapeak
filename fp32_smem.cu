#include "common.h"

__global__ void fp32_smem_01(float *ptr);
__global__ void fp32_smem_02(float *ptr);
__global__ void fp32_smem_04(float *ptr);
__global__ void fp32_smem_08(float *ptr);
__global__ void fp32_smem_16(float *ptr);
__global__ void fp32_smem_32(float *ptr);
__global__ void fp32_smem_64(float *ptr);
__global__ void fp32_smem_128(float *ptr);

void run(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties)
{
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
    int maxBlocksPerSM = 16;

    // Amount of work performed
    unsigned nrRepetitions = 512;
    unsigned fetchPerBlock = 2048;
    unsigned workPerBlock = nrRepetitions * fetchPerBlock;
    unsigned globalBlocks = multiProcessorCount * maxBlocksPerSM * maxThreadsPerBlock;
    double gbytes = 1e-9 * globalBlocks * workPerBlock * sizeof(float4);
    double gflops = gbytes / 2;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount, maxBlocksPerSM);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, fetchPerBlock * 4 * sizeof(float));

    // Run kernels
    double milliseconds;
    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_01, ptr, gridDim, blockDim);
    report("flop:byte ->  1:2", milliseconds, gflops, gbytes/1);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_02, ptr, gridDim, blockDim);
    report("flop:byte ->  1:1", milliseconds, gflops, gbytes/2);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_04, ptr, gridDim, blockDim);
    report("flop:byte ->  2:1", milliseconds, gflops, gbytes/4);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_08, ptr, gridDim, blockDim);
    report("flop:byte ->  4:1", milliseconds, gflops, gbytes/8);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_16, ptr, gridDim, blockDim);
    report("flop:byte ->  8:1", milliseconds, gflops, gbytes/16);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_32, ptr, gridDim, blockDim);
    report("flop:byte -> 16:1", milliseconds, gflops, gbytes/32);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_64, ptr, gridDim, blockDim);
    report("flop:byte -> 32:1", milliseconds, gflops, gbytes/64);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_128, ptr, gridDim, blockDim);
    report("flop:byte -> 64:1", milliseconds, gflops, gbytes/128);

    // Free memory
    cudaFree(ptr);
}
