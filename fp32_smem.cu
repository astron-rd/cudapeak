#include "common.h"

__global__ void fp32_smem_01(float *ptr);
__global__ void fp32_smem_02(float *ptr);
__global__ void fp32_smem_04(float *ptr);
__global__ void fp32_smem_08(float *ptr);
__global__ void fp32_smem_16(float *ptr);
__global__ void fp32_smem_32(float *ptr);

void run(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties)
{
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
    int maxBlocksPerSM = deviceProperties.major >= 5 ? 32 : 16;

    // Amount of work performed
    unsigned workPerBlock = 128 * 512 * 2;
    unsigned globalBlocks = multiProcessorCount * maxBlocksPerSM * maxThreadsPerBlock;
    double gflops = (1e-9 * globalBlocks * workPerBlock);
    double gbytes = (1e-9 * globalBlocks * workPerBlock) * 2;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount, maxBlocksPerSM);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernels
    double milliseconds;
    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_01, ptr, gridDim, blockDim);
    report("flop:byte ->  1:1", milliseconds, gflops*1, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_02, ptr, gridDim, blockDim);
    report("flop:byte ->  2:1", milliseconds, gflops*2, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_04, ptr, gridDim, blockDim);
    report("flop:byte ->  4:1", milliseconds, gflops*4, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_08, ptr, gridDim, blockDim);
    report("flop:byte ->  8:1", milliseconds, gflops*8, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_16, ptr, gridDim, blockDim);
    report("flop:byte -> 16:1", milliseconds, gflops*16, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_smem_32, ptr, gridDim, blockDim);
    report("flop:byte -> 32:1", milliseconds, gflops*32, gbytes);

    // Free memory
    cudaFree(ptr);
}
