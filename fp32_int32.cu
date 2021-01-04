#include "common.h"

__global__ void fp32_int32_1_64(float *ptr);
__global__ void fp32_int32_1_32(float *ptr);
__global__ void fp32_int32_1_16(float *ptr);
__global__ void fp32_int32_1_8(float *ptr);
__global__ void fp32_int32_1_4(float *ptr);
__global__ void fp32_int32_1_2(float *ptr);
__global__ void fp32_int32_1_1(float *ptr);
__global__ void fp32_int32_2_1(float *ptr);
__global__ void fp32_int32_4_1(float *ptr);
__global__ void fp32_int32_8_1(float *ptr);
__global__ void fp32_int32_16_1(float *ptr);
__global__ void fp32_int32_32_1(float *ptr);
__global__ void fp32_int32_64_1(float *ptr);

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

    // Run kernels
    double milliseconds;
    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_1_64, ptr, gridDim, blockDim);
    report("fp32:int32 ->   1:64", milliseconds, gflops/64, gbytes, gflops);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_1_32, ptr, gridDim, blockDim);
    report("fp32:int32 ->   1:32", milliseconds, gflops/32, gbytes, gflops);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_1_16, ptr, gridDim, blockDim);
    report("fp32:int32 ->   1:16", milliseconds, gflops/16, gbytes, gflops);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_1_8, ptr, gridDim, blockDim);
    report("fp32:int32 ->    1:8", milliseconds, gflops/8, gbytes, gflops);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_1_4, ptr, gridDim, blockDim);
    report("fp32:int32 ->    1:4", milliseconds, gflops/4, gbytes, gflops);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_1_2, ptr, gridDim, blockDim);
    report("fp32:int32 ->    1:2", milliseconds, gflops/2, gbytes, gflops);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_1_1, ptr, gridDim, blockDim);
    report("fp32:int32 ->    1:1", milliseconds, gflops, gbytes, gflops);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_2_1, ptr, gridDim, blockDim);
    report("fp32:int32 ->    2:1", milliseconds, gflops, gbytes, gflops/2);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_4_1, ptr, gridDim, blockDim);
    report("fp32:int32 ->    4:1", milliseconds, gflops, gbytes, gflops/4);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_8_1, ptr, gridDim, blockDim);
    report("fp32:int32 ->    8:1", milliseconds, gflops, gbytes, gflops/8);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_16_1, ptr, gridDim, blockDim);
    report("fp32:int32 ->   16:1", milliseconds, gflops, gbytes, gflops/16);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_32_1, ptr, gridDim, blockDim);
    report("fp32:int32 ->   32:1", milliseconds, gflops, gbytes, gflops/32);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_int32_64_1, ptr, gridDim, blockDim);
    report("fp32:int32 ->   64:1", milliseconds, gflops, gbytes, gflops/64);

    // Free memory
    cudaFree(ptr);
}
