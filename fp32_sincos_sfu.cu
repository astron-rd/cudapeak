#include "common.h"

__global__ void fp32_sincos_sfu_1_8(float *ptr);
__global__ void fp32_sincos_sfu_1_4(float *ptr);
__global__ void fp32_sincos_sfu_1_2(float *ptr);
__global__ void fp32_sincos_sfu_1_1(float *ptr);
__global__ void fp32_sincos_sfu_2_1(float *ptr);
__global__ void fp32_sincos_sfu_4_1(float *ptr);
__global__ void fp32_sincos_sfu_8_1(float *ptr);
__global__ void fp32_sincos_sfu_16_1(float *ptr);
__global__ void fp32_sincos_sfu_32_1(float *ptr);
__global__ void fp32_sincos_sfu_64_1(float *ptr);
__global__ void fp32_sincos_sfu_128_1(float *ptr);

void run(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties)
{
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    // Amount of work performed
    int nr_iterations = 512;
    double gflops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) * (1ULL * nr_iterations * 8192 * 2);
    double gbytes = 0;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernels
    double milliseconds;
    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_sincos_sfu_1_8, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->    1:8", milliseconds, gflops, gbytes, gflops*8);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_sincos_sfu_1_4, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->    1:4", milliseconds, gflops, gbytes, gflops*4);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_sincos_sfu_1_2, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->    1:2", milliseconds, gflops, gbytes, gflops*2);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_sincos_sfu_1_1, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->    1:1", milliseconds, gflops, gbytes, gflops*1);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_sincos_sfu_2_1, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->    2:1", milliseconds, gflops, gbytes, gflops/2);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_sincos_sfu_4_1, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->    4:1", milliseconds, gflops, gbytes, gflops/4);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_sincos_sfu_8_1, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->    8:1", milliseconds, gflops, gbytes, gflops/8);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_sincos_sfu_16_1, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->   16:1", milliseconds, gflops, gbytes, gflops/16);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_sincos_sfu_32_1, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->   32:1", milliseconds, gflops, gbytes, gflops/32);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_sincos_sfu_64_1, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->   64:1", milliseconds, gflops, gbytes, gflops/64);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_sincos_sfu_128_1, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->  128:1", milliseconds, gflops, gbytes, gflops/128);

    // Free memory
    cudaFree(ptr);
}

