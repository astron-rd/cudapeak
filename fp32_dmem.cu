#include "common.h"

__global__ void fp32_dmem_01(float4 *ptr);
__global__ void fp32_dmem_02(float4 *ptr);
__global__ void fp32_dmem_04(float4 *ptr);
__global__ void fp32_dmem_08(float4 *ptr);
__global__ void fp32_dmem_16(float4 *ptr);
__global__ void fp32_dmem_32(float4 *ptr);
__global__ void fp32_dmem_64(float4 *ptr);
__global__ void fp32_dmem_128(float4 *ptr);
__global__ void fp32_dmem_256(float4 *ptr);

void run(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties)
{
    // Parameters
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    // Amount of work performed
    unsigned fetchPerBlock = 16;
    int maxItems = deviceProperties.totalGlobalMem / sizeof(float4) / 2;
    int numItems = roundToPowOf2(maxItems);
    double gbytes = (float) (numItems / fetchPerBlock * 2) * sizeof(float4) / 1e9;
    double gflops = gbytes * 4;

    // Kernel dimensions
    dim3 gridDim(numItems / (fetchPerBlock * maxThreadsPerBlock));
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, numItems * sizeof(float4));

    // Run kernels
    double milliseconds;
    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_01, ptr, gridDim, blockDim);
    report("flop:byte ->   1:1", milliseconds, gflops, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_02, ptr, gridDim, blockDim);
    report("flop:byte ->   2:1", milliseconds, 2*gflops, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_04, ptr, gridDim, blockDim);
    report("flop:byte ->   4:1", milliseconds, 4*gflops, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_08, ptr, gridDim, blockDim);
    report("flop:byte ->   8:1", milliseconds, 8*gflops, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_16, ptr, gridDim, blockDim);
    report("flop:byte ->  16:1", milliseconds, 16*gflops, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_32, ptr, gridDim, blockDim);
    report("flop:byte ->  32:1", milliseconds, 32*gflops, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_64, ptr, gridDim, blockDim);
    report("flop:byte ->  64:1", milliseconds, 64*gflops, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_128, ptr, gridDim, blockDim);
    report("flop:byte -> 128:1", milliseconds, 128*gflops, gbytes);

    milliseconds = run_kernel(stream, deviceProperties, (void *) &fp32_dmem_256, ptr, gridDim, blockDim);
    report("flop:byte -> 256:1", milliseconds, 256*gflops, gbytes);

    // Free memory
    cudaFree(ptr);
}
