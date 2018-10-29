#include <iostream>
#include <string>
#include <iomanip>
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cudaProfiler.h>

#include "compute_sp_kernels.cu"
#include "compute_sp_ai_kernels.cu"
#include "compute_sp_sincos_kernels.cu"
#include "mem_global_kernels.cu"


using namespace std;


// Number of times to run each kernel
#define NR_ITERATIONS 5

// Number of times to run each benchmark
#define NR_BENCHMARKS 1

// CUDA variables
cudaStream_t stream;
cudaDeviceProp deviceProperties;


void report(
    string name,
    double milliseconds,
    double gflops = 0,
    double gbytes = 0,
    double gops   = 0)
{
    int w1 = 20;
    int w2 = 7;
    cout << setw(w1) << string(name) << ": ";
    cout << setprecision(2) << fixed;
    cout << setw(w2) << milliseconds << " ms";
    if (gflops != 0) {
        cout << ", " << setw(w2) << gflops / milliseconds * 1e-3 << " TFlops/s";
    }
    if (gbytes != 0) {
        cout << ", " << setw(w2) << gbytes / milliseconds << " GB/s";
    }
    if (gflops != 0 && gbytes != 0) {
        float arithmetic_intensity = gflops / gbytes;
        cout << ", " << setw(w2) << arithmetic_intensity << " Flop/byte";
    }
    if (gops != 0) {
        cout << ", " << setw(w2) << gops / milliseconds * 1e-3 << " TOps/s";
    }
    cout << endl;
}


unsigned roundToPowOf2(unsigned number) {
    double logd = log(number) / log(2);
    logd = floor(logd);

    return (unsigned) pow(2, (int) logd);
}


double run_kernel(
    void *kernel,
    float *ptr,
    dim3 gridDim,
    dim3 blockDim) {
    // Setup events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    ((void (*)(float *)) kernel)<<<gridDim, blockDim, 0, stream>>>(ptr);

    // Benchmark
    cudaEventRecord(start, stream);
    for (int i = 0; i < NR_ITERATIONS; i++) {
        ((void (*)(float *)) kernel)<<<gridDim, blockDim, 0, stream>>>(ptr);
    }
    cudaEventRecord(stop, stream);

    // Finish measurement
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds / NR_ITERATIONS;
}


void run_compute_sp() {
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    // Amount of work performed
    double nr_gflops_total = 0;
    double nr_gbytes_total = 0;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernel
    double milliseconds;
    milliseconds = run_kernel((void *) &compute_sp_v1, ptr, gridDim, blockDim);
    nr_gflops_total = (1e-6 * multiProcessorCount * maxThreadsPerBlock) * (1ULL * 2 * 2 * 64 * 1024 * 8 * 8);
    nr_gbytes_total = 0;
    report("compute_sp_v1", milliseconds, nr_gflops_total, nr_gbytes_total);

    // Free memory
    cudaFree(ptr);
}

void run_compute_sp_ai() {
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;
    int maxBlocksPerSM = deviceProperties.major >= 5 ? 32 : 16;

    // Amount of work performed
    unsigned workPerBlock = 128 * 512 * 2;
    unsigned globalBlocks = multiProcessorCount * maxBlocksPerSM * maxThreadsPerBlock;
    double nr_gflops_total = (1e-6 * globalBlocks * workPerBlock);
    double nr_gbytes_total = (1e-6 * globalBlocks * workPerBlock) * sizeof(float);

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount, maxBlocksPerSM);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernels
    double milliseconds;
    milliseconds = run_kernel((void *) &compute_sp_ai_v1, ptr, gridDim, blockDim);
    report("flop:byte -> 0.25", milliseconds, nr_gflops_total*1, nr_gbytes_total);

    milliseconds = run_kernel((void *) &compute_sp_ai_v2, ptr, gridDim, blockDim);
    report("flop:byte -> 0.50", milliseconds, nr_gflops_total*2, nr_gbytes_total);

    milliseconds = run_kernel((void *) &compute_sp_ai_v3, ptr, gridDim, blockDim);
    report("flop:byte -> 1.00", milliseconds, nr_gflops_total*4, nr_gbytes_total);

    //milliseconds = run_kernel((void *) &compute_sp_ai_v4, ptr, gridDim, blockDim);
    //report("flop:byte -> 1.25", milliseconds, nr_gflops_total*5, nr_gbytes_total);

    milliseconds = run_kernel((void *) &compute_sp_ai_v5, ptr, gridDim, blockDim);
    report("flop:byte -> 2.00", milliseconds, nr_gflops_total*8, nr_gbytes_total);

    milliseconds = run_kernel((void *) &compute_sp_ai_v6, ptr, gridDim, blockDim);
    report("flop:byte -> 4.00", milliseconds, nr_gflops_total*16, nr_gbytes_total);

    //milliseconds = run_kernel((void *) &compute_sp_ai_v7, ptr, gridDim, blockDim);
    //report("flop:byte -> 5.00", milliseconds, nr_gflops_total*20, nr_gbytes_total);

    // Free memory
    cudaFree(ptr);
}

void run_compute_sp_sincos() {
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    // Amount of work performed
    double gflops = (1e-6 * multiProcessorCount * maxThreadsPerBlock) * (1ULL * 2 * 2 * 2048 * 64 * 4 * 8);
    double gbytes = 0;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernels
    double milliseconds;
    milliseconds = run_kernel((void *) &compute_sp_sincos_b0, ptr, gridDim, blockDim);
    report("fma:sincos ->    1:0", milliseconds, 0, gbytes, gflops*4);

    milliseconds = run_kernel((void *) &compute_sp_sincos_b1, ptr, gridDim, blockDim);
    report("fma:sincos ->    0:1", milliseconds, 0, gbytes, gflops*2);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v00, ptr, gridDim, blockDim);
    report("fma:sincos ->    1:8", milliseconds, gflops, gbytes, gflops + gflops*8);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v01, ptr, gridDim, blockDim);
    report("fma:sincos ->    1:4", milliseconds, gflops, gbytes, gflops + gflops*4);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v02, ptr, gridDim, blockDim);
    report("fma:sincos ->    1:2", milliseconds, gflops, gbytes, gflops + gflops*2);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v03, ptr, gridDim, blockDim);
    report("fma:sincos ->    1:1", milliseconds, gflops, gbytes, gflops + gflops*1);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v04, ptr, gridDim, blockDim);
    report("fma:sincos ->    2:1", milliseconds, gflops, gbytes, gflops + gflops/2.0);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v05, ptr, gridDim, blockDim);
    report("fma:sincos ->    4:1", milliseconds, gflops, gbytes, gflops + gflops/4.0);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v06, ptr, gridDim, blockDim);
    report("fma:sincos ->    8:1", milliseconds, gflops, gbytes, gflops + gflops/8.0);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v07, ptr, gridDim, blockDim);
    report("fma:sincos ->   16:1", milliseconds, gflops, gbytes, gflops + gflops/16.0);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v08, ptr, gridDim, blockDim);
    report("fma:sincos ->   32:1", milliseconds, gflops, gbytes, gflops + gflops/32.0);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v09, ptr, gridDim, blockDim);
    report("fma:sincos ->   64:1", milliseconds, gflops, gbytes, gflops + gflops/64.0);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v10, ptr, gridDim, blockDim);
    report("fma:sincos ->  128:1", milliseconds, gflops, gbytes, gflops + gflops/128.0);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v11, ptr, gridDim, blockDim);
    report("fma:sincos ->  256:1", milliseconds, gflops, gbytes, gflops + gflops/256.0);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v12, ptr, gridDim, blockDim);
    report("fma:sincos ->  512:1", milliseconds, gflops, gbytes, gflops + gflops/512.0);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v13, ptr, gridDim, blockDim);
    report("fma:sincos -> 1024:1", milliseconds, gflops, gbytes, gflops + gflops/1024.0);

    milliseconds = run_kernel((void *) &compute_sp_sincos_v14, ptr, gridDim, blockDim);
    report("fma:sincos -> 2048:1", milliseconds, gflops, gbytes, gflops + gflops/2048.0);

    // Free memory
    cudaFree(ptr);
}


void run_mem_global() {
    // Parameters
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    // Amount of work performed
    unsigned fetchPerBlock = 16;
    int maxItems = deviceProperties.totalGlobalMem / sizeof(float) / 2;
    int numItems = roundToPowOf2(maxItems);
    double nr_gbytes_total = (float) (numItems / fetchPerBlock) * sizeof(float) / 1e6;
    double nr_gflops_total = 0;

    // Kernel dimensions
    dim3 gridDim(numItems / (fetchPerBlock * maxThreadsPerBlock));
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, numItems * sizeof(float));

    // Run kernel
    double milliseconds;
    milliseconds = run_kernel((void *) &mem_global_v1, ptr, gridDim, blockDim);
    report("mem_global_v1", milliseconds, nr_gflops_total, nr_gbytes_total);

    // Free memory
    cudaFree(ptr);
}


int main() {
    // Read device number from envirionment
    char *cstr_deviceNumber = getenv("CUDA_DEVICE");
    unsigned deviceNumber = cstr_deviceNumber ? atoi (cstr_deviceNumber) : 0;

    //  Setup CUDA
    cudaSetDevice(deviceNumber);
    cudaStreamCreate(&stream);
    cudaGetDeviceProperties(&deviceProperties, deviceNumber);

    // Print CUDA device information
    std::cout << "Device " << deviceNumber << ": " << deviceProperties.name << std::endl;

    // Run benchmarks
    cuProfilerStart();
    for (int i = 0; i < NR_BENCHMARKS; i++) {
        run_mem_global();
        run_compute_sp();
        run_compute_sp_ai();
        run_compute_sp_sincos();
    }
    cuProfilerStop();

    return EXIT_SUCCESS;
}
