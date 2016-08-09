#include <iostream>
#include <string>
#include <iomanip>
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cudaProfiler.h>

#include "compute_sp_kernels.cu"
#include "compute_sp_ai_kernels.cu"
#include "mem_global_kernels.cu"


using namespace std;


// Number of times to run each kernel
#define NR_ITERATIONS 10

// Number of times to run each benchmark
#define NR_BENCHMARKS 1


// CUDA variables
cudaStream_t stream;
cudaDeviceProp deviceProperties;


void report(string name, double milliseconds, double gflops, double gbytes) {
    cout << setw(10) << string(name) << ": ";
    cout << setprecision(2) << fixed;
    cout << milliseconds << " ms";
    if (gflops != 0)
        cout << ", " << gflops / milliseconds / 1e3 << " TFLOPS";
    if (gbytes != 0)
        cout << ", " << gbytes / milliseconds << " GB/s";
    if (gflops != 0 && gbytes != 0) {
        float arithmetic_intensity = gflops / gbytes;
        cout << ", " << arithmetic_intensity << " Flop/byte";
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

    // Amount of work performed
    double nr_gflops_total = (1e-6 * multiProcessorCount * maxThreadsPerBlock) * 1024;
    double nr_gbytes_total = (1e-6 * multiProcessorCount * maxThreadsPerBlock) * 1024 * 2 * sizeof(float);

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernels
    double milliseconds;
    milliseconds = run_kernel((void *) &compute_sp_ai_v1, ptr, gridDim, blockDim);
    report("compute_sp_ai_v1", milliseconds, nr_gflops_total*2, nr_gbytes_total);

    milliseconds = run_kernel((void *) &compute_sp_ai_v2, ptr, gridDim, blockDim);
    report("compute_sp_ai_v2", milliseconds, nr_gflops_total*4, nr_gbytes_total);

    milliseconds = run_kernel((void *) &compute_sp_ai_v3, ptr, gridDim, blockDim);
    report("compute_sp_ai_v3", milliseconds, nr_gflops_total*6, nr_gbytes_total);

    milliseconds = run_kernel((void *) &compute_sp_ai_v4, ptr, gridDim, blockDim);
    report("compute_sp_ai_v4", milliseconds, nr_gflops_total*8, nr_gbytes_total);

    milliseconds = run_kernel((void *) &compute_sp_ai_v5, ptr, gridDim, blockDim);
    report("compute_sp_ai_v5", milliseconds, nr_gflops_total*16, nr_gbytes_total);

    milliseconds = run_kernel((void *) &compute_sp_ai_v6, ptr, gridDim, blockDim);
    report("compute_sp_ai_v6", milliseconds, nr_gflops_total*32, nr_gbytes_total);

    milliseconds = run_kernel((void *) &compute_sp_ai_v7, ptr, gridDim, blockDim);
    report("compute_sp_ai_v7", milliseconds, nr_gflops_total*40, nr_gbytes_total);

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
    }
    cuProfilerStop();

    return EXIT_SUCCESS;
}
