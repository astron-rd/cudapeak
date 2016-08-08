#include <iostream>
#include <string>
#include <iomanip>
#include <cstdio>
#include <cstdint>
#include <cuda.h>
#include <cudaProfiler.h>

#include "compute_sp_kernels.cu"

// Number of times to run each kernel
#define NR_ITERATIONS 10

// Number of times to run each benchmark
#define NR_BENCHMARKS 1

cudaStream_t stream;
cudaDeviceProp deviceProperties;

using namespace std;

void report_flops(string name, double milliseconds, double gflops) {
    cout << setw(10) << string(name) << ": ";
    cout << setprecision(2) << fixed;
    cout << milliseconds << " ms, ";
    cout << gflops / milliseconds / 1e6 << " TFLOPS";
    cout << endl;
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
    compute_sp_v1<<<gridDim, blockDim, 0, stream>>>(ptr);

    // Benchmark
    cudaEventRecord(start, stream);
    for (int i = 0; i < NR_ITERATIONS; i++) {
        compute_sp_v1<<<gridDim, blockDim, 0, stream>>>(ptr);
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

     // Flops executed per thread block
    uint64_t nr_flops_block = 8192 * 2 * 1000;

     // Total number of flops executed by device
    double nr_gflops_total = ((float) nr_flops_block * multiProcessorCount * maxThreadsPerBlock) / 1e3f;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Kernel data
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernel
    double milliseconds;
    milliseconds = run_kernel((void *) &compute_sp_v1, ptr, gridDim, blockDim);
    report_flops("compute_sp_v1", milliseconds, nr_gflops_total);
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
        run_compute_sp();
    }
    cuProfilerStop();

    return EXIT_SUCCESS;
}
