#include <iostream>
#include <string>
#include <iomanip>
#include <cstdio>
#include <cuda.h>
#include <cudaProfiler.h>

#include "compute_sp_kernels.cu"

cudaStream_t stream;
cudaDeviceProp deviceProperties;

using namespace std;

void report_flops(string name, double milliseconds, float gflops) {
    cout << setw(10) << string(name) << ": ";
    cout << setprecision(2) << fixed;
    cout << milliseconds << " milliseconds, ";
    cout << gflops / milliseconds / 1e6 << " TFLOPS";
    cout << endl;
}

double run_kernel(
    void *kernel,
    float *data,
    float a,
    dim3 gridDim,
    dim3 blockDim) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    compute_sp_v1<<<gridDim, blockDim, 0, stream>>>(data, a);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

void run_compute_sp() {
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

     // Flops executed per thread block
    unsigned nr_flops_block = 8192;

     // Total number of flops executed by device
    unsigned nr_gflops_total = ((float) nr_flops_block * multiProcessorCount * maxThreadsPerBlock) / 1e3f;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Kernel data
    float *data;
    cudaMalloc(&data, multiProcessorCount * maxThreadsPerBlock * sizeof(float));
    float a = 1.0;

    // Run kernel
    double milliseconds;
    milliseconds = run_kernel((void *) &compute_sp_v1, data, a, gridDim, blockDim);
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
    run_compute_sp();
    cuProfilerStop();

    return EXIT_SUCCESS;
}
