#include <iostream>
#include <string>
#include <iomanip>
#include <cstdio>
#include <cstdint>
#include <cmath>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cudaProfiler.h>

__global__ void mem_global_v1(float *ptr);
__global__ void compute_sp_v1(float *ptr);
__global__ void compute_sp_ai_01(float *ptr);
__global__ void compute_sp_ai_02(float *ptr);
__global__ void compute_sp_ai_04(float *ptr);
__global__ void compute_sp_ai_08(float *ptr);
__global__ void compute_sp_ai_16(float *ptr);
__global__ void compute_sp_ai_32(float *ptr);
__global__ void compute_sp_sincos_fpu_01(float *ptr);
__global__ void compute_sp_sincos_fpu_02(float *ptr);
__global__ void compute_sp_sincos_fpu_04(float *ptr);
__global__ void compute_sp_sincos_fpu_08(float *ptr);
__global__ void compute_sp_sincos_fpu_16(float *ptr);
__global__ void compute_sp_sincos_fpu_32(float *ptr);
__global__ void compute_sp_sincos_fpu_64(float *ptr);
__global__ void compute_sp_sincos_sfu_01(float *ptr);
__global__ void compute_sp_sincos_sfu_02(float *ptr);
__global__ void compute_sp_sincos_sfu_04(float *ptr);
__global__ void compute_sp_sincos_sfu_08(float *ptr);
__global__ void compute_sp_sincos_sfu_16(float *ptr);
__global__ void compute_sp_sincos_sfu_32(float *ptr);
__global__ void compute_sp_sincos_sfu_64(float *ptr);

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
    double seconds = milliseconds * 1e-3;
    if (gflops != 0) {
        cout << ", " << setw(w2) << gflops / seconds * 1e-3 << " TFlops/s";
    }
    if (gbytes != 0) {
        cout << ", " << setw(w2) << gbytes / seconds << " GB/s";
    }
    if (gflops != 0 && gbytes != 0) {
        float arithmetic_intensity = gflops / gbytes;
        cout << ", " << setw(w2) << arithmetic_intensity << " Flop/byte";
    }
    if (gops != 0) {
        cout << ", " << setw(w2) << gops / seconds * 1e-3 << " TOps/s";
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
    int nr_iterations = 2048;
    double gflops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) * (1ULL * nr_iterations * 4 * 8192);
    double gbytes = 0;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernel
    double milliseconds;
    milliseconds = run_kernel((void *) &compute_sp_v1, ptr, gridDim, blockDim);
    report("compute_sp_v1", milliseconds, gflops, gbytes);

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
    milliseconds = run_kernel((void *) &compute_sp_ai_01, ptr, gridDim, blockDim);
    report("flop:byte ->  1:1", milliseconds, gflops*1, gbytes);

    milliseconds = run_kernel((void *) &compute_sp_ai_02, ptr, gridDim, blockDim);
    report("flop:byte ->  2:1", milliseconds, gflops*2, gbytes);

    milliseconds = run_kernel((void *) &compute_sp_ai_04, ptr, gridDim, blockDim);
    report("flop:byte ->  4:1", milliseconds, gflops*4, gbytes);

    milliseconds = run_kernel((void *) &compute_sp_ai_08, ptr, gridDim, blockDim);
    report("flop:byte ->  8:1", milliseconds, gflops*8, gbytes);

    milliseconds = run_kernel((void *) &compute_sp_ai_16, ptr, gridDim, blockDim);
    report("flop:byte -> 16:1", milliseconds, gflops*16, gbytes);

    milliseconds = run_kernel((void *) &compute_sp_ai_32, ptr, gridDim, blockDim);
    report("flop:byte -> 32:1", milliseconds, gflops*32, gbytes);

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
    double gbytes = (float) (numItems / fetchPerBlock) * sizeof(float) / 1e9;
    double gflops = 0;

    // Kernel dimensions
    dim3 gridDim(numItems / (fetchPerBlock * maxThreadsPerBlock));
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, numItems * sizeof(float));

    // Run kernel
    double milliseconds;
    milliseconds = run_kernel((void *) &mem_global_v1, ptr, gridDim, blockDim);
    report("mem_global_v1", milliseconds, gflops, gbytes);

    // Free memory
    cudaFree(ptr);
}

void run_compute_sp_sincos_fpu() {
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    // Amount of work performed
    int nr_iterations = 512;
    double gflops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) * (1ULL * nr_iterations * 4 * 8192);
    double gbytes = 0;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernels
    double milliseconds;
    milliseconds = run_kernel((void *) &compute_sp_sincos_fpu_01, ptr, gridDim, blockDim);
    report("fma:sincos (fpu) ->    1:1", milliseconds, gflops, gbytes, gflops);

    milliseconds = run_kernel((void *) &compute_sp_sincos_fpu_02, ptr, gridDim, blockDim);
    report("fma:sincos (fpu) ->    2:1", milliseconds, gflops, gbytes, gflops/2);

    milliseconds = run_kernel((void *) &compute_sp_sincos_fpu_04, ptr, gridDim, blockDim);
    report("fma:sincos (fpu) ->    4:1", milliseconds, gflops, gbytes, gflops/4);

    milliseconds = run_kernel((void *) &compute_sp_sincos_fpu_08, ptr, gridDim, blockDim);
    report("fma:sincos (fpu) ->    8:1", milliseconds, gflops, gbytes, gflops/8);

    milliseconds = run_kernel((void *) &compute_sp_sincos_fpu_16, ptr, gridDim, blockDim);
    report("fma:sincos (fpu) ->   16:1", milliseconds, gflops, gbytes, gflops/16);

    milliseconds = run_kernel((void *) &compute_sp_sincos_fpu_32, ptr, gridDim, blockDim);
    report("fma:sincos (fpu) ->   32:1", milliseconds, gflops, gbytes, gflops/32);

    milliseconds = run_kernel((void *) &compute_sp_sincos_fpu_64, ptr, gridDim, blockDim);
    report("fma:sincos (fpu) ->   64:1", milliseconds, gflops, gbytes, gflops/64);

    // Free memory
    cudaFree(ptr);
}

void run_compute_sp_sincos_sfu() {
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    // Amount of work performed
    int nr_iterations = 512;
    double gflops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) * (1ULL * nr_iterations * 4 * 8192);
    double gbytes = 0;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernels
    double milliseconds;
    milliseconds = run_kernel((void *) &compute_sp_sincos_sfu_01, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->    1:1", milliseconds, gflops, gbytes, gflops);

    milliseconds = run_kernel((void *) &compute_sp_sincos_sfu_02, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->    2:1", milliseconds, gflops, gbytes, gflops/2);

    milliseconds = run_kernel((void *) &compute_sp_sincos_sfu_04, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->    4:1", milliseconds, gflops, gbytes, gflops/4);

    milliseconds = run_kernel((void *) &compute_sp_sincos_sfu_08, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->    8:1", milliseconds, gflops, gbytes, gflops/8);

    milliseconds = run_kernel((void *) &compute_sp_sincos_sfu_16, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->   16:1", milliseconds, gflops, gbytes, gflops/16);

    milliseconds = run_kernel((void *) &compute_sp_sincos_sfu_32, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->   32:1", milliseconds, gflops, gbytes, gflops/32);

    milliseconds = run_kernel((void *) &compute_sp_sincos_sfu_64, ptr, gridDim, blockDim);
    report("fma:sincos (sfu) ->   64:1", milliseconds, gflops, gbytes, gflops/64);

    // Free memory
    cudaFree(ptr);
}

int main() {
    // Read device number from envirionment
    char *cstr_deviceNumber = getenv("CUDA_DEVICE");
    unsigned deviceNumber = cstr_deviceNumber ? atoi (cstr_deviceNumber) : 0;

    // Setup CUDA
    cudaSetDevice(deviceNumber);
    cudaStreamCreate(&stream);
    cudaGetDeviceProperties(&deviceProperties, deviceNumber);

    // Print CUDA device information
    std::cout << "Device " << deviceNumber << ": " << deviceProperties.name;
    std::cout << " (" << deviceProperties.multiProcessorCount <<  "SMs, ";
    std::cout << deviceProperties.clockRate * 1e-6 << " Ghz)" << std::endl;

    // Run benchmarks
    cuProfilerStart();
    for (int i = 0; i < NR_BENCHMARKS; i++) {
        //run_mem_global();
        run_compute_sp();
        run_compute_sp_ai();
        //run_compute_sp_sincos_sfu();
        //run_compute_sp_sincos_fpu();
    }
    cuProfilerStop();

    return EXIT_SUCCESS;
}
