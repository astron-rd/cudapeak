#include "common.h"

#if defined(HAVE_PMT)
#include <pmt.h>
#endif

void report(
    string name,
    double milliseconds,
    double gflops,
    double gbytes,
    double gops)
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
    cudaStream_t stream,
    cudaDeviceProp deviceProperties,
    void *kernel,
    void *ptr,
    dim3 gridDim,
    dim3 blockDim) {
    // Setup events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    ((void (*)(void *)) kernel)<<<gridDim, blockDim, 0, stream>>>(ptr);

    // Benchmark
    cudaEventRecord(start, stream);
    for (int i = 0; i < NR_ITERATIONS; i++) {
        ((void (*)(void *)) kernel)<<<gridDim, blockDim, 0, stream>>>(ptr);
    }
    cudaEventRecord(stop, stream);

    // Finish measurement
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds / NR_ITERATIONS;
}

int main() {
    // Read device number from environment
    char *cstr_deviceNumber = getenv("CUDA_DEVICE");
    unsigned deviceNumber = cstr_deviceNumber ? atoi (cstr_deviceNumber) : 0;

    // Setup CUDA
    cudaSetDevice(deviceNumber);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, deviceNumber);

    // Print CUDA device information
    std::cout << "Device " << deviceNumber << ": " << deviceProperties.name;
    std::cout << " (" << deviceProperties.multiProcessorCount <<  "SMs, ";
    std::cout << deviceProperties.clockRate * 1e-6 << " Ghz)" << std::endl;

    // Run benchmarks
#if defined(HAVE_PMT)
    auto pmt = pmt::nvml::NVML::Create();
#endif
    for (int i = 0; i < NR_BENCHMARKS; i++) {
#if defined(HAVE_PMT)
    pmt::State start = pmt->Read();
#endif
       run(stream, deviceProperties);
#if defined(HAVE_PMT)
    pmt::State end = pmt->Read();
    double watts = pmt::PMT::watts(start, end);
    std::cout << "Watt: " << watts << std::endl;
#endif
    }

    return EXIT_SUCCESS;
}
