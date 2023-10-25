#include "common.h"

void report(
    string name,
    measurement measurement,
    double gflops,
    double gbytes,
    double gops)
{
    double milliseconds = measurement.runtime;
    double power = measurement.power;
    int w1 = 20;
    int w2 = 7;
    cout << setw(w1) << string(name) << ": ";
    cout << setprecision(2) << fixed;
    cout << setw(w2) << milliseconds << " ms";
    double seconds = milliseconds * 1e-3;
    if (gflops != 0) {
        cout << ", " << setw(w2) << gflops / seconds * 1e-3 << " TFlops/s";
    }
    if (power != 0) {
        cout << ", " << setw(w2) << power << " W";
        double efficiency = gflops / seconds / power;
        if (efficiency < 1e4) {
            cout << ", " << setw(w2) << gflops / seconds / power << " GFlops/W";
        }
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


measurement run_kernel(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties,
    void *kernel,
    void *ptr,
    dim3 gridDim,
    dim3 blockDim
#if defined(HAVE_PMT)
    , std::shared_ptr<pmt::PMT> pmt
#endif
    ) {
    // Setup events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    ((void (*)(void *)) kernel)<<<gridDim, blockDim, 0, stream>>>(ptr);

    // Benchmark
    cudaEventRecord(start, stream);
#if defined(HAVE_PMT)
    pmt::State state_start = pmt->Read();
#endif
    for (int i = 0; i < NR_ITERATIONS; i++) {
        ((void (*)(void *)) kernel)<<<gridDim, blockDim, 0, stream>>>(ptr);
    }
    cudaEventRecord(stop, stream);
#if defined(HAVE_PMT)
    pmt::State state_end = pmt->Read();
#endif

    // Finish measurement
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    measurement measurement;
    measurement.runtime = milliseconds / NR_ITERATIONS;
    measurement.energy = 0;
#if defined(HAVE_PMT)
    measurement.energy = pmt::PMT::joules(state_start, state_end) / NR_ITERATIONS;
    measurement.power = pmt::PMT::watts(state_start, state_end);
#endif

    return measurement;
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

    // Run benchmark
#if defined(HAVE_PMT)
    std::unique_ptr<pmt::PMT> pmt_ = pmt::nvml::NVML::Create();
    std::shared_ptr<pmt::PMT> pmt = std::move(pmt_);
#endif
    for (int i = 0; i < NR_BENCHMARKS; i++) {
       run(stream, deviceProperties
#if defined(HAVE_PMT)
       , pmt
#endif
       );
    }

    return EXIT_SUCCESS;
}
