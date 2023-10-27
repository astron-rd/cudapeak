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
    if (power > 1) {
        cout << ", " << setw(w2) << power << " W";
        if (gflops != 0) {
            cout << ", " << setw(w2) << gflops / seconds / power << " GFlops/W";
        }
        if (gops != 0) {
            cout << ", " << setw(w2) << gops / seconds / power << " GOps/W";
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


Benchmark::Benchmark() {
    // Read device number from environment
    char *cstr_device_number = getenv("CUDA_DEVICE");
    unsigned device_number = cstr_device_number ? atoi (cstr_device_number) : 0;

    // Setup CUDA
    cudaSetDevice(device_number);
    cudaStreamCreate(&stream_);
    cudaGetDeviceProperties(&device_properties_, device_number);
    cudaEventCreate(&event_start_);
    cudaEventCreate(&event_end_);

    // Print CUDA device information
    std::cout << "Device " << device_number << ": " << device_properties_.name;
    std::cout << " (" << device_properties_.multiProcessorCount <<  "SMs, ";
    std::cout << device_properties_.clockRate * 1e-6 << " Ghz)" << std::endl;

#if defined(HAVE_PMT)
    pm_= std::move(pmt::nvml::NVML::Create());
#endif
}

Benchmark::~Benchmark() {
    if (data_) {
        cudaFree(data_);
    }
    cudaStreamSynchronize(stream_);
    cudaStreamDestroy(stream_);
    cudaEventDestroy(event_start_);
    cudaEventDestroy(event_end_);
}

void Benchmark::allocate(size_t bytes) {
    if (data_) {
        cudaFree(data_);
    }
    cudaMalloc(&data_, bytes);
    data_bytes_ = bytes;
}

void Benchmark::run(void *kernel, dim3 grid, dim3 block, const char *name, double gflops, double gbytes, double gops) {
    measurement measurement = run_kernel(kernel, grid, block);
    report(name, measurement, gflops, gbytes, gops);
}


measurement Benchmark::run_kernel(
    void *kernel,
    dim3 grid,
    dim3 block
    ) {
    // Warmup
    ((void (*)(void *)) kernel)<<<grid, block, 0, stream_>>>(data_);
    cudaMemset(data_, 1, data_bytes_);

    // Benchmark
    cudaEventRecord(event_start_, stream_);
#if defined(HAVE_PMT)
    pmt::State state_start = pm_->Read();
#endif
    for (int i = 0; i < NR_ITERATIONS; i++) {
        ((void (*)(void *)) kernel)<<<grid, block, 0, stream_>>>(data_);
    }
    cudaEventRecord(event_end_, stream_);
#if defined(HAVE_PMT)
    pmt::State state_end = pm_->Read();
#endif

    // Finish measurement
    cudaEventSynchronize(event_end_);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, event_start_, event_end_);
    measurement measurement;
    measurement.runtime = milliseconds / NR_ITERATIONS;
    measurement.power = 0;
#if defined(HAVE_PMT)
    measurement.power = pmt::PMT::watts(state_start, state_end);
#endif
    return measurement;
}

