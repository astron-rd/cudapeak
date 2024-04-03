#include <thread>
#include <cxxopts.hpp>

#include "common.h"

inline void __checkCudaCall(cudaError_t err, const char* file, int line) {
  if (err != cudaSuccess) {
    std::cerr << "CUDA error in " << file << " at line " << line << ": "
              << cudaGetErrorString(err) << std::endl;
    exit(EXIT_FAILURE);
  }
}

#define checkCudaCall(err) __checkCudaCall(err, __FILE__, __LINE__)

void report(string name, measurement measurement, double gflops, double gbytes,
            double gops) {
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

  return (unsigned)pow(2, (int)logd);
}

cxxopts::Options setupCommandLineParser(const char* argv[]) {
  cxxopts::Options options(argv[0], "Benchmark for BeamFormerKernel");

  const unsigned NR_BENCHMARKS = 1;
  const unsigned NR_ITERATIONS = 1;
#if defined(HAVE_PMT)
  const unsigned MEASURE_POWER = false;
  const unsigned BENCHMARK_DURATION = 4000;  // ms
#endif
  const unsigned DEVICE_ID = 0;

  options.add_options()(
      "nr_benchmarks", "Number of benchmarks",
      cxxopts::value<unsigned>()->default_value(std::to_string(NR_BENCHMARKS)))(
      "nr_iterations", "Number of kernel iteration per benchmark",
      cxxopts::value<unsigned>()->default_value(std::to_string(NR_ITERATIONS)))(
#if defined(HAVE_PMT)
      "measure_power", "Measure power",
      cxxopts::value<bool>()->default_value(std::to_string(MEASURE_POWER)))(
      "benchmark_duration", "Approximate number of ms to run the benchmark",
      cxxopts::value<unsigned>()->default_value(
          std::to_string(BENCHMARK_DURATION)))(
#endif
      "device_id", "Device ID",
      cxxopts::value<unsigned>()->default_value(std::to_string(DEVICE_ID)))(
      "h,help", "Print help");

  return options;
}

cxxopts::ParseResult getCommandLineOptions(int argc, const char* argv[]) {
  cxxopts::Options options = setupCommandLineParser(argv);

  try {
    cxxopts::ParseResult result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      exit(EXIT_SUCCESS);
    }

    return result;

  } catch (const cxxopts::exceptions::exception& e) {
    std::cerr << "Error parsing command-line options: " << e.what()
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

Benchmark::Benchmark(int argc, const char* argv[]) {
  // Parse command-line options
  cxxopts::ParseResult results = getCommandLineOptions(argc, argv);
  const unsigned device_number = results["device_id"].as<unsigned>();
  nr_benchmarks_ = results["nr_benchmarks"].as<unsigned>();
  nr_iterations_ = results["nr_iterations"].as<unsigned>();
#if defined(HAVE_PMT)
  measure_power_ = results["measure_power"].as<bool>();
  benchmark_duration_ = results["benchmark_duration"].as<unsigned>();
#endif

  // Setup CUDA
  checkCudaCall(cudaSetDevice(device_number));
  checkCudaCall(cudaStreamCreate(&stream_));
  checkCudaCall(cudaGetDeviceProperties(&device_properties_, device_number));
  checkCudaCall(cudaEventCreate(&event_start_));
  checkCudaCall(cudaEventCreate(&event_end_));
  checkCudaCall(cudaDeviceSynchronize());

  // Print CUDA device information
  std::cout << "Device " << device_number << ": " << device_properties_.name;
  std::cout << " (" << device_properties_.multiProcessorCount << "SMs, ";
  std::cout << device_properties_.clockRate * 1e-6 << " Ghz)" << std::endl;

#if defined(HAVE_PMT)
  pm_ = std::move(pmt::Create("nvidia"));
#endif
}

Benchmark::~Benchmark() {
  if (data_) {
    checkCudaCall(cudaFree(data_));
  }
  checkCudaCall(cudaStreamSynchronize(stream_));
  checkCudaCall(cudaStreamDestroy(stream_));
  checkCudaCall(cudaEventDestroy(event_start_));
  checkCudaCall(cudaEventDestroy(event_end_));
}

void Benchmark::allocate(size_t bytes) {
  if (data_) {
    checkCudaCall(cudaFree(data_));
  }
  checkCudaCall(cudaMalloc(&data_, bytes));
  checkCudaCall(cudaMemsetAsync(data_, 1, bytes, stream_));
  data_bytes_ = bytes;
}

void Benchmark::run(void* kernel, dim3 grid, dim3 block, const char* name,
                    double gflops, double gbytes, double gops) {
  measurement measurement = run_kernel(kernel, grid, block);
  report(name, measurement, gflops, gbytes, gops);
}

measurement Benchmark::run_kernel(void* kernel, dim3 grid, dim3 block) {
// Benchmark with power measurement
#if defined(HAVE_PMT)
  if (measurePower()) {
    float milliseconds = 0;
    unsigned nr_iterations = 0;

    std::thread thread([&] {
      checkCudaCall(cudaEventRecord(event_start_, stream_));
      ((void (*)(void*))kernel)<<<grid, block, 0, stream_>>>(data_);
      checkCudaCall(cudaEventRecord(event_end_, stream_));
      checkCudaCall(cudaEventSynchronize(event_end_));
      checkCudaCall(
          cudaEventElapsedTime(&milliseconds, event_start_, event_end_));
      nr_iterations = benchmarkDuration() / milliseconds;
      checkCudaCall(cudaEventRecord(event_start_, stream_));
      for (int i = 0; i < nr_iterations; i++) {
        ((void (*)(void*))kernel)<<<grid, block, 0, stream_>>>(data_);
      }
      checkCudaCall(cudaEventRecord(event_end_, stream_));
      checkCudaCall(cudaEventSynchronize(event_end_));
      checkCudaCall(
          cudaEventElapsedTime(&milliseconds, event_start_, event_end_));
    });
    std::this_thread::sleep_for(
        std::chrono::milliseconds(int(0.5 * benchmarkDuration())));
    pmt::State state_start = pm_->Read();
    std::this_thread::sleep_for(
        std::chrono::milliseconds(int(0.2 * benchmarkDuration())));
    pmt::State state_end = pm_->Read();
    if (thread.joinable()) {
      thread.join();
    }

    measurement measurement;
    measurement.runtime = milliseconds / nr_iterations;
    measurement.power = pmt::PMT::watts(state_start, state_end);

    return measurement;
  }
#endif

  // Benchmark (timing only)
  checkCudaCall(cudaEventRecord(event_start_, stream_));
  for (int i = 0; i < nrIterations(); i++) {
    ((void (*)(void*))kernel)<<<grid, block, 0, stream_>>>(data_);
  }
  checkCudaCall(cudaEventRecord(event_end_, stream_));
  cudaEventSynchronize(event_end_);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, event_start_, event_end_);
  measurement measurement;
  measurement.runtime = milliseconds / nrIterations();
  measurement.power = 0;
  return measurement;
}
