#include <cassert>
#include <iostream>
#include <string>
#include <thread>

#include <cxxopts.hpp>

#include "Benchmark.h"
#include "Measurement.h"
#include "common.h"

namespace {

cxxopts::Options setupCommandLineParser(const char *argv[]) {
  cxxopts::Options options(argv[0]);

  const unsigned NR_BENCHMARKS = 1;
  const unsigned NR_ITERATIONS = 1;
#if defined(HAVE_PMT)
  const unsigned MEASURE_POWER = false;
#endif
#if defined(HAVE_FMT)
  const unsigned MEASURE_FREQUENCY = false;
#endif
#if defined(HAVE_PMT) || defined(HAVE_FMT)
  const unsigned BENCHMARK_DURATION = 4000; // ms
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
#endif
#if defined(HAVE_FMT)
      "measure_frequency", "Measure frequency",
      cxxopts::value<bool>()->default_value(std::to_string(MEASURE_FREQUENCY)))(
#endif
#if defined(HAVE_PMT) || defined(HAVE_FMT)
      "benchmark_duration", "Approximate number of ms to run the benchmark",
      cxxopts::value<unsigned>()->default_value(
          std::to_string(BENCHMARK_DURATION)))(
#endif
      "device_id", "Device ID",
      cxxopts::value<unsigned>()->default_value(std::to_string(DEVICE_ID)))(
      "h,help", "Print help");

  return options;
}

cxxopts::ParseResult getCommandLineOptions(int argc, const char *argv[]) {
  cxxopts::Options options = setupCommandLineParser(argv);

  try {
    cxxopts::ParseResult result = options.parse(argc, argv);

    if (result.count("help")) {
      std::cout << options.help() << std::endl;
      exit(EXIT_SUCCESS);
    }

    return result;

  } catch (const cxxopts::exceptions::exception &e) {
    std::cerr << "Error parsing command-line options: " << e.what()
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

} // end namespace

void Benchmark::report(std::string name, double gops, double gbytes,
                       Measurement &m) {
  const double milliseconds = m.runtime;
  const double seconds = milliseconds * 1e-3;
  std::cout << std::setw(w1) << std::string(name) << ": ";
  std::cout << std::setprecision(2) << std::fixed;
  std::cout << std::setw(w2) << milliseconds << " ms";
  m.gops = gops;
  m.gbytes = gbytes;
  std::cout << m;
  std::cout << std::endl;
}

Benchmark::Benchmark(int argc, const char *argv[]) {
  // Parse command-line options
  cxxopts::ParseResult results = getCommandLineOptions(argc, argv);
  const unsigned device_number = results["device_id"].as<unsigned>();
  nr_benchmarks_ = results["nr_benchmarks"].as<unsigned>();
  nr_iterations_ = results["nr_iterations"].as<unsigned>();
#if defined(HAVE_PMT) || defined(HAVE_FMT)
  benchmark_duration_ = results["benchmark_duration"].as<unsigned>();
#endif
#if defined(HAVE_PMT)
  measure_power_ = results["measure_power"].as<bool>();
#endif
#if defined(HAVE_FMT)
  measure_frequency_ = results["measure_frequency"].as<bool>();
#endif

  // Setup CUDA
  cu::init();
  device_ = std::make_unique<cu::Device>(device_number);
  context_ =
      std::make_unique<cu::Context>(CU_CTX_SCHED_BLOCKING_SYNC, *device_);
  stream_ = std::make_unique<cu::Stream>();

  // Print CUDA device information
  std::cout << "Device " << device_number << ": " << device_->getName();
  std::cout << " (" << multiProcessorCount();
#if defined(__HIP_PLATFORM_AMD__)
  if (isCDNA()) {
    std::cout << "CUs, ";
  } else if (isRDNA3()) {
    std::cout << "WGPs, ";
  } else {
    std::cout << "units, ";
  }
#else
  std::cout << "SMs, ";
#endif
  std::cout << clockRate() * 1e-6 << " Ghz)" << std::endl;

#if defined(HAVE_PMT)
#if defined(__HIP_PLATFORM_AMD__)
  pm_ = pmt::rocm::ROCM::Create(device_number);
#else
  pm_ = pmt::nvidia::NVIDIA::Create(device_number);
#endif
#endif

#if defined(HAVE_FMT)
#if defined(__HIP_PLATFORM_AMD__)
  fm_ = std::make_unique<fmt::amdsmi::AMDSMI>(device_number);
#else
  fm_ = std::make_unique<fmt::nvidia::NVIDIA>(*device_);
#endif
#endif
}

#if defined(__HIP_PLATFORM_AMD__)
// architecture checking code based on
// https://github.com/ROCm/rocWMMA/blob/develop/samples/common.hpp
bool Benchmark::isCDNA1() {
  const std::string arch(device_->getArch());
  return (arch.find("gfx908") != std::string::npos);
}

bool Benchmark::isCDNA2() {
  const std::string arch(device_->getArch());
  return (arch.find("gfx90a") != std::string::npos);
}

bool Benchmark::isCDNA3() {
  const std::string arch(device_->getArch());
  return ((arch.find("gfx940") != std::string::npos) ||
          (arch.find("gfx941") != std::string::npos) ||
          (arch.find("gfx942") != std::string::npos));
}

bool Benchmark::isCDNA() { return (isCDNA1() || isCDNA2() || isCDNA3()); }

bool Benchmark::isRDNA2() {
  const std::string arch(device_->getArch());
  return ((arch.find("gfx1030") != std::string::npos) ||
          (arch.find("gfx1031") != std::string::npos) ||
          (arch.find("gfx1032") != std::string::npos));
}

bool Benchmark::isRDNA3() {
  const std::string arch(device_->getArch());
  return ((arch.find("gfx1100") != std::string::npos) ||
          (arch.find("gfx1101") != std::string::npos) ||
          (arch.find("gfx1102") != std::string::npos));
}
#else
bool Benchmark::isVolta() {
  const std::string arch(device_->getArch());
  return (arch.find("sm_70") != std::string::npos);
}

bool Benchmark::isTuring() {
  const std::string arch(device_->getArch());
  return (arch.find("sm_75") != std::string::npos);
}

bool Benchmark::isAda() {
  const std::string arch(device_->getArch());
  return (arch.find("sm_89") != std::string::npos);
}

bool Benchmark::isHopper() {
  const std::string arch(device_->getArch());
  // This also matches sm_90a
  return (arch.find("sm_90") != std::string::npos);
}

bool Benchmark::isBlackwell() {
  const std::string arch(device_->getArch());
  // This also matches sm_100a, sm_101a, and sm_120a
  return ((arch.find("sm_100") != std::string::npos) ||
          (arch.find("sm_101") != std::string::npos) ||
          (arch.find("sm_120") != std::string::npos));
}

#endif

void Benchmark::allocate(size_t bytes) {
  cu::HostMemory h_data(bytes);
  d_data_ = std::make_unique<cu::DeviceMemory>(bytes);
  std::memset(h_data, 1, bytes);
  stream_->memcpyHtoDAsync(*d_data_, h_data, bytes);
  stream_->synchronize();
  args_.resize(1);
  args_[0] = reinterpret_cast<const void *>(static_cast<CUdeviceptr>(*d_data_));
}

void Benchmark::run(void *kernel, dim3 grid, dim3 block, const char *name,
                    double gops, double gbytes) {
  Measurement m = measure_kernel(kernel, grid, block);
  report(name, gops, gbytes, m);
}

int Benchmark::multiProcessorCount() {
  return device_->getAttribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
}

int Benchmark::clockRate() {
  return device_->getAttribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
}

int Benchmark::maxThreadsPerBlock() {
  return device_->getAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
  device_->getAttribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

size_t Benchmark::totalGlobalMem() { return context_->getTotalMemory(); }

float Benchmark::run_kernel(void *kernel, dim3 grid, dim3 block, int n) {
  cu::Event start;
  cu::Event end;
  stream_->record(start);
  for (int i = 0; i < n; i++) {
    launch_kernel(kernel, grid, block, *stream_, args_);
  }
  stream_->record(end);
  end.synchronize();
  return end.elapsedTime(start);
}

double Benchmark::measure_power() {
#if defined(HAVE_PMT)
  pmt::State state_start;

  if (measure_power_) {
    state_start = pm_->Read();
  }
  std::this_thread::sleep_for(
      std::chrono::milliseconds(int(0.2 * benchmarkDuration())));
  if (measure_power_) {
    pmt::State state_end = pm_->Read();
    return pmt::PMT::watts(state_start, state_end);
  }
#endif
  return 0;
}

double Benchmark::measure_frequency() {
#if defined(HAVE_FMT)
  if (measure_frequency_) {
    auto names = fm_->names();
    auto frequency = fm_->get();
#if defined(__HIP_PLATFORM_AMD__)
    assert(names[0].compare("sclk") == 0);
    return frequency[0];
#else
    assert(names[1].compare("sm") == 0);
    return frequency[1];
#endif
  }
#endif
  return 0;
}

Measurement Benchmark::measure_kernel(void *kernel, dim3 grid, dim3 block) {
#if defined(HAVE_PMT) || defined(HAVE_FMT)
  if (measureContinuous()) {
    Measurement m;
    float milliseconds = 0;
    unsigned nr_iterations = 0;
    m.frequency = 0;

    std::thread thread([&] {
      context_->setCurrent();
      milliseconds = run_kernel(kernel, grid, block);
      nr_iterations = benchmarkDuration() / milliseconds;
      milliseconds = run_kernel(kernel, grid, block, nr_iterations);
    });
    std::this_thread::sleep_for(
        std::chrono::milliseconds(int(0.5 * benchmarkDuration())));
    if (measure_power_) {
      m.power = measure_power();
    }
    if (measure_frequency_) {
      m.frequency = measure_frequency();
    }
    if (thread.joinable()) {
      thread.join();
    }

    m.runtime = milliseconds / nr_iterations;

    return m;
  }
#endif

  // Benchmark (timing only)
  const float milliseconds = run_kernel(kernel, grid, block, nrIterations());
  Measurement m;
  m.runtime = milliseconds / nrIterations();
  m.power = 0;
  m.frequency = 0;
  return m;
}
