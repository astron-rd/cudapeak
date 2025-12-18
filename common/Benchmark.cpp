#include <cassert>
#include <iostream>
#include <string>
#include <thread>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>
#include <cxxopts.hpp>
#include <nlohmann/json.hpp>

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
  const bool JSON_OUTPUT = false;

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
      "j,json", "Print output in JSON format",
      cxxopts::value<bool>()->default_value(std::to_string(JSON_OUTPUT)))(
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

void Benchmark::report(const std::string &name, double gops, double gbytes,
                       Measurement &m) {
  const double milliseconds = m.runtime;
  const double seconds = milliseconds * 1e-3;
  m.gops = gops;
  m.gbytes = gbytes;
  if (enable_json_output_) {
    nlohmann::json obj;
    obj["name"] = name;
    obj.update(m.toJson());
    std::cout << obj.dump() << "," << std::endl;
  } else {
    std::cout << std::setw(w1) << std::string(name) << ": ";
    std::cout << std::setprecision(2) << std::fixed;
    std::cout << m;
    std::cout << std::endl;
  }
}

Benchmark::Benchmark(int argc, const char *argv[]) {
  // Parse command-line options
  cxxopts::ParseResult results = getCommandLineOptions(argc, argv);
  const unsigned device_number = results["device_id"].as<unsigned>();
  nr_benchmarks_ = results["nr_benchmarks"].as<unsigned>();
  nr_iterations_ = results["nr_iterations"].as<unsigned>();
  enable_json_output_ = results["json"].as<bool>();

  // Setup CUDA
  cu::init();
  device_ = std::make_unique<cu::Device>(device_number);
  context_ =
      std::make_unique<cu::Context>(CU_CTX_SCHED_BLOCKING_SYNC, *device_);
  stream_ = std::make_unique<cu::Stream>();

  if (enable_json_output_) {
    nlohmann::json dev;
    dev["device_number"] = device_number;
    dev["device_name"] = device_->getName();
    dev["architecture"] = device_->getArch();
    dev["multi_processor_count"] = multiProcessorCount();
    dev["clock_rate"] = clockRate() * 1e-6;
    std::cout << "[\n" << dev.dump() << "," << std::endl;
  } else {
    std::cout << "Device " << device_number << ": " << device_->getName();
    std::cout << " (" << device_->getArch() << ", " << multiProcessorCount()
              << " ";
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
  }

  kernel_runner_ = std::make_unique<KernelRunner>(*device_, *context_);
#if defined(HAVE_PMT)
  if (results["measure_power"].as<bool>()) {
    const unsigned benchmark_duration =
        results["benchmark_duration"].as<unsigned>();
    kernel_runner_->enable_power_measurement(benchmark_duration);
  }
#endif
#if defined(HAVE_FMT)
  if (results["measure_frequency"].as<bool>()) {
    const unsigned benchmark_duration =
        results["benchmark_duration"].as<unsigned>();
    kernel_runner_->enable_frequency_measurement(benchmark_duration);
  }
#endif
}

Benchmark::~Benchmark() {
  if (enable_json_output_) {
    std::cout << "]" << std::endl;
  }
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
  args_[0] = d_data_->parameter();
}

void Benchmark::setArgs(std::vector<const void *> args) { args_ = args; }

void Benchmark::run(std::shared_ptr<cu::Function> function, dim3 grid,
                    dim3 block, const std::string &name, double gops,
                    double gbytes) {
  Measurement m = kernel_runner_->run(*stream_, *function, nr_iterations_, grid,
                                      block, args_);
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
