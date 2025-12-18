#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <memory>
#include <string>

#include <cudawrappers/cu.hpp>
#include <nlohmann/json.hpp>

#include "KernelRunner.h"
#include "Measurement.h"

class Benchmark {
public:
  Benchmark(int argc, const char *argv[]);
  ~Benchmark();

#if defined(__HIP_PLATFORM_AMD__)
  bool isCDNA();
  bool isCDNA1();
  bool isCDNA2();
  bool isCDNA3();
  bool isRDNA2();
  bool isRDNA3();
#else
  bool isVolta();
  bool isTuring();
  bool isAda();
  bool isHopper();
  bool isBlackwell();
#endif

  cu::Device &getDevice() { return *device_; }

  void allocate(size_t bytes);
  void setArgs(std::vector<const void *> args);
  void run(std::shared_ptr<cu::Function> function, dim3 grid, dim3 block,
           const std::string &name, double gops = 0, double gbytes = 0);
  void report(const std::string &name, double gops, double gbytes,
              Measurement &m);

  int multiProcessorCount();
  int clockRate();
  int maxThreadsPerBlock();
  size_t totalGlobalMem();

  unsigned nrBenchmarks() { return nr_benchmarks_; }
  unsigned nrIterations() { return nr_iterations_; }

protected:
  std::vector<const void *> args_;
  unsigned nr_benchmarks_;
  unsigned nr_iterations_;
  bool enable_json_output_;
  std::unique_ptr<cu::Device> device_;
  std::unique_ptr<cu::Context> context_;
  std::unique_ptr<cu::Stream> stream_;
  std::unique_ptr<cu::DeviceMemory> d_data_;
  std::unique_ptr<KernelRunner> kernel_runner_;

  nlohmann::json json_output_;
};

#endif // end BENCHMARK_H
