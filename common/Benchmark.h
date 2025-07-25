#ifndef BENCHMARK_H
#define BENCHMARK_H

#include <memory>
#include <string>

#include <cudawrappers/cu.hpp>
#include <cudawrappers/nvrtc.hpp>

#if defined(HAVE_PMT)
#include <pmt.h>
#endif

#if defined(HAVE_FMT)
#include <fmt/fmt.h>
#endif

#include "Measurement.h"

class Benchmark {
public:
  Benchmark(int argc, const char *argv[]);

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

  void allocate(size_t bytes);
  void setArgs(std::vector<const void *> args);
  void run(std::shared_ptr<cu::Function> function, dim3 grid, dim3 block,
           const std::string &name, double gops = 0, double gbytes = 0);
  void report(const std::string &name, double gops, double gbytes,
              Measurement &m);

  std::shared_ptr<cu::Function> compileKernel(const std::string &kernel_source,
                                              const std::string &kernel_name);
  std::vector<std::shared_ptr<cu::Function>>
  compileKernels(const std::string &kernel_source,
                 const std::vector<std::string> &kernel_name);

  int multiProcessorCount();
  int clockRate();
  int maxThreadsPerBlock();
  size_t totalGlobalMem();

  unsigned nrBenchmarks() { return nr_benchmarks_; }
  unsigned nrIterations() { return nr_iterations_; }
#if defined(HAVE_PMT) || defined(HAVE_FMT)
  unsigned benchmarkDuration() { return benchmark_duration_; }
#endif
#if defined(HAVE_PMT) || defined(HAVE_FMT)
  bool measureContinuous() {
    bool result = false;
#if defined(HAVE_PMT)
    result |= measure_power_;
#endif
#if defined(HAVE_FMT)
    result |= measure_frequency_;
#endif
    return result;
  }
#endif

protected:
  double measure_power();
  double measure_frequency();
  float run_function(std::shared_ptr<cu::Function> function, dim3 grid,
                     dim3 block, int n = 1);
  Measurement measure_function(std::shared_ptr<cu::Function> function,
                               dim3 grid, dim3 block);

  std::vector<const void *> args_;
  unsigned nr_benchmarks_;
  unsigned nr_iterations_;
  std::unique_ptr<cu::Device> device_;
  std::unique_ptr<cu::Context> context_;
  std::unique_ptr<cu::Stream> stream_;
  std::unique_ptr<cu::DeviceMemory> d_data_;
  bool measure_power_ = false;
  bool measure_frequency_ = false;
#if defined(HAVE_PMT)
  std::shared_ptr<pmt::PMT> pm_;
#endif
#if defined(HAVE_FMT)
  std::shared_ptr<fmt::FMT> fm_;
#endif
#if defined(HAVE_PMT) || defined(HAVE_FMT)
  unsigned benchmark_duration_;
#endif

  std::unique_ptr<cu::Module> module_;
  std::unique_ptr<nvrtc::Program> program_;
};

#endif // end BENCHMARK_H
