#include <memory>

#include <cudawrappers/cu.hpp>

#include "Measurement.h"

class KernelRunner {

public:
  KernelRunner(cu::Device &device, cu::Context &context);
  ~KernelRunner();
  void enable_power_measurement(unsigned int benchmark_duration);
  void enable_frequency_measurement(unsigned int benchmark_duration);
  Measurement run(cu::Stream &stream, cu::Function &function,
                  unsigned int nr_iterations, dim3 grid, dim3 block,
                  std::vector<const void *> &args);

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};