#if defined(HAVE_PMT)
#include <pmt.h>
#endif

#if defined(HAVE_FMT)
#include <fmt/fmt.h>
#endif

#include "KernelRunner.h"
#include "Measurement.h"

class KernelRunner::Impl {
public:
  Impl(cu::Device &device, cu::Context &context)
      : device_(device), context_(context) {
#if defined(HAVE_PMT)
#if defined(__HIP_PLATFORM_AMD__)
    pm_ = pmt::rocm::ROCM::Create(0);
#else
    pm_ = pmt::nvml::NVML::Create(device_);
#endif
#endif

#if defined(HAVE_FMT)
#if defined(__HIP_PLATFORM_AMD__)
    fm_ = std::make_unique<fmt::amdsmi::AMDSMI>(0);
#else
    fm_ = std::make_unique<fmt::nvidia::NVIDIA>(device_);
#endif
#endif
  }

  void enable_power_measurement(unsigned int benchmark_duration) {
    measure_power_ = true;
    benchmark_duration_ = benchmark_duration;
  }

  void enable_frequency_measurement(unsigned int benchmark_duration) {
    measure_frequency_ = true;
    benchmark_duration_ = benchmark_duration;
  }

  bool measure_continuous() {
    bool result = false;
#if defined(HAVE_PMT)
    result |= measure_power_;
#endif
#if defined(HAVE_FMT)
    result |= measure_frequency_;
#endif
    return result;
  }

  Measurement run(cu::Stream &stream, cu::Function function,
                  unsigned int nr_iterations, dim3 grid, dim3 block,
                  std::vector<const void *> &args);

private:
  cu::Device &device_;
  cu::Context &context_;

  bool measure_power_ = false;
  bool measure_frequency_ = false;
  unsigned int benchmark_duration_;

#if defined(HAVE_PMT)
  std::shared_ptr<pmt::PMT> pm_;
  float measure_power();
#endif

#if defined(HAVE_FMT)
  std::shared_ptr<fmt::FMT> fm_;
  float measure_frequency();
#endif
};

KernelRunner::KernelRunner(cu::Device &device, cu::Context &context)
    : impl_(std::make_unique<Impl>(device, context)) {}

void KernelRunner::enable_power_measurement(unsigned int benchmark_duration) {
#if defined(HAVE_PMT)
  impl_->enable_power_measurement(benchmark_duration);
#endif
}

KernelRunner::~KernelRunner() = default;

void KernelRunner::enable_frequency_measurement(
    unsigned int benchmark_duration) {
#if defined(HAVE_FMT)
  impl_->enable_frequency_measurement(benchmark_duration);
#endif
}

Measurement KernelRunner::run(cu::Stream &stream, cu::Function &function,
                              unsigned int nr_iterations, dim3 grid, dim3 block,
                              std::vector<const void *> &args) {
  return impl_->run(stream, function, nr_iterations, grid, block, args);
}

float run_function(cu::Stream &stream, cu::Function &function, dim3 grid,
                   dim3 block, int nr_iterations,
                   std::vector<const void *> &args) {
  cu::Event start;
  cu::Event end;
  stream.record(start);
  for (unsigned int i = 0; i < nr_iterations; i++) {
    stream.launchKernel(function, grid.x, grid.y, grid.z, block.x, block.y,
                        block.z, 0, args);
  }
  stream.record(end);
  end.synchronize();
  return end.elapsedTime(start);
}

float KernelRunner::Impl::measure_power() {
#if defined(HAVE_PMT)
  pmt::State state_start;

  if (measure_power_) {
    state_start = pm_->Read();
  }
  std::this_thread::sleep_for(
      std::chrono::milliseconds(int(0.2 * benchmark_duration_)));
  if (measure_power_) {
    pmt::State state_end = pm_->Read();
    return pmt::PMT::watts(state_start, state_end);
  }
#endif
  return 0;
}

float KernelRunner::Impl::measure_frequency() {
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

Measurement KernelRunner::Impl::run(cu::Stream &stream, cu::Function function,
                                    unsigned int nr_iterations, dim3 grid,
                                    dim3 block,
                                    std::vector<const void *> &args) {
#if defined(HAVE_PMT) || defined(HAVE_FMT)
  if (measure_continuous()) {
    Measurement m;
    float milliseconds = 0;
    unsigned nr_iterations = 0;

    std::thread thread([&] {
      context_.setCurrent();
      milliseconds = run_function(stream, function, grid, block, 1, args);
      nr_iterations = benchmark_duration_ / milliseconds;
      milliseconds =
          run_function(stream, function, grid, block, nr_iterations, args);
      m.runtime = milliseconds / nr_iterations;
    });
    std::this_thread::sleep_for(
        std::chrono::milliseconds(int(0.5 * benchmark_duration_)));
    m.power = measure_power_ ? measure_power() : 0;
    m.frequency = measure_frequency_ ? measure_frequency() : 0;
    if (thread.joinable()) {
      thread.join();
    }

    return m;
  }
#endif

  // Benchmark (timing only)
  const float milliseconds =
      run_function(stream, function, grid, block, nr_iterations, args);
  Measurement m;
  m.runtime = milliseconds / nr_iterations;
  m.power = 0;
  m.frequency = 0;
  return m;
}
