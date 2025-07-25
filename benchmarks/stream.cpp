#include <cstring>
#include <type_traits>

#include "common/common.h"

#include "kernels/stream.cu.o.h"

template <typename T> class Stream {
public:
  Stream(int argc, const char *argv[]) : benchmark_(argc, argv) {}

  void run() {

    const size_t N = benchmark_.totalGlobalMem() / sizeof(T) / 4;
    N_ = roundToPowOf2(N);
    allocate(N_ * sizeof(T));

    std::vector<const void *> args_{d_a_->parameter(), d_b_->parameter(),
                                    d_c_->parameter(), &scale_, &N_};

    const double gops = 0;
    const double gbytes[4] = {
        2 * sizeof(T) * (double)N_ / 1.e9, 2 * sizeof(T) * (double)N_ / 1.e9,
        3 * sizeof(T) * (double)N_ / 1.e9, 3 * sizeof(T) * (double)N_ / 1.e9};

    dim3 block(benchmark_.maxThreadsPerBlock());
    dim3 grid(N_ / block.x);

    std::string names[4] = {"copy", "scale", "add", "triad"};
    std::string kernel_names[4] = {"STREAM_Copy", "STREAM_Scale", "STREAM_Add",
                                   "STREAM_Triad"};

    std::string suffix;
    if constexpr (std::is_same_v<T, double>) {
      suffix = "_double";
    } else if constexpr (std::is_same_v<T, float>) {
      suffix = "_float";
    } else {
      throw std::runtime_error("Unsupported data type");
    }

    for (int i = 0; i < 4; i++) {
      names[i] = names[i] + suffix;
      kernel_names[i] = kernel_names[i] + suffix;
    }

    KernelFactory kernel_factory(stream_source);
    auto kernels = kernel_factory.compileKernels(
        benchmark_.getDevice(),
        std::vector<std::string>(kernel_names, std::end(kernel_names)));

    benchmark_.setArgs(args_);

    for (int i = 0; i < benchmark_.nrBenchmarks(); i++) {
      for (int j = 0; j < 4; j++) {
        benchmark_.run(kernels[j], grid, block, names[j], gops, gbytes[j]);
      }
    }
  }

private:
  void allocate(size_t bytes) {
    cu::HostMemory h_data(bytes);
    std::memset(h_data, 1, bytes);
    d_a_ = std::make_unique<cu::DeviceMemory>(bytes);
    d_b_ = std::make_unique<cu::DeviceMemory>(bytes);
    d_c_ = std::make_unique<cu::DeviceMemory>(bytes);
    cu::Stream stream;
    stream.memcpyHtoDAsync(*d_a_, h_data, bytes);
    stream.memcpyHtoDAsync(*d_b_, h_data, bytes);
    stream.memcpyHtoDAsync(*d_c_, h_data, bytes);
    stream.synchronize();
  }

  Benchmark benchmark_;

  const T scale_ = 2.0;
  size_t N_;

  std::unique_ptr<cu::DeviceMemory> d_a_;
  std::unique_ptr<cu::DeviceMemory> d_b_;
  std::unique_ptr<cu::DeviceMemory> d_c_;
};

int main(int argc, const char *argv[]) {
  Stream<float> stream(argc, argv);
  stream.run();
  return EXIT_SUCCESS;
}
