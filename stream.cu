#include <cstring>
#include <type_traits>

#include "common.h"

#define STREAM_arguments_float                                         \
  float *__restrict__ a, float *__restrict__ b, float *__restrict__ c, \
      float scalar, int len

#define STREAM_arguments_double                                           \
  double *__restrict__ a, double *__restrict__ b, double *__restrict__ c, \
      double scalar, int len

__global__ void STREAM_Triad_float(STREAM_arguments_float);

__global__ void STREAM_Add_float(STREAM_arguments_float);

__global__ void STREAM_Scale_float(STREAM_arguments_float);

__global__ void STREAM_Copy_float(STREAM_arguments_float);

__global__ void STREAM_Triad_double(STREAM_arguments_double);

__global__ void STREAM_Add_double(STREAM_arguments_double);

__global__ void STREAM_Scale_double(STREAM_arguments_double);

__global__ void STREAM_Copy_double(STREAM_arguments_double);

template <typename T>
class BenchmarkStream : public Benchmark {
 public:
  BenchmarkStream(int argc, const char* argv[]) : Benchmark(argc, argv) {
    const size_t N = totalGlobalMem() / sizeof(T) / 4;
    N_ = roundToPowOf2(N);
    allocate(N_ * sizeof(T));
    args_.resize(5);
    args_[0] = reinterpret_cast<const void*>(static_cast<CUdeviceptr>(*d_a_));
    args_[1] = reinterpret_cast<const void*>(static_cast<CUdeviceptr>(*d_b_));
    args_[2] = reinterpret_cast<const void*>(static_cast<CUdeviceptr>(*d_c_));
    args_[3] = &scale_;
    args_[4] = &N_;
  }

  void benchmark() {
    const double gops = 0;
    const double gbytes[4] = {
        2 * sizeof(T) * (double)N_ / 1.e9, 2 * sizeof(T) * (double)N_ / 1.e9,
        3 * sizeof(T) * (double)N_ / 1.e9, 3 * sizeof(T) * (double)N_ / 1.e9};

    dim3 block(maxThreadsPerBlock());
    dim3 grid(N_ / block.x);

    void* copy_kernel;
    void* scale_kernel;
    void* add_kernel;
    void* triad_kernel;

    std::string names[4] = {"copy", "scale", "add", "triad"};

    if constexpr (std::is_same_v<T, double>) {
      copy_kernel = reinterpret_cast<void*>(&STREAM_Copy_double);
      scale_kernel = reinterpret_cast<void*>(&STREAM_Scale_double);
      add_kernel = reinterpret_cast<void*>(&STREAM_Add_double);
      triad_kernel = reinterpret_cast<void*>(&STREAM_Triad_double);
      for (std::string& name : names) {
        name = name + "_double";
      }
    } else if constexpr (std::is_same_v<T, float>) {
      copy_kernel = reinterpret_cast<void*>(&STREAM_Copy_float);
      scale_kernel = reinterpret_cast<void*>(&STREAM_Scale_float);
      add_kernel = reinterpret_cast<void*>(&STREAM_Add_float);
      triad_kernel = reinterpret_cast<void*>(&STREAM_Triad_float);
      for (std::string& name : names) {
        name = name + "_float";
      }
    } else {
      throw std::runtime_error("Unsupported data type");
    }

    for (int i = 0; i < nrBenchmarks(); i++) {
      run(copy_kernel, grid, block, names[0].c_str(), gops, gbytes[0]);
      run(scale_kernel, grid, block, names[1].c_str(), gops, gbytes[1]);
      run(add_kernel, grid, block, names[2].c_str(), gops, gbytes[2]);
      run(triad_kernel, grid, block, names[3].c_str(), gops, gbytes[3]);
    }
  }

 private:
  virtual void launch_kernel(void* kernel, dim3 grid, dim3 block,
                             cu::Stream& stream,
                             const std::vector<const void*>& args) override;

  void allocate(size_t bytes) {
    cu::HostMemory h_data(bytes);
    std::memset(h_data, 1, bytes);
    d_a_ = std::make_unique<cu::DeviceMemory>(bytes);
    d_b_ = std::make_unique<cu::DeviceMemory>(bytes);
    d_c_ = std::make_unique<cu::DeviceMemory>(bytes);
    stream_->memcpyHtoDAsync(*d_a_, h_data, bytes);
    stream_->memcpyHtoDAsync(*d_b_, h_data, bytes);
    stream_->memcpyHtoDAsync(*d_c_, h_data, bytes);
    stream_->synchronize();
  }

  const T scale_ = 2.0;
  size_t N_;

  std::unique_ptr<cu::DeviceMemory> d_a_;
  std::unique_ptr<cu::DeviceMemory> d_b_;
  std::unique_ptr<cu::DeviceMemory> d_c_;
};

int main(int argc, const char* argv[]) {
  BenchmarkStream<float> benchmark(argc, argv);
  benchmark.benchmark();
  return EXIT_SUCCESS;
}
