#include "common/common.h"

#include "kernels/fp32_sincos_fpu.cu.o.h"

std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

std::vector<std::string>
transform(const std::vector<std::string> &kernel_names) {
  std::vector<std::string> result;

  for (const auto &name : kernel_names) {
    auto parts = split(name, '_');
    if (parts.size() < 5) {
      continue;
    }

    std::ostringstream oss;
    oss << "fma:" << parts[1] << " (" << parts[2] << ") -> " << std::setw(4)
        << parts[3] << ":" << parts[4];

    result.push_back(oss.str());
  }

  return result;
}

int main(int argc, const char *argv[]) {
  Benchmark benchmark(argc, argv);

  KernelFactory kernel_factory(fp32_sincos_fpu_source);

  const std::vector<std::string> kernel_names{
      "fp32_sincos_fpu_1_8",  "fp32_sincos_fpu_1_4",  "fp32_sincos_fpu_1_2",
      "fp32_sincos_fpu_1_1",  "fp32_sincos_fpu_2_1",  "fp32_sincos_fpu_4_1",
      "fp32_sincos_fpu_8_1",  "fp32_sincos_fpu_16_1", "fp32_sincos_fpu_32_1",
      "fp32_sincos_fpu_64_1", "fp32_sincos_fpu_128_1"};

  auto kernels =
      kernel_factory.compileKernels(benchmark.getDevice(), kernel_names);
  const std::vector<std::string> names = transform(kernel_names);

  // Parameters
  int multiProcessorCount = benchmark.multiProcessorCount();
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  // Amount of work performed
  const int nr_outer = 4096;
  const int nr_inner = 1024;
  const double gops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) *
                      (1ULL * nr_outer * nr_inner * 2);
  const double gbytes = 0;

  // Kernel dimensions
  dim3 grid(multiProcessorCount);
  dim3 block(maxThreadsPerBlock);

  // Allocate memory
  benchmark.allocate(multiProcessorCount * maxThreadsPerBlock * sizeof(float));

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    for (int j = 0; j < kernels.size(); j++) {
      benchmark.run(kernels[j], grid, block, names[j], gops, gbytes);
    }
  }

  return EXIT_SUCCESS;
}
