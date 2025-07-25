#include "common/common.h"

#include "kernels/fp16.cu.o.h"

int main(int argc, const char *argv[]) {
  Benchmark benchmark(argc, argv);
  KernelFactory kernel_factory(fp16_source);
  std::vector<std::string> kernel_names = {"fp16_kernel"};
#if !defined(__HIP_PLATFORM_AMD__)
  kernel_names.push_back("fp16x2_kernel");
#endif
  auto kernels =
      kernel_factory.compileKernels(benchmark.getDevice(), kernel_names);

  // Parameters
  int multiProcessorCount = benchmark.multiProcessorCount();
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  // Amount of work performed
  const int nr_outer = 4096;
  const int nr_inner = 1024;
  const double gops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) *
                      (1ULL * nr_outer * nr_inner * 8);
  const double gbytes = 0;

  // Kernel dimensions
  dim3 grid(multiProcessorCount);
  dim3 block(maxThreadsPerBlock);

  benchmark.allocate(multiProcessorCount * maxThreadsPerBlock * sizeof(float));

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(kernels[0], grid, block, "fp16", gops, gbytes);
#if !defined(__HIP_PLATFORM_AMD__)
    benchmark.run(kernels[1], grid, block, "fp16x2", gops, gbytes);
#endif
  }

  return EXIT_SUCCESS;
}
