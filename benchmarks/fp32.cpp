#include "common/common.h"

#include "kernels/fp32.cu.o.h"

int main(int argc, const char *argv[]) {
  Benchmark benchmark(argc, argv);
  KernelFactory kernel_factory(fp32_source);
  auto kernel =
      kernel_factory.compileKernel(benchmark.getDevice(), "fp32_kernel");

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
    benchmark.run(kernel, grid, block, "fp32", gops, gbytes);
  }

  return EXIT_SUCCESS;
}
