#include "common/common.h"

#include "kernels/int24.cu.o.h"

int main(int argc, const char *argv[]) {
  Benchmark benchmark(argc, argv);
  KernelFactory kernel_factory(int24_source);
  auto kernel =
      kernel_factory.compileKernel(benchmark.getDevice(), "int24_kernel");

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

  // Allocate memory
  benchmark.allocate(multiProcessorCount * maxThreadsPerBlock * sizeof(int));

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(kernel, grid, block, "int24", gops, gbytes);
  }

  return EXIT_SUCCESS;
}
