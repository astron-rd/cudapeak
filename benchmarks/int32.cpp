#include "common/common.h"

#include "kernels/int32.cu.o.h"

__global__ void int32_kernel(int *ptr);

int main(int argc, const char *argv[]) {
  Benchmark benchmark(argc, argv);
  KernelFactory kernel_factory(int32_source);
  auto kernel =
      kernel_factory.compileKernel(benchmark.getDevice(), "int32_kernel");

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
    benchmark.run(kernel, grid, block, "int32", gops, gbytes);
  }

  return EXIT_SUCCESS;
}
