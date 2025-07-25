#include "common/common.h"

#include "kernels/fp16.cu.o.h"

int main(int argc, const char *argv[]) {
  Benchmark benchmark(argc, argv);

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
  auto kernels =
      benchmark.compileKernels(fp16_source, {"fp16_kernel", "fp16x2_kernel"});

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(kernels[0], grid, block, "fp16", gops, gbytes);
#if !defined(__HIP_PLATFORM_AMD__)
    benchmark.run(kernels[1], grid, block, "fp16x2", gops, gbytes);
#endif
  }

  return EXIT_SUCCESS;
}
