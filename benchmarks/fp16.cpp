#include "kernels/fp16.h"
#include "common/common.h"

__global__ void fp16_kernel(half *ptr);
__global__ void fp16x2_kernel(half *ptr);

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

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(reinterpret_cast<void *>(&fp16_kernel), grid, block, "fp16",
                  gops, gbytes);
#if !defined(__HIP_PLATFORM_AMD__)
    benchmark.run(reinterpret_cast<void *>(&fp16x2_kernel), grid, block,
                  "fp16x2", gops, gbytes);
#endif
  }

  return EXIT_SUCCESS;
}
