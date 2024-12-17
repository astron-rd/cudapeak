#include "common.h"

__global__ void dmem_kernel(float* ptr);

int main(int argc, const char* argv[]) {
  Benchmark benchmark(argc, argv);

  // Parameters
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  // Amount of work performed
  unsigned fetchPerBlock = 16;
  const size_t maxItems = benchmark.totalGlobalMem() / sizeof(float) / 2;
  const size_t numItems = roundToPowOf2(maxItems);
  const double gops = 0;
  const double gbytes =
      (double)(numItems / fetchPerBlock) * sizeof(float) / 1e9;

  // Kernel dimensions
  dim3 grid(numItems / (fetchPerBlock * maxThreadsPerBlock));
  dim3 block(maxThreadsPerBlock);

  // Allocate memory
  benchmark.allocate(numItems * sizeof(float));

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(reinterpret_cast<void*>(&dmem_kernel), grid, block, "dmem",
                  gops, gbytes);
  }

  return EXIT_SUCCESS;
}
