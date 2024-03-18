#include "common.h"

__global__ void dmem_kernel(float* ptr);

int main(int argc, const char* argv[]) {
  Benchmark benchmark(argc, argv);

  // Parameters
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  // Amount of work performed
  unsigned fetchPerBlock = 16;
  int maxItems = benchmark.totalGlobalMem() / sizeof(float) / 2;
  int numItems = roundToPowOf2(maxItems);
  double gbytes = (float)(numItems / fetchPerBlock) * sizeof(float) / 1e9;
  double gflops = 0;

  // Kernel dimensions
  dim3 grid(numItems / (fetchPerBlock * maxThreadsPerBlock));
  dim3 block(maxThreadsPerBlock);

  // Allocate memory
  benchmark.allocate(numItems * sizeof(float));

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(reinterpret_cast<void*>(&dmem_kernel), grid, block, "dmem",
                  gflops, gbytes);
  }

  return EXIT_SUCCESS;
}
