#include <sstream>

#include "common.h"

__global__ void fp32_kernel(float* ptr);

int main(int argc, char* argv[]) {
  Benchmark benchmark;

  // Parameters
  int multiProcessorCount = benchmark.multiProcessorCount();
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  for (int i = 0; i < NR_BENCHMARKS; i++) {
    for (int m = multiProcessorCount; m > 1; m--) {
      // Amount of work performed
      int nr_iterations = 2048;
      double gflops =
          (1e-9 * m * maxThreadsPerBlock) * (1ULL * nr_iterations * 8 * 4096);
      double gbytes = 0;

      // Kernel dimensions
      dim3 grid(m);
      dim3 block(maxThreadsPerBlock);

      benchmark.allocate(m * maxThreadsPerBlock * sizeof(float));

      // Run benchmark
      std::stringstream name;
      name << "fp32_" << m;
      benchmark.run(reinterpret_cast<void*>(&fp32_kernel), grid, block,
                    name.str().c_str(), gflops, gbytes);
    }
  }

  return EXIT_SUCCESS;
}
