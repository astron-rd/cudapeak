#include "common.h"

__global__ void bmma_kernel(void* ptr);

int main(int argc, const char* argv[]) {
  Benchmark benchmark(argc, argv);

  // Parameters
  int multiProcessorCount = benchmark.multiProcessorCount();
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  // Kernel dimensions
  int nr_thread_blocks = multiProcessorCount * 512;
  int nr_warps_per_thread_block = 4;
  dim3 grid(nr_thread_blocks);
  dim3 block(32, nr_warps_per_thread_block);

  // MMA size
  int m = 8;
  int n = 8;
  int k = 128;
  int nr_ops_per_mma = (m * n * k * (1 /* xnor */ + 1 /* add */));

  size_t sizeof_data = nr_warps_per_thread_block * m * n * sizeof(int);

  // Amount of work performed
  int nr_iterations = 32768;
  double gflops = 1e-9 * nr_ops_per_mma * nr_iterations *
                  nr_warps_per_thread_block * nr_thread_blocks;
  double gbytes = 0;

  benchmark.allocate(sizeof_data);

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(reinterpret_cast<void*>(&bmma_kernel), grid, block, "bmma",
                  gflops, gbytes);
  }

  return EXIT_SUCCESS;
}
