#include "common.h"

__global__ void mma8_kernel(void* ptr);
__global__ void mma16_kernel(void* ptr);
__global__ void mma32_kernel(void* ptr);

int main(int argc, char* argv[]) {
  Benchmark benchmark;

  // Parameters
  int multiProcessorCount = benchmark.multiProcessorCount();
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  // Kernel dimensions
  int nr_thread_blocks = multiProcessorCount * 512;
  int nr_warps_per_thread_block = 4;
  dim3 grid(nr_thread_blocks);
  dim3 block(32, nr_warps_per_thread_block);

  // MMA size
  int m = 16;
  int n = 16;
  int k = 16;
  int nr_ops_per_mma = (m * n * k * (1 /* mul */ + 1 /* add */));

  size_t sizeof_data =
      nr_thread_blocks * nr_warps_per_thread_block * m * n * sizeof(int);

  // Amount of work performed
  int nr_iterations = 32768;
  double gflops = 1e-9 * nr_ops_per_mma * nr_iterations *
                  nr_warps_per_thread_block * nr_thread_blocks;
  double gbytes = 0;

  benchmark.allocate(sizeof_data);

  // Run benchmark
  for (int i = 0; i < NR_BENCHMARKS; i++) {
    benchmark.run(reinterpret_cast<void*>(&mma8_kernel), grid, block,
                  "mma_8bit", gflops, gbytes);  // k = 16
    benchmark.run(reinterpret_cast<void*>(&mma16_kernel), grid, block,
                  "mma_16bit", gflops, gbytes);  // k = 16
    benchmark.run(reinterpret_cast<void*>(&mma32_kernel), grid, block,
                  "mma_32bit", gflops / 2, gbytes);  // k = 8
  }

  return EXIT_SUCCESS;
}
