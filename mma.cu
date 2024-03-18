#include "common.h"

__global__ void mma8_kernel(void* ptr);
__global__ void mma16_kernel(void* ptr);
__global__ void mma32_kernel(void* ptr);

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

  // MMA size (m * n * k)
  int nr_ops_per_mma_8 = (16 * 16 * 16 * (1 /* mul */ + 1 /* add */));
  int nr_ops_per_mma_16 = (16 * 16 * 16 * (1 /* mul */ + 1 /* add */));
  int nr_ops_per_mma_32 = (16 * 16 * 8 * (1 /* mul */ + 1 /* add */));

  size_t sizeof_data = nr_warps_per_thread_block * 16 * 16 * sizeof(int);

  // Amount of work performed
  int nr_iterations = 32768;
  double gflops = 1e-9 * nr_iterations *
                  nr_warps_per_thread_block * nr_thread_blocks;
  double gflops_8 = gflops * nr_ops_per_mma_8;
  double gflops_16 = gflops * nr_ops_per_mma_16;
  double gflops_32 = gflops * nr_ops_per_mma_32;
  double gbytes = 0;

  benchmark.allocate(sizeof_data);

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(reinterpret_cast<void*>(&mma8_kernel), grid, block,
                  "mma_8bit", gflops_8, gbytes);
    benchmark.run(reinterpret_cast<void*>(&mma16_kernel), grid, block,
                  "mma_16bit", gflops_16, gbytes);
    benchmark.run(reinterpret_cast<void*>(&mma32_kernel), grid, block,
                  "mma_32bit", gflops_32, gbytes);
  }

  return EXIT_SUCCESS;
}
