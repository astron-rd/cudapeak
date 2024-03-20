#include "common.h"

__global__ void mma_s4(void* ptr);
__global__ void mma_s8(void* ptr);
__global__ void mma_f16(void* ptr);
__global__ void mma_tf32(void* ptr);

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

  size_t sizeof_data = nr_warps_per_thread_block * 16 * 16 * sizeof(int);

  // Amount of work performed
  int nr_iterations = 32768;
  double gflops =
      1e-9 * nr_iterations * nr_warps_per_thread_block * nr_thread_blocks;
  double gbytes = 0;

  benchmark.allocate(sizeof_data);

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(reinterpret_cast<void*>(&mma_s4), grid, block, "mma_s4",
                  gflops * (8 * 8 * 32 * 2), gbytes);
    benchmark.run(reinterpret_cast<void*>(&mma_s8), grid, block, "mma_s8",
                  gflops * (16 * 16 * 16 * 2), gbytes);
    benchmark.run(reinterpret_cast<void*>(&mma_f16), grid, block, "mma_f16",
                  gflops * (16 * 16 * 16 * 2), gbytes);
    benchmark.run(reinterpret_cast<void*>(&mma_tf32), grid, block, "mma_tf32",
                  gflops * (16 * 16 * 8 * 2), gbytes);
  }

  return EXIT_SUCCESS;
}
