#include "common.h"

__global__ void fp32_int32_1_64(float* ptr);
__global__ void fp32_int32_1_32(float* ptr);
__global__ void fp32_int32_1_16(float* ptr);
__global__ void fp32_int32_1_8(float* ptr);
__global__ void fp32_int32_1_4(float* ptr);
__global__ void fp32_int32_1_2(float* ptr);
__global__ void fp32_int32_1_1(float* ptr);
__global__ void fp32_int32_2_1(float* ptr);
__global__ void fp32_int32_4_1(float* ptr);
__global__ void fp32_int32_8_1(float* ptr);
__global__ void fp32_int32_16_1(float* ptr);
__global__ void fp32_int32_32_1(float* ptr);
__global__ void fp32_int32_64_1(float* ptr);

int main(int argc, char* argv[]) {
  Benchmark benchmark;

  // Parameters
  int multiProcessorCount = benchmark.multiProcessorCount();
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  // Amount of work performed
  int nr_iterations = 2048;
  double gflops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) *
                  (1ULL * nr_iterations * 8 * 4096);
  double gbytes = 0;

  // Kernel dimensions
  dim3 grid(multiProcessorCount);
  dim3 block(maxThreadsPerBlock);

  // Allocate memory
  benchmark.allocate(multiProcessorCount * maxThreadsPerBlock * sizeof(float));

  // Run benchmark
  for (int i = 0; i < NR_BENCHMARKS; i++) {
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_64), grid, block,
                  "fp32:int32 ->  1:64", gflops / 64, gbytes, gflops);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_32), grid, block,
                  "fp32:int32 ->  1:32", gflops / 32, gbytes, gflops);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_16), grid, block,
                  "fp32:int32 ->  1:16", gflops / 16, gbytes, gflops);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_8), grid, block,
                  "fp32:int32 ->   1:8", gflops / 8, gbytes, gflops);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_4), grid, block,
                  "fp32:int32 ->   1:4", gflops / 4, gbytes, gflops);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_2), grid, block,
                  "fp32:int32 ->   1:2", gflops / 2, gbytes, gflops);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_1), grid, block,
                  "fp32:int32 ->   1:1", gflops, gbytes, gflops);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_2_1), grid, block,
                  "fp32:int32 ->   2:1", gflops, gbytes, gflops / 2);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_4_1), grid, block,
                  "fp32:int32 ->   4:1", gflops, gbytes, gflops / 4);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_8_1), grid, block,
                  "fp32:int32 ->   8:1", gflops, gbytes, gflops / 8);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_16_1), grid, block,
                  "fp32:int32 ->  16:1", gflops, gbytes, gflops / 16);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_32_1), grid, block,
                  "fp32:int32 ->  32:1", gflops, gbytes, gflops / 32);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_64_1), grid, block,
                  "fp32:int32 ->  64:1", gflops, gbytes, gflops / 64);
  }

  return EXIT_SUCCESS;
}
