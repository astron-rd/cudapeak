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

int main(int argc, const char* argv[]) {
  Benchmark benchmark(argc, argv);

  // Parameters
  int multiProcessorCount = benchmark.multiProcessorCount();
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  // Amount of work performed
  int nr_iterations = 2048;
  const double gops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) *
                      (1ULL * nr_iterations * 8 * 4096);
  const double gbytes = 0;

  // Kernel dimensions
  dim3 grid(multiProcessorCount);
  dim3 block(maxThreadsPerBlock);

  // Allocate memory
  benchmark.allocate(multiProcessorCount * maxThreadsPerBlock * sizeof(float));

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_64), grid, block,
                  "fp32:int32 ->  1:64", gops / 64, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_32), grid, block,
                  "fp32:int32 ->  1:32", gops / 32, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_16), grid, block,
                  "fp32:int32 ->  1:16", gops / 16, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_8), grid, block,
                  "fp32:int32 ->   1:8", gops / 8, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_4), grid, block,
                  "fp32:int32 ->   1:4", gops / 4, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_2), grid, block,
                  "fp32:int32 ->   1:2", gops / 2, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_1_1), grid, block,
                  "fp32:int32 ->   1:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_2_1), grid, block,
                  "fp32:int32 ->   2:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_4_1), grid, block,
                  "fp32:int32 ->   4:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_8_1), grid, block,
                  "fp32:int32 ->   8:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_16_1), grid, block,
                  "fp32:int32 ->  16:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_32_1), grid, block,
                  "fp32:int32 ->  32:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_int32_64_1), grid, block,
                  "fp32:int32 ->  64:1", gops, gbytes);
  }

  return EXIT_SUCCESS;
}
