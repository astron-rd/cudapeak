#include "common.h"

__global__ void fp32_smem_01(float* ptr);
__global__ void fp32_smem_02(float* ptr);
__global__ void fp32_smem_04(float* ptr);
__global__ void fp32_smem_08(float* ptr);
__global__ void fp32_smem_16(float* ptr);
__global__ void fp32_smem_32(float* ptr);
__global__ void fp32_smem_64(float* ptr);
__global__ void fp32_smem_128(float* ptr);

int main(int argc, const char* argv[]) {
  Benchmark benchmark(argc, argv);

  // Parameters
  int multiProcessorCount = benchmark.multiProcessorCount();
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();
  int maxBlocksPerSM = 16;

  // Amount of work performed
  unsigned nrRepetitions = 512;
  unsigned fetchPerBlock = 2048;
  unsigned workPerBlock = nrRepetitions * fetchPerBlock;
  unsigned globalBlocks =
      multiProcessorCount * maxBlocksPerSM * maxThreadsPerBlock;
  double gbytes = 1e-9 * globalBlocks * workPerBlock * sizeof(float4);
  double gflops = gbytes / 2;

  // Kernel dimensions
  dim3 grid(multiProcessorCount, maxBlocksPerSM);
  dim3 block(maxThreadsPerBlock);

  // Allocate memory
  benchmark.allocate(fetchPerBlock * 4 * sizeof(float));

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(reinterpret_cast<void*>(&fp32_smem_01), grid, block,
                  "flop:byte ->  1:2", gflops, gbytes / 1);
    benchmark.run(reinterpret_cast<void*>(&fp32_smem_02), grid, block,
                  "flop:byte ->  1:1", gflops, gbytes / 2);
    benchmark.run(reinterpret_cast<void*>(&fp32_smem_04), grid, block,
                  "flop:byte ->  2:1", gflops, gbytes / 4);
    benchmark.run(reinterpret_cast<void*>(&fp32_smem_08), grid, block,
                  "flop:byte ->  4:1", gflops, gbytes / 8);
    benchmark.run(reinterpret_cast<void*>(&fp32_smem_16), grid, block,
                  "flop:byte ->  8:1", gflops, gbytes / 16);
    benchmark.run(reinterpret_cast<void*>(&fp32_smem_32), grid, block,
                  "flop:byte -> 16:1", gflops, gbytes / 32);
    benchmark.run(reinterpret_cast<void*>(&fp32_smem_64), grid, block,
                  "flop:byte -> 32:1", gflops, gbytes / 64);
    benchmark.run(reinterpret_cast<void*>(&fp32_smem_128), grid, block,
                  "flop:byte -> 64:1", gflops, gbytes / 128);
  }

  return EXIT_SUCCESS;
}
