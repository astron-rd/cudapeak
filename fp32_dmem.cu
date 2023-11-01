#include "common.h"

__global__ void fp32_dmem_01(float4* ptr);
__global__ void fp32_dmem_02(float4* ptr);
__global__ void fp32_dmem_04(float4* ptr);
__global__ void fp32_dmem_08(float4* ptr);
__global__ void fp32_dmem_16(float4* ptr);
__global__ void fp32_dmem_32(float4* ptr);
__global__ void fp32_dmem_64(float4* ptr);
__global__ void fp32_dmem_128(float4* ptr);
__global__ void fp32_dmem_256(float4* ptr);

int main(int argc, char* argv[]) {
  Benchmark benchmark;

  // Parameters
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  // Amount of work performed
  unsigned fetchPerBlock = 16;
  int maxItems = benchmark.totalGlobalMem() / sizeof(float4) / 2;
  int numItems = roundToPowOf2(maxItems);
  double gbytes = (float)(numItems / fetchPerBlock * 2) * sizeof(float4) / 1e9;
  double gflops = gbytes * 4;

  // Kernel dimensions
  dim3 grid(numItems / (fetchPerBlock * maxThreadsPerBlock));
  dim3 block(maxThreadsPerBlock);

  // Allocate memory
  benchmark.allocate(numItems * sizeof(float4));

  // Run benchmark
  for (int i = 0; i < NR_BENCHMARKS; i++) {
    benchmark.run(reinterpret_cast<void*>(&fp32_dmem_01), grid, block,
                  "flop:byte ->   1:1", 1*gflops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_dmem_02), grid, block,
                  "flop:byte ->   2:1", 2*gflops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_dmem_04), grid, block,
                  "flop:byte ->   4:1", 4*gflops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_dmem_08), grid, block,
                  "flop:byte ->   8:1", 8*gflops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_dmem_16), grid, block,
                  "flop:byte ->  16:1", 16*gflops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_dmem_32), grid, block,
                  "flop:byte ->  32:1", 32*gflops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_dmem_64), grid, block,
                  "flop:byte ->  64:1", 64*gflops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_dmem_128), grid, block,
                  "flop:byte -> 128:1", 128*gflops, gbytes);
    benchmark.run(reinterpret_cast<void*>(&fp32_dmem_256), grid, block,
                  "flop:byte -> 256:1", 256*gflops, gbytes);
  }

  return EXIT_SUCCESS;
}
