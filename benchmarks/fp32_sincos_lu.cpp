#include "common/common.h"

__global__ void fp32_sincos_lu_1_8(float *ptr);
__global__ void fp32_sincos_lu_1_4(float *ptr);
__global__ void fp32_sincos_lu_1_2(float *ptr);
__global__ void fp32_sincos_lu_1_1(float *ptr);
__global__ void fp32_sincos_lu_2_1(float *ptr);
__global__ void fp32_sincos_lu_4_1(float *ptr);
__global__ void fp32_sincos_lu_8_1(float *ptr);
__global__ void fp32_sincos_lu_16_1(float *ptr);
__global__ void fp32_sincos_lu_32_1(float *ptr);
__global__ void fp32_sincos_lu_64_1(float *ptr);
__global__ void fp32_sincos_lu_128_1(float *ptr);

int main(int argc, const char *argv[]) {
  Benchmark benchmark(argc, argv);

  // Parameters
  int multiProcessorCount = benchmark.multiProcessorCount();
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  // Amount of work performed
  const int nr_outer = 4096;
  const int nr_inner = 1024;
  const double gops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) *
                      (1ULL * nr_outer * nr_inner * 2);
  const double gbytes = 0;

  // Kernel dimensions
  dim3 grid(multiProcessorCount);
  dim3 block(maxThreadsPerBlock);

  // Allocate memory
  benchmark.allocate(multiProcessorCount * maxThreadsPerBlock * sizeof(float));

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_1_8), grid, block,
                  "fma:sincos (lu) ->    1:8", gops, gbytes);
    benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_1_4), grid, block,
                  "fma:sincos (lu) ->    1:4", gops, gbytes);
    benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_1_2), grid, block,
                  "fma:sincos (lu) ->    1:2", gops, gbytes);
    benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_1_1), grid, block,
                  "fma:sincos (lu) ->    1:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_2_1), grid, block,
                  "fma:sincos (lu) ->    2:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_4_1), grid, block,
                  "fma:sincos (lu) ->    4:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_8_1), grid, block,
                  "fma:sincos (lu) ->    8:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_16_1), grid, block,
                  "fma:sincos (lu) ->   16:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_32_1), grid, block,
                  "fma:sincos (lu) ->   32:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_64_1), grid, block,
                  "fma:sincos (lu) ->   64:1", gops, gbytes);
    benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_128_1), grid, block,
                  "fma:sincos (lu) ->  128:1", gops, gbytes);
  }

  return EXIT_SUCCESS;
}
