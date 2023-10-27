#include "common.h"

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

int main(int argc, char *argv[]) {
    Benchmark benchmark;

    // Parameters
    int multiProcessorCount = benchmark.multiProcessorCount();
    int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

    // Amount of work performed
    int nr_iterations = 512;
    double gflops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) * (1ULL * nr_iterations * 8192 * 2);
    double gbytes = 0;

    // Kernel dimensions
    dim3 grid(multiProcessorCount);
    dim3 block(maxThreadsPerBlock);

    // Allocate memory
    benchmark.allocate(multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run benchmark
    for (int i = 0; i < NR_BENCHMARKS; i++) {
        benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_1_8),   grid, block, "fma:sincos (lu) ->    1:8", gflops, gbytes, gflops*8);
        benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_1_4),   grid, block, "fma:sincos (lu) ->    1:4", gflops, gbytes, gflops*4);
        benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_1_2),   grid, block, "fma:sincos (lu) ->    1:2", gflops, gbytes, gflops*2);
        benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_1_1),   grid, block, "fma:sincos (lu) ->    1:1", gflops, gbytes, gflops*1);
        benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_2_1),   grid, block, "fma:sincos (lu) ->    2:1", gflops, gbytes, gflops/2);
        benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_4_1),   grid, block, "fma:sincos (lu) ->    4:1", gflops, gbytes, gflops/4);
        benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_8_1),   grid, block, "fma:sincos (lu) ->    8:1", gflops, gbytes, gflops/8);
        benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_16_1),  grid, block, "fma:sincos (lu) ->   16:1", gflops, gbytes, gflops/16);
        benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_32_1),  grid, block, "fma:sincos (lu) ->   32:1", gflops, gbytes, gflops/32);
        benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_64_1),  grid, block, "fma:sincos (lu) ->   64:1", gflops, gbytes, gflops/64);
        benchmark.run(reinterpret_cast<void *>(&fp32_sincos_lu_128_1), grid, block, "fma:sincos (lu) ->  128:1", gflops, gbytes, gflops/128);
    }

    return EXIT_SUCCESS;
}
