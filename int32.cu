#include "common.h"

__global__ void int32_kernel(int *ptr);

int main(int argc, char *argv[]) {
    Benchmark benchmark;

    // Parameters
    int multiProcessorCount = benchmark.multiProcessorCount();
    int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

    // Amount of work performed
    int nr_iterations = 2048;
    double gflops = 0;
    double gbytes = 0;
    double gops   = (1e-9 * multiProcessorCount * maxThreadsPerBlock) * (1ULL * nr_iterations * 8 * 4096);

    // Kernel dimensions
    dim3 grid(multiProcessorCount);
    dim3 block(maxThreadsPerBlock);

    // Allocate memory
    benchmark.allocate(multiProcessorCount * maxThreadsPerBlock * sizeof(int));

    // Run benchmark
    for (int i = 0; i < NR_BENCHMARKS; i++) {
        benchmark.run(reinterpret_cast<void *>(&int32_kernel),  grid, block, "int32", gflops, gbytes, gops);
    }

    return EXIT_SUCCESS;
}
