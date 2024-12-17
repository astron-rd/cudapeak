#include "common.h"

__global__ void mma_f16_16_16_16(void* ptr);
__global__ void mma_bf16_16_16_16(void* ptr);
#if not defined(__HIP_PLATFORM_AMD__)
__global__ void bmma_b1_8_8_128_xor(void* ptr);
__global__ void bmma_b1_16_8_256_xor(void* ptr);
__global__ void bmma_b1_8_8_128_and(void* ptr);
__global__ void bmma_b1_16_8_256_and(void* ptr);
__global__ void mma_s4_8_8_32(void* ptr);
__global__ void mma_s8_16_16_16(void* ptr);
__global__ void mma_tf32_16_16_8(void* ptr);
#else
__global__ void mma_fp8_16_16_32(void* ptr);
__global__ void mma_bf8_16_16_32(void* ptr);
__global__ void mma_s8_16_16_32(void* ptr);
__global__ void mma_f32_16_16_16(void* ptr);
__global__ void mma_f64_16_16_16(void* ptr);
__global__ void mma_xf32_16_16_8(void* ptr);
#endif

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
  const double gops =
      1e-9 * nr_iterations * nr_warps_per_thread_block * nr_thread_blocks;
  const double gbytes = 0;

  benchmark.allocate(sizeof_data);

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
#if defined(__HIP_PLATFORM_AMD__)
    benchmark.run(reinterpret_cast<void*>(&mma_s8_16_16_32), grid, block,
                  "mma_s8_16_16_32", gops * (16 * 16 * 32 * 2), gbytes);

    // FP8 / BF8 / XF32 are only available on CDNA3
    if (benchmark.isCDNA3()) {
      benchmark.run(reinterpret_cast<void*>(&mma_fp8_16_16_32), grid, block,
                    "mma_fp8_16_16_32", gops * (16 * 16 * 32 * 2), gbytes);
      benchmark.run(reinterpret_cast<void*>(&mma_bf8_16_16_32), grid, block,
                    "mma_bf8_16_16_32", gops * (16 * 16 * 32 * 2), gbytes);
      benchmark.run(reinterpret_cast<void*>(&mma_xf32_16_16_8), grid, block,
                    "mma_xf32_16_16_8", gops * (16 * 16 * 8 * 2), gbytes);
    }
#else
    benchmark.run(reinterpret_cast<void*>(&bmma_b1_16_8_256_xor), grid, block,
                  "bmma_b1_16_8_256_xor", gops * (16 * 8 * 256 * 2), gbytes);
    benchmark.run(reinterpret_cast<void*>(&bmma_b1_16_8_256_and), grid, block,
                  "bmma_b1_16_8_256_and", gops * (16 * 8 * 256 * 2), gbytes);
    benchmark.run(reinterpret_cast<void*>(&bmma_b1_8_8_128_xor), grid, block,
                  "bmma_b1_8_8_128_xor", gops * (8 * 8 * 128 * 2), gbytes);
    benchmark.run(reinterpret_cast<void*>(&bmma_b1_8_8_128_and), grid, block,
                  "bmma_b1_8_8_128_and", gops * (8 * 8 * 128 * 2), gbytes);
    benchmark.run(reinterpret_cast<void*>(&mma_s4_8_8_32), grid, block,
                  "mma_s4_8_8_32", gops * (8 * 8 * 32 * 2), gbytes);
    benchmark.run(reinterpret_cast<void*>(&mma_s8_16_16_16), grid, block,
                  "mma_s8_16_16_16", gops * (16 * 16 * 16 * 2), gbytes);
#endif
    benchmark.run(reinterpret_cast<void*>(&mma_f16_16_16_16), grid, block,
                  "mma_f16_16_16_16", gops * (16 * 16 * 16 * 2), gbytes);
    benchmark.run(reinterpret_cast<void*>(&mma_bf16_16_16_16), grid, block,
                  "mma_bf16_16_16_16", gops * (16 * 16 * 16 * 2), gbytes);

#if defined(__HIP_PLATFORM_AMD__)
    // F32 is only available on CDNA
    if (benchmark.isCDNA()) {
      benchmark.run(reinterpret_cast<void*>(&mma_f32_16_16_16), grid, block,
                    "mma_f32_16_16_16", gops * (16 * 16 * 16 * 2), gbytes);
    }
    // F64 is only available on CDNA2+
    if (benchmark.isCDNA2() || benchmark.isCDNA3()) {
      benchmark.run(reinterpret_cast<void*>(&mma_f64_16_16_16), grid, block,
                    "mma_f64_16_16_16", gops * (16 * 16 * 16 * 2), gbytes);
    }
#else
    benchmark.run(reinterpret_cast<void*>(&mma_tf32_16_16_8), grid, block,
                  "mma_tf32_16_16_8", gops * (16 * 16 * 8 * 2), gbytes);
#endif
  }

  return EXIT_SUCCESS;
}
