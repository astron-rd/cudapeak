#include <algorithm>
#include <iostream>
#include <tuple>

#include "common/common.h"

#include "kernels/mma.cu.o.h"

std::vector<std::string> split(const std::string &s, char delimiter) {
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(s);

  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }

  return tokens;
}

int getBitSize(const std::string &kernel_name) {
  const std::string type = split(kernel_name, '_')[1];

  // Handle special cases first
  if (type == "e4m3" || type == "e5m2")
    return 8;

  // Default case: try to extract number from type (e.g., "16" from "f16")
  for (char c : type) {
    if (isdigit(c)) {
      return atoi(type.substr(type.find_first_of("0123456789")).c_str());
    }
  }

  // Unknown type
  return 0;
}

std::vector<std::string>
sortKernelsByBitSize(std::vector<std::string> &kernels) {
  // Create a vector of pairs (bit_size, original_index) for stable sorting
  std::vector<std::pair<int, size_t>> bit_sizes;
  for (size_t i = 0; i < kernels.size(); ++i) {
    bit_sizes.emplace_back(getBitSize(kernels[i]), i);
  }

  // Stable sort based on bit size
  std::stable_sort(
      bit_sizes.begin(), bit_sizes.end(),
      [](const auto &a, const auto &b) { return a.first > b.first; });

  // Reorder the kernels based on the sorted indices
  std::vector<std::string> sorted_kernels;
  for (const auto &[bit_size, idx] : bit_sizes) {
    sorted_kernels.push_back(kernels[idx]);
  }

  return sorted_kernels;
}

std::vector<std::string> getSupportedKernels(Benchmark &benchmark) {
  std::vector<std::string> kernel_names;

  // Always supported across all platforms
  kernel_names.push_back("mma_f16_16_16_16");

#if defined(__HIP_PLATFORM_AMD__)
  // AMD-specific kernels
  kernel_names.push_back("mma_s8_16_16_32");

  if (benchmark.isCDNA()) {
    kernel_names.push_back("mma_f32_16_16_16");
  }
  if (benchmark.isCDNA2() || benchmark.isCDNA3()) {
    kernel_names.push_back("mma_f64_16_16_16");
  }
  if (benchmark.isCDNA3()) {
    kernel_names.push_back("mma_fp8_16_16_32");
    kernel_names.push_back("mma_bf8_16_16_32");
    kernel_names.push_back("mma_xf32_16_16_8");
  }
#else
  // NVIDIA-specific kernels
  if (!benchmark.isVolta()) {
    kernel_names.push_back("mma_s4_8_8_32");
    kernel_names.push_back("mma_s8_16_16_16");

    kernel_names.push_back("bmma_b1_8_8_128_xor");
    if (!benchmark.isTuring()) {
      kernel_names.push_back("bmma_b1_8_8_128_and");
      kernel_names.push_back("bmma_b1_16_8_256_xor");
      kernel_names.push_back("bmma_b1_16_8_256_and");
      kernel_names.push_back("mma_bf16_16_16_16");
      kernel_names.push_back("mma_tf32_16_16_8");
    }
  }

  if (benchmark.isAda() || benchmark.isHopper() || benchmark.isBlackwell()) {
    kernel_names.push_back("mma_e4m3_16_8_32");
    kernel_names.push_back("mma_e5m2_16_8_32");
  }
#endif

  return sortKernelsByBitSize(kernel_names);
}

std::tuple<int, int, int> extractFragmentSizes(const std::string &input) {
  // Split the string by underscores
  auto parts = split(input, '_');

  // We need at least 5 parts (with optional parts after)
  if (parts.size() < 5) {
    throw std::invalid_argument("Input string doesn't contain enough parts");
  }

  try {
    int m = std::stoi(parts[2]);
    int n = std::stoi(parts[3]);
    int k = std::stoi(parts[4]);
    return std::make_tuple(m, n, k);
  } catch (const std::invalid_argument &) {
    throw std::invalid_argument("Failed to convert fragment sizes to integers");
  } catch (const std::out_of_range &) {
    throw std::invalid_argument("Fragment size value is out of range");
  }
}

int main(int argc, const char *argv[]) {
  Benchmark benchmark(argc, argv);

#if defined(__HIP_PLATFORM_AMD__)
  if (benchmark.isRDNA2()) {
    std::cout << "RDNA2 is not supported." << std::endl;
    return EXIT_SUCCESS;
  }
#endif

  KernelFactory kernel_factory(mma_source);
  const std::vector<std::string> kernel_names = getSupportedKernels(benchmark);
  auto kernels =
      kernel_factory.compileKernels(benchmark.getDevice(), kernel_names);

  // Parameters
  int multiProcessorCount = benchmark.multiProcessorCount();
  int maxThreadsPerBlock = benchmark.maxThreadsPerBlock();

  // Kernel dimensions
  int nr_thread_blocks = multiProcessorCount * 512;
  int nr_warps_per_thread_block = 4;
  dim3 grid(nr_thread_blocks);
  unsigned warp_size = 32;
#if defined(__HIP_PLATFORM_AMD__)
  if (benchmark.isCDNA()) {
    warp_size = 64;
  }
#endif
  dim3 block(warp_size, nr_warps_per_thread_block);

  size_t sizeof_data = nr_warps_per_thread_block * 16 * 16 * sizeof(int);

  // Amount of work performed
  int nr_iterations = 32768;
  const double gops =
      1e-9 * nr_iterations * nr_warps_per_thread_block * nr_thread_blocks;
  const double gbytes = 0;

  benchmark.allocate(sizeof_data);

  // Run benchmark
  for (int i = 0; i < benchmark.nrBenchmarks(); i++) {
    for (int j = 0; j < kernel_names.size(); j++) {
      const std::string &kernel_name = kernel_names[j];
      auto [m, n, k] = extractFragmentSizes(kernel_name);
      benchmark.run(kernels[j], grid, block, kernel_name,
                    gops * (m * n * k * 2), gbytes);
    }
  }

  return EXIT_SUCCESS;
}
