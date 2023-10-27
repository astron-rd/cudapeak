#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

#ifndef COMMON_H
#define COMMON_H

#if defined(HAVE_PMT)
#include <pmt.h>
#endif

using namespace std;

// Number of times to run each kernel
#define NR_ITERATIONS 1

// Number of times to run each benchmark
#define NR_BENCHMARKS 5

// Helper functions
unsigned roundToPowOf2(unsigned number);

typedef struct {
  double runtime;  // milliseconds
  double power;    // watts
} measurement;

// Function to report kernel performance
void report(string name, measurement measurement, double gflops = 0,
            double gbytes = 0, double gops = 0);

class Benchmark {
 public:
  Benchmark();
  ~Benchmark();

  void allocate(size_t bytes);
  void run(void* kernel, dim3 grid, dim3 block, const char* name,
           double gflops = 0, double gbytes = 0, double gops = 0);

  int multiProcessorCount() { return device_properties_.multiProcessorCount; }
  int maxThreadsPerBlock() { return device_properties_.maxThreadsPerBlock; }
  size_t totalGlobalMem() { return device_properties_.totalGlobalMem; }

 protected:
  measurement run_kernel(void* kernel, dim3 grid, dim3 block);

  cudaStream_t stream_;
  cudaDeviceProp device_properties_;
  cudaEvent_t event_start_;
  cudaEvent_t event_end_;
  void* data_;
  size_t data_bytes_;
#if defined(HAVE_PMT)
  std::shared_ptr<pmt::PMT> pm_;
#endif
};

#endif  // end COMMON_H
