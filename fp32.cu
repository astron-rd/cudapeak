#include "common.h"

__global__ void fp32_kernel(float *ptr);

void run(
    cudaStream_t stream,
    cudaDeviceProp deviceProperties
#if defined(HAVE_PMT)
    , std::shared_ptr<pmt::PMT> pmt
#endif
)
{
    // Parameters
    int multiProcessorCount = deviceProperties.multiProcessorCount;
    int maxThreadsPerBlock = deviceProperties.maxThreadsPerBlock;

    // Amount of work performed
    int nr_iterations = 2048;
    double gflops = (1e-9 * multiProcessorCount * maxThreadsPerBlock) * (1ULL * nr_iterations * 8 * 4096);
    double gbytes = 0;

    // Kernel dimensions
    dim3 gridDim(multiProcessorCount);
    dim3 blockDim(maxThreadsPerBlock);

    // Allocate memory
    float *ptr;
    cudaMalloc(&ptr, multiProcessorCount * maxThreadsPerBlock * sizeof(float));

    // Run kernel
    measurement measurement;
    measurement = run_kernel(stream, deviceProperties, (void *) &fp32_kernel, ptr, gridDim, blockDim
#if defined(HAVE_PMT)
    , pmt
#endif
        );
    report("fp32", measurement, gflops, gbytes);

    // Free memory
    cudaFree(ptr);
}
