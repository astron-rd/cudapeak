#include <iostream>
#include <cuda.h>
#include <cudaProfiler.h>

int main() {
    // Read device number from envirionment
    char *cstr_deviceNumber = getenv("CUDA_DEVICE");
    unsigned deviceNumber = cstr_deviceNumber ? atoi (cstr_deviceNumber) : 0;

    //  Setup CUDA
    cudaSetDevice(deviceNumber);
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Print CUDA device information
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, deviceNumber);
    std::cout << "Device " << deviceNumber << ": " << deviceProperties.name << std::endl;

    // Run benchmarks
    cuProfilerStart();
    cuProfilerStop();

    return EXIT_SUCCESS;
}
