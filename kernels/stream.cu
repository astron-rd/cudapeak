#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#endif

template <typename T>
__device__ void STREAM_Copy(T const *__restrict__ const a,
                            T *__restrict__ const b, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    b[idx] = a[idx];
}

template <typename T>
__device__ void STREAM_Scale(T const *__restrict__ const a,
                             T *__restrict__ const b, T scale, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    b[idx] = scale * a[idx];
}

template <typename T>
__device__ void STREAM_Add(T const *__restrict__ const a,
                           T const *__restrict__ const b,
                           T *__restrict__ const c, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    c[idx] = a[idx] + b[idx];
}

template <typename T>
__device__ void STREAM_Triad(T const *__restrict__ a, T const *__restrict__ b,
                             T *__restrict__ const c, T scalar, int len) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < len)
    c[idx] = a[idx] + scalar * b[idx];
}

#define STREAM_arguments_double                                                \
  double *__restrict__ a, double *__restrict__ b, double *__restrict__ c,      \
      double scale, int len

__global__ void STREAM_Triad_double(STREAM_arguments_double) {
  STREAM_Triad(a, b, c, scale, len);
}

__global__ void STREAM_Add_double(STREAM_arguments_double) {
  STREAM_Add(a, b, c, len);
}

__global__ void STREAM_Scale_double(STREAM_arguments_double) {
  STREAM_Scale(a, b, scale, len);
}

__global__ void STREAM_Copy_double(STREAM_arguments_double) {
  STREAM_Copy(a, b, len);
}

#define STREAM_arguments_float                                                 \
  float *__restrict__ a, float *__restrict__ b, float *__restrict__ const c,   \
      float scale, int len

__global__ void STREAM_Triad_float(STREAM_arguments_float) {
  STREAM_Triad(a, b, c, scale, len);
}

__global__ void STREAM_Add_float(STREAM_arguments_float) {
  STREAM_Add(a, b, c, len);
}

__global__ void STREAM_Scale_float(STREAM_arguments_float) {
  STREAM_Scale(a, b, scale, len);
}

__global__ void STREAM_Copy_float(STREAM_arguments_float) {
  STREAM_Copy(a, b, len);
}
