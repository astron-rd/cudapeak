#ifndef CUDAPEAK_OPAQUE_H
#define CUDAPEAK_OPAQUE_H

#if defined(__HIP_PLATFORM_AMD__)

#include "fp16.h"

// Hide a value from the optimizer.
//
// The peak throughput kernels use loop-invariant operands that are
// compile-time constants. Left alone, the compiler folds their products into
// constants, and -ffast-math then collapses the loop into a few adds, so the
// reported throughput says nothing about the hardware. Passing the operands
// through opaque() makes them unknown values, and the kernels then use explicit
// fused multiply-adds to make sure every operation is issued.
//
// On NVIDIA the kernels use inline PTX for this purpose.

__device__ __forceinline__ float opaque(float x) {
  asm volatile("" : "+v"(x));
  return x;
}

__device__ __forceinline__ half opaque(half x) {
  unsigned bits = __half_as_ushort(x);
  asm volatile("" : "+v"(bits));
  return __ushort_as_half(bits);
}

#endif // __HIP_PLATFORM_AMD__

#endif // CUDAPEAK_OPAQUE_H
