#include <cassert>
#include "Benchmark.h"

template <typename T>
class BenchmarkStream : public Benchmark {
  virtual void launch_kernel(void* kernel, dim3 grid, dim3 block,
                             cu::Stream& stream,
                             const std::vector<const void*>& args) {
    assert(args.size() == 5);
    T* a = const_cast<T*>(reinterpret_cast<const T*>(args[0]));
    T* b = const_cast<T*>(reinterpret_cast<const T*>(args[1]));
    T* c = const_cast<T*>(reinterpret_cast<const T*>(args[2]));
    T scale = *reinterpret_cast<const T*>(args[3]);
    size_t N = *reinterpret_cast<const size_t*>(args[4]);
    ((void (*)(T*, T*, T*, T, int))kernel)<<<grid, block, 0, *stream_>>>(
        a, b, c, scale, N);
  }
};

template class BenchmarkStream<float>;