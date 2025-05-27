#include "Benchmark.h"
#include <cassert>

void Benchmark::launch_kernel(void *kernel, dim3 grid, dim3 block,
                              cu::Stream &stream,
                              const std::vector<const void *> &args) {
  assert(args.size() == 1);
  ((void (*)(void *))kernel)<<<grid, block, 0, stream>>>(
      const_cast<void *>(args[0]));
}
