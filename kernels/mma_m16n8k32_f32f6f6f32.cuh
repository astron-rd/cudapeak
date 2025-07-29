#include <cuda_fp6.h>

using namespace nvcuda::wmma;

template <>
class fragment<matrix_a, 16, 8, 32, __nv_fp6_e3m2, row_major>
    : public __frag_base<int, 4> {};
template <>
class fragment<matrix_b, 16, 8, 32, __nv_fp6_e2m3, col_major>
    : public __frag_base<int, 2> {};

inline __device__ void
mma_sync_ptx(fragment<accumulator, 16, 8, 32, float> &d,
             const fragment<matrix_a, 16, 8, 32, __nv_fp6_e3m2, row_major> &a,
             const fragment<matrix_b, 16, 8, 32, __nv_fp6_e2m3, col_major> &b,
             const fragment<accumulator, 16, 8, 32, float> &c) {

  asm("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e3m2.e2m3.f32 {%0, "
      "%1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
      : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
        "r"(b.x[1]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]));
}