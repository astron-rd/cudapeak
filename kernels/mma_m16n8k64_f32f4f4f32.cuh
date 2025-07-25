#include <mma.h>

typedef unsigned int uint32_t;
typedef unsigned short uint16_t;

using namespace nvcuda::wmma;

template <>
class fragment<matrix_a, 16, 8, 64, __nv_fp4_e2m1, row_major>
    : public __frag_base<int, 4> {};
template <>
class fragment<matrix_b, 16, 8, 64, __nv_fp4_e2m1, col_major>
    : public __frag_base<int, 2> {};

template <>
class fragment<accumulator, 16, 8, 64, float> : public __frag_base<float, 4> {};

inline __device__ void
mma_sync_ptx(fragment<accumulator, 16, 8, 64, float> &d,
             const fragment<matrix_a, 16, 8, 64, __nv_fp4_e2m1, row_major> &a,
             const fragment<matrix_b, 16, 8, 64, __nv_fp4_e2m1, col_major> &b,
             const fragment<accumulator, 16, 8, 64, float> &c) {
  static constexpr uint32_t sfa0 = 0x7F7F;
  static constexpr uint16_t tidA = 0;
  static constexpr uint16_t bidA = 0;
  static constexpr uint32_t sfb0 = 0x7F7F;
  static constexpr uint16_t tidB = 0;
  static constexpr uint16_t bidB = 0;

  asm("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row."
      "col.f32.e2m1.e2m1.f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, "
      "%9}, {%10, %11, %12, %13}, {%14}, {%15, %16}, {%17}, {%18, %19};"
      : "=f"(d.x[0]), "=f"(d.x[1]), "=f"(d.x[2]), "=f"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
        "r"(b.x[1]), "f"(c.x[0]), "f"(c.x[1]), "f"(c.x[2]), "f"(c.x[3]),
        "r"(uint32_t(sfa0)), "h"(bidA), "h"(tidA), "r"(uint32_t(sfb0)),
        "h"(bidB), "h"(tidB));
}

inline __device__ void
store_matrix_sync(float *p, const fragment<accumulator, 16, 8, 64, float> &d,
                  unsigned ldm, layout_t layout) {
  if (layout == mem_row_major) {
    ((float2 *)p)[ldm / 2 * (laneid() / 4) + laneid() % 4] =
        make_float2(d.x[0], d.x[1]);
    ((float2 *)p)[ldm / 2 * (laneid() / 4 + 8) + laneid() % 4] =
        make_float2(d.x[2], d.x[3]);
  } else {
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4)] = d.x[0];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4) + ldm] = d.x[1];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4 + 8)] = d.x[2];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4 + 8) + ldm] = d.x[3];
  }
}