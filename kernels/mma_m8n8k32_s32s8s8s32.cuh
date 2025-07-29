using namespace nvcuda::wmma;

template <>
class fragment<matrix_a, 16, 8, 32, signed char, row_major>
    : public __frag_base<int, 4> {};
template <>
class fragment<matrix_b, 16, 8, 32, signed char, col_major>
    : public __frag_base<int, 2> {};

template <>
class wmma::fragment<wmma::accumulator, 16, 8, 32, int>
    : public __frag_base<int, 4> {};

inline __device__ void
mma_sync_ptx(fragment<accumulator, 16, 8, 32, int> &d,
             const fragment<matrix_a, 16, 8, 32, signed char, row_major> &a,
             const fragment<matrix_b, 16, 8, 32, signed char, col_major> &b,
             const fragment<accumulator, 16, 8, 32, int> &c) {
  asm volatile(
      "mma.sync.aligned.row.col.m16n8k32.s32.s8.s8.s32 {%0, %1, %2, %3}, "
      "{%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};"
      : "=r"(d.x[0]), "=r"(d.x[1]), "=r"(d.x[2]), "=r"(d.x[3])
      : "r"(a.x[0]), "r"(a.x[1]), "r"(a.x[2]), "r"(a.x[3]), "r"(b.x[0]),
        "r"(b.x[1]), "r"(c.x[0]), "r"(c.x[1]), "r"(c.x[2]), "r"(c.x[3]));
}

inline __device__ void
store_matrix_sync(int *p, const fragment<accumulator, 16, 8, 32, int> &d,
                  unsigned ldm, layout_t layout) {
  if (layout == mem_row_major) {
    ((int2 *)p)[ldm / 2 * (laneid() / 4) + laneid() % 4] =
        make_int2(d.x[0], d.x[1]);
    ((int2 *)p)[ldm / 2 * (laneid() / 4 + 8) + laneid() % 4] =
        make_int2(d.x[2], d.x[3]);
  } else {
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4)] = d.x[0];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4) + ldm] = d.x[1];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4 + 8)] = d.x[2];
    p[(laneid() % 4) * 2 * ldm + (laneid() / 4 + 8) + ldm] = d.x[3];
  }
}