using namespace nvcuda::wmma;

inline __device__ void mma_sync_ptx(
    fragment<accumulator, 8, 8, 32, int> &d,
    const fragment<matrix_a, 8, 8, 32, experimental::precision::s4, row_major>
        &a,
    const fragment<matrix_b, 8, 8, 32, experimental::precision::s4, col_major>
        &b,
    const fragment<accumulator, 8, 8, 32, int> &c) {
  asm volatile(
      "mma.sync.aligned.row.col.m8n8k32.s32.s4.s4.s32 {%0, %1}, {%2}, {%3}, "
      "{%4, %5};\n"
      : "=r"(d.x[0]), "=r"(d.x[1])
      : "r"(a.x[0]), "r"(b.x[0]), "r"(c.x[0]), "r"(c.x[1]));
}