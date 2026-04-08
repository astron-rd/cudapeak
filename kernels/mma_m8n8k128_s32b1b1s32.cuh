using namespace nvcuda::wmma;

// wmma::bmma_sync with bmmaBitOpAND is broken in CUDA 13.2 for Blackwell
// (sm_120a): the __bmma_m8n8k128_mma_and_popc_b1 intrinsic triggers
// "unsupported operation". The underlying PTX instruction remains valid, use
// inline PTX as a workaround.
inline __device__ void bmma_m8n8k128_and_sync(
    fragment<accumulator, 8, 8, 128, int> &d,
    const fragment<matrix_a, 8, 8, 128, experimental::precision::b1, row_major>
        &a,
    const fragment<matrix_b, 8, 8, 128, experimental::precision::b1, col_major>
        &b,
    const fragment<accumulator, 8, 8, 128, int> &c) {
#if __CUDACC_VER_MAJOR__ > 13 ||                                               \
    (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)
  asm("wmma.mma.and.popc.sync.aligned.row.col.m8n8k128.s32.b1.b1.s32"
      " {%0, %1}, {%2}, {%3}, {%4, %5};"
      : "=r"(d.x[0]), "=r"(d.x[1])
      : "r"(a.x[0]), "r"(b.x[0]), "r"(c.x[0]), "r"(c.x[1]));
#else
  bmma_sync(d, a, b, c, experimental::bmmaBitOpAND);
#endif
}
