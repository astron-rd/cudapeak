#define BITS 11
#define NR_ENTRIES (1 << (BITS))

/*
 *  Lookup table contains sine values in the range [0 ... (2*PI)]
 */

// Floating-point PI values
#define PI float(M_PI)
#define TWO_PI float(2 * M_PI)
#define HLF_PI float(M_PI_2)

// Integer representations of PI
#define TWO_PI_INT NR_ENTRIES
#define PI_INT TWO_PI_INT / 2
#define HLF_PI_INT TWO_PI_INT / 4

//__constant__ float c_cosisin_table[NR_ENTRIES];
__shared__ float s_cosisin_table[NR_ENTRIES];

__host__ void cosisin_init(float *lookup) {
  for (unsigned i = 0; i < NR_ENTRIES; i++) {
    lookup[i] = sinf(i * (TWO_PI / TWO_PI_INT));
  }
}

inline __device__ void cosisin(const float x, float *sin, float *cos) {
  unsigned index = __float2uint_rn(x * (TWO_PI_INT / TWO_PI));
  index &= (TWO_PI_INT - 1);
  *cos = s_cosisin_table[(index + HLF_PI_INT) & (TWO_PI_INT - 1)];
  *sin = s_cosisin_table[index];
}
