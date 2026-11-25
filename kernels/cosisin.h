#define BITS 11
#define NR_ENTRIES (1 << (BITS))

/*
 *  Lookup table contains sine values in the range [0 ... (2*PI)]
 */

// Floating-point PI values
#define M_PI 3.14159265358979323846
#define PI float(M_PI)
#define TWO_PI float(2 * M_PI)
#define HLF_PI float(M_PI_2)

// Integer representations of PI
#define TWO_PI_INT NR_ENTRIES
#define PI_INT TWO_PI_INT / 2
#define HLF_PI_INT TWO_PI_INT / 4

__shared__ float s_cosisin_table[NR_ENTRIES];

// Fill the table, called by all threads of the block
__device__ void cosisin_init() {
  for (unsigned i = threadIdx.x; i < NR_ENTRIES; i += blockDim.x) {
    s_cosisin_table[i] = sinf(i * (TWO_PI / TWO_PI_INT));
  }
  __syncthreads();
}

inline __device__ void cosisin(const float x, float *sin, float *cos) {
  // Signed, so that negative angles wrap around
  unsigned index = __float2int_rn(x * (TWO_PI_INT / TWO_PI));
  index &= (TWO_PI_INT - 1);
  *cos = s_cosisin_table[(index + HLF_PI_INT) & (TWO_PI_INT - 1)];
  *sin = s_cosisin_table[index];
}
