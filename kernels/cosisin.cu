#define BITS        11
#define NR_ENTRIES  (1 << (BITS))

#if 1
/*
 * V1
 *  lookup table contains sine values in the range [0 ... (2*PI)]
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

__host__ void cosisin_init(float* lookup) {
  for (unsigned i = 0; i < NR_ENTRIES; i++) {
    lookup[i] = sinf(i * (TWO_PI / TWO_PI_INT));
  }
}

inline __device__ void cosisin(const float x, float* sin, float* cos) {
    unsigned index = __float2uint_rn(x * (TWO_PI_INT / TWO_PI));
    index &= (TWO_PI_INT - 1);
    *cos = s_cosisin_table[(index + HLF_PI_INT) & (TWO_PI_INT - 1)];
    *sin = s_cosisin_table[index];
}

#else

/*
 * V2
 *  lookup table contains sine and cosine values in the range [0 ... (PI/4)]
 */
#define BIT(A,N)    (((A) >> (N)) & 1)
#define PI          float(M_PI)

//__constant__ float2 c_cosisin_table[NR_ENTRIES];
__shared__ float2 s_cosisin_table[NR_ENTRIES];

__host__ void cosisin_init(float2* cosisin_table) {
  for (unsigned i = 0; i < NR_ENTRIES; i++) {
    float x = (i  / (float) NR_ENTRIES) * (PI / 4.0f);
    cosisin_table[i] = make_float2(cosf(x), sinf(x));
  }
}

inline __device__ void cosisin(const float x, float* sin, float* cos) {
    float x_rounded = __float2uint_rn(x);
    uint arg_int = x * (4 * NR_ENTRIES / PI);
    uint index   = arg_int & (NR_ENTRIES - 1);
    uint octant  = arg_int >> BITS;

    if (BIT(octant, 0)) {
        index = ~index & (NR_ENTRIES - 1);
    }

    float2 val = s_cosisin_table[index];
    *sin = val.x;
    *cos = val.y;

    if (BIT(octant, 0) ^ BIT(octant, 1)) {
        *sin = val.y;
        *cos = val.x;
    }

    if (BIT(octant, 1) ^ BIT(octant, 2)) {
        *sin = -*sin;
    }

    if (BIT(octant, 2)) {
        *cos = -*cos;
    }
}
#endif
