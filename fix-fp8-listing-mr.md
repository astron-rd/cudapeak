# Enable FP8 mma kernels on Tegra Blackwell (sm_110, sm_121)

## Problem

The FP8 mma kernels (`mma_e4m3_16_8_32`, `mma_e5m2_16_8_32`) never run on Thor or GB10. `Benchmark::isBlackwell()` in `common/Benchmark.cpp` matches sm_100/sm_101/sm_120 but not sm_110/sm_121, so `getSupportedKernels()` in `benchmarks/mma.cpp` never lists them there. The kernel sources themselves already support these architectures (the `ENABLE_FP8` gate in `kernels/mma_nvidia.cuh` includes 1100/1210 since #21), so the gate in `Benchmark.cpp` is the only thing missing.

This is why `blackwell-Spark-mma.json` in cudapeak-results has no FP8 entries. Thor has them because that measurement predates the refactor.

## Fix

- `Benchmark::isBlackwell()`: add sm_110 and sm_121.
- New `Benchmark::isTegra()`: sm_110/sm_121. FP4/FP6 block_scale has no hardware support there and the kernels compile to empty bodies that report bogus performance (3.3M TOps). `benchmarks/mma.cpp` now keeps FP4/FP6 out of the list on Tegra instead.

## Verification

Measured on the real hardware, hostname and identity printed:

```
Device 0: NVIDIA GB10 (sm_121, 48 SMs, 2.418 Ghz)   [spark01/spark02]
    mma_e4m3_16_8_32:  105.11 ms,  251.05 TOps/s
    mma_e5m2_16_8_32:  105.12 ms,  251.03 TOps/s

Device 0: NVIDIA Thor (sm_110, 20 SMs, 1.04858 Ghz)  [thor01]
    mma_e4m3_16_8_32:   64.19 ms,  171.28 TOps/s
    mma_e5m2_16_8_32:   64.19 ms,  171.29 TOps/s
```

On GB10 FP8 runs at exactly 2x the fp16 mma rate (127 to 251 TOps), which is what the hardware should do. The Thor values reproduce the numbers already in cudapeak-results (171.29) to 0.01. FP4/FP6 kernels are no longer listed on either device.
