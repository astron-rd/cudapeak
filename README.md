# cudapeak

This program measures peak performance of a CUDA or HIP capable device.

## Building

```
cmake -S . -B build
cmake --build build
```

Options:

| Option | Description |
| --- | --- |
| `BUILD_WITH_HIP` | Build for AMD GPUs with HIP instead of CUDA. Defaults to `ON` when no CUDA toolkit is found but a HIP compiler is available, otherwise `OFF`. Pass `-DBUILD_WITH_HIP=ON` or `OFF` to override. |
| `BUILD_WITH_PMT` | Enable `--measure_power` (Power Measurement Toolkit) |
| `BUILD_WITH_FMT` | Enable `--measure_frequency` (Frequency Measurement Toolkit) |

The `BUILD_WITH_HIP` default is only evaluated when a build directory is first configured. An already-configured build directory keeps its cached value, so use a fresh build directory (or pass the flag explicitly) to re-trigger detection.

### AMD GPUs

`BUILD_WITH_PMT` and `BUILD_WITH_FMT` together with `BUILD_WITH_HIP` need the
AMD SMI library (headers and `libamd_smi.so`), which is not always installed
with ROCm. On Arch it is the `amdsmi` package. On Debian, install
`libamd-smi-dev` (it pulls in the library); `amd-smi-lib` is the equivalent
package from the AMD ROCm apt repository. Without it, CMake fails with "Could
not find a package configuration file provided by amd_smi".

Not every AMD GPU reports power and clock through AMD SMI (for example some
APUs). In that case `--measure_power` or `--measure_frequency` prints an error
and exits; run without these options.
