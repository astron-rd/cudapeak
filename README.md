# cudapeak

This program measures peak performance of a CUDA or HIP capable device.

## Building

```
cmake -S . -B build
cmake --build build
```

With CMake 3.25 or newer the presets in `CMakePresets.json` do the same and keep a build directory per configuration (`build-<preset>`), which is handy when the same checkout is built for both CUDA and HIP:

```
cmake --preset hip          # or cuda
cmake --build --preset hip -j
```

Extra options can be added to the first command, for example `cmake --preset hip -DBUILD_WITH_PMT=ON`.

Every benchmark is also registered as a CTest test that runs it for one iteration, as a smoke test (it needs a GPU, the tests are labeled `gpu`, and they always run one at a time, also with `ctest -j`, because benchmarks disturb each other when they share a GPU). With CMake 3.25 or newer one command configures, builds and tests:

```
cmake --workflow --preset hip
```

or, in an existing build, `ctest --preset hip`. The benchmarks print their results when run directly, see `--help`.

To run a single test, select it by name with `-R`, which takes a regular expression. The test names are the executable names, for example `benchmark-fp32`, and `benchmark-fp32` also matches `benchmark-fp32-sincos-sfu`, so anchor the name to run only that one:

```
ctest --preset hip -N                          # list the tests without running them
ctest --preset hip -R '^benchmark-fp32$'       # only benchmark-fp32
ctest --preset hip -R 'sincos'                 # all sincos benchmarks
```

The tests have to be built first (`cmake --build --preset hip -j` or `--target benchmark-fp32`). A test runs the benchmark for one iteration. The output of all benchmarks of a `ctest` run is collected in one file, `build-<preset>/benchmarks/results-<host>-<timestamp>.txt` (also available as `results-latest.txt`), which starts with the date, host and commit and has a section per benchmark. It is created by the test `benchmark-results-file`, which `ctest` runs first, also when only some of the tests are selected. `ctest` itself only shows whether the tests passed, and the error output of a test that failed. To choose the number of iterations or other options run the executable directly, for example `build-hip/benchmarks/benchmark-fp32 --nr_iterations 10`.

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
