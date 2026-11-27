file(GLOB BENCHMARK_SOURCES *.cpp)

# The tests of one ctest run write to a single file,
# results-<host>-<timestamp>.txt in the build directory of the benchmarks, also
# available as results-latest.txt. The file is created by a test that all
# benchmark tests depend on (a fixture), so that they share one timestamp.
# Arguments: the directory of the results, the source directory.
set(RESULTS_FILE_SCRIPT
    [=[
dir=$1; src=$2
out="$dir/results-$(hostname -s)-$(date +%Y%m%d-%H%M%S).txt"
{
  echo "date:   $(date -Is)"
  echo "host:   $(hostname -f)"
  echo "commit: $(git -C "$src" rev-parse HEAD 2>/dev/null || echo unknown)"
} > "$out"
ln -sf "$(basename "$out")" "$dir/results-latest.txt"
]=])
add_test(NAME benchmark-results-file
         COMMAND sh -c "${RESULTS_FILE_SCRIPT}" sh ${CMAKE_CURRENT_BINARY_DIR}
                 ${CMAKE_SOURCE_DIR})
set_tests_properties(benchmark-results-file PROPERTIES FIXTURES_SETUP
                                                       benchmark-results)

# A benchmark appends its output to the results file. The exit status of the
# shell is the exit status of the benchmark. Arguments: the executable, the name
# of the benchmark, the directory of the results.
set(BENCHMARK_SCRIPT
    [=[
{ echo; echo "=== $2"; "$1" --nr_iterations 1; } >> "$3/results-latest.txt"
]=])

foreach(source_file ${BENCHMARK_SOURCES})
  get_filename_component(EXECUTABLE_NAME ${source_file} NAME_WE)
  get_filename_component(kernel_file ${EXECUTABLE_NAME}.cu NAME)
  string(REPLACE "_" "-" EXECUTABLE_NAME ${EXECUTABLE_NAME})
  set(EXECUTABLE_NAME "benchmark-${EXECUTABLE_NAME}")
  add_executable(${EXECUTABLE_NAME} ${source_file})
  target_include_directories(${EXECUTABLE_NAME} PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(
    ${EXECUTABLE_NAME}
    PRIVATE $<TARGET_OBJECTS:common> cudawrappers::cu cudawrappers::nvrtc
            cxxopts nlohmann_json::nlohmann_json)
  if(BUILD_WITH_PMT)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE pmt)
  endif()
  if(BUILD_WITH_FMT)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE fmt)
  endif()
  set(KERNEL_FILE "${CMAKE_SOURCE_DIR}/kernels/${kernel_file}")
  target_embed_source(${EXECUTABLE_NAME} ${KERNEL_FILE})

  # Smoke test: the benchmark runs to completion. Needs a GPU, run with ctest.
  # Benchmarks measure peak performance and disturb each other when they share
  # the GPU, RUN_SERIAL keeps ctest from running them in parallel (ctest -j).
  add_test(
    NAME ${EXECUTABLE_NAME}
    COMMAND sh -c "${BENCHMARK_SCRIPT}" sh $<TARGET_FILE:${EXECUTABLE_NAME}>
            ${EXECUTABLE_NAME} ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(
    ${EXECUTABLE_NAME}
    PROPERTIES LABELS
               gpu
               TIMEOUT
               900
               RUN_SERIAL
               TRUE
               FIXTURES_REQUIRED
               benchmark-results)
endforeach()
