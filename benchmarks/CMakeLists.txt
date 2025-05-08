file(GLOB BENCHMARK_SOURCES *.cpp)
list(APPEND BENCHMARK_SOURCES stream.cu)

if(BUILD_WITH_HIP)
  set(BENCHMARK_SOURCES fp16.cpp fp32.cpp mma.cpp stream.cu)
endif()

foreach(source_file ${BENCHMARK_SOURCES})
  get_filename_component(executable_name ${source_file} NAME_WE)
  get_filename_component(kernel_file ${executable_name}.cu NAME)
  set(KERNEL_FILE "${CMAKE_SOURCE_DIR}/kernels/${kernel_file}")
  add_executable(${executable_name} ${source_file} ${KERNEL_FILE})
  target_include_directories(${executable_name} PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(${executable_name} $<TARGET_OBJECTS:common>
                        $<TARGET_OBJECTS:launch> cudawrappers::cu cxxopts)
  if(NOT BUILD_WITH_HIP AND BUILD_KERNEL_DEBUG)
    target_compile_options(
      ${executable_name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: -g
                                 --generate-line-info >)
  endif()
  if(BUILD_WITH_HIP)
    if(source_file STREQUAL "stream.cu")
      set_source_files_properties(${source_file} PROPERTIES LANGUAGE HIP)
    endif()
    set_source_files_properties(${KERNEL_FILE} PROPERTIES LANGUAGE HIP)
  endif()
endforeach()
