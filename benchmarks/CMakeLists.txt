file(GLOB BENCHMARK_SOURCES *.cpp)

if(BUILD_WITH_HIP)
  set(BENCHMARK_SOURCES fp16.cpp fp32.cpp mma.cpp stream.cu)
endif()

foreach(source_file ${BENCHMARK_SOURCES})
  get_filename_component(EXECUTABLE_NAME ${source_file} NAME_WE)
  get_filename_component(kernel_file ${EXECUTABLE_NAME}.cu NAME)
  set(EXECUTABLE_NAME "benchmark-${EXECUTABLE_NAME}")
  add_executable(${EXECUTABLE_NAME} ${source_file})
  target_include_directories(${EXECUTABLE_NAME} PRIVATE ${CMAKE_SOURCE_DIR})
  target_link_libraries(
    ${EXECUTABLE_NAME} PRIVATE $<TARGET_OBJECTS:common> cudawrappers::cu
                               cudawrappers::nvrtc cxxopts)
  set(KERNEL_FILE "${CMAKE_SOURCE_DIR}/kernels/${kernel_file}")
  target_embed_source(${EXECUTABLE_NAME} ${KERNEL_FILE})
endforeach()
