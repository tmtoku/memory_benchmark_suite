cmake_minimum_required(VERSION 3.10)
project(memory_benchmark_suite LANGUAGES C CXX)

set(MEMORY_BENCHMARK_SUITE_ARCH "native" CACHE STRING "Target architecture (e.g., native, skylake, znver2)")
message(STATUS "MEMORY_BENCHMARK_SUITE_ARCH: ${MEMORY_BENCHMARK_SUITE_ARCH}")

#
# Build type configuration
#
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, default to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()

#
# C++17 standards
#
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#
# Link-time optimization (LTO)
#
include(CheckIPOSupported)
check_ipo_supported(RESULT is_lto_supported)

if(is_lto_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
else()
    message(WARNING "Link-time optimization (LTO) is not supported by the compiler.")
endif()

#
# External libraries
#
add_subdirectory(externals/perf-counter)

#
# Common utilities
#
add_library(memory_benchmark_common INTERFACE)
target_include_directories(memory_benchmark_common INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

#
# Compile options
#
add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_options(-g -O3 -march=${MEMORY_BENCHMARK_SUITE_ARCH})

#
# Subdirectories
#
add_subdirectory(latency_benchmark)

