add_executable(cache_throughput
    src/cache_throughput.cpp
    src/utils.hpp
)

target_link_libraries(cache_throughput PRIVATE
    micro_benchmark_common
    perf_counter::perf_counter
    OpenMP::OpenMP_CXX
)
