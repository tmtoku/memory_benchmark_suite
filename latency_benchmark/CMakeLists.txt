add_executable(latency_benchmark
    src/main.cpp
    src/utils.hpp
)

target_link_libraries(latency_benchmark PRIVATE
    memory_benchmark_common
    perf_counter::perf_counter
)
