#include "common.hpp"
#include "perf_counter.h"
#include "utils.hpp"

#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <vector>

namespace cache_throughput
{
    constexpr auto CYCLES_EVENT = "CYCLES";
#ifdef __znver2__
    constexpr auto LOADS_EVENT = "amd64_fam17h_zen2::LS_DISPATCH:LD_DISPATCH";
    constexpr auto LOAD_QUEUE_STALLS_EVENT =
        "amd64_fam17h_zen2::DISPATCH_RESOURCE_STALL_CYCLES_1:LOAD_QUEUE_RSRC_STALL";
    constexpr auto DRAM_REFILLS_EVENT =
        "amd64_fam17h_zen2::DATA_CACHE_REFILLS_FROM_SYSTEM"
        ":LS_MABRESP_LCL_DRAM"
        ":LS_MABRESP_RMT_DRAM";
#else
    constexpr auto DRAM_REFILLS_EVENT = "LLC-LOAD-MISSES";
#endif

    struct BenchmarkResult
    {
        std::uint64_t elapsed_cycles = std::numeric_limits<std::uint64_t>::max();
        std::uint64_t total_cycles = 0;
        std::uint64_t load_ops = 0;
        std::uint64_t load_queue_stalls = 0;
        std::uint64_t dram_refills = 0;
    };

    struct alignas(std::hardware_destructive_interference_size) ThreadResult
    {
        std::uint64_t cycles = 0;
        std::uint64_t loads = 0;
        std::uint64_t load_queue_stalls = 0;
        std::uint64_t dram_refills = 0;
    };
    static_assert(alignof(ThreadResult) == std::hardware_destructive_interference_size);

    using BufferPtr = std::unique_ptr<std::uint64_t, void (*)(void*)>;

    void load_kernel(const std::uint64_t* const buffer, const std::size_t num_elements, const std::size_t num_reps)
    {
        constexpr auto UNROLL_COUNT = std::size_t{16};
        constexpr auto SIMD_ELEMENTS = sizeof(__m256i) / sizeof(std::uint64_t);

        for (std::size_t r = 0; r < num_reps; ++r)
        {
            for (std::size_t i = 0; i < num_elements; i += UNROLL_COUNT * SIMD_ELEMENTS)
            {
                for (std::size_t k = 0; k < UNROLL_COUNT; ++k)
                {
                    const auto v = _mm256_load_si256((const __m256i*)&buffer[i + (k * SIMD_ELEMENTS)]);
                    asm volatile("" : : "x"(v));
                }
            }
        }
    }

    void print_csv_header()
    {
        std::cout
            << "Threads,BufferSize,ElapsedCycles,TotalCycles,LoadOps,BytesPerLoad,LoadQueueStallCycles,DRAMRefills\n";
    }

    template <std::size_t BYTES_PER_LOAD>
    void print_csv_row(const std::int32_t num_threads, const std::size_t buffer_size, const BenchmarkResult& result)
    {
        std::cout << num_threads << "," << buffer_size << "," << result.elapsed_cycles << "," << result.total_cycles
                  << "," << result.load_ops << "," << BYTES_PER_LOAD << "," << result.load_queue_stalls << ","
                  << result.dram_refills << "\n";
    }

    void update_best_result(const std::vector<ThreadResult>& thread_results, BenchmarkResult& result)
    {
        const auto max_cycles = std::accumulate(thread_results.begin(), thread_results.end(), std::uint64_t{0},
                                                [](const std::uint64_t max_cycles, const ThreadResult& thread_result) {
                                                    return std::max(max_cycles, thread_result.cycles);
                                                });

        if (max_cycles > 0 && max_cycles < result.elapsed_cycles)
        {
            auto total_cycles = std::uint64_t{0};
            auto total_loads = std::uint64_t{0};
            auto total_load_queue_stalls = std::uint64_t{0};
            auto total_dram_refills = std::uint64_t{0};

            for (const auto& thread_result : thread_results)
            {
                total_cycles += thread_result.cycles;
                total_loads += thread_result.loads;
                total_load_queue_stalls += thread_result.load_queue_stalls;
                total_dram_refills += thread_result.dram_refills;
            }

            result.elapsed_cycles = max_cycles;
            result.total_cycles = total_cycles;
            result.load_ops = total_loads;
            result.load_queue_stalls = total_load_queue_stalls;
            result.dram_refills = total_dram_refills;
        }
    }

    void run_benchmark(const std::vector<BufferPtr>& thread_local_buffers, const std::size_t buffer_size,
                       const std::int32_t num_threads)
    {
        constexpr auto NUM_WARMUPS = std::int32_t{3};
        constexpr auto NUM_TRIALS = std::int32_t{20};

        constexpr auto TOTAL_BYTES_TO_LOAD = 1 * common::GiB;
        constexpr auto BYTES_PER_LOAD = std::size_t{32};

        const auto num_reps = (TOTAL_BYTES_TO_LOAD + buffer_size - 1) / buffer_size;
        const auto num_elements = buffer_size / sizeof(std::uint64_t);

        BenchmarkResult result;

        const auto open_counter = [](const char* name, const std::int32_t group_fd) {
            const auto counter = perf_counter_open_by_name(name, group_fd);
            if (!perf_counter_is_valid(&counter))
            {
                std::cerr << "Error: Failed to open performance counter for event '" << name << "'.\n";
            }
            return counter;
        };

        std::vector<ThreadResult> thread_results(num_threads);

#pragma omp parallel num_threads(num_threads)
        {
            const auto tid = omp_get_thread_num();
            const auto* const buffer = thread_local_buffers[tid].get();

            auto cycle_counter = open_counter(CYCLES_EVENT, -1);
            auto load_counter = open_counter(LOADS_EVENT, cycle_counter.fd);
            auto load_queue_stall_counter = open_counter(LOAD_QUEUE_STALLS_EVENT, cycle_counter.fd);
            auto dram_refill_counter = open_counter(DRAM_REFILLS_EVENT, cycle_counter.fd);

            if (perf_counter_is_valid(&cycle_counter))
            {
                perf_counter_enable(&cycle_counter);

                for (std::int32_t i = 0; i < NUM_WARMUPS + NUM_TRIALS; ++i)
                {
                    load_kernel(buffer, num_elements, 1);
#pragma omp barrier
                    const auto start_cycles = perf_counter_read(&cycle_counter);
                    const auto start_loads = perf_counter_read(&load_counter);
                    const auto start_stalls = perf_counter_read(&load_queue_stall_counter);
                    const auto start_refills = perf_counter_read(&dram_refill_counter);

                    load_kernel(buffer, num_elements, num_reps);

                    const auto current_cycles = perf_counter_read(&cycle_counter) - start_cycles;
                    const auto current_loads = perf_counter_read(&load_counter) - start_loads;
                    const auto current_stalls = perf_counter_read(&load_queue_stall_counter) - start_stalls;
                    const auto current_refills = perf_counter_read(&dram_refill_counter) - start_refills;

                    thread_results[tid] = {current_cycles, current_loads, current_stalls, current_refills};
#pragma omp barrier
#pragma omp single
                    {
                        if (i >= NUM_WARMUPS)
                        {
                            update_best_result(thread_results, result);
                        }
                    }
                }

                perf_counter_disable(&cycle_counter);
            }

            perf_counter_close(&dram_refill_counter);
            perf_counter_close(&load_queue_stall_counter);
            perf_counter_close(&load_counter);
            perf_counter_close(&cycle_counter);
        }

        print_csv_row<BYTES_PER_LOAD>(num_threads, buffer_size, result);
    }
}  // namespace cache_throughput

int main()
{
    using namespace cache_throughput;

    print_csv_header();

    try
    {
        const auto run_benchmark_for_threads = [](const std::vector<BufferPtr>& thread_local_buffers,
                                                  const std::int32_t num_threads, const std::size_t min_size,
                                                  const std::size_t max_size) {
            for (auto size = min_size; size <= max_size; size *= 2)
            {
                run_benchmark(thread_local_buffers, size, num_threads);
            }
        };

        const auto max_threads = omp_get_max_threads();
        std::vector<BufferPtr> thread_local_buffers;
        thread_local_buffers.reserve(max_threads);
        for (std::int32_t i = 0; i < max_threads; ++i)
        {
            thread_local_buffers.emplace_back(nullptr, std::free);
        }

        constexpr auto MIN_BUFFER_SIZE = 8 * common::KiB;
        const auto max_buffer_size = memory_throughput::get_cache_size(3);

#pragma omp parallel num_threads(max_threads)
        {
            const auto tid = omp_get_thread_num();
            auto buffer = common::allocate_aligned_buffer<std::uint64_t>(max_buffer_size, common::get_hugepage_size());
            if (madvise(static_cast<void*>(buffer.get()), max_buffer_size, MADV_HUGEPAGE) != 0)
            {
                std::cerr << "Warning: madvise(MADV_HUGEPAGE) failed: " << std::strerror(errno) << "\n";
            }
            std::memset(buffer.get(), 1, max_buffer_size);
            thread_local_buffers[tid] = std::move(buffer);
        }

        for (std::int32_t num_threads = 1; num_threads < max_threads; num_threads *= 2)
        {
            run_benchmark_for_threads(thread_local_buffers, num_threads, MIN_BUFFER_SIZE, max_buffer_size);
        }
        run_benchmark_for_threads(thread_local_buffers, max_threads, MIN_BUFFER_SIZE, max_buffer_size);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
