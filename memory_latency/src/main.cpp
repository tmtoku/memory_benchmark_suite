#include "common.hpp"
#include "perf_counter.h"
#include "utils.hpp"

#include <sys/mman.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>

namespace memory_latency
{
    constexpr auto CYCLES_EVENT = "CYCLES";
    constexpr auto TLB_MISS_EVENT = "DTLB-LOAD-MISSES";
#ifdef __znver2__
    constexpr auto L1D_MISS_EVENT =
        "amd64_fam17h_zen2::DATA_CACHE_REFILLS_FROM_SYSTEM"
        ":MABRESP_LCL_L2"
        ":LS_MABRESP_LCL_CACHE"
        ":LS_MABRESP_LCL_DRAM"
        ":LS_MABRESP_RMT_CACHE"
        ":LS_MABRESP_RMT_DRAM";
    constexpr auto L2_MISS_EVENT =
        "amd64_fam17h_zen2::DATA_CACHE_REFILLS_FROM_SYSTEM"
        ":LS_MABRESP_LCL_CACHE"
        ":LS_MABRESP_LCL_DRAM"
        ":LS_MABRESP_RMT_CACHE"
        ":LS_MABRESP_RMT_DRAM";
    constexpr auto L3_MISS_EVENT =
        "amd64_fam17h_zen2::DATA_CACHE_REFILLS_FROM_SYSTEM"
        ":LS_MABRESP_LCL_DRAM"
        ":LS_MABRESP_RMT_DRAM";
#else
    constexpr auto L1D_MISS_EVENT = "L1-DCACHE-LOAD-MISSES";
    constexpr auto L2_MISS_EVENT = "LLC-LOAD-MISSES";
    constexpr auto L3_MISS_EVENT = "LLC-LOAD-MISSES";
#endif

    struct BenchmarkResult
    {
        std::uint64_t cycle_count = std::numeric_limits<uint64_t>::max();
        std::uint64_t l1d_miss_count = 0;
        std::uint64_t l2_miss_count = 0;
        std::uint64_t l3_miss_count = 0;
        std::uint64_t tlb_miss_count = 0;
    };

    void print_csv_header()
    {
        std::cout
            << "BufferSize,PaddedElementSize,PageSize,NumLogicalLoads,Cycles,L1DMisses,L2Misses,L3Misses,TLBMisses\n";
    }

    void print_csv_row(const std::size_t buffer_size, const std::size_t padded_element_size,
                       const std::size_t page_size, const std::int32_t num_logical_loads, const BenchmarkResult& result)
    {
        std::cout << buffer_size << "," << padded_element_size << "," << page_size << "," << num_logical_loads << ","
                  << result.cycle_count << "," << result.l1d_miss_count << "," << result.l2_miss_count << ","
                  << result.l3_miss_count << "," << result.tlb_miss_count << "\n";
    }

    void run_benchmark(const std::size_t buffer_size_in_bytes, const std::size_t padded_bytes_per_element,
                       const bool use_hugepage)
    {
        constexpr auto NUM_LOGICAL_LOADS = std::int32_t{1'000'000};
        constexpr auto NUM_TRIALS = std::int32_t{10};
        constexpr auto NUM_WARMUPS = std::int32_t{3};
        constexpr auto RAND_SEED = std::uint64_t{12345};

        if (buffer_size_in_bytes % padded_bytes_per_element != 0)
        {
            std::cerr << "Error: `buffer_size_in_bytes` must be a multiple of `padded_bytes_per_element`\n";
            return;
        }
        const auto num_elements = buffer_size_in_bytes / padded_bytes_per_element;
        const auto page_size = use_hugepage ? common::get_hugepage_size() : common::get_page_size();

        auto buffer = common::allocate_aligned_buffer<MemoryAddress>(buffer_size_in_bytes, page_size);

        if (use_hugepage)
        {
            if (madvise(static_cast<void*>(buffer.get()), buffer_size_in_bytes, MADV_HUGEPAGE) != 0)
            {
                std::cerr << "Warning: madvise(MADV_HUGEPAGE) failed: " << std::strerror(errno) << "\n";
            }
        }
        else
        {
            if (madvise(static_cast<void*>(buffer.get()), buffer_size_in_bytes, MADV_NOHUGEPAGE) != 0)
            {
                std::cerr << "Warning: madvise(MADV_NOHUGEPAGE) failed: " << std::strerror(errno) << "\n";
            }
        }

        auto* const start_ptr =
            generate_random_pointer_chasing(buffer.get(), num_elements, padded_bytes_per_element, RAND_SEED);

        const auto open_counter = [](const char* name, const std::int32_t group_fd) {
            const auto counter = perf_counter_open_by_name(name, group_fd);
            if (!perf_counter_is_valid(&counter))
            {
                std::cerr << "Error: Failed to open performance counter for event '" << name << "'.\n";
            }
            return counter;
        };

        auto cycle_counter = open_counter(CYCLES_EVENT, -1);
        if (!perf_counter_is_valid(&cycle_counter))
        {
            return;
        }

        const auto group_fd = cycle_counter.fd;

        auto l1d_miss_counter = open_counter(L1D_MISS_EVENT, group_fd);
        auto l2_miss_counter = open_counter(L2_MISS_EVENT, group_fd);
        auto l3_miss_counter = open_counter(L3_MISS_EVENT, group_fd);
        auto tlb_miss_counter = open_counter(TLB_MISS_EVENT, group_fd);

        if (!perf_counter_is_valid(&l1d_miss_counter) || !perf_counter_is_valid(&l2_miss_counter) ||
            !perf_counter_is_valid(&l3_miss_counter) || !perf_counter_is_valid(&tlb_miss_counter))
        {
            const auto close_counter = [](perf_counter* const counter) {
                if (perf_counter_is_valid(counter))
                {
                    perf_counter_close(counter);
                }
            };

            close_counter(&l1d_miss_counter);
            close_counter(&l2_miss_counter);
            close_counter(&l3_miss_counter);
            close_counter(&tlb_miss_counter);
            close_counter(&cycle_counter);

            return;
        }

        perf_counter_enable(&cycle_counter);

        auto* volatile kernel = walk_pointer_chain<NUM_LOGICAL_LOADS>;

        BenchmarkResult result;

        for (std::int32_t i = 0; i < NUM_WARMUPS + NUM_TRIALS; ++i)
        {
            const auto start_l1d_misses = perf_counter_read(&l1d_miss_counter);
            const auto start_l2_misses = perf_counter_read(&l2_miss_counter);
            const auto start_l3_misses = perf_counter_read(&l3_miss_counter);
            const auto start_tlb_misses = perf_counter_read(&tlb_miss_counter);

            const auto start_cycles = perf_counter_read(&cycle_counter);

            kernel(start_ptr);

            const auto end_cycles = perf_counter_read(&cycle_counter);

            const auto end_tlb_misses = perf_counter_read(&tlb_miss_counter);
            const auto end_l3_misses = perf_counter_read(&l3_miss_counter);
            const auto end_l2_misses = perf_counter_read(&l2_miss_counter);
            const auto end_l1d_misses = perf_counter_read(&l1d_miss_counter);

            const auto latency_cycles = end_cycles - start_cycles;

            if (i >= NUM_WARMUPS && latency_cycles < result.cycle_count)
            {
                result.cycle_count = latency_cycles;
                result.l1d_miss_count = end_l1d_misses - start_l1d_misses;
                result.l2_miss_count = end_l2_misses - start_l2_misses;
                result.l3_miss_count = end_l3_misses - start_l3_misses;
                result.tlb_miss_count = end_tlb_misses - start_tlb_misses;
            }
        }

        perf_counter_disable(&cycle_counter);
        perf_counter_close(&tlb_miss_counter);
        perf_counter_close(&l3_miss_counter);
        perf_counter_close(&l2_miss_counter);
        perf_counter_close(&l1d_miss_counter);
        perf_counter_close(&cycle_counter);

        print_csv_row(buffer_size_in_bytes, padded_bytes_per_element, page_size, NUM_LOGICAL_LOADS, result);
    }

}  // namespace memory_latency

int main()
{
    memory_latency::print_csv_header();

    try
    {
        const auto cache_line_bytes = common::get_cache_line_bytes();
        const auto page_size = common::get_page_size();

        for (auto size = 16 * common::KiB; size <= 1 * common::GiB; size *= 2)  // NOLINT(readability-magic-numbers)
        {
            memory_latency::run_benchmark(size, cache_line_bytes, true);

            memory_latency::run_benchmark(size, page_size, true);
            memory_latency::run_benchmark(size, page_size, false);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
