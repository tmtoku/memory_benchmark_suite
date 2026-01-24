#pragma once

#include <unistd.h>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>

#ifdef __GNUC__
#define FORCE_INLINE __attribute__((always_inline)) inline
#else
#define FORCE_INLINE inline
#endif

namespace common
{
    constexpr auto KiB = std::size_t{1024};
    constexpr auto MiB = std::size_t{1024} * KiB;
    constexpr auto GiB = std::size_t{1024} * MiB;

    [[nodiscard]] inline std::size_t get_cache_line_bytes()
    {
        const auto cache_line_bytes = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
        if (cache_line_bytes < 1)
        {
            throw std::runtime_error("Failed to get the cache line size.");
        }

        return static_cast<std::size_t>(cache_line_bytes);
    }

    [[nodiscard]] inline std::size_t get_page_size()
    {
        const auto page_size = sysconf(_SC_PAGESIZE);
        if (page_size < 1)
        {
            throw std::runtime_error("Failed to get the page size.");
        }
        return static_cast<std::size_t>(page_size);
    }

    [[nodiscard]] inline std::size_t get_hugepage_size()
    {
        static const auto hugepage_size = []() {
            std::ifstream ifs("/proc/meminfo");
            std::string line;
            if (!ifs.is_open())
            {
                throw std::runtime_error("Failed to open /proc/meminfo to get the hugepage size.");
            }
            while (std::getline(ifs, line))
            {
                if (line.find("Hugepagesize:") != std::string::npos)
                {
                    std::stringstream ss(line);
                    std::string label;
                    std::size_t value = 0;
                    std::string unit;

                    ss >> label >> value >> unit;

                    if (unit == "kB")
                    {
                        return value * KiB;
                    }
                    if (unit == "MB")
                    {
                        return value * MiB;
                    }
                    throw std::runtime_error("Unknown unit for Hugepagesize in /proc/meminfo: " + unit);
                }
            }
            throw std::runtime_error("Could not find 'Hugepagesize' entry in /proc/meminfo.");
        }();

        return hugepage_size;
    }

    template <typename T>
    [[nodiscard]] std::unique_ptr<T, void (*)(void*)> allocate_aligned_buffer(const std::size_t buffer_size_in_bytes,
                                                                              const std::size_t alignment_bytes)
    {
        if (alignment_bytes == 0 || (alignment_bytes & (alignment_bytes - 1)) != 0)
        {
            throw std::invalid_argument("`alignment_bytes` must be a power of 2.");
        }

        if (alignment_bytes % sizeof(void*) != 0)
        {
            throw std::invalid_argument("`alignment_bytes` must be a multiple of " + std::to_string(sizeof(void*)) +
                                        ".");
        }

        const auto alignment_mask = ~(alignment_bytes - 1);
        const auto size = (buffer_size_in_bytes + (alignment_bytes - 1)) & alignment_mask;

        void* const buffer = std::aligned_alloc(alignment_bytes, size);
        if (buffer == nullptr)
        {
            throw std::bad_alloc();
        }

        return {static_cast<T*>(buffer), std::free};
    }
}  // namespace common
