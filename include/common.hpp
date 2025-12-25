#pragma once

#include <unistd.h>
#include <cstdlib>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>

namespace common
{
    [[nodiscard]] inline std::size_t get_cache_line_bytes()
    {
        const auto cache_line_bytes = sysconf(_SC_LEVEL1_DCACHE_LINESIZE);
        if (cache_line_bytes < 1)
        {
            throw std::runtime_error("Failed to get the cache line size.");
        }

        return static_cast<std::size_t>(cache_line_bytes);
    }

    template <typename T>
    [[nodiscard]] std::unique_ptr<T, void (*)(void*)> allocate_aligned_buffer(const std::size_t num_elements,
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
        const auto raw_size = num_elements * sizeof(T);
        const auto size = (raw_size + (alignment_bytes - 1)) & alignment_mask;

        void* const ptr = std::aligned_alloc(alignment_bytes, size);
        if (ptr == nullptr)
        {
            throw std::bad_alloc();
        }

        return {static_cast<T*>(ptr), std::free};
    }
}  // namespace common
