#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace latency_benchmark
{
    using MemoryAddress = void*;

    namespace detail
    {
        [[nodiscard]] inline std::vector<std::size_t> generate_random_permutation(const std::size_t num_elements,
                                                                                  const std::uint64_t seed)
        {
            auto indices = std::vector<std::size_t>(num_elements);
            std::iota(indices.begin(), indices.end(), 0);

            auto rng = std::mt19937_64(seed);
            std::shuffle(indices.begin(), indices.end(), rng);

            return indices;
        }

        [[nodiscard]] inline MemoryAddress* get_element_location(MemoryAddress* const buffer, const std::size_t index,
                                                                 const std::size_t padded_bytes_per_element) noexcept
        {
            auto* const element_ptr = reinterpret_cast<unsigned char*>(buffer) + (index * padded_bytes_per_element);
            return reinterpret_cast<MemoryAddress*>(element_ptr);
        }
    }  // namespace detail

    [[nodiscard]] inline MemoryAddress generate_random_pointer_chasing(MemoryAddress* const buffer,
                                                                       const std::size_t num_elements,
                                                                       const std::size_t padded_bytes_per_element,
                                                                       const std::uint64_t seed)
    {
        if (buffer == nullptr || num_elements == 0)
        {
            return nullptr;
        }

        if (reinterpret_cast<std::uintptr_t>(buffer) % alignof(MemoryAddress) != 0)
        {
            throw std::invalid_argument("`buffer` must be aligned to " + std::to_string(alignof(MemoryAddress)) +
                                        " bytes.");
        }

        if (padded_bytes_per_element % alignof(MemoryAddress) != 0)
        {
            throw std::invalid_argument("`padded_bytes_per_element` must be a multiple of " +
                                        std::to_string(alignof(MemoryAddress)) + ".");
        }

        if (padded_bytes_per_element < sizeof(MemoryAddress))
        {
            throw std::invalid_argument("`padded_bytes_per_element` must be at least " +
                                        std::to_string(sizeof(MemoryAddress)) + ".");
        }

        const auto indices = detail::generate_random_permutation(num_elements, seed);

        // Link elements according to the shuffled indices
        for (std::size_t i = 0; i < num_elements - 1; ++i)
        {
            // indices[i] -> indices[i+1]
            MemoryAddress* current_ptr = detail::get_element_location(buffer, indices[i], padded_bytes_per_element);
            MemoryAddress* next_ptr = detail::get_element_location(buffer, indices[i + 1], padded_bytes_per_element);
            *current_ptr = reinterpret_cast<MemoryAddress>(next_ptr);
        }

        // indices[num_elements-1] -> indices[0]
        MemoryAddress* last_ptr =
            detail::get_element_location(buffer, indices[num_elements - 1], padded_bytes_per_element);
        MemoryAddress* first_ptr = detail::get_element_location(buffer, indices[0], padded_bytes_per_element);
        *last_ptr = reinterpret_cast<MemoryAddress>(first_ptr);

        // Return the entry point of the cyclic list
        return reinterpret_cast<MemoryAddress>(first_ptr);
    }

}  // namespace latency_benchmark
