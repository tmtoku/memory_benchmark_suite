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

#define REP10(x) x x x x x x x x x x
#define REP100(x) REP10(REP10(x))
#define REP1000(x) REP10(REP100(x))

namespace memory_latency
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

    [[nodiscard]] inline MemoryAddress* generate_random_pointer_chasing(MemoryAddress* const buffer,
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
            MemoryAddress* const current_ptr =
                detail::get_element_location(buffer, indices[i], padded_bytes_per_element);
            MemoryAddress* const next_ptr =
                detail::get_element_location(buffer, indices[i + 1], padded_bytes_per_element);
            *current_ptr = reinterpret_cast<MemoryAddress>(next_ptr);
        }

        // indices[num_elements-1] -> indices[0]
        MemoryAddress* const last_ptr =
            detail::get_element_location(buffer, indices[num_elements - 1], padded_bytes_per_element);
        MemoryAddress* const first_ptr = detail::get_element_location(buffer, indices[0], padded_bytes_per_element);
        *last_ptr = reinterpret_cast<MemoryAddress>(first_ptr);

        // Return the entry point of the cyclic list
        return first_ptr;
    }

    template <std::int32_t NUM_STEPS>
    MemoryAddress* walk_pointer_chain(MemoryAddress* const start_ptr)
    {
        constexpr auto UNROLL_COUNT = std::int32_t{1000};
        static_assert(NUM_STEPS % UNROLL_COUNT == 0, "`NUM_STEPS` must be a multiple of `UNROLL_COUNT`");

        auto* current_ptr = start_ptr;
        for (std::int32_t i = 0; i < NUM_STEPS; i += UNROLL_COUNT)
        {
            // current_ptr = *current_ptr
            __asm__ volatile(REP1000("mov (%0), %0\n\t") : "+r"(current_ptr) : : "memory");
        }
        return current_ptr;
    }

}  // namespace memory_latency

#undef REP1000
#undef REP100
#undef REP10
