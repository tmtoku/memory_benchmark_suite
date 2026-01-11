#pragma once

#include "common.hpp"

#include <cstdint>
#include <fstream>

namespace memory_throughput
{
    std::size_t get_cache_size(const std::int32_t cache_index)
    {
        const auto file_path = "/sys/devices/system/cpu/cpu0/cache/index" + std::to_string(cache_index) + "/size";
        std::ifstream ifs(file_path);
        if (!ifs.is_open())
        {
            throw std::runtime_error("Failed to open a file: " + file_path);
        }

        std::string line;
        std::getline(ifs, line);

        while (!line.empty() && std::isspace(static_cast<unsigned char>(line.back())))
        {
            line.pop_back();
        }

        if (line.empty())
        {
            throw std::runtime_error("Empty cache size file: " + file_path);
        }

        auto cache_size = std::stoul(line);
        const auto suffix = line.back();
        if (suffix == 'K')
        {
            cache_size *= common::KiB;
        }
        else if (suffix == 'M')
        {
            cache_size *= common::MiB;
        }
        return cache_size;
    }
}  // namespace memory_throughput
