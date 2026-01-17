#pragma once

namespace perf_events
{
    constexpr auto CYCLES = "CYCLES";
    constexpr auto TLB_MISS = "DTLB-LOAD-MISSES";
#ifdef __znver2__
    constexpr auto L1D_MISS =
        "amd64_fam17h_zen2::DATA_CACHE_REFILLS_FROM_SYSTEM"
        ":MABRESP_LCL_L2"
        ":LS_MABRESP_LCL_CACHE"
        ":LS_MABRESP_LCL_DRAM"
        ":LS_MABRESP_RMT_CACHE"
        ":LS_MABRESP_RMT_DRAM";
    constexpr auto L2_MISS =
        "amd64_fam17h_zen2::DATA_CACHE_REFILLS_FROM_SYSTEM"
        ":LS_MABRESP_LCL_CACHE"
        ":LS_MABRESP_LCL_DRAM"
        ":LS_MABRESP_RMT_CACHE"
        ":LS_MABRESP_RMT_DRAM";
    constexpr auto L3_MISS =
        "amd64_fam17h_zen2::DATA_CACHE_REFILLS_FROM_SYSTEM"
        ":LS_MABRESP_LCL_DRAM"
        ":LS_MABRESP_RMT_DRAM";
    constexpr auto LOADS =
        "amd64_fam17h_zen2::LS_DISPATCH"
        ":LD_DISPATCH";
    constexpr auto LOAD_QUEUE_STALL_CYCLES =
        "amd64_fam17h_zen2::DISPATCH_RESOURCE_STALL_CYCLES_1"
        ":LOAD_QUEUE_RSRC_STALL";
#else
    constexpr auto L1D_MISS = "L1-DCACHE-LOAD-MISSES";
    constexpr auto L2_MISS = "LLC-LOAD-MISSES";
    constexpr auto L3_MISS = "LLC-LOAD-MISSES";
    constexpr auto LOADS_EVENT = "";
    constexpr auto LOAD_QUEUE_STALL_CYCLES = "";
#endif

}  // namespace perf_events
