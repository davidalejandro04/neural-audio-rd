#pragma once
#include <cstdint>
namespace nfx {
extern thread_local bool trackAllocations;
extern thread_local std::uint64_t allocationCount;
}
