#include "allocation_probe.h"
#include <cstdlib>
#include <new>
#ifdef _WIN32
#include <malloc.h>
#endif
namespace nfx { thread_local bool trackAllocations = false; thread_local std::uint64_t allocationCount = 0; }
void* operator new(std::size_t size) {
    if (nfx::trackAllocations) ++nfx::allocationCount;
    if (void* p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size) { return ::operator new(size); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t size, std::align_val_t alignment) {
    if (nfx::trackAllocations) ++nfx::allocationCount;
#ifdef _WIN32
    void* p = _aligned_malloc(size ? size : 1, static_cast<std::size_t>(alignment));
#else
    void* p = nullptr;
    if (posix_memalign(&p, static_cast<std::size_t>(alignment), size ? size : 1) != 0) p = nullptr;
#endif
    if (p) return p;
    throw std::bad_alloc();
}
void* operator new[](std::size_t size, std::align_val_t alignment) { return ::operator new(size, alignment); }
void operator delete(void* p, std::align_val_t) noexcept {
#ifdef _WIN32
    _aligned_free(p);
#else
    std::free(p);
#endif
}
void operator delete[](void* p, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete(void* p, std::size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete[](void* p, std::size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }
