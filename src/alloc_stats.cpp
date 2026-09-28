#include "alloc_stats.hpp"
#include <atomic>
#include <cstdlib>
#include <new>

#ifdef _WIN32
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {
std::atomic<long long> g_count{0}, g_bytes{0}, g_live{0}, g_peak{0};
constexpr std::size_t kHeader = 16;   // keeps malloc's 16-byte alignment

void* counted_alloc(std::size_t n) noexcept {
    void* p = std::malloc(n + kHeader);
    if (!p) return nullptr;
    *static_cast<std::size_t*>(p) = n;
    g_count.fetch_add(1, std::memory_order_relaxed);
    g_bytes.fetch_add((long long)n, std::memory_order_relaxed);
    long long live = g_live.fetch_add((long long)n, std::memory_order_relaxed) + (long long)n;
    long long peak = g_peak.load(std::memory_order_relaxed);
    while (live > peak && !g_peak.compare_exchange_weak(peak, live, std::memory_order_relaxed)) {}
    return static_cast<char*>(p) + kHeader;
}
void counted_free(void* q) noexcept {
    if (!q) return;
    char* p = static_cast<char*>(q) - kHeader;
    g_live.fetch_sub((long long)*reinterpret_cast<std::size_t*>(p), std::memory_order_relaxed);
    std::free(p);
}
void* counted_alloc_or_throw(std::size_t n) {
    void* p = counted_alloc(n);
    if (!p) throw std::bad_alloc();
    return p;
}
} // namespace

void* operator new(std::size_t n) { return counted_alloc_or_throw(n); }
void* operator new[](std::size_t n) { return counted_alloc_or_throw(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { return counted_alloc(n); }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { return counted_alloc(n); }
void operator delete(void* p) noexcept { counted_free(p); }
void operator delete[](void* p) noexcept { counted_free(p); }
void operator delete(void* p, std::size_t) noexcept { counted_free(p); }
void operator delete[](void* p, std::size_t) noexcept { counted_free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { counted_free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { counted_free(p); }

AllocStats alloc_stats() {
    AllocStats s;
    s.allocations = g_count.load();
    s.bytes_allocated = g_bytes.load();
    s.live_bytes = g_live.load();
    s.peak_live_bytes = g_peak.load();
    return s;
}

ProcessStats process_stats() {
    ProcessStats s;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc{};
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof pmc)) {
        s.peak_working_set = (long long)pmc.PeakWorkingSetSize;
        s.peak_private = (long long)pmc.PeakPagefileUsage;
    }
    FILETIME c, e, k, u;
    if (GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) {
        auto sec = [](const FILETIME& f) { return (double)(((unsigned long long)f.dwHighDateTime << 32) | f.dwLowDateTime) * 1e-7; };
        s.cpu_kernel = sec(k);
        s.cpu_user = sec(u);
    }
#else
    rusage ru{};
    if (getrusage(RUSAGE_SELF, &ru) == 0) {
        s.peak_working_set = (long long)ru.ru_maxrss * 1024;   // Linux reports KiB
        s.cpu_user = ru.ru_utime.tv_sec + ru.ru_utime.tv_usec * 1e-6;
        s.cpu_kernel = ru.ru_stime.tv_sec + ru.ru_stime.tv_usec * 1e-6;
    }
#endif
    return s;
}
