// alloc_stats.hpp -- memory and CPU accounting for one solver run.
//
// src/alloc_stats.cpp replaces the global operator new / delete of the
// sovereign_solve executable (not of the library or the tests): every heap
// allocation made through C++ -- by the engines, the standard containers and
// the CUDA host code alike -- is counted with atomic counters, so the numbers
// are exact and thread-safe. Each block carries a 16-byte size header.
// Allocations made directly by the OS or by driver DLLs (e.g. GPU device
// memory) are not included; the OS figures below cover the whole process.
#pragma once
#include <cstdint>

struct AllocStats {
    long long allocations = 0;      // number of operator new calls
    long long bytes_allocated = 0;  // total bytes requested over the run
    long long live_bytes = 0;       // bytes still allocated now
    long long peak_live_bytes = 0;  // largest live heap at any moment
};
AllocStats alloc_stats();

struct ProcessStats {
    long long peak_working_set = 0; // bytes: largest physical memory footprint (RSS)
    long long peak_private = 0;     // bytes: largest committed private memory (0 where unknown)
    double cpu_user = 0, cpu_kernel = 0;   // seconds of CPU time, all threads
};
ProcessStats process_stats();
