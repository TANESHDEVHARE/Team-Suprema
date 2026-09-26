// pdlp_gpu.hpp -- the GPU engine's entry point, only declared (and only
// compiled) when SOVEREIGN_WITH_CUDA is defined by CMake, which happens only
// when CMake's check_language(CUDA) actually finds an nvcc on this machine.
// On a machine with no CUDA toolchain, this header is never included and
// the whole GPU engine is simply absent from the build -- solve_mps()
// falls back to the single CPU path it has always had.
#pragma once
#ifdef SOVEREIGN_WITH_CUDA
#include "pdlp.hpp"
#include <atomic>

// Same contract as solve_pdhg() (src/pdlp.cpp) -- same inputs, same
// PdlpResult, same restart/convergence logic, same stop_flag cooperative
// cancellation -- so solve_mps_race() (src/solve.cpp) can run this and the
// CPU engine side by side and treat whichever finishes first identically.
// The only difference is *where* the sparse matvecs and elementwise PDHG
// updates execute: this version keeps A, x, y resident on the GPU for the
// whole solve and only copies back to the host every `check_every`
// iterations, to call the exact same host-side verify() (include/verify.hpp)
// the CPU engine uses -- both engines are judged by the same independent
// checker, at the same cadence, so a race between them is apples-to-apples.
PdlpResult solve_pdhg_gpu(const RangedLP& unscaled_form, const RangedLP& scaled_form,
                           const std::vector<double>& Dr, const std::vector<double>& Dc,
                           double eta, int max_iterations, int check_every, double tol,
                           std::atomic<bool>* stop_flag);
#endif
