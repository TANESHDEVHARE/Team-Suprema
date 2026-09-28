cmake_minimum_required(VERSION 3.18)
project(sovereign_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# -O2 is GCC/Clang syntax -- MSVC (cl.exe) silently ignores it and, because
# this used to overwrite the whole flags variable, would lose its own
# correct /O2 default entirely. Only touch it for GCC/Clang; leave MSVC's
# own Release defaults (which already include /O2 /DNDEBUG) alone.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(Threads REQUIRED)

# ---- CUDA: probe for it, never require it -------------------------------
# check_language(CUDA) tries to find a working nvcc WITHOUT failing
# configure if there isn't one -- this is what lets the exact same
# CMakeLists.txt build a CPU-only binary on a laptop with no GPU (today's
# working build) and a full CPU/GPU race binary on a CUDA-capable machine,
# with zero manual flags either way.
#
# SOVEREIGN_FORCE_CPU: deliberately skip the CUDA probe even on a machine
# that has a working nvcc. This exists for ONE reason: to get an honest
# CPU-only timing baseline on the SAME hardware as a GPU-enabled build, so
# a claimed GPU speedup is measured against a real number instead of
# against "whatever the CPU thread's time would have been" (which a race
# never reports for the losing engine). Use a SEPARATE build directory --
# e.g. `cmake -S . -B build_cpu -DCMAKE_BUILD_TYPE=Release
# -DSOVEREIGN_FORCE_CPU=ON` -- so this never overwrites your working
# GPU-enabled `build` directory.
option(SOVEREIGN_FORCE_CPU "Skip CUDA detection and force a CPU-only build, even if nvcc is found" OFF)
include(CheckLanguage)
if(NOT SOVEREIGN_FORCE_CPU)
  check_language(CUDA)
endif()
if(CMAKE_CUDA_COMPILER AND NOT SOVEREIGN_FORCE_CPU)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # 86 = compute capability 8.6 = RTX 3050 (and other Ampere laptop/desktop
  # GPUs). Plain "86" (no arch= all list) tells nvcc to emit a REAL native
  # binary for this exact GPU, not just a virtual PTX target that the
  # driver would have to JIT-compile at the first kernel launch -- that
  # JIT step is exactly what produced "the provided PTX was compiled with
  # an unsupported toolchain": this machine's driver (reports CUDA 13.1 as
  # its max) is older than the CUDA 13.4 toolkit's PTX ISA version, so its
  # JIT compiler refuses it. A real sm_86 binary needs no JIT at all, so
  # the driver/toolkit version gap never matters at runtime.
  #
  # FORCE is deliberate and required here: enable_language(CUDA) above
  # creates its OWN cached value for CMAKE_CUDA_ARCHITECTURES the first
  # time it runs (CMake's internal default, observed to land on "75" on
  # this exact machine+toolkit combination) -- an ordinary set(), or a
  # set() guarded by if(NOT CMAKE_CUDA_ARCHITECTURES), only ever writes to
  # a NORMAL variable, which cannot override a value that is already in
  # CMakeCache.txt. That mismatch is exactly why the previous version of
  # this file kept losing to a stale "75" even across "Delete Cache and
  # Reconfigure" if the cache wasn't actually fully removed. CACHE STRING
  # ... FORCE writes directly into the cache, unconditionally, every time
  # this file is processed, so there is no stale value left for it to lose
  # to. If you build this on a DIFFERENT GPU, change 86 to that card's
  # compute capability (e.g. 75 for a GTX 1650/RTX 2060 class card).
  set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "" FORCE)
  set(SOVEREIGN_HAVE_CUDA TRUE)
  message(STATUS "sovereign_cpp: CUDA compiler found (${CMAKE_CUDA_COMPILER}) -- building the real CPU/GPU race engine")
else()
  set(SOVEREIGN_HAVE_CUDA FALSE)
  if(SOVEREIGN_FORCE_CPU)
    message(STATUS "sovereign_cpp: SOVEREIGN_FORCE_CPU=ON -- skipping CUDA probe, building CPU-only baseline for timing comparison")
  else()
    message(STATUS "sovereign_cpp: no CUDA compiler found -- building CPU-only (this is the normal, expected result on a laptop with no NVIDIA GPU / no CUDA toolkit installed)")
  endif()
endif()

set(SOVEREIGN_ENGINE_SOURCES
  src/mps_reader.cpp
  src/presolve.cpp
  src/scaling.cpp
  src/verify.cpp
  src/pdlp.cpp
  src/basis_lu.cpp
  src/simplex.cpp
  src/mip.cpp
  src/mip_probe.cpp
  src/report_io.cpp
  src/sparse_ldl.cpp
  src/qp_ipm.cpp
  src/solve.cpp
)
if(SOVEREIGN_HAVE_CUDA)
  list(APPEND SOVEREIGN_ENGINE_SOURCES gpu/pdlp_gpu_solve.cu)
endif()

add_library(sovereign_engine ${SOVEREIGN_ENGINE_SOURCES})
target_link_libraries(sovereign_engine PUBLIC Threads::Threads)

if(SOVEREIGN_HAVE_CUDA)
  target_compile_definitions(sovereign_engine PUBLIC SOVEREIGN_WITH_CUDA=1)
  # Set the architecture on the TARGET explicitly too, not just the global
  # CMAKE_CUDA_ARCHITECTURES variable above -- on at least one real build
  # (Visual Studio generator + CUDA 13.4) the global variable alone did not
  # reliably reach the actual nvcc invocation, and only the default
  # compute_75 architecture was emitted regardless of what the variable
  # said. Setting it again here, directly on the target, is the belt-and-
  # suspenders fix that's guaranteed to take effect.
  set_target_properties(sovereign_engine PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  # NOTE: deliberately NOT setting CUDA_SEPARABLE_COMPILATION here. Turning
  # that on makes nvcc compile with -rdc=true (relocatable device code),
  # which requires a separate nvcc device-link step before the final host
  # link. CMake does not automatically add that step when a STATIC library
  # containing the .cu file is linked into a *different* executable target
  # (exactly this project's layout: sovereign_engine -> sovereign_solve /
  # run_checks) -- the real, observed symptom is a host-link failure like
  # "LNK2019: unresolved external symbol __cudaRegisterLinkedBinary_...".
  # This project has no cross-translation-unit device code (one .cu file,
  # no extern __device__/__global__ declarations shared across files), so
  # separable compilation was never actually needed -- ordinary whole-
  # program device compilation (the nvcc default, -rdc=false) links cleanly
  # through a static library with no extra CMake plumbing required.
endif()

# alloc_stats.cpp replaces global operator new/delete for this executable only
# (allocation accounting reported in --json output); tests and tools are unaffected.
add_executable(sovereign_solve src/main.cpp src/alloc_stats.cpp)
target_link_libraries(sovereign_solve sovereign_engine)
if(WIN32)
  target_link_libraries(sovereign_solve psapi)
endif()

enable_testing()
add_executable(run_checks tests/run_checks.cpp)
target_link_libraries(run_checks sovereign_engine)
add_test(NAME run_checks COMMAND run_checks)
add_executable(test_lu tests/test_lu.cpp)
target_link_libraries(test_lu sovereign_engine)
add_test(NAME test_lu COMMAND test_lu)
add_executable(test_ldl tests/test_ldl.cpp)
target_link_libraries(test_ldl sovereign_engine)
add_test(NAME test_ldl COMMAND test_ldl)

# gen_lp: standalone MPS generator for large synthetic LPs, used to give the
# CPU/GPU race enough parallel work to actually show a GPU advantage (the
# Netlib fixtures above are correctness tests, not performance benchmarks --
# see tools/gen_lp.cpp's header comment). No dependency on sovereign_engine.
add_executable(gen_lp tools/gen_lp.cpp)

# audit_lp: runs the CPU and GPU PDHG engines INDEPENDENTLY (no race, no
# stop_flag) on the same input and reports both engines' own objective and
# KKT residuals side by side, plus known Netlib reference objectives where
# available. This is what actually answers "is the GPU engine correct",
# as opposed to solve_mps()'s race, which only ever reports the winner and
# never shows you what the other engine would have said. See
# tools/audit_lp.cpp's header comment for the full rationale.
add_executable(audit_lp tools/audit_lp.cpp)
target_link_libraries(audit_lp sovereign_engine)
