cmake_minimum_required(VERSION 3.18)
project(sovereign_cpp CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)

find_package(Threads REQUIRED)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # 86 = compute capability 8.6 = your RTX 3050. CACHE STRING ... FORCE is
  # required: enable_language(CUDA) already wrote its own default ("75") into
  # CMakeCache.txt the first time it ran, and a plain set() (or one guarded by
  # if(NOT CMAKE_CUDA_ARCHITECTURES)) can never overwrite an existing cache
  # entry -- only CACHE ... FORCE writes into the cache unconditionally, every
  # single configure, with no stale value left for it to lose to.
  set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "" FORCE)
  set(SOVEREIGN_HAVE_CUDA TRUE)
  message(STATUS "sovereign_cpp: CUDA compiler found (${CMAKE_CUDA_COMPILER}) -- building the real CPU/GPU race engine")
else()
  set(SOVEREIGN_HAVE_CUDA FALSE)
  # SOVEREIGN_FORCE_CPU: deliberately skip the CUDA probe even on a machine
# that has a working nvcc. This exists for ONE reason: to get an honest
# CPU-only timing baseline on the SAME hardware as a GPU-enabled build, so
# a claimed GPU speedup is measured against a real number instead of
# against "whatever the CPU thread's time would have been" (which a race
# never reports for the losing engine). Use a SEPARATE build directory so
# this never overwrites your working GPU-enabled `build` directory.
option(SOVEREIGN_FORCE_CPU "Skip CUDA detection and force a CPU-only build, even if nvcc is found" OFF)
include(CheckLanguage)
if(NOT SOVEREIGN_FORCE_CPU)
  check_language(CUDA)
endif()
if(CMAKE_CUDA_COMPILER AND NOT SOVEREIGN_FORCE_CPU)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES "86" CACHE STRING "" FORCE)
  set(SOVEREIGN_HAVE_CUDA TRUE)
  message(STATUS "sovereign_cpp: CUDA compiler found (${CMAKE_CUDA_COMPILER}) -- building the real CPU/GPU race engine")
else()
  set(SOVEREIGN_HAVE_CUDA FALSE)
  if(SOVEREIGN_FORCE_CPU)
    message(STATUS "sovereign_cpp: SOVEREIGN_FORCE_CPU=ON -- skipping CUDA probe, building CPU-only baseline for timing comparison")
  else()
    message(STATUS "sovereign_cpp: no CUDA compiler found -- building CPU-only")
  endif()
endif()
endif()

set(SOVEREIGN_ENGINE_SOURCES
  src/mps_reader.cpp
  src/presolve.cpp
  src/scaling.cpp
  src/verify.cpp
  src/pdlp.cpp
  src/solve.cpp
)
if(SOVEREIGN_HAVE_CUDA)
  list(APPEND SOVEREIGN_ENGINE_SOURCES gpu/pdlp_gpu_solve.cu)
endif()

add_library(sovereign_engine ${SOVEREIGN_ENGINE_SOURCES})
target_link_libraries(sovereign_engine PUBLIC Threads::Threads)

if(SOVEREIGN_HAVE_CUDA)
  target_compile_definitions(sovereign_engine PUBLIC SOVEREIGN_WITH_CUDA=1)
  set_target_properties(sovereign_engine PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
endif()

add_executable(sovereign_solve src/main.cpp)
target_link_libraries(sovereign_solve sovereign_engine)

enable_testing()
add_executable(run_checks tests/run_checks.cpp)
target_link_libraries(run_checks sovereign_engine)
add_test(NAME run_checks COMMAND run_checks)

# gen_lp: standalone MPS generator for large synthetic LPs, used to give the
# CPU/GPU race enough parallel work to actually show a GPU advantage (the
# Netlib fixtures above are correctness tests, not performance benchmarks).
add_executable(gen_lp tools/gen_lp.cpp)