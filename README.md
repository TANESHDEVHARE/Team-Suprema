# Sovereign LP Solver — C++ POC

A direct C++17 port of the working `sovereign_pdlp` Python engine: MPS reader
→ Presolve (R1-R4) → Scaling (Ruiz + Pock-Chambolle) → restarted PDHG/PDLP →
Postsolve → an **independent KKT verifier** (shares no code with the solver).
No external dependencies for the CPU path — builds with nothing but a C++17
compiler and CMake. The GPU path additionally needs the CUDA toolkit, and
CMake detects it automatically (see below).

## Honest status (read this before presenting)

- **CPU pipeline: working, tested, today.** Builds and runs now; matches the
  Python POC's results to 8+ significant figures on real Netlib instances.
- **CPU/GPU race: real code, wired into the build, not yet executed on a GPU
  anywhere.** `gpu/pdlp_gpu_solve.cu` is a complete host+device implementation
  — not a stub — that mirrors `src/pdlp.cpp`'s `solve_pdhg()` iteration for
  iteration, restart rule for restart rule, and calls the exact same
  independent `verify()` the CPU engine calls, at the same cadence. When
  CMake finds a CUDA compiler (`check_language(CUDA)`), it is compiled in and
  `solve_mps()` races it against the CPU engine on two `std::thread`s sharing
  one `std::atomic<bool>` stop flag — first verified answer wins, exactly as
  described in the architecture diagram. When CMake finds no CUDA compiler
  (true of every machine this project has been built on so far — this
  sandbox and the Windows/MSYS2 toolchain), the GPU file is silently left out
  of the build and `solve_mps()` falls back to the single CPU path, unchanged.
  **No run of this project, anywhere, has yet produced a real GPU timing
  number.** Don't claim one until `build/sovereign_solve ... ` has actually
  printed `Engine: gpu` on real hardware.
- `gpu/pdlp_spmv_kernel.cu` (a separate file) is kept as the original,
  documented kernel design referenced in the deck — `pdlp_gpu_solve.cu` is
  the file CMake actually compiles.
- **MILP/QP: not built.** Out of scope for this POC, per the project's own
  scope decision (LP only through the 3-month milestone).

## Build & run — CPU only (no GPU needed, works everywhere)

```
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
./run_checks                                   # correctness suite
./sovereign_solve sample_problems/afiro.mps    # solve one problem
```

CMake will print one of these two lines during configure — that line tells
you, honestly, whether this build includes the GPU race:

```
-- sovereign_cpp: no CUDA compiler found -- building CPU-only ...
-- sovereign_cpp: CUDA compiler found (...) -- building the real CPU/GPU race engine
```

## Getting an actual GPU run (do this before quoting a GPU number)

You need an NVIDIA GPU and the CUDA toolkit. Two ways to get one before a
deadline:

1. **Check your own laptop first** — Windows PowerShell:
   ```
   nvidia-smi
   ```
   If that prints a GPU table, you have an NVIDIA GPU. Then install the CUDA
   Toolkit from developer.nvidia.com/cuda-downloads, open a fresh MSYS2
   MinGW64 terminal (or a regular terminal with `nvcc` on PATH), and re-run
   the build steps above from a clean `build/` directory — CMake will detect
   `nvcc` and compile the race in automatically. No code changes needed.

2. **No local GPU — use a free cloud GPU (Google Colab).** Upload this whole
   `sovereign_cpp/` folder, set Runtime → Change runtime type → GPU, then in
   a Colab cell:
   ```
   !apt-get install -y cmake
   !cd sovereign_cpp && rm -rf build && mkdir build && cd build && \
     cmake .. -DCMAKE_BUILD_TYPE=Release && make -j4 && \
     ./run_checks && ./sovereign_solve ../sample_problems/afiro.mps 1e-8 200000
   ```
   Colab ships `nvcc` on its GPU runtimes already, so this should just work.
   The CLI output's `Engine:` line tells you, honestly, which engine actually
   won that particular race — it may say `cpu` even on a GPU-enabled build if
   the CPU engine happens to converge first on a problem this small (AFIRO is
   tiny; the GPU engine's real advantage, per the published PDLP literature,
   only shows up at much larger problem sizes — see the earlier discussion of
   O(nnz) scaling in this chat before assuming `gpu` should always win here).

## Verified result (AFIRO, real Netlib instance, 27 rows x 32 cols) — CPU

```
Status:     optimal
Objective:  -464.7531429
Engine:     cpu
Iterations: 4672  (restarts: 14)
eps_P:      9.0e-12   eps_D: 4.9e-11   eps_G: 3.5e-10
Time:       0.0021 s
```

Cross-checked against the Python POC's independent run: `-464.7531428522593`
— agrees to 8 significant figures. `run_checks` also passes on BLEND, SC50A,
a RANGES-section instance, and an RHS-omitted-vector-name instance (the same
MPS-format edge cases the Python test suite covers).

## Structure

```
include/    sparse.hpp, lp_problem.hpp, ranged_lp.hpp, presolve.hpp,
            scaling.hpp, verify.hpp, pdlp.hpp, pdlp_gpu.hpp, solve.hpp
src/        matching .cpp for each header, + main.cpp (CLI)
gpu/        pdlp_spmv_kernel.cu   -- original documented kernel design (reference only)
            pdlp_gpu_solve.cu     -- the real host+device engine CMake compiles
                                     when a CUDA compiler is found
tests/      run_checks.cpp -- correctness suite against real Netlib data
sample_problems/   Netlib MPS files copied from the Python POC
```

Every CPU module is a direct, line-by-line port of the equivalent Python file
in `sovereign_pdlp/engine/` — same algorithm, same formulas, same variable
names where possible — so the Python version can keep serving as the answer
key for any future change here. The GPU engine is a device-parallel port of
the same `solve_pdhg()` loop, not a different algorithm — see
`gpu/pdlp_gpu_solve.cu`'s header comment for exactly how the two correspond.
