# Sovereign Optimization Solver (SIH 2026 · PS 26119 · MRPL)

A from-scratch mathematical optimization engine for **LP, MILP and convex QP**,
written in C++17 (plus CUDA for the GPU path). No third-party solver or
linear-algebra library is used anywhere in the solve path: the sparse LU, the
sparse LDLᵀ, the orderings, the simplex, the interior-point method, the PDLP
engine and every branch-and-cut component are implemented here from the
published mathematics. The design follows the pipeline documents in `Math/`
(with the corrections recorded in `Math/PIPELINE_NOTES.md`).

Every answer is checked by an **independent verifier** (`src/verify.cpp`)
against the original, unscaled, unpresolved problem, and the reported status
follows that verifier — not the engine's own view.

## What is in the box

| Problem | Engine | Where |
|---|---|---|
| LP | **Dual revised simplex** — Markowitz LU (Suhl–Suhl dual storage) with Forrest–Tomlin updates, dual steepest edge, bound-flipping + Harris ratio test, dual phase 1 by artificial bounding, cost perturbation, Bland fallback, primal simplex cleanup | `src/basis_lu.cpp`, `src/simplex.cpp` |
| LP | **Restarted PDLP** (adaptive steps, primal weight, KKT restarts) — one algorithm (`include/pdlp_algo.hpp`), two backends: CPU and hand-written CUDA kernels (no cuBLAS/cuSPARSE), deterministic reductions | `src/pdlp.cpp`, `gpu/pdlp_gpu_solve.cu` |
| LP | **Crossover** PDLP point → simplex basis (pivoting crash), then simplex to machine precision | `DualSimplex::crossover_start` |
| LP | **Auto race**: dual simplex vs PDLP(GPU)+crossover — first verified answer wins | `solve_mps_simplex(..., race=true)` |
| LP, QP | **Primal-dual interior point** — Mehrotra predictor-corrector + Gondzio centrality correctors, quasidefinite augmented system, sparse LDLᵀ with minimum-degree ordering, dynamic regularization (inertia control), iterative refinement, dependent-row removal | `src/qp_ipm.cpp`, `src/sparse_ldl.cpp` |
| MILP | **Branch-and-cut** — warm-started dual simplex at every node; Gomory mixed-integer, c-MIR with variable-upper-bound substitution (flow-cover strength), knapsack cover cuts; reliability branching (strong branching → pseudocosts); best-bound + plunging; reduced-cost fixing; rounding, feasibility pump, RENS, RINS, diving; **multi-threaded tree search** | `src/mip.cpp` |
| all | MPS/QPS reader (free + fixed format, RANGES, all BOUNDS types, integer markers, QUADOBJ/QMATRIX), presolve + postsolve, Ruiz/Pock–Chambolle scaling (integer columns never scaled, Q scaled symmetrically) | `src/mps_reader.cpp`, `src/presolve.cpp`, `src/scaling.cpp` |

## Build

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release          # finds nvcc if present
cmake --build build --config Release -j 8
```

CMake prints whether the CUDA engine is included (`CUDA compiler found` /
`no CUDA compiler found`). `-DSOVEREIGN_FORCE_CPU=ON` forces a CPU-only build
(for honest CPU-vs-GPU timing on the same machine). `CMAKE_CUDA_ARCHITECTURES`
is set to 86 (RTX 30xx); change it for other GPUs.

## Run

```
sovereign_solve            model.mps [time_limit]   # engine picked by type: MILP -> branch-and-cut,
                                                    #   QP -> interior point, LP -> dual simplex
sovereign_solve --simplex  model.mps [time_limit]   # LP: dual simplex
sovereign_solve --auto     model.mps [time_limit]   # LP: simplex vs PDLP(GPU)+crossover race
sovereign_solve --crossover model.mps               # LP: PDLP -> crossover -> simplex
sovereign_solve --ipm      model.mps|model.qps       # LP or convex QP: interior point
sovereign_solve --mip [--threads=8] [--gap=1e-4] model.mps [time_limit]   # MILP
sovereign_solve --pdlp     model.mps [tol] [iters]  # LP: PDLP only (CPU/GPU race)
sovereign_solve --stats    model.mps                # what the reader parsed
options: --no-presolve, --no-cuts, --no-heuristics, --nodes=N, -v / -vv / -vvv
```

Output always includes the verifier's residuals: `eps_P` (primal), `eps_D`
(dual), `eps_G` (duality gap) — relative, on the original problem — and for
MILP the integrality violation, the proven bound and the gap. Statuses:
`optimal`, `near_optimal` (verified residuals ≤ 1e-6 but above 1e-8 /
engine stopped on a stall), `infeasible`, `unbounded`,
`unbounded_or_infeasible`, `time_limit`, `node_limit`, `iteration_limit`.

## Tests and benchmarks

```
build/Release/run_checks     # LP (PDLP, simplex, IPM), QP and MILP on sample_problems/
build/Release/test_lu        # LU factorization + Forrest-Tomlin updates
build/Release/test_ldl       # sparse LDL^T on random quasidefinite systems

pip install highspy          # optional: the comparison solver
python tools/benchmark.py --exe build/Release/sovereign_solve.exe --set all --highs --time 60
```

`tools/benchmark.py` downloads Netlib (LP), MIPLIB 3 (MILP) and the
Maros–Mészáros set (QP) into `bench_data/`, runs this solver and — with
`--highs` — HiGHS on the same instances, and writes `bench_results/report.md`.
A result counts as **OK** only if it is verified *and* agrees with the
reference; **WRONG** (a verified-looking answer that disagrees) must stay 0.

### Results (RTX 3050 laptop, 12 threads; reference HiGHS 1.15.1; 2026-09-27)

| Set | Engine | Result | Notes |
|---|---|---|---|
| Netlib LP, 91 problems | dual simplex | **91/91 optimal**, objective within 1e-6 of HiGHS (worst 2.9e-9); verifier eps ≤ 4.4e-8 | 46 s total vs HiGHS 19 s; median 1.3x HiGHS's pivot count; Bland's fallback never fired |
| Netlib LP | PDLP → crossover → simplex | **91/91 optimal** | |
| Netlib LP | interior point | 85/91 optimal, 2 near-optimal | bnl2, finnis, greenbea, dfl001 stall (simplex solves them) |
| Maros–Mészáros convex QP, 134 problems | interior point | **119/134 certified optimal** (110 match the published optimum, 9 without a reference certified to gap ≤ 1e-11), 4 near-optimal | 97 s total; LISWET family + YAO stall |
| MIPLIB 3, 63 problems, 60 s limit, 4 threads | branch-and-cut | **47/63 proven optimal, 0 wrong**, 16 at the time limit, 15 of them with a verified feasible incumbent | HiGHS (1 thread) proves 48/63. Faster than HiGHS on 24 instances, incl. misc07 5.8 s vs 32.4 s, stein45 7.2 s vs 33.4 s; mas76, pk1, qiu proven where HiGHS hits the limit. HiGHS proves air05, harp2, modglob, set1ch that we do not |
| Refinery planning (tools/gen_refinery.py) | branch-and-cut / IPM | 12×12 and 30×24 MILP and the QP variant match HiGHS exactly | 30×24 (744 binaries): 1.7 s vs HiGHS 0.95 s |

GPU (PDLP, hand-written CUDA kernels, RTX 3050), 1,000,000-variable
transportation LP (2M nonzeros), both engines to 1e-4:
**CPU PDLP 48.4 s, GPU PDLP 8.1 s (6.0x)** with the same iteration count
(2185 vs 2179) and matching objectives. On the same LP the dual simplex reaches
a verified vertex in 17.5 s; PDLP(GPU)+crossover+simplex takes 49 s because
the post-crossover simplex pivots on dense rows. The GPU engine therefore is a
measured win over CPU first-order solving, while the dual simplex remains the
fastest route to a certified vertex on every LP tried — which is why plain LP
defaults to the simplex and `--auto` races both.

## Honest limitations (read before presenting)

- **Speed is behind HiGHS**, correctness is not. The simplex takes a median
  1.3x HiGHS's pivots but each pivot costs more (dense-vector FTRAN/BTRAN; no
  hypersparse solves yet). MILP node throughput and root strength trail a
  mature solver on harder MIPLIB instances (see the table: time-limit rows).
- **MILP presolve is basic** (integer bound rounding, coefficient tightening, singleton rows,
  fixed/empty columns). Probing, clique detection, GCD tightening and
  symmetry handling from the MILP document are not implemented yet; neither
  are lifted cover / clique cuts, local cuts in the tree, or GPU cut scoring.
- **Interior point** stalls on the LISWET family and YAO (degenerate QPs with
  long chains of second-difference constraints) and on four Netlib LPs
  (bnl2, finnis, greenbea, dfl001); those LPs are solved by the simplex.
- **GPU**: only PDLP runs on the GPU. It beats the CPU PDLP (2x at 160k
  variables, 6x at 1M), but a certified vertex is still reached fastest by the
  dual simplex on every LP tried: the post-crossover simplex pivots on dense
  rows (no hypersparse / partial pricing yet). The QP document's GPU ADMM warm
  start is not implemented.
- **Crossover** is a pivoting crash, not the full Megiddo primal/dual push;
  it helps most when PDLP converges well (large, well-scaled LPs).
- Synthetic refinery model (`tools/gen_refinery.py`) is representative in
  structure, not calibrated to MRPL data.

## Layout

```
include/, src/   engine (one header per module), main.cpp = CLI
gpu/             pdlp_gpu_solve.cu (CUDA backend of the PDLP engine);
                 pdlp_spmv_kernel.cu (original kernel design, reference only)
tests/           run_checks.cpp, test_lu.cpp, test_ldl.cpp
tools/           benchmark.py, gen_lp.cpp (large synthetic LPs), audit_lp.cpp (CPU vs GPU PDLP audit)
sample_problems/ small Netlib / MIPLIB 3 / Maros-Meszaros instances used by run_checks
Math/            the pipeline documents and PIPELINE_NOTES.md (agreed corrections)
```
