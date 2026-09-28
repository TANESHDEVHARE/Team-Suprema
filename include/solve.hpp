// solve.hpp -- wires the whole pipeline together (port of solve.py):
//   MPS -> Presolve -> PDLP-form + Scaling -> PDHG -> Unscale -> Postsolve
//        -> KKT verification
#pragma once
#include <string>
#include <map>
#include "ranged_lp.hpp"

struct Solution {
    std::string status;   // "optimal" | "infeasible" | "unbounded" | "iteration_limit"
    double objective = 0;
    bool has_solution = false;
    std::map<std::string,double> x, y;
    int iterations = 0, restarts = 0;
    double eps_P = 0, eps_D = 0, eps_G = 0;
    // "cpu", "gpu", or "n/a" (box-shortcut path never runs an engine at all).
    // Only ever "gpu" when this binary was built with SOVEREIGN_WITH_CUDA
    // AND the GPU engine actually won the race on the machine that ran it.
    std::string engine_used = "n/a";
    // Crossover pipeline breakdown (solve_mps_simplex with crossover = true).
    int pdlp_iterations = 0;
    double pdlp_seconds = 0, simplex_seconds = 0;
    // MILP (solve_mps_mip): optimality is certified by the gap, not by duals.
    bool is_mip = false;
    double mip_bound = 0, mip_gap = 0, max_int_violation = 0, root_lp = 0, root_after_cuts = 0;
    long long nodes = 0;
    int cuts = 0;
};

// max_iterations default: raised from 200000 to 500000 after audit_lp
// (tools/audit_lp.cpp) showed that at the old default, 3 of 11 real Netlib
// fixtures (kb2, sc105, scagr7) were being reported as "iteration_limit" --
// not because PDLP couldn't solve them, but because it just hadn't been
// given enough iterations yet (they converge cleanly by ~330k-480k iters,
// still well under half a second each on CPU). This was silently hiding
// real convergence failures behind a budget that was too tight for some
// perfectly solvable problems -- see the audit's findings for the two
// fixtures (share1b, beaconfd) that still do NOT converge even at 1e6
// iterations, which is a real, unresolved issue, not a budget problem.
Solution solve_mps(const std::string& path, double tol = 1e-6, int max_iterations = 500000, double eta = 0.99);

// Same pipeline, but the continuous solve is the dual revised simplex
// (src/simplex.cpp) instead of PDLP: MPS -> Presolve -> Scaling -> Dual
// Simplex -> Unscale -> Postsolve -> KKT verification. Returns a vertex
// solution at simplex tolerances (1e-7 scaled) rather than PDLP's 1e-4..1e-6.
// Solution::iterations is the simplex pivot count; engine_used = "simplex".
struct SimplexOptions;
//
// crossover = true runs LP Steps 3-5: PDLP (CPU/GPU race) to pdlp_tol, then
// crossover (DualSimplex::crossover_start) and the simplex to full accuracy.
// MILP pipeline (MILP pipeline Steps 1-12): MPS (integrality markers) ->
// presolve (integer bounds rounded) -> scaling (integer columns unscaled) ->
// branch-and-cut (src/mip.cpp) -> unscale -> postsolve -> verification of
// primal feasibility, bounds and integrality on the original problem.
// Interior-point pipeline for LP and convex QP (QUADOBJ/QMATRIX in the MPS
// file): presolve (Q-aware) -> scaling (Q scaled symmetrically) -> Mehrotra
// IPM (src/qp_ipm.cpp) -> unscale -> postsolve -> KKT verification. For QP
// the verifier runs on the linearized cost c + Qx, which makes its dual
// residual and gap exactly the QP optimality conditions.
struct IpmOptions;
Solution solve_mps_ipm(const std::string& path, const IpmOptions& opt, bool use_presolve = true);


// Concurrent LP portfolio: dual simplex, primal simplex, IPM+crossover and
// PDLP(GPU)+crossover on their own threads after one shared presolve; the
// first answer the independent verifier certifies wins and stops the rest.
// engines: bit 0 dual, 1 primal, 2 ipm, 3 pdlp (default all four).
Solution solve_mps_concurrent(const std::string& path, const SimplexOptions& opt, bool use_presolve = true,
                              double pdlp_tol = 1e-4, unsigned engines = 0xFu);

// Independent check of an external solution (any solver's): reads the model
// and a text file of "name value" lines -- column values, then optionally a
// line "DUAL" followed by row duals -- and runs the same KKT verifier on the
// original problem. eps_D / eps_G are NaN when no duals are given.
Solution check_solution(const std::string& path, const std::string& solution_file);

// Seconds spent parsing the MPS file in the most recent solve_mps* call on this thread.
double last_read_seconds();
struct MipOptions;
Solution solve_mps_mip(const std::string& path, const MipOptions& opt, bool use_presolve = true);

Solution solve_mps_simplex(const std::string& path, const SimplexOptions& opt, bool use_presolve = true,
                           bool crossover = false, double pdlp_tol = 1e-4, int pdlp_max_iterations = 200000,
                           bool race = false);   // race = true: dual simplex vs PDLP+crossover, first answer wins
