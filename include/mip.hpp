// mip.hpp -- branch-and-cut for MILP (MILP pipeline, Steps 5-11).
//
// Operates on a presolved, scaled RangedLP whose `integer` flags mark the
// integer columns (integer columns are never column-scaled, so integrality is
// checked and branched on directly in this space).
//
//   root:        dual simplex; up to `cut_rounds` rounds of Gomory mixed-
//                integer and knapsack cover cuts with efficacy/parallelism
//                filtering and a tailing-off stop; rounding + diving heuristics
//   tree:        warm-started dual simplex at every node (a basis optimal for
//                one node is dual feasible for all of them -- bounds do not
//                affect dual feasibility -- so every node re-enters phase 2);
//                reliability branching (pseudocosts, strong branching until a
//                variable's pseudocosts are reliable); best-bound node
//                selection with plunging; reduced-cost fixing; periodic diving
//   termination: relative gap, node limit, time limit
//
// Results are in the scaled space passed in; the caller unscales/postsolves.
#pragma once
#include <string>
#include <vector>
#include <limits>
#include "ranged_lp.hpp"

struct MipOptions {
    double time_limit = 3600.0;        // seconds
    double rel_gap = 1e-4;             // |bound - incumbent| / max(1, |incumbent|)
    double abs_gap = 1e-6;
    long long node_limit = 100000000;
    double int_tol = 1e-6;
    int threads = 1;                   // parallel tree search workers
    bool cuts = true;
    int cut_rounds = 20;
    bool heuristics = true;
    int dive_frequency = 200;          // nodes between in-tree dives
    bool rins = true;                  // RINS sub-MIPs (root + every 5 dives)
    int strong_branch_candidates = 8;
    int reliability = 4;               // pseudocost observations before trusting them
    int strong_branch_iterations = 60;
    int verbose = 1;
};

struct MipResult {
    std::string status;        // "optimal", "infeasible", "unbounded", "time_limit", "node_limit"
    bool has_solution = false;
    double objective = std::numeric_limits<double>::infinity();   // min form, incl. offset
    double bound = -std::numeric_limits<double>::infinity();      // best proven lower bound
    double gap = std::numeric_limits<double>::infinity();
    double root_lp = 0.0, root_after_cuts = 0.0;
    std::vector<double> x;     // scaled space, length n
    long long nodes = 0;
    long long lp_iterations = 0;
    int cuts_added = 0;
    int heuristic_solutions = 0;
    double seconds = 0.0;
};

MipResult solve_mip(const RangedLP& lp, const MipOptions& opt = MipOptions());
