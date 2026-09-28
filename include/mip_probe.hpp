// mip_probe.hpp -- MILP root reasoning: bound propagation, probing on binary
// variables, a clique table, and the implication list behind implied-bound
// cuts (MILP pipeline, presolve and cut steps).
//
//   propagate():  activity-based bound tightening over the rows, driven by a
//                 queue of changed columns (integers rounded inward).
//   probe():      for each binary x_j (time-limited): propagate x_j = 0 and
//                 x_j = 1. One side infeasible -> x_j fixed to the other;
//                 both infeasible -> the MILP is infeasible; a bound implied by
//                 both sides -> global tightening. Every bound a side implies
//                 beyond the global one is kept as an implication.
//   CliqueTable:  cliques of literals (x_j or 1 - x_j) that cannot be 1
//                 together, from set-packing / knapsack rows and binary
//                 implications; greedy separation of violated clique cuts.
#pragma once
#include <vector>
#include <tuple>
#include "ranged_lp.hpp"
#include "simplex.hpp"

struct Implication {
    int bin;        // binary column
    int val;        // 0 or 1
    int col;        // implied column
    bool upper;     // true: x_col <= bound, false: x_col >= bound
    double bound;
};

class Propagator {
public:
    Propagator(const RangedLP& lp, const std::vector<char>& is_int);
    // Tighten lo/hi from the rows, starting from the rows of `seed` columns
    // (all rows if seed is empty). Records every change in `log` as
    // (col, old lo, old hi) so the caller can undo. False if infeasible.
    bool propagate(std::vector<double>& lo, std::vector<double>& hi, const std::vector<int>& seed,
                   std::vector<std::tuple<int,double,double>>* log, long long work_limit = 2000000) const;   // thread-safe
private:
    int m_, n_;
    std::vector<int> rp_, ri_, cp_, ci_;     // CSR rows (cols in ri_), CSC columns (rows in ci_)
    std::vector<double> rv_;
    std::vector<double> rL_, rU_;
    std::vector<char> is_int_;
};

struct ProbeResult {
    bool infeasible = false;
    int fixed = 0, tightened = 0, probed = 0;
    std::vector<Implication> impl;
};

// Probes the binaries of lp within `seconds` on `threads` threads; lo/hi
// (global bounds) are tightened in place.
ProbeResult probe(const RangedLP& lp, const std::vector<char>& is_int, std::vector<double>& lo,
                  std::vector<double>& hi, double seconds, int threads = 1);

class CliqueTable {
public:
    // literal = 2*j (x_j) or 2*j+1 (1 - x_j)
    void add_clique(std::vector<int> lits);
    void build_from_rows(const RangedLP& lp, const std::vector<char>& is_int,
                         const std::vector<double>& lo, const std::vector<double>& hi);
    void add_implications(const std::vector<Implication>& impl, const std::vector<char>& is_binary);
    // Violated clique cuts at the LP point x (structural values).
    int separate(const std::vector<double>& x, std::vector<DualSimplex::Row>& out, int max_cuts) const;
    int size() const { return (int)cliques_.size(); }
private:
    bool adjacent(int a, int b) const;
    std::vector<std::vector<int>> cliques_;      // sorted literal lists
    std::vector<std::vector<int>> of_lit_;       // literal -> clique ids
};
