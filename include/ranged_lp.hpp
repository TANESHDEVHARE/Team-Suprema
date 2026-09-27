// ranged_lp.hpp -- port of the Python POC's ranged_lp.py
// The one struct presolve/scaling/pdlp/verify all operate on:
//   minimize c^T x   s.t.  rL <= A x <= rU ,  l <= x <= u
#pragma once
#include <string>
#include <vector>
#include "sparse.hpp"
#include "lp_problem.hpp"

struct RangedLP {
    std::string name;
    std::vector<std::string> col_names;
    std::vector<std::string> row_names;
    std::vector<double> c;
    SparseMatrix A;
    std::vector<double> rL, rU;
    std::vector<double> l, u;
    double obj_offset = 0.0;
    std::string original_sense;   // "min" or "max" -- for reporting only
    std::vector<char> integer;    // per column, 1 = integer (empty = pure LP)
    SparseMatrix Q;               // QP: lower triangle (i >= j) of Q, objective 1/2 x'Qx + c'x; empty for LP

    int m() const { return static_cast<int>(row_names.size()); }
    int n() const { return static_cast<int>(col_names.size()); }
};

inline void row_bounds(char rtype, double rhs, bool has_range, double rng, double& lo, double& hi) {
    if (!has_range) {
        if (rtype == 'L') { lo = -INF; hi = rhs; return; }
        if (rtype == 'G') { lo = rhs; hi = INF; return; }
        if (rtype == 'E') { lo = rhs; hi = rhs; return; }
    }
    double r = std::fabs(rng);
    if (rtype == 'L') { lo = rhs - r; hi = rhs; return; }
    if (rtype == 'G') { lo = rhs; hi = rhs + r; return; }
    if (rtype == 'E') { if (rng >= 0) { lo = rhs; hi = rhs + r; } else { lo = rhs - r; hi = rhs; } return; }
}

inline RangedLP to_ranged_lp(const LPProblem& p) {
    int m = p.m(), n = p.n();
    RangedLP r;
    r.name = p.name;
    r.col_names = p.col_names;
    r.row_names = p.row_names;
    r.rL.resize(m); r.rU.resize(m);
    for (int i = 0; i < m; ++i) {
        auto it = p.ranges.find(p.row_names[i]);
        bool has_range = it != p.ranges.end();
        row_bounds(p.row_types[i], p.row_rhs[i], has_range, has_range ? it->second : 0.0, r.rL[i], r.rU[i]);
    }
    bool flip = p.sense == "max";
    r.c.resize(n);
    for (int j = 0; j < n; ++j) r.c[j] = flip ? -p.obj[j] : p.obj[j];
    r.A = p.A;
    r.l = p.lo; r.u = p.hi;
    r.obj_offset = flip ? -p.obj_constant : p.obj_constant;
    r.integer = p.integer;
    r.Q = p.Q;
    if (flip) for (auto& v : r.Q.val) v = -v;
    r.original_sense = p.sense;
    return r;
}

inline double report_objective(const RangedLP& r, double min_cost_objective) {
    return r.original_sense == "max" ? -min_cost_objective : min_cost_objective;
}
