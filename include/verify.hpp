// verify.hpp -- the independent KKT verifier (port of verify.py).
// Shares no code with presolve/scaling/pdlp; takes only (x,y) and the
// RangedLP, derives z = c - A^T y itself, never trusts the caller's z.
#pragma once
#include "ranged_lp.hpp"

struct KKTReport {
    double eps_P = 0, eps_D = 0, eps_G = 0;
    double objective = 0, dual_objective = 0;
    double max_row_violation = 0, max_bound_violation = 0;
};

KKTReport verify(const RangedLP& ranged, const std::vector<double>& x, const std::vector<double>& y);
