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

// Same checks, with CSR(A) and CSR(A^T) supplied by the caller -- for engines
// that verify repeatedly (PDLP checks every few dozen iterations) and must not
// rebuild both matrices from COO on every call. The matrices must be built
// from ranged.A; nothing else about the verification changes.
KKTReport verify(const RangedLP& ranged, const CSR& A, const CSR& AT,
                 const std::vector<double>& x, const std::vector<double>& y);
