// presolve.hpp -- port of the Python POC's presolve.py (R1-R4 cheap tier).
#pragma once
#include <vector>
#include <string>
#include "ranged_lp.hpp"

struct Step {
    std::string kind;              // "R1" | "R2" | "R3" | "R4"
    int row_id = -1, col_id = -1;
    double value = 0.0, cost = 0.0, a_ij = 0.0;
    double lo_from_row = 0.0, hi_from_row = 0.0, lo_before = 0.0, hi_before = 0.0;
    std::vector<std::pair<int,double>> column;   // R3's fixed-column (row_id, coeff) pairs
};

struct PresolveResult {
    bool has_reduced = false;
    RangedLP reduced;
    std::vector<Step> steps;
    std::vector<int> row_ids;      // reduced row position -> original row index
    std::vector<int> col_ids;      // reduced col position -> original col index
    std::string status;            // "ok" | "infeasible" | "unbounded"
};

// mip_mode: also apply integer-only reductions (coefficient tightening), which
// keep every integer point but change the LP relaxation -- MILP pipeline only.
PresolveResult presolve(const RangedLP& ranged, int max_iterations = 500, bool mip_mode = false);

// x,y,z sized n0,m0,n0 on return.
void postsolve(int n0, int m0, const std::vector<int>& row_ids, const std::vector<int>& col_ids,
               const std::vector<double>& x_reduced, const std::vector<double>& y_reduced,
               const std::vector<double>& z_reduced, const std::vector<Step>& steps,
               std::vector<double>& x, std::vector<double>& y, std::vector<double>& z);
