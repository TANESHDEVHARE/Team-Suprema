// presolve.hpp -- LP / MILP / convex QP presolve with primal-dual postsolve.
//
// Reductions (all on original row/column indices; nothing is renumbered
// until the reduced problem is built):
//   R1  empty row                     RR  redundant row (activity within bounds)
//   R2  empty column                  FR  forcing row (activity bound == row bound)
//   R3  fixed column / dual fixing    FCS implied-free variable substituted out of an
//                                         equality row (column singletons and the aggregator)
//   R4  singleton row -> bound        DT  doubleton equation (substitute one variable)
// Every step records what postsolve needs to rebuild x and y for the problem
// as it was before that step; postsolve runs the steps in reverse, keeping
// the reduced costs z of the current stage up to date.
#pragma once
#include <vector>
#include <string>
#include "ranged_lp.hpp"

struct ForcedVar {
    int col = -1;
    double a = 0.0, value = 0.0, cost = 0.0;
    std::vector<std::pair<int,double>> others;   // (row, coeff) of the column in the other rows
};

struct Step {
    std::string kind;              // see the table above
    int row_id = -1, col_id = -1, col2 = -1;
    double value = 0.0, cost = 0.0, a_ij = 0.0, a2 = 0.0;
    double lo_from_row = 0.0, hi_from_row = 0.0, lo_before = 0.0, hi_before = 0.0;
    bool lo_from = false, hi_from = false;       // DT: kept variable's bound came from the eliminated one
    std::vector<std::pair<int,double>> column;   // R3/DT/FCS: (row, coeff) of the removed column (FCS: other rows)
    std::vector<std::pair<int,double>> row;      // FCS: (col, coeff) of the removed row, minus the singleton
    std::vector<ForcedVar> forced;               // FR
};

struct PresolveResult {
    bool has_reduced = false;
    RangedLP reduced;
    std::vector<Step> steps;
    std::vector<int> row_ids;      // reduced row position -> original row index
    std::vector<int> col_ids;      // reduced col position -> original col index
    std::string status;            // "ok" | "infeasible" | "unbounded"
    int passes = 0;
};

// max_iterations: rule passes (0 = no reductions beyond integer bound rounding).
// mip_mode: never eliminate integer columns, and also apply coefficient
// tightening (keeps every integer point, tightens the LP relaxation).
PresolveResult presolve(const RangedLP& ranged, int max_iterations = 500, bool mip_mode = false);

// x,y,z sized n0,m0,n0 on return.
void postsolve(int n0, int m0, const std::vector<int>& row_ids, const std::vector<int>& col_ids,
               const std::vector<double>& x_reduced, const std::vector<double>& y_reduced,
               const std::vector<double>& z_reduced, const std::vector<Step>& steps,
               std::vector<double>& x, std::vector<double>& y, std::vector<double>& z);
