// qp_ipm.hpp -- primal-dual interior-point method for convex QP (and LP):
//   min 1/2 x^T Q x + c^T x   s.t.  rL <= A x <= rU,  l <= x <= u,  Q PSD.
// (QP pipeline Steps 4-6; LP is the special case Q = 0.)
//
// Formulation: w = A x for inequality rows (equality rows stay A x = b);
// every finite bound on x and w carries a slack/dual pair, and x, w stay
// strictly inside their bounds. Each iteration factors the quasidefinite
// augmented system
//     [ -(Q + Dx + dp I)   A^T          ] [dx]
//     [      A            Dw^-1 + dd I  ] [dl]
// once (SparseLDL: minimum-degree ordering, inertia-controlled LDL^T) and
// solves it twice -- Mehrotra predictor and corrector -- with iterative
// refinement. Divergence of the iterates with collapsing steps is reported
// as infeasible_or_unbounded (QP Step 5.4, a heuristic monitor, not a
// certificate).
//
// Duals follow verify()'s convention: y_i > 0 means row i at its lower bound,
// z = Q x + c - A^T y (positive at lower bounds).
#pragma once
#include <string>
#include <atomic>
#include <vector>
#include "ranged_lp.hpp"

struct IpmOptions {
    double tol = 1e-8;              // relative primal/dual residual and gap
    int max_iterations = 200;
    double time_limit = 1e30;
    int verbose = 0;
    // Cooperative cancellation (concurrent LP portfolio): checked every
    // iteration; when set, solve_ipm returns "stopped".
    std::atomic<bool>* stop_flag = nullptr;
};

struct IpmResult {
    std::string status;             // "optimal", "infeasible_or_unbounded", "iteration_limit", "time_limit", "numerical_error"
    std::vector<double> x, y, z;
    double objective = 0.0;         // 1/2 x'Qx + c'x + offset
    int iterations = 0;
    int regularized_pivots = 0;
    double pres = 0, dres = 0, gap = 0;
    double seconds = 0;
    long long nnz_L = 0;
};

// lp.Q: lower triangle of Q (COO, i >= j), may be empty.
IpmResult solve_ipm(const RangedLP& lp, const IpmOptions& opt = IpmOptions());
