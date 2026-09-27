// simplex.hpp -- bounded revised simplex engine (LP pipeline, Steps 5-6).
//
// Internal form: every row i gets a logical variable s_i with column +e_i,
//     A x + s = 0,   l <= x <= u,   -rU <= s <= -rL
// so variables are indexed 0..n-1 (structural) and n..n+m-1 (logical), and
// the all-logical basis B = I is always available as a starting point.
//
// Duals follow verify()'s convention directly: d_j = c_j - a_j^T y, and
// y_i > 0 means row i is held at its lower bound rL_i.
//
// Main algorithm: dual simplex
//   - pricing:     dual steepest edge (Forrest-Goldfarb weight update)
//   - ratio test:  bound-flipping ratio test combined with Harris' two-pass
//                  tolerance rule (largest |alpha| among near-ties)
//   - phase 1:     auxiliary box problem ("artificial bounding"): minimises
//                  the sum of dual infeasibilities with the dual simplex itself
//   - degeneracy:  deterministic cost perturbation (dual degeneracy comes from
//                  zero reduced costs, so costs -- not bounds -- are perturbed),
//                  Bland's least-index rule after a long stall, cost shifting
//                  for tolerance-level dual infeasibilities
//   - stability:   Forrest-Tomlin LU updates, refactorization every
//                  `refactor_frequency` updates, FTRAN/row pivot consistency
//                  check, iterative refinement of x_B after refactorization
// Cleanup: primal simplex (Devex pricing, Harris ratio test) removes any dual
// infeasibility left after perturbations/shifts are taken out.
#pragma once
#include <string>
#include <vector>
#include <atomic>
#include "ranged_lp.hpp"
#include "basis_lu.hpp"

struct SimplexOptions {
    double primal_tol = 1e-7;
    double dual_tol = 1e-7;
    double pivot_tol = 1e-7;
    int max_iterations = 5000000;
    double time_limit = 1e30;          // seconds
    int refactor_frequency = 100;
    bool perturb_costs = true;
    int stall_limit = 1000;                         // non-improving pivots before Bland's rule
    // Robustness features, switchable for ablation studies ("what does a
    // textbook implementation do on this problem?"). All on by default.
    bool steepest_edge = true;         // off: Dantzig pricing (largest infeasibility)
    bool harris_bfrt = true;           // off: textbook min-ratio test, no bound flipping
    bool bland_fallback = true;        // off: no anti-cycling rule
    bool scaling = true;               // off: solve the unscaled (presolved) matrix
    int verbose = 0;                   // 0 silent, 1 summary, 2 periodic log
    // Warm start: make the starting basis dual feasible by cost shifting
    // instead of dual phase 1 (the shifts are removed at the end and a primal
    // simplex cleans up). Right for a near-optimal basis, e.g. after crossover.
    bool shift_instead_of_phase1 = false;
    // Cooperative cancellation (racing engines): checked every 64 pivots;
    // when set, solve() returns "stopped".
    std::atomic<bool>* stop_flag = nullptr;
};

enum class VarStatus : signed char { Basic = 0, AtLower = 1, AtUpper = 2, AtZero = 3 };

struct SimplexStats {
    int iterations = 0;
    int phase1_iterations = 0;
    int primal_iterations = 0;
    int refactorizations = 0;
    int bound_flips = 0;
    int bland_pivots = 0;
    int degenerate_pivots = 0;      // dual pivots with a zero dual step
    int longest_stall = 0;          // longest run of pivots without objective progress
    // time profile (seconds), for performance work
    double t_factor = 0, t_btran = 0, t_row = 0, t_ftran = 0, t_dse = 0, t_update = 0, t_price = 0, t_ratio = 0;
    int singular_repairs = 0;
    double seconds = 0.0;
};

class DualSimplex {
public:
    // Load an LP (typically presolved + scaled). Resets the basis to all-logical.
    void load(const RangedLP& lp);

    // Solve from the current basis. Returns "optimal", "infeasible",
    // "unbounded", "dual_infeasible", "iteration_limit", "time_limit" or
    // "numerical_error".
    std::string solve(const SimplexOptions& opt = SimplexOptions());

    // Replace the structural costs (length n). The basis is kept; the next
    // solve() restores dual feasibility (feasibility pump, sub-MIP objectives).
    void set_costs(const std::vector<double>& c);

    // Change a structural variable's bounds (branch-and-bound / warm start).
    // Keeps the basis; the next solve() re-enters from it.
    void set_col_bounds(int j, double lo, double hi);

    // Basis get/set over all n+m variables (warm start, crossover, MILP nodes).
    std::vector<VarStatus> get_basis() const { return status_; }
    void set_basis(const std::vector<VarStatus>& status);

    // Crossover (LP Step 4): build a starting basis from an approximate
    // primal-dual point (x length n, y length m, in the loaded LP's space),
    // e.g. a PDLP solution. Variables far from their bounds with small reduced
    // cost become basic; the LU repair removes dependent columns; the rest go
    // to their nearest bound. Follow with solve() using shift_instead_of_phase1.
    // Returns the number of structural variables placed in the basis.
    int crossover_start(const std::vector<double>& x, const std::vector<double>& y);

    // ---- MILP support (cuts, branching) ----
    struct Row { std::vector<std::pair<int,double>> entries; double lo, hi; };
    // Append rows lo <= a^T x <= hi. Each new row's logical becomes basic, so
    // the basis stays square and the next solve() continues with the dual
    // simplex from the current basis (a violated cut = a primal infeasibility).
    void add_rows(const std::vector<Row>& rows);
    // Remove rows (by index >= first_removable) whose logical is basic, i.e.
    // inactive cuts. Rows whose logical is nonbasic are kept. Returns the
    // number removed. Remaining rows keep their relative order.
    int remove_inactive_rows(int first_removable, double slack_tol = 1e-6);
    // Row `slot` of the tableau B^{-1} [A I], dense over all n+m variables.
    void tableau_row(int slot, std::vector<double>& alpha);
    int basic_var(int slot) const { return basis_[slot]; }
    int slot_of(int j) const { return slot_of_[j]; }
    VarStatus status(int j) const { return status_[j]; }
    double value(int j) const { return x_[j]; }         // j over n+m (logical: s = -a_i x)
    double lower(int j) const { return lb_orig_[j]; }
    double upper(int j) const { return ub_orig_[j]; }
    double reduced_cost(int j) const { return d_[j]; }
    double cost(int j) const { return c_orig_[j]; }
    // Row i of A (the loaded LP plus any added rows).
    void row_entries(int i, std::vector<std::pair<int,double>>& out) const;
    double row_lower(int i) const { return -ub_orig_[n_ + i]; }
    double row_upper(int i) const { return -lb_orig_[n_ + i]; }

    // Results in the loaded LP's space.
    std::vector<double> primal() const;         // x, length n
    std::vector<double> row_duals() const;      // y, length m
    std::vector<double> reduced_costs() const;  // d over structurals, length n
    double objective() const;                   // c^T x + obj_offset (min form)
    const SimplexStats& stats() const { return stats_; }

    int n() const { return n_; }
    int m() const { return m_; }

private:
    // ---- problem data ----
    int n_ = 0, m_ = 0;
    std::vector<int> cptr_, cidx_; std::vector<double> cval_;   // CSC of A
    std::vector<int> rptr_, ridx_; std::vector<double> rval_;   // CSR of A
    std::vector<double> c_orig_, lb_orig_, ub_orig_;            // length n+m
    std::vector<double> cost_, lb_, ub_;                        // working (perturbed/shifted/aux)
    double obj_offset_ = 0.0;

    // ---- basis state ----
    std::vector<VarStatus> status_;
    std::vector<int> basis_;        // slot -> var
    std::vector<int> slot_of_;      // var -> slot or -1
    std::vector<double> x_, d_, y_;
    std::vector<double> dse_;       // dual steepest-edge weights per slot
    std::vector<double> devex_;     // primal Devex weights per var
    BasisLU lu_;
    bool have_factor_ = false;

    SimplexOptions opt_;
    SimplexStats stats_;
    double t_start_ = 0.0;

    // ---- helpers ----
    int N() const { return n_ + m_; }
    void column(int j, std::vector<int>& rows, std::vector<double>& vals) const;
    void add_column(int j, double scale, std::vector<double>& dense) const;
    double dot_column(int j, const std::vector<double>& dense) const;
    bool refactor();
    void ensure_factor();          // refactor only if the factor does not match the basis
    void compute_primal();
    void compute_duals();
    void place_nonbasic(int j);
    void row_of_tableau(const std::vector<double>& rho, std::vector<double>& alpha, std::vector<int>& nz) const;
    mutable std::vector<char> row_mark_;
    double primal_infeasibility(int j) const;
    int count_dual_infeasibilities(double tol) const;
    bool out_of_time() const;

    std::string dual_phase1();
    std::string dual_loop(bool phase1);
    std::string primal_loop();
    void perturb_costs();
    void restore_costs();
};
