#include "simplex.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <limits>

namespace {

double now_seconds() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// Deterministic value in [0,1) from an index -- the perturbation must be
// reproducible run to run, so no global RNG state.
double hash01(int j) {
    uint64_t z = static_cast<uint64_t>(j) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * (1.0 / 9007199254740992.0);
}

// Adds the elapsed time of a scope to a stats counter.
struct ScopeTimer {
    double& acc; double t0;
    explicit ScopeTimer(double& a) : acc(a), t0(now_seconds()) {}
    ~ScopeTimer() { acc += now_seconds() - t0; }
};

} // namespace

// ============================================================ data setup ==

void DualSimplex::load(const RangedLP& lp) {
    n_ = lp.n(); m_ = lp.m();
    const SparseMatrix& A = lp.A;

    cptr_.assign(n_ + 1, 0); rptr_.assign(m_ + 1, 0);
    for (int k = 0; k < A.nnz(); ++k) { cptr_[A.col_idx[k] + 1]++; rptr_[A.row_idx[k] + 1]++; }
    for (int j = 0; j < n_; ++j) cptr_[j + 1] += cptr_[j];
    for (int i = 0; i < m_; ++i) rptr_[i + 1] += rptr_[i];
    cidx_.assign(A.nnz(), 0); cval_.assign(A.nnz(), 0.0);
    ridx_.assign(A.nnz(), 0); rval_.assign(A.nnz(), 0.0);
    {
        std::vector<int> cc(cptr_.begin(), cptr_.end() - 1), rc(rptr_.begin(), rptr_.end() - 1);
        for (int k = 0; k < A.nnz(); ++k) {
            int i = A.row_idx[k], j = A.col_idx[k];
            int p = cc[j]++; cidx_[p] = i; cval_[p] = A.val[k];
            int q = rc[i]++; ridx_[q] = j; rval_[q] = A.val[k];
        }
    }

    const int NN = N();
    c_orig_.assign(NN, 0.0); lb_orig_.assign(NN, 0.0); ub_orig_.assign(NN, 0.0);
    for (int j = 0; j < n_; ++j) { c_orig_[j] = lp.c[j]; lb_orig_[j] = lp.l[j]; ub_orig_[j] = lp.u[j]; }
    for (int i = 0; i < m_; ++i) { lb_orig_[n_ + i] = -lp.rU[i]; ub_orig_[n_ + i] = -lp.rL[i]; }
    obj_offset_ = lp.obj_offset;
    cost_ = c_orig_; lb_ = lb_orig_; ub_ = ub_orig_;

    status_.assign(NN, VarStatus::AtLower);
    basis_.resize(m_); slot_of_.assign(NN, -1);
    for (int i = 0; i < m_; ++i) { basis_[i] = n_ + i; slot_of_[n_ + i] = i; status_[n_ + i] = VarStatus::Basic; }
    x_.assign(NN, 0.0); d_.assign(NN, 0.0); y_.assign(m_, 0.0);
    for (int j = 0; j < n_; ++j) {
        d_[j] = c_orig_[j];   // y = 0 for the slack basis
        place_nonbasic(j);
    }
    dse_.assign(m_, 1.0);     // exact for B = I
    devex_.assign(NN, 1.0);
    have_factor_ = false;
}

void DualSimplex::set_costs(const std::vector<double>& c) {
    for (int j = 0; j < n_; ++j) { c_orig_[j] = c[j]; cost_[j] = c[j]; }
}

void DualSimplex::set_col_bounds(int j, double lo, double hi) {
    lb_orig_[j] = lo; ub_orig_[j] = hi;
    lb_[j] = lo; ub_[j] = hi;
    if (status_[j] != VarStatus::Basic) place_nonbasic(j);
}

void DualSimplex::set_basis(const std::vector<VarStatus>& status) {
    status_ = status;
    int slot = 0;
    slot_of_.assign(N(), -1);
    for (int j = 0; j < N() && slot < m_; ++j)
        if (status_[j] == VarStatus::Basic) { basis_[slot] = j; slot_of_[j] = slot; ++slot; }
    // Too few basics: pad with logicals (the LU repair would do the same).
    for (int i = 0; slot < m_ && i < m_; ++i) {
        int j = n_ + i;
        if (slot_of_[j] < 0) { basis_[slot] = j; slot_of_[j] = slot; status_[j] = VarStatus::Basic; ++slot; }
    }
    // Too many basics: demote the surplus.
    for (int j = 0; j < N(); ++j)
        if (status_[j] == VarStatus::Basic && slot_of_[j] < 0) { status_[j] = VarStatus::AtLower; place_nonbasic(j); }
    dse_.assign(m_, 1.0);
    have_factor_ = false;
}

int DualSimplex::crossover_start(const std::vector<double>& xin, const std::vector<double>& yin) {
    const int NN = N();
    std::vector<double> xv(NN, 0.0), dv(NN, 0.0);
    for (int j = 0; j < n_; ++j) xv[j] = std::min(std::max(xin[j], lb_orig_[j]), ub_orig_[j]);
    // logical s = -A x, and reduced costs d = c - A^T y (logical: d = -y)
    for (int i = 0; i < m_; ++i) { xv[n_ + i] = 0.0; dv[n_ + i] = -yin[i]; }
    for (int j = 0; j < n_; ++j) {
        double dj = c_orig_[j];
        for (int p = cptr_[j]; p < cptr_[j + 1]; ++p) {
            xv[n_ + cidx_[p]] -= cval_[p] * xv[j];
            dj -= cval_[p] * yin[cidx_[p]];
        }
        dv[j] = dj;
    }
    double cscale = 1.0;
    for (int j = 0; j < n_; ++j) cscale = std::max(cscale, std::fabs(c_orig_[j]));

    // Basicness score in [0,1]: primal slack from the bounds vs dual magnitude.
    struct Sc { int j; double s; };
    std::vector<Sc> sc(NN);
    for (int j = 0; j < NN; ++j) {
        double lo = lb_orig_[j], hi = ub_orig_[j], v = xv[j];
        double pd;
        if (lo == hi) pd = 0.0;
        else {
            double a = std::isfinite(lo) ? v - lo : INF, b = std::isfinite(hi) ? hi - v : INF;
            pd = std::min(a, b);
            pd = std::isfinite(pd) ? std::max(pd, 0.0) / (1.0 + std::fabs(v)) : 1e30;
        }
        double dd = std::fabs(dv[j]) / cscale;
        double s = pd / (pd + dd + 1e-12);
        if (j >= n_) s += 1e-9;      // ties: prefer logicals (well conditioned)
        sc[j] = {j, s};
    }
    std::vector<double> score(NN);
    for (const auto& e : sc) score[e.j] = e.s;
    std::stable_sort(sc.begin(), sc.end(), [](const Sc& a, const Sc& b) { return a.s > b.s; });

    // ---- pivoting crash from the slack basis ----
    // Structurals enter in basicness order, each replacing a basic *logical*
    // (never an already chosen structural), so the basis stays nonsingular by
    // construction. Among logical slots with an acceptable pivot (>= 0.1 of
    // the largest), the least basic logical -- the most active row -- leaves.
    std::vector<VarStatus> st(NN, VarStatus::AtLower);
    for (int i = 0; i < m_; ++i) st[n_ + i] = VarStatus::Basic;
    set_basis(st);
    refactor();
    std::vector<double> col(m_), alpha, spike;
    int structural_basics = 0;
    for (const auto& e : sc) {
        int j = e.j;
        // Every structural is offered, best first; a low-scoring one (at a bound
        // with near-zero reduced cost: a degenerate basic) may only displace a
        // logical that is even less basic -- typically an active row's slack,
        // whose presence would force y_i = 0 where the dual says otherwise.
        if (j >= n_ || e.s <= 0.0) continue;
        if (lb_orig_[j] == ub_orig_[j]) continue;
        std::fill(col.begin(), col.end(), 0.0);
        add_column(j, 1.0, col);
        lu_.ftran(col, alpha, &spike);
        double amax = 0.0;
        for (int k = 0; k < m_; ++k) if (basis_[k] >= n_) amax = std::max(amax, std::fabs(alpha[k]));
        if (amax < 1e-7) continue;                   // dependent on the structurals already in
        int r = -1;
        for (int k = 0; k < m_; ++k) {
            if (basis_[k] < n_ || std::fabs(alpha[k]) < 0.1 * amax) continue;
            if (r < 0 || score[basis_[k]] < score[basis_[r]]) r = k;
        }
        if (score[basis_[r]] >= e.s) continue;       // no less-basic logical to displace
        int leaving = basis_[r];
        slot_of_[leaving] = -1; status_[leaving] = VarStatus::AtLower;
        basis_[r] = j; slot_of_[j] = r; status_[j] = VarStatus::Basic;
        ++structural_basics;
        if (lu_.num_updates() >= opt_.refactor_frequency || !lu_.update(r, spike, alpha[r])) refactor();
    }

    // Nonbasic variables go to the bound nearest the approximate point.
    for (int j = 0; j < NN; ++j) {
        if (status_[j] == VarStatus::Basic) continue;
        double lo = lb_orig_[j], hi = ub_orig_[j], v = xv[j];
        if (std::isfinite(lo) && (!std::isfinite(hi) || v - lo <= hi - v)) status_[j] = VarStatus::AtLower;
        else if (std::isfinite(hi)) status_[j] = VarStatus::AtUpper;
        else status_[j] = VarStatus::AtZero;
        x_[j] = status_[j] == VarStatus::AtLower ? lo : (status_[j] == VarStatus::AtUpper ? hi : 0.0);
    }
    dse_.assign(m_, 1.0);
    return structural_basics;
}

// Choose a legal nonbasic position for j given its bounds, preferring the
// side its reduced cost says is dual feasible, and set x_j to match.
void DualSimplex::place_nonbasic(int j) {
    double lo = lb_[j], hi = ub_[j];
    bool flo = std::isfinite(lo), fhi = std::isfinite(hi);
    VarStatus s;
    if (flo && fhi) {
        if (lo == hi) s = VarStatus::AtLower;
        else if (status_[j] == VarStatus::AtUpper || status_[j] == VarStatus::AtLower) s = status_[j];
        else s = d_[j] < 0 ? VarStatus::AtUpper : VarStatus::AtLower;
    } else if (flo) s = VarStatus::AtLower;
    else if (fhi) s = VarStatus::AtUpper;
    else s = VarStatus::AtZero;
    status_[j] = s;
    x_[j] = s == VarStatus::AtLower ? lo : (s == VarStatus::AtUpper ? hi : 0.0);
}

void DualSimplex::column(int j, std::vector<int>& rows, std::vector<double>& vals) const {
    rows.clear(); vals.clear();
    if (j < n_) {
        for (int p = cptr_[j]; p < cptr_[j + 1]; ++p) { rows.push_back(cidx_[p]); vals.push_back(cval_[p]); }
    } else {
        rows.push_back(j - n_); vals.push_back(1.0);
    }
}

void DualSimplex::add_column(int j, double scale, std::vector<double>& dense) const {
    if (j < n_) { for (int p = cptr_[j]; p < cptr_[j + 1]; ++p) dense[cidx_[p]] += scale * cval_[p]; }
    else dense[j - n_] += scale;
}

double DualSimplex::dot_column(int j, const std::vector<double>& dense) const {
    if (j >= n_) return dense[j - n_];
    double s = 0.0;
    for (int p = cptr_[j]; p < cptr_[j + 1]; ++p) s += cval_[p] * dense[cidx_[p]];
    return s;
}

bool DualSimplex::out_of_time() const {
    return now_seconds() - t_start_ > opt_.time_limit;
}

// ======================================================= linear algebra ==

void DualSimplex::ensure_factor() {
    if (!have_factor_ || lu_.num_updates() >= opt_.refactor_frequency) refactor();
}

bool DualSimplex::refactor() {
    auto fn = [this](int s, std::vector<int>& rows, std::vector<double>& vals) { column(basis_[s], rows, vals); };
    std::vector<std::pair<int,int>> rep;
    { ScopeTimer st(stats_.t_factor); rep = lu_.factorize(m_, fn); }
    stats_.refactorizations++;
    for (const auto& pr : rep) {
        int s = pr.first, r = pr.second;
        int old = basis_[s];
        slot_of_[old] = -1;
        // Put the evicted variable on the bound nearest its current value.
        double lo = lb_[old], hi = ub_[old], v = x_[old];
        if (std::isfinite(lo) && (!std::isfinite(hi) || std::fabs(v - lo) <= std::fabs(v - hi))) { status_[old] = VarStatus::AtLower; x_[old] = lo; }
        else if (std::isfinite(hi)) { status_[old] = VarStatus::AtUpper; x_[old] = hi; }
        else { status_[old] = VarStatus::AtZero; x_[old] = 0.0; }
        int nv = n_ + r;
        basis_[s] = nv; slot_of_[nv] = s; status_[nv] = VarStatus::Basic;
        dse_[s] = 1.0;
        stats_.singular_repairs++;
    }
    have_factor_ = true;
    return true;
}

// x_B = B^{-1} (-N x_N), plus one step of iterative refinement (Step 6.2).
void DualSimplex::compute_primal() {
    std::vector<double> rhs(m_, 0.0);
    for (int j = 0; j < N(); ++j)
        if (status_[j] != VarStatus::Basic && x_[j] != 0.0) add_column(j, -x_[j], rhs);
    std::vector<double> rhs0 = rhs, xb;
    lu_.ftran(rhs, xb);
    for (int k = 0; k < m_; ++k) x_[basis_[k]] = xb[k];

    std::vector<double> res = rhs0;
    for (int k = 0; k < m_; ++k) add_column(basis_[k], -xb[k], res);
    double rmax = 0.0;
    for (double v : res) rmax = std::max(rmax, std::fabs(v));
    if (rmax > 1e-11) {
        std::vector<double> dx;
        lu_.ftran(res, dx);
        for (int k = 0; k < m_; ++k) x_[basis_[k]] += dx[k];
    }
}

void DualSimplex::compute_duals() {
    std::vector<double> e(m_);
    for (int k = 0; k < m_; ++k) e[k] = cost_[basis_[k]];
    lu_.btran(e, y_);
    for (int j = 0; j < N(); ++j)
        d_[j] = status_[j] == VarStatus::Basic ? 0.0 : cost_[j] - dot_column(j, y_);
}

// alpha_j = rho^T a_j, computed row-wise. `alpha` is dense (length n+m) and
// is left holding only this row: entries listed in `nz` from the previous
// call are zeroed first, so the cost is O(nnz touched), never O(n+m).
void DualSimplex::row_of_tableau(const std::vector<double>& rho, std::vector<double>& alpha,
                                 std::vector<int>& nz) const {
    if ((int)alpha.size() != N()) { alpha.assign(N(), 0.0); row_mark_.assign(N(), 0); nz.clear(); }
    for (int j : nz) { alpha[j] = 0.0; row_mark_[j] = 0; }
    nz.clear();

    int rho_nnz = 0;
    for (int i = 0; i < m_; ++i) rho_nnz += rho[i] != 0.0;
    if (rho_nnz > m_ / 10) {
        // Dense rho: accumulate without per-entry bookkeeping, then collect
        // the nonzeros in one pass (the ratio test and dual update loop over
        // nz, so zeros and basic columns are left out).
        for (int i = 0; i < m_; ++i) {
            double ri = rho[i];
            if (ri == 0.0) continue;
            for (int p = rptr_[i]; p < rptr_[i + 1]; ++p) alpha[ridx_[p]] += ri * rval_[p];
            alpha[n_ + i] = ri;
        }
        const int NN = N();
        for (int j = 0; j < NN; ++j)
            if (alpha[j] != 0.0) {
                if (status_[j] == VarStatus::Basic) alpha[j] = 0.0;
                else { row_mark_[j] = 1; nz.push_back(j); }
            }
        return;
    }
    for (int i = 0; i < m_; ++i) {
        double ri = rho[i];
        if (ri == 0.0) continue;
        for (int p = rptr_[i]; p < rptr_[i + 1]; ++p) {
            int j = ridx_[p];
            if (!row_mark_[j]) { row_mark_[j] = 1; nz.push_back(j); }
            alpha[j] += ri * rval_[p];
        }
        alpha[n_ + i] = ri;
        row_mark_[n_ + i] = 1; nz.push_back(n_ + i);
    }
}

double DualSimplex::primal_infeasibility(int j) const {
    if (x_[j] < lb_[j] - opt_.primal_tol) return x_[j] - lb_[j];
    if (x_[j] > ub_[j] + opt_.primal_tol) return x_[j] - ub_[j];
    return 0.0;
}

int DualSimplex::count_dual_infeasibilities(double tol) const {
    int cnt = 0;
    for (int j = 0; j < N(); ++j) {
        if (status_[j] == VarStatus::Basic || lb_[j] == ub_[j]) continue;
        double dj = d_[j];
        bool bad = (status_[j] == VarStatus::AtLower && dj < -tol) ||
                   (status_[j] == VarStatus::AtUpper && dj > tol) ||
                   (status_[j] == VarStatus::AtZero && std::fabs(dj) > tol);
        cnt += bad;
    }
    return cnt;
}

// ============================================== costs: perturb / restore ==

void DualSimplex::perturb_costs() {
    for (int j = 0; j < n_; ++j) {
        if (lb_[j] == ub_[j]) continue;
        if (!std::isfinite(lb_[j]) && !std::isfinite(ub_[j])) continue;
        double xi = 5e-7 * (1.0 + std::fabs(c_orig_[j])) * (1.0 + hash01(j));
        bool up = status_[j] == VarStatus::AtLower ||
                  (status_[j] == VarStatus::Basic && std::isfinite(lb_[j]));
        cost_[j] += up ? xi : -xi;
    }
    compute_duals();
}

void DualSimplex::restore_costs() {
    cost_ = c_orig_;
    compute_duals();
}

// Make the current basis dual feasible: boxed variables flip to the bound
// their reduced cost prefers; anything else gets its cost shifted so d_j = 0.
// Shifts are removed by restore_costs() and cleaned up by the primal simplex.
static bool enforce_dual_feasibility(int NN, const std::vector<double>& lb, const std::vector<double>& ub,
                                     std::vector<VarStatus>& status, std::vector<double>& x,
                                     std::vector<double>& d, std::vector<double>& cost, double tol) {
    bool flipped = false;
    for (int j = 0; j < NN; ++j) {
        VarStatus s = status[j];
        if (s == VarStatus::Basic || lb[j] == ub[j]) continue;
        double dj = d[j];
        bool boxed = std::isfinite(lb[j]) && std::isfinite(ub[j]);
        if (s == VarStatus::AtLower && dj < -tol) {
            if (boxed) { status[j] = VarStatus::AtUpper; x[j] = ub[j]; flipped = true; }
            else { cost[j] -= dj; d[j] = 0.0; }
        } else if (s == VarStatus::AtUpper && dj > tol) {
            if (boxed) { status[j] = VarStatus::AtLower; x[j] = lb[j]; flipped = true; }
            else { cost[j] -= dj; d[j] = 0.0; }
        } else if (s == VarStatus::AtZero && std::fabs(dj) > tol) {
            cost[j] -= dj; d[j] = 0.0;
        }
    }
    return flipped;
}

// ========================================================= dual simplex ==

std::string DualSimplex::dual_loop(bool phase1) {
    const int NN = N();
    std::vector<double> e(m_), rho, alpha_row, col(m_), alpha_q, spike, tau, work(m_), dxb;
    std::vector<int> row_nz;
    struct Cand { int j; double ratio; double abar; };
    std::vector<Cand> cands;
    std::vector<int> flips;

    int stall = 0;
    bool bland = false;
    double best_obj = -std::numeric_limits<double>::infinity();
    int consecutive_rejects = 0;

    auto recompute = [&]() {
        refactor();
        compute_primal();
        compute_duals();
        if (enforce_dual_feasibility(NN, lb_, ub_, status_, x_, d_, cost_, opt_.dual_tol)) compute_primal();
    };
    // Entry: reuse the factorization when it still matches the basis (B&B
    // children start from exactly the basis their parent finished with).
    ensure_factor();
    compute_primal();
    compute_duals();
    if (enforce_dual_feasibility(NN, lb_, ub_, status_, x_, d_, cost_, opt_.dual_tol)) compute_primal();

    while (true) {
        if (stats_.iterations >= opt_.max_iterations) return "iteration_limit";
        if ((stats_.iterations & 63) == 0) {
            if (out_of_time()) return "time_limit";
            if (opt_.stop_flag && opt_.stop_flag->load(std::memory_order_relaxed)) return "stopped";
        }
        if (lu_.num_updates() >= opt_.refactor_frequency) recompute();

        // ---- CHUSR: leaving row by dual steepest edge ----
        double t_mark = now_seconds();
        int r = -1; double best = 0.0;
        for (int k = 0; k < m_; ++k) {
            double inf = primal_infeasibility(basis_[k]);
            if (inf == 0.0) continue;
            if (bland) {
                if (r < 0 || basis_[k] < basis_[r]) r = k;
            } else {
                double score = opt_.steepest_edge ? inf * inf / dse_[k] : std::fabs(inf);
                if (score > best) { best = score; r = k; }
            }
        }
        stats_.t_price += now_seconds() - t_mark;
        if (r < 0) return "optimal";

        const int p = basis_[r];
        const bool to_lower = x_[p] < lb_[p];
        const double delta = to_lower ? x_[p] - lb_[p] : x_[p] - ub_[p];

        // ---- BTRAN + pivot row ----
        std::fill(e.begin(), e.end(), 0.0); e[r] = 1.0;
        { ScopeTimer st(stats_.t_btran); lu_.btran(e, rho); }
        { ScopeTimer st(stats_.t_row); row_of_tableau(rho, alpha_row, row_nz); }

        // ---- ratio test: bound flipping + Harris ----
        t_mark = now_seconds();
        cands.clear();
        for (int j : row_nz) {
            VarStatus s = status_[j];
            if (s == VarStatus::Basic || lb_[j] == ub_[j]) continue;
            double a = alpha_row[j];
            if (std::fabs(a) < 1e-9) continue;
            double abar = to_lower ? -a : a;
            bool ok = (s == VarStatus::AtLower && abar > 0) || (s == VarStatus::AtUpper && abar < 0) ||
                      (s == VarStatus::AtZero);
            if (!ok) continue;
            cands.push_back({j, d_[j] / abar, abar});
        }

        int q = -1;
        flips.clear();
        if (bland) {
            double tmin = std::numeric_limits<double>::infinity();
            for (const auto& c : cands) tmin = std::min(tmin, std::max(c.ratio, 0.0));
            for (const auto& c : cands)
                if (std::max(c.ratio, 0.0) <= tmin + 1e-12 && (q < 0 || c.j < q)) q = c.j;
            stats_.bland_pivots++;
        } else if (!opt_.harris_bfrt) {
            // Textbook ratio test: exact minimum ratio, first candidate on ties.
            double tmin = std::numeric_limits<double>::infinity();
            for (const auto& c : cands)
                if (c.ratio < tmin) { tmin = c.ratio; q = c.j; }
        } else {
            double slope = std::fabs(delta);
            // Most pivots take the first Harris group, which two O(c) scans find.
            // Only when bound flipping must pass that group are the candidates
            // sorted by breakpoint, making each later group a contiguous run.
            {
                double theta_max = std::numeric_limits<double>::infinity();
                for (const auto& c : cands) theta_max = std::min(theta_max, c.ratio + opt_.dual_tol / std::fabs(c.abar));
                double dec = 0.0, amax = -1.0;
                int best = -1;
                for (const auto& c : cands) {
                    if (c.ratio > theta_max) continue;
                    dec += std::fabs(c.abar) * (ub_[c.j] - lb_[c.j]);
                    if (std::fabs(c.abar) > amax) { amax = std::fabs(c.abar); best = c.j; }
                }
                if (!(slope - dec > opt_.primal_tol)) q = best;
            }
            if (q < 0 && !cands.empty())
                std::sort(cands.begin(), cands.end(), [](const Cand& a, const Cand& b) { return a.ratio < b.ratio; });
            size_t pos = q < 0 ? 0 : cands.size();
            while (pos < cands.size()) {
                // Harris pass 1 over the remaining candidates: every candidate with
                // ratio <= theta_max belongs to this group. Sorted order means the
                // run can stop at the first ratio beyond the current bound.
                double theta_max = std::numeric_limits<double>::infinity();
                size_t end = pos;
                while (end < cands.size() && cands[end].ratio <= theta_max) {
                    theta_max = std::min(theta_max, cands[end].ratio + opt_.dual_tol / std::fabs(cands[end].abar));
                    ++end;
                }
                double dec = 0.0;
                for (size_t k = pos; k < end; ++k)
                    dec += std::fabs(cands[k].abar) * (ub_[cands[k].j] - lb_[cands[k].j]);   // inf if not boxed
                // The slope is the leaving variable's remaining infeasibility, so
                // only pass the group if it stays infeasible beyond tolerance.
                if (slope - dec > opt_.primal_tol) {   // pass the whole group, flipping it
                    slope -= dec;
                    for (size_t k = pos; k < end; ++k) flips.push_back(cands[k].j);
                    pos = end;
                    continue;
                }
                // Harris pass 2: the largest |alpha| in the group is the most stable pivot.
                double amax = -1.0;
                for (size_t k = pos; k < end; ++k)
                    if (std::fabs(cands[k].abar) > amax) { amax = std::fabs(cands[k].abar); q = cands[k].j; }
                break;
            }
        }
        stats_.t_ratio += now_seconds() - t_mark;
        if (q < 0) {
            // No entering candidate: the dual ray is unbounded => primal infeasible.
            if (lu_.num_updates() > 0) { recompute(); continue; }   // confirm on a fresh factor
            if (opt_.verbose && phase1) {
                std::printf("  simplex: phase 1 found no entering candidate (row %d)\n", r);
                std::printf("    leaving var %d x=%.6e lb=%.3e ub=%.3e delta=%.3e\n", p, x_[p], lb_[p], ub_[p], delta);
                for (int j : row_nz) {
                    if (status_[j] == VarStatus::Basic || std::fabs(alpha_row[j]) < 1e-12) continue;
                    std::printf("    nb %d st=%d x=%.3e [%.3e, %.3e] alpha=%.3e d=%.3e\n", j, (int)status_[j], x_[j], lb_[j], ub_[j], alpha_row[j], d_[j]);
                }
            }
            return phase1 ? "numerical_error" : "infeasible";
        }

        // ---- FTRAN entering column, pivot consistency check ----
        std::fill(col.begin(), col.end(), 0.0);
        add_column(q, 1.0, col);
        { ScopeTimer st(stats_.t_ftran); lu_.ftran(col, alpha_q, &spike); }
        const double piv = alpha_q[r];
        const double piv_row = alpha_row[q];
        if (std::fabs(piv - piv_row) > 1e-7 * (1.0 + std::fabs(piv)) || std::fabs(piv) < opt_.pivot_tol) {
            if (lu_.num_updates() > 0 && consecutive_rejects < 3) { ++consecutive_rejects; recompute(); continue; }
            if (std::fabs(piv) < 1e-11) {
                if (opt_.verbose) std::printf("  simplex: pivot %.3e (row says %.3e) unusable on a fresh factor\n", piv, piv_row);
                return "numerical_error";
            }
        }
        consecutive_rejects = 0;

        // ---- dual update (Harris may leave d_q with a tolerance-level wrong sign) ----
        {
            double abar_q = to_lower ? -piv_row : piv_row;
            if (d_[q] / abar_q < 0.0) { cost_[q] -= d_[q]; d_[q] = 0.0; }
        }
        const double theta_d = d_[q] / piv_row;
        if (theta_d == 0.0) stats_.degenerate_pivots++;
        if (theta_d != 0.0)
            for (int j : row_nz)
                if (status_[j] != VarStatus::Basic) d_[j] -= theta_d * alpha_row[j];
        d_[p] = -theta_d;
        d_[q] = 0.0;

        // ---- bound flips ----
        if (!flips.empty()) {
            std::fill(work.begin(), work.end(), 0.0);
            for (int j : flips) {
                double nx;
                if (status_[j] == VarStatus::AtLower) { status_[j] = VarStatus::AtUpper; nx = ub_[j]; }
                else { status_[j] = VarStatus::AtLower; nx = lb_[j]; }
                add_column(j, nx - x_[j], work);
                x_[j] = nx;
            }
            lu_.ftran(work, dxb);
            for (int k = 0; k < m_; ++k) x_[basis_[k]] -= dxb[k];
            stats_.bound_flips += (int)flips.size();
        }

        // ---- primal step ----
        const double target = to_lower ? lb_[p] : ub_[p];
        const double theta_p = (x_[p] - target) / piv;
        for (int k = 0; k < m_; ++k) if (alpha_q[k] != 0.0) x_[basis_[k]] -= theta_p * alpha_q[k];
        x_[q] += theta_p;

        // ---- dual steepest-edge weights ----
        if (opt_.steepest_edge) {
            std::vector<double>& rho_copy = dse_tmp_;   // reused buffer: no allocation per pivot
            rho_copy = rho;
            ScopeTimer st(stats_.t_dse);
            lu_.ftran(rho_copy, tau);
            double wr = 0.0;
            for (double v : rho) wr += v * v;
            for (int k = 0; k < m_; ++k) {
                if (k == r || alpha_q[k] == 0.0) continue;
                double ratio = alpha_q[k] / piv;
                double w = dse_[k] + ratio * (ratio * wr - 2.0 * tau[k]);
                dse_[k] = std::max(w, 1e-8);
            }
            dse_[r] = std::max(wr / (piv * piv), 1e-8);
        }

        // ---- basis change ----
        status_[p] = to_lower ? VarStatus::AtLower : VarStatus::AtUpper;
        x_[p] = target;
        slot_of_[p] = -1;
        basis_[r] = q; slot_of_[q] = r; status_[q] = VarStatus::Basic;
        bool upd_ok;
        { ScopeTimer st(stats_.t_update); upd_ok = lu_.update(r, spike, alpha_q[r]); }
        if (!upd_ok) recompute();

        ++stats_.iterations;
        if (phase1) ++stats_.phase1_iterations;

        // ---- stall detection -> Bland's rule (Step 6.1 fallback) ----
        // The objective is an O(n+m) sum, so it is sampled every 16 pivots
        // (or every pivot while logging / in Bland mode, so Bland exits promptly).
        double obj = 0.0;
        const int stride = (bland || opt_.verbose >= 2) ? 1 : 16;
        if (stats_.iterations % stride == 0) {
            for (int j = 0; j < NN; ++j) obj += cost_[j] * x_[j];
            // (best_obj starts at -inf; the tolerance term must not turn that into NaN.)
            if (!std::isfinite(best_obj) || obj > best_obj + 1e-9 * (1.0 + std::fabs(best_obj))) {
                best_obj = obj; stall = 0; bland = false;
            } else {
                stall += stride;
                stats_.longest_stall = std::max(stats_.longest_stall, stall);
                if (stall > opt_.stall_limit && opt_.bland_fallback) bland = true;
            }
        }

        if (opt_.verbose >= 3 || (opt_.verbose >= 2 && stats_.iterations % 1000 == 0))
            std::printf("  dual %s it %7d  obj % .10e  flips %d\n", phase1 ? "ph1" : "ph2",
                        stats_.iterations, obj, stats_.bound_flips);
    }
}

// Dual phase 1 by artificial bounding: every variable gets a small box
// ([0,0] if boxed, [0,1] / [-1,0] if one-sided, [-1,1] if free), which makes
// every basis dual feasible after bound flips. Optimizing that problem with
// the dual simplex minimizes the sum of dual infeasibilities of the original.
std::string DualSimplex::dual_phase1() {
    const int NN = N();
    for (int j = 0; j < NN; ++j) {
        bool flo = std::isfinite(lb_orig_[j]), fhi = std::isfinite(ub_orig_[j]);
        if (flo && fhi) { lb_[j] = 0.0; ub_[j] = 0.0; }
        else if (flo) { lb_[j] = 0.0; ub_[j] = 1.0; }
        else if (fhi) { lb_[j] = -1.0; ub_[j] = 0.0; }
        else { lb_[j] = -1.0; ub_[j] = 1.0; }
        if (status_[j] != VarStatus::Basic) {
            if (lb_[j] == ub_[j]) status_[j] = VarStatus::AtLower;
            else status_[j] = d_[j] < 0 ? VarStatus::AtUpper : VarStatus::AtLower;
            x_[j] = status_[j] == VarStatus::AtLower ? lb_[j] : ub_[j];
        }
    }
    std::string st = dual_loop(true);
    lb_ = lb_orig_; ub_ = ub_orig_;
    cost_ = c_orig_;                     // drop any Harris shifts made during phase 1
    compute_duals();
    for (int j = 0; j < NN; ++j) if (status_[j] != VarStatus::Basic) {
        status_[j] = VarStatus::Basic;   // forces place_nonbasic to decide from d_j
        place_nonbasic(j);
    }
    if (st != "optimal") return st;
    ensure_factor();
    compute_primal();
    compute_duals();
    if (count_dual_infeasibilities(1e3 * opt_.dual_tol) > 0) return "dual_infeasible";
    return "optimal";
}

// ================================================== primal simplex cleanup ==

std::string DualSimplex::primal_loop() {
    const int NN = N();
    std::vector<double> col(m_), alpha_q, spike, e(m_), rho, alpha_row;
    std::vector<int> row_nz;
    devex_.assign(NN, 1.0);
    int stall = 0, consecutive_rejects = 0;
    bool bland = false, fresh_unbounded_check = false;
    double best_obj = std::numeric_limits<double>::infinity();

    auto recompute = [&]() { refactor(); compute_primal(); compute_duals(); };
    ensure_factor(); compute_primal(); compute_duals();

    while (true) {
        if (stats_.iterations >= opt_.max_iterations) return "iteration_limit";
        if ((stats_.iterations & 63) == 0) {
            if (out_of_time()) return "time_limit";
            if (opt_.stop_flag && opt_.stop_flag->load(std::memory_order_relaxed)) return "stopped";
        }
        if (lu_.num_updates() >= opt_.refactor_frequency) recompute();

        // ---- CHUZC: entering variable by Devex ----
        int q = -1; double best = 0.0;
        for (int j = 0; j < NN; ++j) {
            VarStatus s = status_[j];
            if (s == VarStatus::Basic || lb_[j] == ub_[j]) continue;
            double dj = d_[j];
            bool bad = (s == VarStatus::AtLower && dj < -opt_.dual_tol) ||
                       (s == VarStatus::AtUpper && dj > opt_.dual_tol) ||
                       (s == VarStatus::AtZero && std::fabs(dj) > opt_.dual_tol);
            if (!bad) continue;
            if (bland) { if (q < 0) q = j; continue; }
            double score = dj * dj / devex_[j];
            if (score > best) { best = score; q = j; }
        }
        if (q < 0) return "optimal";
        const double dir = d_[q] < 0 ? 1.0 : -1.0;

        std::fill(col.begin(), col.end(), 0.0);
        add_column(q, 1.0, col);
        { ScopeTimer st(stats_.t_ftran); lu_.ftran(col, alpha_q, &spike); }

        // ---- Harris two-pass ratio test on the basic variables ----
        double theta_max = std::numeric_limits<double>::infinity();
        for (int k = 0; k < m_; ++k) {
            double a = dir * alpha_q[k];
            if (std::fabs(a) < opt_.pivot_tol) continue;
            int j = basis_[k];
            if (a > 0 && std::isfinite(lb_[j])) theta_max = std::min(theta_max, (x_[j] - lb_[j] + opt_.primal_tol) / a);
            if (a < 0 && std::isfinite(ub_[j])) theta_max = std::min(theta_max, (ub_[j] - x_[j] + opt_.primal_tol) / -a);
        }
        const double range = ub_[q] - lb_[q];
        if (!std::isfinite(theta_max) && !std::isfinite(range)) {
            // Only believe an unbounded ray computed on a fresh factorization.
            if (lu_.num_updates() > 0 || !fresh_unbounded_check) { fresh_unbounded_check = true; recompute(); continue; }
            return "unbounded";
        }
        fresh_unbounded_check = false;

        if (range <= theta_max) {
            // Entering variable reaches its own opposite bound first: bound flip.
            for (int k = 0; k < m_; ++k) x_[basis_[k]] -= dir * range * alpha_q[k];
            if (status_[q] == VarStatus::AtLower) { status_[q] = VarStatus::AtUpper; x_[q] = ub_[q]; }
            else { status_[q] = VarStatus::AtLower; x_[q] = lb_[q]; }
            ++stats_.iterations; ++stats_.primal_iterations; ++stats_.bound_flips;
            continue;
        }
        int r = -1; double amax = 0.0, theta = 0.0;
        for (int k = 0; k < m_; ++k) {
            double a = dir * alpha_q[k];
            if (std::fabs(a) < opt_.pivot_tol) continue;
            int j = basis_[k];
            double t;
            if (a > 0 && std::isfinite(lb_[j])) t = (x_[j] - lb_[j]) / a;
            else if (a < 0 && std::isfinite(ub_[j])) t = (ub_[j] - x_[j]) / -a;
            else continue;
            if (t <= theta_max && (bland ? (r < 0 || basis_[k] < basis_[r]) : std::fabs(a) > amax)) {
                amax = std::fabs(a); r = k; theta = t;
            }
        }
        if (r < 0) {
            if (opt_.verbose) std::printf("  simplex: primal ratio test found no leaving row (theta_max %.3e)\n", theta_max);
            return "numerical_error";
        }
        theta = std::max(theta, 0.0);
        const int p = basis_[r];
        const bool leave_lower = dir * alpha_q[r] > 0;

        // ---- pivot row, and the same FTRAN/row consistency check as the dual ----
        std::fill(e.begin(), e.end(), 0.0); e[r] = 1.0;
        { ScopeTimer st(stats_.t_btran); lu_.btran(e, rho); }
        { ScopeTimer st(stats_.t_row); row_of_tableau(rho, alpha_row, row_nz); }
        const double piv = alpha_q[r];
        if (std::fabs(piv - alpha_row[q]) > 1e-7 * (1.0 + std::fabs(piv))) {
            if (lu_.num_updates() > 0 && consecutive_rejects < 3) { ++consecutive_rejects; recompute(); continue; }
        }
        consecutive_rejects = 0;

        for (int k = 0; k < m_; ++k) if (alpha_q[k] != 0.0) x_[basis_[k]] -= dir * theta * alpha_q[k];
        x_[q] += dir * theta;

        // ---- dual update from the pivot row ----
        const double theta_d = d_[q] / piv;
        for (int j : row_nz) if (status_[j] != VarStatus::Basic) d_[j] -= theta_d * alpha_row[j];
        d_[p] = -theta_d;
        d_[q] = 0.0;

        const double wq = devex_[q];
        for (int j : row_nz) {
            if (status_[j] == VarStatus::Basic || alpha_row[j] == 0.0) continue;
            double ratio = alpha_row[j] / piv;
            devex_[j] = std::max(devex_[j], ratio * ratio * wq);
        }
        devex_[p] = std::max(wq / (piv * piv), 1.0);

        status_[p] = leave_lower ? VarStatus::AtLower : VarStatus::AtUpper;
        x_[p] = leave_lower ? lb_[p] : ub_[p];
        slot_of_[p] = -1;
        basis_[r] = q; slot_of_[q] = r; status_[q] = VarStatus::Basic;
        dse_[r] = 1.0;
        bool upd_ok;
        { ScopeTimer st(stats_.t_update); upd_ok = lu_.update(r, spike, alpha_q[r]); }
        if (!upd_ok) recompute();

        ++stats_.iterations; ++stats_.primal_iterations;

        const int stride = bland ? 1 : 16;
        if (stats_.iterations % stride == 0) {
            double obj = 0.0;
            for (int j = 0; j < NN; ++j) obj += cost_[j] * x_[j];
            if (!std::isfinite(best_obj) || obj < best_obj - 1e-9 * (1.0 + std::fabs(best_obj))) {
                best_obj = obj; stall = 0; bland = false;
            } else {
                stall += stride;
                stats_.longest_stall = std::max(stats_.longest_stall, stall);
                if (stall > opt_.stall_limit && opt_.bland_fallback) bland = true;
            }
        }
    }
}

// ============================================================ top level ==

// ================================================== primal phase 1 ==
//
// Minimizes the sum of infeasibilities of the basic variables. The phase-1
// cost of a basic variable is -1 below its lower bound, +1 above its upper
// bound and 0 inside its box; nonbasic variables sit at bounds and cost 0.
// Costs and duals are rebuilt every iteration (the infeasible set changes as
// the basis moves). The ratio test stops at the first breakpoint: a feasible
// basic variable reaching a bound, or an infeasible one reaching the bound it
// violates (becoming feasible) -- so the sum of infeasibilities never rises.
std::string DualSimplex::primal_phase1() {
    const int NN = N();
    const double INF = std::numeric_limits<double>::infinity();
    const double tol = opt_.primal_tol;
    std::vector<double> col(m_), alpha_q, spike, e(m_), rho, alpha_row;
    std::vector<int> row_nz;
    devex_.assign(NN, 1.0);
    int consecutive_rejects = 0, stall = 0;
    bool bland = false, confirmed = false;
    double best_inf = INF;

    ensure_factor();
    compute_primal();
    while (true) {
        if (stats_.iterations >= opt_.max_iterations) return "iteration_limit";
        if ((stats_.iterations & 63) == 0) {
            if (out_of_time()) return "time_limit";
            if (opt_.stop_flag && opt_.stop_flag->load(std::memory_order_relaxed)) return "stopped";
        }
        if (lu_.num_updates() >= opt_.refactor_frequency) { refactor(); compute_primal(); }

        // ---- phase-1 costs and duals ----
        double sinf = 0.0;
        std::fill(cost_.begin(), cost_.end(), 0.0);
        for (int k = 0; k < m_; ++k) {
            int j = basis_[k];
            if (x_[j] < lb_[j] - tol) { cost_[j] = -1.0; sinf += lb_[j] - x_[j]; }
            else if (x_[j] > ub_[j] + tol) { cost_[j] = 1.0; sinf += x_[j] - ub_[j]; }
        }
        if (sinf == 0.0) return "optimal";
        compute_duals();

        if (sinf < best_inf * (1.0 - 1e-12)) { best_inf = sinf; stall = 0; bland = false; }
        else if (++stall > opt_.stall_limit && opt_.bland_fallback) bland = true;

        // ---- CHUZC: entering variable by Devex ----
        int q = -1; double best = 0.0;
        for (int j = 0; j < NN; ++j) {
            VarStatus s = status_[j];
            if (s == VarStatus::Basic || lb_[j] == ub_[j]) continue;
            double dj = d_[j];
            bool bad = (s == VarStatus::AtLower && dj < -opt_.dual_tol) ||
                       (s == VarStatus::AtUpper && dj > opt_.dual_tol) ||
                       (s == VarStatus::AtZero && std::fabs(dj) > opt_.dual_tol);
            if (!bad) continue;
            if (bland) { if (q < 0) q = j; continue; }
            double score = dj * dj / devex_[j];
            if (score > best) { best = score; q = j; }
        }
        if (q < 0) {
            // Phase-1 optimum with positive infeasibility: confirm on a fresh factor.
            if (!confirmed) { confirmed = true; refactor(); compute_primal(); continue; }
            return "infeasible";
        }
        confirmed = false;
        const double dir = d_[q] < 0 ? 1.0 : -1.0;

        std::fill(col.begin(), col.end(), 0.0);
        add_column(q, 1.0, col);
        { ScopeTimer st(stats_.t_ftran); lu_.ftran(col, alpha_q, &spike); }

        // ---- ratio test to the first breakpoint (Harris two-pass) ----
        // Basic x_j moves by -dir * alpha_q[k] * theta.
        auto breakpoint = [&](int k, bool harris) -> double {
            double a = dir * alpha_q[k];
            if (std::fabs(a) < opt_.pivot_tol) return INF;
            int j = basis_[k];
            double slack = harris ? tol : 0.0;
            if (a > 0) {                                   // x_j decreases
                if (x_[j] > ub_[j] + tol) return (x_[j] - ub_[j] + slack) / a;
                if (x_[j] >= lb_[j] - tol && std::isfinite(lb_[j])) return (x_[j] - lb_[j] + slack) / a;
            } else {                                       // x_j increases
                if (x_[j] < lb_[j] - tol) return (lb_[j] - x_[j] + slack) / -a;
                if (x_[j] <= ub_[j] + tol && std::isfinite(ub_[j])) return (ub_[j] - x_[j] + slack) / -a;
            }
            return INF;
        };
        double theta_max = INF;
        for (int k = 0; k < m_; ++k) theta_max = std::min(theta_max, breakpoint(k, true));
        const double range = ub_[q] - lb_[q];
        if (range <= theta_max) {
            if (!std::isfinite(range)) {
                if (opt_.verbose) std::printf("  simplex: primal phase 1 found no breakpoint\n");
                return "numerical_error";
            }
            for (int k = 0; k < m_; ++k) x_[basis_[k]] -= dir * range * alpha_q[k];
            if (status_[q] == VarStatus::AtLower) { status_[q] = VarStatus::AtUpper; x_[q] = ub_[q]; }
            else { status_[q] = VarStatus::AtLower; x_[q] = lb_[q]; }
            ++stats_.iterations; ++stats_.phase1_iterations; ++stats_.bound_flips;
            continue;
        }
        int r = -1; double amax = 0.0, theta = 0.0;
        for (int k = 0; k < m_; ++k) {
            double t = breakpoint(k, false);
            if (t > theta_max) continue;
            double a = std::fabs(alpha_q[k]);
            if (bland ? (r < 0 || basis_[k] < basis_[r]) : a > amax) { amax = a; r = k; theta = t; }
        }
        if (r < 0) return "numerical_error";
        theta = std::max(theta, 0.0);
        const int p = basis_[r];
        const bool decreasing = dir * alpha_q[r] > 0;
        // The bound p stops at: the violated one if it was infeasible, else the one it reached.
        const bool leave_lower = decreasing ? !(x_[p] > ub_[p] + tol) : (x_[p] < lb_[p] - tol);

        std::fill(e.begin(), e.end(), 0.0); e[r] = 1.0;
        { ScopeTimer st(stats_.t_btran); lu_.btran(e, rho); }
        { ScopeTimer st(stats_.t_row); row_of_tableau(rho, alpha_row, row_nz); }
        const double piv = alpha_q[r];
        if (std::fabs(piv - alpha_row[q]) > 1e-7 * (1.0 + std::fabs(piv))) {
            if (lu_.num_updates() > 0 && consecutive_rejects < 3) {
                ++consecutive_rejects; refactor(); compute_primal(); continue;
            }
        }
        consecutive_rejects = 0;

        for (int k = 0; k < m_; ++k) if (alpha_q[k] != 0.0) x_[basis_[k]] -= dir * theta * alpha_q[k];
        x_[q] += dir * theta;

        const double wq = devex_[q];
        for (int j : row_nz) {
            if (status_[j] == VarStatus::Basic || alpha_row[j] == 0.0) continue;
            double ratio = alpha_row[j] / piv;
            devex_[j] = std::max(devex_[j], ratio * ratio * wq);
        }
        devex_[p] = std::max(wq / (piv * piv), 1.0);

        status_[p] = leave_lower ? VarStatus::AtLower : VarStatus::AtUpper;
        x_[p] = leave_lower ? lb_[p] : ub_[p];
        slot_of_[p] = -1;
        basis_[r] = q; slot_of_[q] = r; status_[q] = VarStatus::Basic;
        dse_[r] = 1.0;
        bool upd_ok;
        { ScopeTimer st(stats_.t_update); upd_ok = lu_.update(r, spike, alpha_q[r]); }
        if (!upd_ok) { refactor(); compute_primal(); }
        ++stats_.iterations; ++stats_.phase1_iterations;
        if (opt_.verbose >= 2 && stats_.iterations % 1000 == 0)
            std::printf("  primal ph1 it %7d  sum infeasibility %.6e\n", stats_.iterations, sinf);
    }
}

std::string DualSimplex::solve_primal(const SimplexOptions& opt) {
    opt_ = opt;
    stats_ = SimplexStats();
    t_start_ = now_seconds();
    cost_ = c_orig_; lb_ = lb_orig_; ub_ = ub_orig_;
    for (int j = 0; j < N(); ++j) if (status_[j] != VarStatus::Basic) place_nonbasic(j);
    std::string st = primal_phase1();
    if (opt_.verbose) std::printf("  simplex: primal phase 1 -> %s after %d its\n", st.c_str(), stats_.phase1_iterations);
    if (st == "optimal") {
        cost_ = c_orig_;
        ensure_factor();
        compute_primal();
        compute_duals();
        st = primal_loop();
        if (opt_.verbose) std::printf("  simplex: primal phase 2 -> %s after %d its\n", st.c_str(), stats_.primal_iterations);
    }
    cost_ = c_orig_;
    if (st == "optimal") { ensure_factor(); compute_primal(); compute_duals(); }
    stats_.seconds = now_seconds() - t_start_;
    if (opt_.verbose)
        std::printf("  simplex: primal %s  its %d (ph1 %d)  flips %d  refactors %d  %.3fs\n", st.c_str(),
                    stats_.iterations, stats_.phase1_iterations, stats_.bound_flips, stats_.refactorizations, stats_.seconds);
    return st;
}

std::string DualSimplex::solve(const SimplexOptions& opt) {
    opt_ = opt;
    stats_ = SimplexStats();
    t_start_ = now_seconds();
    auto finish = [&](const std::string& s) {
        stats_.seconds = now_seconds() - t_start_;
        if (opt_.verbose)
            std::printf("  simplex: %s  its %d (ph1 %d, primal %d)  degenerate %d  stall %d  bland %d  flips %d  refactors %d  repairs %d  %.3fs\n",
                        s.c_str(), stats_.iterations, stats_.phase1_iterations, stats_.primal_iterations,
                        stats_.degenerate_pivots, stats_.longest_stall, stats_.bland_pivots, stats_.bound_flips, stats_.refactorizations,
                        stats_.singular_repairs, stats_.seconds);
        if (opt_.verbose)
            std::printf("  simplex time: factor %.3f btran %.3f row %.3f ftran %.3f dse %.3f update %.3f price %.3f ratio %.3f other %.3f  (LU fill %lld)\n",
                        stats_.t_factor, stats_.t_btran, stats_.t_row, stats_.t_ftran, stats_.t_dse, stats_.t_update,
                        stats_.t_price, stats_.t_ratio,
                        stats_.seconds - stats_.t_factor - stats_.t_btran - stats_.t_row - stats_.t_ftran - stats_.t_dse - stats_.t_update
                            - stats_.t_price - stats_.t_ratio,
                        lu_.fill());
        return s;
    };

    cost_ = c_orig_; lb_ = lb_orig_; ub_ = ub_orig_;
    for (int j = 0; j < N(); ++j) if (status_[j] != VarStatus::Basic) place_nonbasic(j);
    ensure_factor();
    compute_primal();
    compute_duals();

    // Boxed dual infeasibilities are fixed by flipping; anything else needs phase 1.
    int bad = 0;
    double bad_sum = 0.0, cmax = 0.0;
    for (int j = 0; j < N(); ++j) cmax = std::max(cmax, std::fabs(c_orig_[j]));
    for (int j = 0; j < N(); ++j) {
        VarStatus s = status_[j];
        if (s == VarStatus::Basic || lb_[j] == ub_[j]) continue;
        bool boxed = std::isfinite(lb_[j]) && std::isfinite(ub_[j]);
        double dj = d_[j];
        bool wrong = (s == VarStatus::AtLower && dj < -opt_.dual_tol) || (s == VarStatus::AtUpper && dj > opt_.dual_tol) ||
                     (s == VarStatus::AtZero && std::fabs(dj) > opt_.dual_tol);
        if (wrong && !boxed) { ++bad; bad_sum += std::fabs(dj); }
    }
    // With shift_instead_of_phase1, dual_loop's enforce_dual_feasibility shifts
    // these costs instead and restore_costs + the primal cleanup undo the
    // shifts -- but only when the infeasibility is small. Large shifts turn the
    // cleanup into a long primal simplex on a distorted problem (seen on
    // DEGEN3/DFL001 after crossover), so then a real phase 1 runs instead.
    bool shift_ok = opt_.shift_instead_of_phase1 && bad_sum <= 1e-3 * (1.0 + cmax) && bad <= std::max(10, N() / 100);
    if (opt_.verbose && opt_.shift_instead_of_phase1)
        std::printf("  simplex: warm start has %d dual infeasibilities (sum %.2e): %s\n", bad, bad_sum,
                    shift_ok || bad == 0 ? "cost shifting" : "dual phase 1");
    if (bad > 0 && !shift_ok) {
        std::string st = dual_phase1();
        if (opt_.verbose) std::printf("  simplex: dual phase 1 (%d infeasible) -> %s after %d its\n", bad, st.c_str(), stats_.phase1_iterations);
        // "dual_infeasible" after phase 1 is a tolerance-based verdict (dual
        // infeasibilities left above 1e3 * dual_tol). Rather than stop, shift
        // those costs, run phase 2, and let the primal cleanup remove the
        // shifts: it ends in the true optimum or in a genuine unbounded ray.
        if (st == "dual_infeasible") {
            if (opt_.verbose) std::printf("  simplex: continuing with cost shifting; the primal cleanup decides\n");
            st = "optimal";
        }
        if (st != "optimal") return finish(st);
    }

    if (opt_.perturb_costs) perturb_costs();
    for (int round = 0; round < 5; ++round) {
        std::string st = dual_loop(false);
        if (opt_.verbose) std::printf("  simplex: dual phase 2 round %d -> %s after %d its\n", round, st.c_str(), stats_.iterations);
        if (st != "optimal") return finish(st);

        restore_costs();
        if (count_dual_infeasibilities(opt_.dual_tol) > 0) {
            st = primal_loop();
            if (opt_.verbose) std::printf("  simplex: primal cleanup -> %s after %d primal its\n", st.c_str(), stats_.primal_iterations);
            if (st != "optimal") return finish(st);
        }
        ensure_factor();
        compute_primal();
        compute_duals();
        bool primal_ok = true;
        for (int k = 0; k < m_; ++k) if (primal_infeasibility(basis_[k]) != 0.0) { primal_ok = false; break; }
        if (primal_ok && count_dual_infeasibilities(opt_.dual_tol) == 0) return finish("optimal");
        if (opt_.verbose) std::printf("  simplex: primal_ok=%d dual_infeas=%d\n", (int)primal_ok, count_dual_infeasibilities(opt_.dual_tol));
        if (opt_.verbose) std::printf("  simplex: cleanup round %d left infeasibilities, re-entering dual\n", round);
    }
    return finish("numerical_error");
}

// ======================================================== MILP support ==

void DualSimplex::row_entries(int i, std::vector<std::pair<int,double>>& out) const {
    out.clear();
    for (int p = rptr_[i]; p < rptr_[i + 1]; ++p) out.emplace_back(ridx_[p], rval_[p]);
}

// Rebuild CSR and CSC from row lists (structural columns only).
static void build_csr_csc(int n, const std::vector<std::vector<std::pair<int,double>>>& rows,
                          std::vector<int>& rptr, std::vector<int>& ridx, std::vector<double>& rval,
                          std::vector<int>& cptr, std::vector<int>& cidx, std::vector<double>& cval) {
    const int m = (int)rows.size();
    rptr.assign(m + 1, 0);
    for (int i = 0; i < m; ++i) rptr[i + 1] = rptr[i] + (int)rows[i].size();
    ridx.resize(rptr[m]); rval.resize(rptr[m]);
    cptr.assign(n + 1, 0);
    for (int i = 0; i < m; ++i)
        for (size_t k = 0; k < rows[i].size(); ++k) {
            ridx[rptr[i] + k] = rows[i][k].first; rval[rptr[i] + k] = rows[i][k].second;
            cptr[rows[i][k].first + 1]++;
        }
    for (int j = 0; j < n; ++j) cptr[j + 1] += cptr[j];
    cidx.resize(cptr[n]); cval.resize(cptr[n]);
    std::vector<int> cc(cptr.begin(), cptr.end() - 1);
    for (int i = 0; i < m; ++i)
        for (const auto& e : rows[i]) { int q = cc[e.first]++; cidx[q] = i; cval[q] = e.second; }
}

void DualSimplex::add_rows(const std::vector<Row>& rows) {
    if (rows.empty()) return;
    std::vector<std::vector<std::pair<int,double>>> all(m_);
    for (int i = 0; i < m_; ++i) row_entries(i, all[i]);
    for (const auto& r : rows) all.push_back(r.entries);
    build_csr_csc(n_, all, rptr_, ridx_, rval_, cptr_, cidx_, cval_);
    for (const auto& r : rows) {
        double act = 0.0;
        for (const auto& e : r.entries) act += e.second * x_[e.first];
        c_orig_.push_back(0.0); cost_.push_back(0.0);
        lb_orig_.push_back(-r.hi); ub_orig_.push_back(-r.lo);
        lb_.push_back(-r.hi); ub_.push_back(-r.lo);
        status_.push_back(VarStatus::Basic);
        x_.push_back(-act); d_.push_back(0.0); devex_.push_back(1.0);
        slot_of_.push_back(m_);
        basis_.push_back(n_ + m_);
        dse_.push_back(1.0);
        y_.push_back(0.0);
        ++m_;
    }
    row_mark_.clear();
    have_factor_ = false;
}

int DualSimplex::remove_inactive_rows(int first, double slack_tol) {
    std::vector<char> keep(m_, 1);
    int removed = 0;
    for (int i = first; i < m_; ++i) {
        int j = n_ + i;
        if (status_[j] != VarStatus::Basic) continue;
        double s = x_[j];   // logical = -activity; inactive if strictly inside its bounds
        if (s > lb_orig_[j] + slack_tol && s < ub_orig_[j] - slack_tol) { keep[i] = 0; ++removed; }
    }
    if (!removed) return 0;
    std::vector<int> new_row(m_, -1);
    int nm = 0;
    for (int i = 0; i < m_; ++i) if (keep[i]) new_row[i] = nm++;
    std::vector<std::vector<std::pair<int,double>>> all;
    all.reserve(nm);
    for (int i = 0; i < m_; ++i) if (keep[i]) { all.emplace_back(); row_entries(i, all.back()); }
    build_csr_csc(n_, all, rptr_, ridx_, rval_, cptr_, cidx_, cval_);

    auto remap = [&](auto& v) {   // per-variable arrays: drop removed logicals
        std::remove_reference_t<decltype(v)> w;
        w.reserve(n_ + nm);
        for (int j = 0; j < n_; ++j) w.push_back(v[j]);
        for (int i = 0; i < m_; ++i) if (keep[i]) w.push_back(v[n_ + i]);
        v.swap(w);
    };
    remap(c_orig_); remap(cost_); remap(lb_orig_); remap(ub_orig_); remap(lb_); remap(ub_);
    remap(status_); remap(x_); remap(d_); remap(devex_);
    auto new_var = [&](int j) { return j < n_ ? j : n_ + new_row[j - n_]; };
    std::vector<int> nb; std::vector<double> nd;
    for (int k = 0; k < m_; ++k) {
        int j = basis_[k];
        if (j >= n_ && !keep[j - n_]) continue;          // removed logical: its slot goes away
        nb.push_back(new_var(j)); nd.push_back(dse_[k]);
    }
    basis_ = nb; dse_ = nd;
    y_.assign(nm, 0.0);
    m_ = nm;
    slot_of_.assign(n_ + m_, -1);
    for (int k = 0; k < m_; ++k) slot_of_[basis_[k]] = k;
    row_mark_.clear();
    have_factor_ = false;
    return removed;
}

void DualSimplex::tableau_row(int slot, std::vector<double>& alpha) const {
    std::vector<double> e(m_, 0.0), rho;
    e[slot] = 1.0;
    lu_.btran(e, rho);
    alpha.assign(N(), 0.0);
    for (int i = 0; i < m_; ++i) {
        double ri = rho[i];
        if (ri == 0.0) continue;
        for (int p = rptr_[i]; p < rptr_[i + 1]; ++p) alpha[ridx_[p]] += ri * rval_[p];
        alpha[n_ + i] = ri;
    }
}

// ============================================================== results ==

std::vector<double> DualSimplex::primal() const { return std::vector<double>(x_.begin(), x_.begin() + n_); }
std::vector<double> DualSimplex::row_duals() const { return y_; }
std::vector<double> DualSimplex::reduced_costs() const { return std::vector<double>(d_.begin(), d_.begin() + n_); }

double DualSimplex::objective() const {
    double s = obj_offset_;
    for (int j = 0; j < n_; ++j) s += c_orig_[j] * x_[j];
    return s;
}
