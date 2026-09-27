#include "mip.hpp"
#include "simplex.hpp"
#include <cmath>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

constexpr double kInf = std::numeric_limits<double>::infinity();

struct BoundChange { int col; double lo, hi; };

struct Node {
    std::vector<BoundChange> bounds;                  // full list from the root; later entries win
    double lb = -kInf;                                // parent's LP bound
    int depth = 0;
    int branch_var = -1;                              // for pseudocost updates
    int branch_dir = 0;                               // -1 down, +1 up
    double branch_frac = 0.0;                         // distance moved (f or 1-f)
    std::shared_ptr<std::vector<VarStatus>> basis;    // parent's optimal basis (warm start)
};
struct NodeOrder {   // min-heap on the LP bound; among (near-)equal bounds, deeper first
    bool operator()(const Node& a, const Node& b) const {
        double tol = 1e-9 * (1.0 + std::fabs(a.lb));
        if (std::fabs(a.lb - b.lb) > tol) return a.lb > b.lb;
        return a.depth < b.depth;
    }
};

// ---------------------------------------------------------------------------
// State shared by all tree-search workers (MILP Step 9.2). One mutex guards
// the node pool, the incumbent (value AND solution, updated as one guarded
// operation -- MILP Step 9.3), the pseudocosts and the per-worker bounds.
// ---------------------------------------------------------------------------
struct Shared {
    const RangedLP& lp;
    MipOptions opt;
    int n = 0, m0 = 0;
    std::vector<char> is_int;
    std::vector<double> root_lo, root_hi;
    CSR Arows;
    double t0 = 0.0;

    std::mutex mu;
    std::condition_variable cv;
    std::priority_queue<Node, std::vector<Node>, NodeOrder> open;
    int active = 0;                                   // workers currently holding a node
    bool stop = false;
    std::string status = "optimal";
    std::vector<double> worker_lb;                    // bound of each worker's current node (inf = idle)
    long long stored_basis_entries = 0;

    std::atomic<double> inc_obj{kInf};                // readable without the lock
    std::vector<double> inc_x;

    std::vector<double> pc_sum[2];
    std::vector<int> pc_cnt[2];
    double pc_avg_sum[2] = {0, 0};                    // running totals for the unobserved-variable default
    int pc_avg_n[2] = {0, 0};

    std::atomic<long long> nodes{0}, lp_iterations{0};
    std::atomic<int> heuristic_solutions{0};
    MipResult res;

    Shared(const RangedLP& l, const MipOptions& o) : lp(l), opt(o) {}
    double elapsed() const { return now_s() - t0; }

    // Caller holds mu.
    double global_bound_locked() const {
        double b = inc_obj.load();
        if (!open.empty()) b = std::min(b, open.top().lb);
        for (double w : worker_lb) b = std::min(b, w);
        return b;
    }
    bool gap_closed_locked() const {
        double inc = inc_obj.load();
        if (!std::isfinite(inc)) return false;
        return inc - global_bound_locked() <= std::max(opt.abs_gap, opt.rel_gap * std::max(1.0, std::fabs(inc)));
    }
};

// ---------------------------------------------------------------------------
// One tree-search worker: its own LP (a copy of the root LP after cuts).
// ---------------------------------------------------------------------------
class BranchAndCut {
public:
    BranchAndCut(Shared& s, int id) : S(s), id_(id), opt_(s.opt), n_(s.n), m0_(s.m0), is_int_(s.is_int) {}

    bool root();                           // worker 0: root LP, cuts, heuristics. false = done already
    void tree(Node start, bool have_start);
    DualSimplex spx_;

private:
    Shared& S;
    int id_;
    MipOptions opt_;
    int n_, m0_;
    std::vector<char> is_int_;
    std::vector<int> applied_;             // columns whose bounds differ from the root in spx_

    double elapsed() const { return S.elapsed(); }
    bool out_of_time() const { return elapsed() > opt_.time_limit; }
    double inc_obj() const { return S.inc_obj.load(); }

    std::string solve_lp(int max_iterations = -1, bool perturb = false) {
        SimplexOptions o;
        o.perturb_costs = perturb;
        o.time_limit = std::max(1.0, opt_.time_limit - elapsed());
        if (max_iterations > 0) o.max_iterations = max_iterations;
        std::string st = spx_.solve(o);
        S.lp_iterations += spx_.stats().iterations;
        return st;
    }

    void apply_bounds(const std::vector<BoundChange>& bc) {
        for (int j : applied_) spx_.set_col_bounds(j, S.root_lo[j], S.root_hi[j]);
        applied_.clear();
        for (const auto& b : bc) { spx_.set_col_bounds(b.col, b.lo, b.hi); applied_.push_back(b.col); }
    }

    bool is_fractional(int j, double& v) const {
        v = spx_.value(j);
        return std::fabs(v - std::round(v)) > opt_.int_tol;
    }
    void fractional_list(std::vector<int>& out) const {
        out.clear();
        double v;
        for (int j = 0; j < n_; ++j) if (is_int_[j] && is_fractional(j, v)) out.push_back(j);
    }

    // Accept x as incumbent if integral and feasible for the original rows.
    // The feasibility check runs outside the lock; the compare-and-update of
    // (value, solution) runs inside it, as one operation.
    bool try_incumbent(std::vector<double> x, const char* source) {
        for (int j = 0; j < n_; ++j) {
            if (is_int_[j]) {
                if (std::fabs(x[j] - std::round(x[j])) > 1e-5) return false;
                x[j] = std::round(x[j]);
            }
            double tol = 1e-6 * (1.0 + std::fabs(x[j]));
            if (x[j] < S.root_lo[j] - tol || x[j] > S.root_hi[j] + tol) return false;
            x[j] = std::min(std::max(x[j], S.root_lo[j]), S.root_hi[j]);
        }
        auto ax = matvec(S.Arows, x);
        for (int i = 0; i < m0_; ++i) {
            // (|inf| * 0 would be NaN and silently pass every check -- test finiteness explicitly)
            double mag = 0.0;
            if (std::isfinite(S.lp.rL[i])) mag = std::max(mag, std::fabs(S.lp.rL[i]));
            if (std::isfinite(S.lp.rU[i])) mag = std::max(mag, std::fabs(S.lp.rU[i]));
            double tol = 1e-6 * (1.0 + mag);
            if (ax[i] < S.lp.rL[i] - tol || ax[i] > S.lp.rU[i] + tol) return false;
        }
        double obj = S.lp.obj_offset;
        for (int j = 0; j < n_; ++j) obj += S.lp.c[j] * x[j];
        {
            std::lock_guard<std::mutex> lk(S.mu);
            double cur = S.inc_obj.load();
            if (obj >= cur - 1e-9 * (1.0 + std::fabs(cur))) return false;
            S.inc_x = x;
            S.inc_obj.store(obj);
        }
        if (source[0] != 'n') S.heuristic_solutions++;
        if (opt_.verbose)
            std::printf("  * incumbent %.10g  (%s, %lld nodes, %.2fs)\n", obj, source, S.nodes.load(), elapsed());
        return true;
    }

    double cutoff() const {
        double inc = inc_obj();
        return inc - 1e-9 * (1.0 + std::fabs(inc));
    }

    // ---------------------------------------------------------------- cuts
    int gmi_cuts(std::vector<DualSimplex::Row>& out, int max_cuts);
    int cover_cuts(std::vector<DualSimplex::Row>& out);
    int mir_cuts(std::vector<DualSimplex::Row>& out);
    bool finish_cut(std::vector<double>& g, double R, DualSimplex::Row& row) const;
    void root_cuts();

    // ----------------------------------------------------------- heuristics
    void rounding_heuristic();
    void dive(int max_depth);
    void feasibility_pump(int max_rounds);
    void rins(long long node_budget, bool rens = false);

    // ------------------------------------------------------------ branching
    int select_branch(const std::vector<int>& frac, double lp_obj, bool& node_infeasible,
                      std::vector<BoundChange>& fixings);
    double pseudocost_locked(int j, int dir) const {
        if (S.pc_cnt[dir][j] > 0) return S.pc_sum[dir][j] / S.pc_cnt[dir][j];
        return S.pc_avg_n[dir] ? S.pc_avg_sum[dir] / S.pc_avg_n[dir] : 1.0;
    }
    void update_pseudocost(int j, int dir, double delta, double dist) {
        if (dist <= 1e-9 || !std::isfinite(delta)) return;
        double unit = std::max(delta, 0.0) / dist;
        std::lock_guard<std::mutex> lk(S.mu);
        S.pc_sum[dir][j] += unit;
        S.pc_cnt[dir][j]++;
        S.pc_avg_sum[dir] += unit;
        S.pc_avg_n[dir]++;
    }

    void push_node(Node&& nd) {
        std::lock_guard<std::mutex> lk(S.mu);
        if (nd.basis) S.stored_basis_entries += (long long)nd.basis->size();
        S.open.push(std::move(nd));
        S.cv.notify_one();
    }
};

// ======================================================================= cuts

// Clean, scale and accept a cut sum g_j x_j >= R (dense g over structurals).
bool BranchAndCut::finish_cut(std::vector<double>& g, double R, DualSimplex::Row& row) const {
    double gmax = 0.0;
    for (int j = 0; j < n_; ++j) gmax = std::max(gmax, std::fabs(g[j]));
    if (gmax < 1e-9) return false;
    // Drop tiny coefficients, relaxing R by the worst case over the bounds (safe).
    for (int j = 0; j < n_; ++j) {
        if (g[j] == 0.0 || std::fabs(g[j]) >= 1e-9 * gmax) continue;
        double lo = spx_.lower(j), hi = spx_.upper(j);
        double worst = g[j] > 0 ? g[j] * hi : g[j] * lo;
        if (!std::isfinite(worst)) return false;
        R -= worst;
        g[j] = 0.0;
    }
    double gmin = kInf, norm = 0.0, act = 0.0;
    row.entries.clear();
    for (int j = 0; j < n_; ++j) {
        if (g[j] == 0.0) continue;
        gmin = std::min(gmin, std::fabs(g[j]));
        norm += g[j] * g[j];
        act += g[j] * spx_.value(j);
        row.entries.emplace_back(j, g[j] / gmax);
    }
    if (row.entries.empty() || gmax / gmin > 1e6) return false;     // numerically unsafe
    norm = std::sqrt(norm);
    double viol = R - act;
    if (viol / norm < 1e-5 || viol < 1e-6 * (1.0 + std::fabs(R))) return false;
    row.lo = R / gmax; row.hi = kInf;
    return true;
}

// Gomory mixed-integer cuts from the rows of fractional basic integer variables.
int BranchAndCut::gmi_cuts(std::vector<DualSimplex::Row>& out, int max_cuts) {
    const int m = spx_.m(), NN = n_ + m;
    struct Cand { int slot; double score; };
    std::vector<Cand> cand;
    for (int k = 0; k < m; ++k) {
        int j = spx_.basic_var(k);
        if (j >= n_ || !is_int_[j]) continue;
        double v = spx_.value(j), f = v - std::floor(v);
        if (f < 0.01 || f > 0.99) continue;
        cand.push_back({k, std::fabs(f - 0.5)});
    }
    std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.score < b.score; });
    if ((int)cand.size() > max_cuts) cand.resize(max_cuts);

    std::vector<double> alpha, g(n_);
    std::vector<std::pair<int,double>> rowent;
    int made = 0;
    for (const auto& c : cand) {
        spx_.tableau_row(c.slot, alpha);
        const int bj = spx_.basic_var(c.slot);
        const double beta = spx_.value(bj);
        const double f0 = beta - std::floor(beta);
        std::fill(g.begin(), g.end(), 0.0);
        double R = 1.0;       // sum pi_j t_j >= 1, with t_j = z_j - l_j or u_j - z_j
        bool ok = true;
        for (int j = 0; j < NN && ok; ++j) {
            if (j == bj || spx_.status(j) == VarStatus::Basic) continue;
            double a = alpha[j];
            if (std::fabs(a) < 1e-11) continue;
            double lo = spx_.lower(j), hi = spx_.upper(j);
            if (lo == hi) continue;                              // fixed: t_j = 0
            VarStatus s = spx_.status(j);
            if (s == VarStatus::AtZero) { ok = false; break; }   // free nonbasic: no GMI
            // x_bj + sum abar_j t_j = beta: abar = alpha at lower, -alpha at upper
            double abar = s == VarStatus::AtLower ? a : -a;
            bool integral = j < n_ && is_int_[j];
            double pi;
            if (integral) {
                double fj = abar - std::floor(abar);
                pi = fj <= f0 ? fj / f0 : (1.0 - fj) / (1.0 - f0);
            } else {
                pi = abar >= 0 ? abar / f0 : -abar / (1.0 - f0);
            }
            if (pi == 0.0) continue;
            // pi t_j in terms of z_j: lower -> pi (z - l), upper -> pi (u - z)
            double coef = s == VarStatus::AtLower ? pi : -pi;
            R += s == VarStatus::AtLower ? pi * lo : -pi * hi;
            if (j < n_) g[j] += coef;
            else {                                               // logical z = -a_i^T x
                spx_.row_entries(j - n_, rowent);
                for (const auto& e : rowent) g[e.first] -= coef * e.second;
            }
        }
        if (!ok) continue;
        DualSimplex::Row row;
        if (finish_cut(g, R, row)) { out.push_back(std::move(row)); ++made; }
    }
    return made;
}

// Knapsack cover cuts on rows whose variables are all binary.
int BranchAndCut::cover_cuts(std::vector<DualSimplex::Row>& out) {
    int made = 0;
    std::vector<std::pair<int,double>> ent;
    for (int i = 0; i < m0_; ++i) {
        spx_.row_entries(i, ent);
        if (ent.size() < 2) continue;
        bool binary = true;
        for (const auto& e : ent)
            if (!is_int_[e.first] || spx_.lower(e.first) != 0.0 || spx_.upper(e.first) != 1.0) { binary = false; break; }
        if (!binary) continue;
        for (int side = 0; side < 2; ++side) {
            // side 0: a^T x <= rU ; side 1: -a^T x <= -rL
            double b = side == 0 ? spx_.row_upper(i) : -spx_.row_lower(i);
            if (!std::isfinite(b)) continue;
            double sgn = side == 0 ? 1.0 : -1.0;
            // complement negative coefficients: x' = 1 - x
            struct It { int j; double a; bool comp; double xs; };
            std::vector<It> items;
            double bb = b;
            for (const auto& e : ent) {
                double a = sgn * e.second;
                double xv = spx_.value(e.first);
                if (a < 0) { items.push_back({e.first, -a, true, 1.0 - xv}); bb -= a; }
                else if (a > 0) items.push_back({e.first, a, false, xv});
            }
            if (bb < 0) continue;
            std::sort(items.begin(), items.end(), [](const It& p, const It& q) {
                return (1.0 - p.xs) / p.a < (1.0 - q.xs) / q.a;
            });
            double wsum = 0.0; size_t k = 0;
            while (k < items.size() && wsum <= bb + 1e-9) wsum += items[k++].a;
            if (wsum <= bb + 1e-9) continue;                     // no cover
            // make it minimal: drop items (smallest x*) while still a cover
            std::vector<It> cover(items.begin(), items.begin() + k);
            std::sort(cover.begin(), cover.end(), [](const It& p, const It& q) { return p.xs < q.xs; });
            for (size_t t = 0; t < cover.size(); ) {
                if (wsum - cover[t].a > bb + 1e-9) { wsum -= cover[t].a; cover.erase(cover.begin() + t); }
                else ++t;
            }
            double lhs = 0.0;
            for (const auto& it : cover) lhs += it.xs;
            double rhs = (double)cover.size() - 1.0;
            if (lhs <= rhs + 1e-4) continue;
            // sum x'_C <= |C|-1 with x' = 1 - x for complemented items
            DualSimplex::Row row;
            double hi = rhs;
            for (const auto& it : cover) {
                if (it.comp) { row.entries.emplace_back(it.j, -1.0); hi -= 1.0; }
                else row.entries.emplace_back(it.j, 1.0);
            }
            row.lo = -kInf; row.hi = hi;
            out.push_back(std::move(row));
            ++made;
        }
    }
    return made;
}

// Complemented mixed-integer rounding (c-MIR, Marchand & Wolsey) on single
// rows. For each side a^T x <= b: substitute every variable by its nearer
// bound (x = l + t or x = u - t, t >= 0), drop continuous terms with a
// positive coefficient (valid: they are >= 0), then for a few divisors delta
// apply the MIR formula
//   sum (floor(g_j/d) + max(0, f_j - f0)/(1 - f0)) z_j - s/(d (1 - f0)) <= floor(b/d)
// and keep the most violated cut, mapped back to x.
int BranchAndCut::mir_cuts(std::vector<DualSimplex::Row>& out) {
    int made = 0;
    std::vector<std::pair<int,double>> ent;
    const int m = spx_.m();
    // Variable upper bounds x_j <= u * y_k (y binary), read off two-entry rows
    // a x + b y <= 0 (or >= 0). They let the MIR see fixed-charge structure:
    // substituting x = u y - t turns flow rows into flow-cover-strength cuts.
    std::vector<int> vub_y(n_, -1);
    std::vector<double> vub_u(n_, 0.0);
    for (int i = 0; i < m0_; ++i) {
        spx_.row_entries(i, ent);
        if (ent.size() != 2) continue;
        for (int side = 0; side < 2; ++side) {
            double rhs = side == 0 ? spx_.row_upper(i) : -spx_.row_lower(i);
            if (!std::isfinite(rhs) || rhs != 0.0) continue;
            double sg = side == 0 ? 1.0 : -1.0;
            for (int k = 0; k < 2; ++k) {
                int jx = ent[k].first, jy = ent[1 - k].first;
                double ax = sg * ent[k].second, ay = sg * ent[1 - k].second;
                if (is_int_[jx] || !is_int_[jy] || spx_.lower(jy) != 0.0 || spx_.upper(jy) != 1.0) continue;
                if (ax > 0 && ay < 0 && spx_.lower(jx) >= 0.0) { vub_y[jx] = jy; vub_u[jx] = -ay / ax; }
            }
        }
    }

    for (int i = 0; i < m; ++i) {
        spx_.row_entries(i, ent);
        if (ent.size() < 2) continue;
        for (int side = 0; side < 2; ++side) {
            double b = side == 0 ? spx_.row_upper(i) : -spx_.row_lower(i);
            if (!std::isfinite(b)) continue;
            const double sg = side == 0 ? 1.0 : -1.0;
            double act = 0.0;
            for (const auto& e : ent) act += sg * e.second * spx_.value(e.first);
            if (b - act > 0.1 * (1.0 + std::fabs(b))) continue;       // far from binding

            // continuous terms after bound / VUB substitution; kind 0: t = x - l,
            // 1: t = u - x, 2: t = u*y - x (VUB). Integer coefficients accumulate.
            struct CT { int j; double g; int kind; double tval; };
            std::vector<CT> cont;
            std::vector<std::pair<int,double>> gint;
            auto add_int = [&](int j, double g) {
                for (auto& p : gint) if (p.first == j) { p.second += g; return; }
                gint.emplace_back(j, g);
            };
            double bb = b;
            bool ok = true;
            for (const auto& e : ent) {
                int j = e.first; double a = sg * e.second;
                if (is_int_[j]) { add_int(j, a); continue; }
                double lo = spx_.lower(j), hi = spx_.upper(j), xv = spx_.value(j);
                double s_lo = std::isfinite(lo) ? xv - lo : kInf, s_hi = std::isfinite(hi) ? hi - xv : kInf;
                double s_vub = vub_y[j] >= 0 ? vub_u[j] * spx_.value(vub_y[j]) - xv : kInf;
                if (std::isfinite(s_vub) && s_vub <= s_lo && s_vub <= s_hi) {
                    add_int(vub_y[j], a * vub_u[j]);
                    cont.push_back({j, -a, 2, std::max(s_vub, 0.0)});
                } else if (std::isfinite(s_lo) && s_lo <= s_hi) {
                    bb -= a * lo; cont.push_back({j, a, 0, s_lo});
                } else if (std::isfinite(s_hi)) {
                    bb -= a * hi; cont.push_back({j, -a, 1, s_hi});
                } else { ok = false; break; }
            }
            if (!ok || gint.empty()) continue;
            struct IT { int j; double g; bool comp; double tval; };
            std::vector<IT> ints;
            for (const auto& p : gint) {
                int j = p.first; double g = p.second;
                if (g == 0.0) continue;
                double lo = spx_.lower(j), hi = spx_.upper(j), xv = spx_.value(j);
                bool use_lo = std::isfinite(lo) && (!std::isfinite(hi) || xv - lo <= hi - xv);
                if (!use_lo && !std::isfinite(hi)) { ok = false; break; }
                if (use_lo) { bb -= g * lo; ints.push_back({j, g, false, xv - lo}); }
                else        { bb -= g * hi; ints.push_back({j, -g, true, hi - xv}); }
            }
            if (!ok || ints.empty()) continue;

            std::vector<double> deltas;
            for (const auto& t : ints) if (t.tval > 1e-6 && std::fabs(t.g) > 1e-6) deltas.push_back(std::fabs(t.g));
            std::sort(deltas.begin(), deltas.end());
            deltas.erase(std::unique(deltas.begin(), deltas.end()), deltas.end());
            if (deltas.size() > 8) deltas.resize(8);
            if (deltas.empty()) continue;
            size_t nd = deltas.size();
            for (size_t k = 0; k < nd; ++k) for (double dv : {2.0, 4.0, 8.0}) deltas.push_back(deltas[k] / dv);

            double best_viol = 1e-6, best_rhs = 0.0;
            std::vector<double> best_pi, best_psi, pi(ints.size()), psi(cont.size());
            for (double d : deltas) {
                double beta = bb / d, f0 = beta - std::floor(beta);
                if (f0 < 0.05 || f0 > 0.95) continue;
                double lhs = 0.0, norm = 0.0;
                for (size_t k = 0; k < ints.size(); ++k) {
                    double q = ints[k].g / d, fj = q - std::floor(q);
                    pi[k] = std::floor(q) + std::max(0.0, fj - f0) / (1.0 - f0);
                    lhs += pi[k] * ints[k].tval; norm += pi[k] * pi[k];
                }
                for (size_t k = 0; k < cont.size(); ++k) {
                    psi[k] = cont[k].g < 0 ? cont[k].g / (d * (1.0 - f0)) : 0.0;
                    lhs += psi[k] * cont[k].tval; norm += psi[k] * psi[k];
                }
                double viol = (lhs - std::floor(beta)) / std::sqrt(std::max(norm, 1e-12));
                if (viol > best_viol) { best_viol = viol; best_pi = pi; best_psi = psi; best_rhs = std::floor(beta); }
            }
            if (best_pi.empty()) continue;
            // back to x:  sum pi t_int + sum psi t_cont <= R
            std::vector<double> g(n_, 0.0);
            double R = best_rhs;
            for (size_t k = 0; k < ints.size(); ++k) {
                double p = best_pi[k]; if (p == 0.0) continue;
                int j = ints[k].j;
                if (!ints[k].comp) { g[j] += p; R += p * spx_.lower(j); }
                else               { g[j] -= p; R -= p * spx_.upper(j); }
            }
            for (size_t k = 0; k < cont.size(); ++k) {
                double p = best_psi[k]; if (p == 0.0) continue;
                int j = cont[k].j;
                if (cont[k].kind == 0)      { g[j] += p; R += p * spx_.lower(j); }
                else if (cont[k].kind == 1) { g[j] -= p; R -= p * spx_.upper(j); }
                else                        { g[j] -= p; g[vub_y[j]] += p * vub_u[j]; }
            }
            for (auto& v : g) v = -v;           // finish_cut takes g^T x >= R
            DualSimplex::Row row;
            if (finish_cut(g, -R, row)) { out.push_back(std::move(row)); ++made; }
        }
    }
    return made;
}

void BranchAndCut::root_cuts() {
    double obj = spx_.objective();
    double gap_ref = std::fabs(obj);
    int stall_rounds = 0;
    std::vector<int> frac;
    const double cut_deadline = elapsed() + 0.2 * opt_.time_limit;   // root cuts: at most 20% of the budget
    for (int round = 0; round < opt_.cut_rounds && !out_of_time() && elapsed() < cut_deadline; ++round) {
        fractional_list(frac);
        if (frac.empty()) break;
        std::vector<DualSimplex::Row> cuts;
        gmi_cuts(cuts, 100);
        cover_cuts(cuts);
        mir_cuts(cuts);
        if (cuts.empty()) break;
        // parallelism filter: keep a cut only if it is not nearly parallel to one kept
        std::vector<DualSimplex::Row> kept;
        std::vector<double> dense(n_, 0.0);
        for (auto& c : cuts) {
            double nc = 0.0;
            for (const auto& e : c.entries) nc += e.second * e.second;
            bool parallel = false;
            for (const auto& k : kept) {
                for (const auto& e : k.entries) dense[e.first] = e.second;
                double dot = 0.0, nk = 0.0;
                for (const auto& e : c.entries) dot += e.second * dense[e.first];
                for (const auto& e : k.entries) { nk += e.second * e.second; dense[e.first] = 0.0; }
                if (std::fabs(dot) > 0.999 * std::sqrt(nc * nk)) { parallel = true; break; }
            }
            if (!parallel) kept.push_back(std::move(c));
            if (kept.size() >= 200) break;
        }
        spx_.add_rows(kept);
        S.res.cuts_added += (int)kept.size();
        std::string st = solve_lp();
        if (st == "infeasible") { S.res.status = "infeasible"; return; }
        if (st != "optimal") break;
        double nobj = spx_.objective();
        if (opt_.verbose)
            std::printf("  cut round %2d: %3zu cuts, bound %.10g\n", round + 1, kept.size(), nobj);
        // tailing off (MILP Step 7.5): three rounds in a row that each close
        // less than 1% of the gap to the incumbent (or, without one, move the
        // bound by less than 1e-4 relative)
        double inc = inc_obj();
        double need = std::isfinite(inc) ? 1e-2 * std::max(inc - nobj, 1e-9) : 1e-4 * (1.0 + std::fabs(nobj));
        (void)gap_ref;
        if (nobj - obj < need) {
            if (++stall_rounds >= 3) { obj = nobj; break; }
        } else stall_rounds = 0;
        obj = nobj;
    }
    // Keep only binding cuts in the tree LP.
    if (spx_.m() > m0_ && spx_.remove_inactive_rows(m0_) > 0) solve_lp();
}

// ================================================================ heuristics

void BranchAndCut::rounding_heuristic() {
    std::vector<double> x = spx_.primal();
    for (int j = 0; j < n_; ++j) if (is_int_[j]) x[j] = std::round(x[j]);
    try_incumbent(x, "rounding");
}

// Feasibility pump (Fischetti, Glover & Lodi; MILP Step 10.2) on a copy of the
// root LP. Alternates rounding x* to x~ with an LP that minimizes the L1
// distance to x~ over the binaries (plus a fading share of the objective),
// perturbing the rounding when it cycles. Any LP point whose integer
// variables are all integral is feasible for the MILP.
void BranchAndCut::feasibility_pump(int max_rounds) {
    std::vector<int> bins;
    for (int j = 0; j < n_; ++j)
        if (is_int_[j] && spx_.lower(j) == 0.0 && spx_.upper(j) == 1.0) bins.push_back(j);
    if (bins.empty()) return;
    DualSimplex fp = spx_;
    std::vector<double> c0(n_), cfp(n_);
    double cn = 0.0;
    for (int j = 0; j < n_; ++j) { c0[j] = fp.cost(j); cn += c0[j] * c0[j]; }
    cn = std::sqrt(cn);
    const double cscale = cn > 0 ? std::sqrt((double)bins.size()) / cn : 0.0;
    std::vector<double> xs = spx_.primal(), xt(n_), prev(n_, -1.0);
    double alpha = 1.0;
    unsigned seed = 12345;
    auto frac_of = [&](const std::vector<double>& x) {
        for (int j = 0; j < n_; ++j) if (is_int_[j] && std::fabs(x[j] - std::round(x[j])) > opt_.int_tol) return true;
        return false;
    };
    for (int r = 0; r < max_rounds && !out_of_time(); ++r) {
        if (!frac_of(xs)) { try_incumbent(xs, "feasibility pump"); return; }
        for (int j = 0; j < n_; ++j) xt[j] = is_int_[j] ? std::round(xs[j]) : xs[j];
        if (try_incumbent(xt, "feasibility pump")) return;
        bool cycle = true;
        for (int j : bins) if (xt[j] != prev[j]) { cycle = false; break; }
        if (cycle) {   // flip the binaries whose LP value is furthest from the rounding
            std::vector<std::pair<double,int>> d;
            for (int j : bins) d.push_back({std::fabs(xs[j] - xt[j]), j});
            std::sort(d.rbegin(), d.rend());
            seed = seed * 1103515245u + 12345u;
            size_t T = std::min(d.size(), (size_t)(10 + (seed >> 16) % 20));
            for (size_t k = 0; k < T; ++k) xt[d[k].second] = 1.0 - xt[d[k].second];
        }
        prev = xt;
        alpha *= 0.9;
        for (int j = 0; j < n_; ++j) cfp[j] = alpha * cscale * c0[j];
        for (int j : bins) cfp[j] += xt[j] < 0.5 ? 1.0 : -1.0;
        fp.set_costs(cfp);
        SimplexOptions o;
        o.perturb_costs = false;
        o.max_iterations = std::max(5000, 3 * fp.m());
        o.time_limit = std::max(1.0, opt_.time_limit - elapsed());
        std::string st = fp.solve(o);
        S.lp_iterations += fp.stats().iterations;
        if (st != "optimal") return;
        xs = fp.primal();
    }
}

// RINS (Danna, Rothberg & Le Pape; MILP Step 10.3): fix every integer
// variable on which the incumbent and the current LP solution agree and solve
// the restricted sub-MIP with a node budget (same branch-and-cut, one thread,
// no nested RINS). Needs at least 30% of the integers fixed to be worth it.
void BranchAndCut::rins(long long node_budget, bool rens) {
    // RINS: fix integers where incumbent and LP agree. RENS (no incumbent
    // needed): fix integers that are already integral in the LP solution.
    std::vector<double> xinc;
    if (!rens) {
        std::lock_guard<std::mutex> lk(S.mu);
        if (S.inc_x.empty()) return;
        xinc = S.inc_x;
    }
    RangedLP sub = S.lp;
    int nint = 0, fixed = 0;
    for (int j = 0; j < n_; ++j) {
        if (!is_int_[j]) continue;
        ++nint;
        double v = spx_.value(j);
        if (rens) {
            if (std::fabs(v - std::round(v)) < opt_.int_tol) { sub.l[j] = sub.u[j] = std::round(v); ++fixed; }
        } else if (std::fabs(v - xinc[j]) < opt_.int_tol) { sub.l[j] = sub.u[j] = xinc[j]; ++fixed; }
    }
    if (opt_.verbose >= 2) std::printf("  %s candidate: %d of %d integers fixable\n", rens ? "RENS" : "RINS", fixed, nint);
    if (nint == 0 || fixed < 0.3 * nint || fixed == nint) return;
    const RangedLP sub_full = rens ? sub : RangedLP();
    MipOptions o = opt_;
    o.threads = 1; o.verbose = 0; o.rins = false;
    o.node_limit = node_budget;
    o.cut_rounds = 5;
    o.time_limit = std::min(std::max(1.0, opt_.time_limit - elapsed()), 0.1 * opt_.time_limit + 1.0);
    MipResult r = solve_mip(sub, o);
    if (rens && r.status == "infeasible") {
        // Too tight: keep only the variables integral at a non-lower value fixed
        // (e.g. the ones of an assignment), free the rest.
        sub = sub_full;
        int kept = 0;
        for (int j = 0; j < n_; ++j) {
            if (!is_int_[j] || sub.l[j] != sub.u[j]) continue;
            if (sub.l[j] == S.lp.l[j]) { sub.l[j] = S.lp.l[j]; sub.u[j] = S.lp.u[j]; }
            else ++kept;
        }
        if (kept > 0) r = solve_mip(sub, o);
    }
    if (opt_.verbose >= 2)
        std::printf("  %s: fixed %d/%d integers -> %s, %lld nodes, %.2fs\n", rens ? "RENS" : "RINS", fixed, nint,
                    r.status.c_str(), r.nodes, r.seconds);
    if (r.has_solution) try_incumbent(r.x, rens ? "RENS" : "RINS");
}

// Fractional diving: fix the least fractional variable to its nearest integer
// and re-solve, until integral, infeasible or worse than the incumbent.
void BranchAndCut::dive(int max_depth) {
    auto saved_basis = spx_.get_basis();
    std::vector<std::pair<int, std::pair<double,double>>> changed;
    std::vector<int> frac;
    for (int d = 0; d < max_depth && !out_of_time(); ++d) {
        fractional_list(frac);
        if (frac.empty()) { try_incumbent(spx_.primal(), "diving"); break; }
        int best = -1; double bf = kInf;
        for (int j : frac) {
            double v = spx_.value(j), f = std::fabs(v - std::round(v));
            if (f < bf) { bf = f; best = j; }
        }
        double v = std::round(spx_.value(best));
        changed.push_back({best, {spx_.lower(best), spx_.upper(best)}});
        spx_.set_col_bounds(best, v, v);
        std::string st = solve_lp(std::max(1000, 2 * spx_.m()));
        if (st != "optimal" || spx_.objective() >= cutoff()) break;
    }
    for (auto it = changed.rbegin(); it != changed.rend(); ++it) spx_.set_col_bounds(it->first, it->second.first, it->second.second);
    spx_.set_basis(saved_basis);
}

// ================================================================= branching

// Reliability branching. Returns the branching column; may report the node
// infeasible, or add bound fixings proven by strong branching.
int BranchAndCut::select_branch(const std::vector<int>& frac, double lp_obj, bool& node_infeasible,
                                std::vector<BoundChange>& fixings) {
    node_infeasible = false;
    std::vector<double> xval(n_);
    for (int j : frac) xval[j] = spx_.value(j);
    auto score = [](double dn, double up) { return std::max(dn, 1e-6) * std::max(up, 1e-6); };

    struct C { int j; double s; bool reliable; };
    std::vector<C> cs;
    {
        std::lock_guard<std::mutex> lk(S.mu);
        for (int j : frac) {
            double f = xval[j] - std::floor(xval[j]);
            bool rel = std::min(S.pc_cnt[0][j], S.pc_cnt[1][j]) >= opt_.reliability;
            cs.push_back({j, score(pseudocost_locked(j, 0) * f, pseudocost_locked(j, 1) * (1 - f)), rel});
        }
    }
    std::sort(cs.begin(), cs.end(), [](const C& a, const C& b) { return a.s > b.s; });

    int best = cs[0].j; double best_s = -1.0;
    auto basis = spx_.get_basis();
    int sb_done = 0;
    for (const auto& c : cs) {
        if (c.reliable || sb_done >= opt_.strong_branch_candidates || out_of_time()) {
            if (c.s > best_s) { best_s = c.s; best = c.j; }
            continue;
        }
        ++sb_done;
        const int j = c.j;
        const double v = xval[j], f = v - std::floor(v);
        const double lo = spx_.lower(j), hi = spx_.upper(j);
        double delta[2];
        for (int dir = 0; dir < 2; ++dir) {
            if (dir == 0) spx_.set_col_bounds(j, lo, std::floor(v));
            else spx_.set_col_bounds(j, std::ceil(v), hi);
            std::string st = solve_lp(opt_.strong_branch_iterations);
            delta[dir] = st == "infeasible" ? kInf : std::max(spx_.objective() - lp_obj, 0.0);
            spx_.set_col_bounds(j, lo, hi);
            // No basis restore per trial: the dual simplex leaves a dual-feasible
            // basis even at its iteration limit, so the next trial starts from
            // here and the current LU factorization stays valid (a restore would
            // force a full refactorization every trial). Restored once below.
            if (std::isfinite(delta[dir]) && st == "optimal") update_pseudocost(j, dir, delta[dir], dir == 0 ? f : 1 - f);
        }
        if (!std::isfinite(delta[0]) && !std::isfinite(delta[1])) { node_infeasible = true; return j; }
        if (!std::isfinite(delta[0])) { fixings.push_back({j, std::ceil(v), hi}); continue; }
        if (!std::isfinite(delta[1])) { fixings.push_back({j, lo, std::floor(v)}); continue; }
        double s = score(delta[0], delta[1]);
        if (s > best_s) { best_s = s; best = j; }
    }
    if (sb_done > 0) spx_.set_basis(basis);
    // If the chosen variable was fixed by strong branching, fall back to another.
    for (const auto& fx : fixings) if (fx.col == best) {
        best = -1;
        for (const auto& c : cs) {
            bool fixed = false;
            for (const auto& f2 : fixings) fixed |= f2.col == c.j;
            if (!fixed) { best = c.j; break; }
        }
        break;
    }
    return best;
}

// ====================================================================== root

bool BranchAndCut::root() {
    std::string st = solve_lp(-1, true);
    S.res.root_lp = spx_.objective();
    if (st == "infeasible") { S.res.status = "infeasible"; return false; }
    if (st == "dual_infeasible" || st == "unbounded") { S.res.status = "unbounded"; return false; }
    if (st != "optimal") { S.res.status = st; return false; }
    if (opt_.verbose) std::printf("  root LP: %.10g  (%lld its, %.2fs)\n", S.res.root_lp, S.lp_iterations.load(), elapsed());

    if (opt_.heuristics) rounding_heuristic();
    if (opt_.cuts) {
        root_cuts();
        if (S.res.status == "infeasible") return false;
    }
    S.res.root_after_cuts = spx_.objective();
    if (opt_.heuristics) {
        rounding_heuristic();
        if (opt_.rins) rins(500, true);                        // RENS
        if (!std::isfinite(inc_obj())) feasibility_pump(100);
        dive(n_);
        if (opt_.rins && std::isfinite(inc_obj())) rins(500);
    }
    // The tree's first node re-solves the root LP: a dive leaves spx_ holding
    // the dive's last solution, not the root's.
    return true;
}

// ===================================================================== tree

void BranchAndCut::tree(Node cur, bool have_cur) {
    bool cur_solved = false;
    std::vector<int> frac;
    int plunge_depth = 0;
    double last_log = elapsed();
    const long long basis_budget = 200000000;          // ~200 MB of stored node bases

    auto release = [&]() {                              // drop the current node
        have_cur = false;
        std::lock_guard<std::mutex> lk(S.mu);
        S.active--;
        S.worker_lb[id_] = kInf;
        if (S.active == 0 && S.open.empty()) { S.stop = true; S.cv.notify_all(); }
    };
    auto set_status_stop = [&](const char* st) {
        std::lock_guard<std::mutex> lk(S.mu);
        if (!S.stop) S.status = st;
        S.stop = true;
        S.cv.notify_all();
    };

    if (have_cur) {
        std::lock_guard<std::mutex> lk(S.mu);
        S.active++;
        S.worker_lb[id_] = cur.lb;
    }

    std::string st;
    while (true) {
        if (!have_cur) {
            std::unique_lock<std::mutex> lk(S.mu);
            while (S.open.empty() && !S.stop) {
                if (S.active == 0) { S.stop = true; S.cv.notify_all(); break; }
                S.cv.wait(lk);
            }
            if (S.stop) break;
            cur = S.open.top(); S.open.pop();
            if (cur.basis) S.stored_basis_entries -= (long long)cur.basis->size();
            S.active++;
            S.worker_lb[id_] = cur.lb;
            have_cur = true; cur_solved = false; plunge_depth = 0;
        }
        {
            std::lock_guard<std::mutex> lk(S.mu);
            bool stop = S.stop;
            if (!stop && S.gap_closed_locked()) { S.stop = true; S.cv.notify_all(); stop = true; }
            if (stop) {   // hand the unexplored node back so the final bound accounts for it
                S.open.push(cur);
                S.active--; S.worker_lb[id_] = kInf;
                have_cur = false;
                break;
            }
        }
        if (out_of_time()) { set_status_stop("time_limit"); continue; }
        if (S.nodes.load() >= opt_.node_limit) { set_status_stop("node_limit"); continue; }
        if (cur.lb >= cutoff()) { release(); continue; }

        if (!cur_solved) {
            apply_bounds(cur.bounds);
            if (cur.basis) spx_.set_basis(*cur.basis);
            st = solve_lp();
        } else st = "optimal";
        cur_solved = false;
        S.nodes++;

        if (st == "infeasible") { release(); continue; }
        if (st != "optimal") {
            if (out_of_time()) { set_status_stop("time_limit"); continue; }
            release(); continue;          // numerical trouble at this node: drop it
        }
        const double obj = spx_.objective();
        if (cur.branch_var >= 0) update_pseudocost(cur.branch_var, cur.branch_dir > 0 ? 1 : 0, obj - cur.lb, cur.branch_frac);
        cur.branch_var = -1;
        cur.lb = std::max(cur.lb, obj);
        if (cur.lb >= cutoff()) { release(); continue; }
        { std::lock_guard<std::mutex> lk(S.mu); S.worker_lb[id_] = cur.lb; }

        fractional_list(frac);
        if (frac.empty()) { try_incumbent(spx_.primal(), "node"); release(); continue; }

        if (opt_.heuristics && opt_.dive_frequency > 0 && S.nodes.load() % opt_.dive_frequency == 0) {
            dive(50);
            if (opt_.rins && id_ == 0 && std::isfinite(inc_obj()) && S.nodes.load() % (5 * opt_.dive_frequency) == 0) rins(200);
            if (cur.lb >= cutoff()) { release(); continue; }
            // the dive left its own last LP in spx_: re-solve this node (restored basis, ~0 pivots)
            st = solve_lp();
            if (st != "optimal") { release(); continue; }
            fractional_list(frac);
            if (frac.empty()) { try_incumbent(spx_.primal(), "node"); release(); continue; }
        }

        // reduced-cost fixing (local to this subtree)
        std::vector<BoundChange> fix;
        const double inc = inc_obj();
        if (std::isfinite(inc)) {
            double gap = inc - cur.lb;
            for (int j = 0; j < n_; ++j) {
                if (!is_int_[j] || spx_.status(j) == VarStatus::Basic) continue;
                double d = spx_.reduced_cost(j), lo = spx_.lower(j), hi = spx_.upper(j);
                if (lo == hi) continue;
                if (spx_.status(j) == VarStatus::AtLower && d > 1e-9) {
                    double nh = lo + std::floor(gap / d + 1e-9);
                    if (nh < hi) fix.push_back({j, lo, nh});
                } else if (spx_.status(j) == VarStatus::AtUpper && d < -1e-9) {
                    double nl = hi - std::floor(gap / -d + 1e-9);
                    if (nl > lo) fix.push_back({j, nl, hi});
                }
            }
        }

        bool infeasible = false;
        int j = select_branch(frac, obj, infeasible, fix);
        if (infeasible) { release(); continue; }
        if (j < 0) {   // every candidate got fixed: re-solve this node with the fixings
            cur.bounds.insert(cur.bounds.end(), fix.begin(), fix.end());
            continue;
        }
        // Strong branching restored the basis but not the solution values, and
        // may have added fixings: re-solve (usually zero pivots) before branching.
        double xv = 0.0;
        {
            apply_bounds(cur.bounds);
            for (const auto& b : fix) { spx_.set_col_bounds(b.col, b.lo, b.hi); applied_.push_back(b.col); }
            st = solve_lp();
            if (st != "optimal" || spx_.objective() >= cutoff()) { release(); continue; }
            double vv;
            if (!is_fractional(j, vv)) {   // fixings changed the LP solution: treat as a fresh node
                cur.bounds.insert(cur.bounds.end(), fix.begin(), fix.end());
                cur.lb = std::max(cur.lb, spx_.objective());
                cur_solved = true;
                continue;
            }
            xv = vv;
        }
        const double f = xv - std::floor(xv);
        const double lo = spx_.lower(j), hi = spx_.upper(j);

        Node down, up;
        down.bounds = cur.bounds; down.bounds.insert(down.bounds.end(), fix.begin(), fix.end());
        up.bounds = down.bounds;
        down.bounds.push_back({j, lo, std::floor(xv)});
        up.bounds.push_back({j, std::ceil(xv), hi});
        down.lb = up.lb = cur.lb;
        down.depth = up.depth = cur.depth + 1;
        down.branch_var = up.branch_var = j;
        down.branch_dir = -1; up.branch_dir = +1;
        down.branch_frac = f; up.branch_frac = 1.0 - f;

        // Plunge into the child the LP solution leans towards; queue the other.
        bool go_up = f > 0.5;
        Node& next = go_up ? up : down;
        Node& other = go_up ? down : up;
        if (S.stored_basis_entries < basis_budget)
            other.basis = std::make_shared<std::vector<VarStatus>>(spx_.get_basis());
        push_node(std::move(other));
        bool switch_to_best = false;
        {
            std::lock_guard<std::mutex> lk(S.mu);
            switch_to_best = ++plunge_depth > 200 && !S.open.empty() && next.lb > S.open.top().lb + 1e-9;
        }
        if (switch_to_best) {   // long plunge while better nodes wait: go back to best-bound
            if (S.stored_basis_entries < basis_budget)
                next.basis = std::make_shared<std::vector<VarStatus>>(spx_.get_basis());
            push_node(std::move(next));
            release();
        } else {
            cur = std::move(next);
            cur_solved = false;
            std::lock_guard<std::mutex> lk(S.mu);
            S.worker_lb[id_] = cur.lb;
        }

        if (id_ == 0 && opt_.verbose && elapsed() - last_log > 5.0) {
            last_log = elapsed();
            std::lock_guard<std::mutex> lk(S.mu);
            double b = S.global_bound_locked(), incv = inc_obj();
            std::printf("  nodes %8lld  open %7zu  incumbent %-16.10g bound %-16.10g gap %6.2f%%  %.1fs\n",
                        S.nodes.load(), S.open.size(), incv, b,
                        std::isfinite(incv) ? 100.0 * (incv - b) / std::max(1.0, std::fabs(incv)) : 100.0, elapsed());
        }
    }
}

} // namespace

MipResult solve_mip(const RangedLP& lp, const MipOptions& opt) {
    Shared S(lp, opt);
    S.t0 = now_s();
    S.n = lp.n(); S.m0 = lp.m();
    S.is_int.assign(S.n, 0);
    for (int j = 0; j < S.n && j < (int)lp.integer.size(); ++j) S.is_int[j] = lp.integer[j];
    for (int d = 0; d < 2; ++d) { S.pc_sum[d].assign(S.n, 0.0); S.pc_cnt[d].assign(S.n, 0); }
    S.Arows = to_csr(lp.A);
    S.root_lo = lp.l; S.root_hi = lp.u;
    const int T = std::max(1, opt.threads);
    S.worker_lb.assign(T, kInf);

    auto finish = [&](MipResult r) {
        r.nodes = S.nodes.load();
        r.lp_iterations = S.lp_iterations.load();
        r.heuristic_solutions = S.heuristic_solutions.load();
        r.seconds = S.elapsed();
        return r;
    };

    BranchAndCut w0(S, 0);
    w0.spx_.load(lp);
    if (!w0.root()) {
        MipResult r = S.res;
        return finish(r);
    }

    Node root;
    root.lb = S.res.root_after_cuts;
    if (T == 1) {
        w0.tree(root, true);
    } else {
        // Every worker gets its own copy of the root LP (cuts included).
        std::vector<std::unique_ptr<BranchAndCut>> workers;
        for (int t = 1; t < T; ++t) {
            workers.push_back(std::make_unique<BranchAndCut>(S, t));
            workers.back()->spx_ = w0.spx_;
        }
        std::vector<std::thread> threads;
        for (auto& w : workers) threads.emplace_back([&w]() { w->tree(Node(), false); });
        w0.tree(root, true);
        for (auto& th : threads) th.join();
    }

    MipResult r = S.res;
    r.status = S.status;
    double inc = S.inc_obj.load();
    r.has_solution = std::isfinite(inc);
    r.objective = inc;
    r.x = S.inc_x;
    {
        std::lock_guard<std::mutex> lk(S.mu);
        r.bound = S.global_bound_locked();
    }
    if (r.has_solution) {
        r.bound = std::min(r.bound, inc);
        r.gap = (inc - r.bound) / std::max(1.0, std::fabs(inc));
    }
    if (r.status == "optimal" && !r.has_solution) r.status = "infeasible";
    return finish(r);
}
