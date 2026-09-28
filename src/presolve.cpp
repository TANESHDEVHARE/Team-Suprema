#include "presolve.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

namespace {

constexpr double INFV = std::numeric_limits<double>::infinity();
constexpr double ZERO_TOL = 1e-12;

struct Entry { int r, c; double v; bool alive; };

// Working copy of the problem. Rows and columns keep their original indices;
// the matrix is an entry pool indexed from both sides, with lazy deletion.
class Presolver {
public:
    Presolver(const RangedLP& lp, bool mip) : mip_(mip) {
        m0_ = lp.m(); n0_ = lp.n();
        c_ = lp.c; rL_ = lp.rL; rU_ = lp.rU; l_ = lp.l; u_ = lp.u; offset_ = lp.obj_offset;
        rowAct_.assign(m0_, 1); colAct_.assign(n0_, 1);
        inQ_.assign(n0_, 0);
        for (int k = 0; k < lp.Q.nnz(); ++k) { inQ_[lp.Q.row_idx[k]] = 1; inQ_[lp.Q.col_idx[k]] = 1; }
        isInt_.assign(n0_, 0);
        if (!lp.integer.empty()) for (int j = 0; j < n0_; ++j) isInt_[j] = lp.integer[j];
        rowE_.assign(m0_, {}); colE_.assign(n0_, {});
        aok_.assign(m0_, 0); amin_.assign(m0_, 0.0); amax_.assign(m0_, 0.0); anmin_.assign(m0_, 0); anmax_.assign(m0_, 0);
        rowLen_.assign(m0_, 0); colLen_.assign(n0_, 0);
        pool_.reserve(lp.A.nnz() + lp.A.nnz() / 4);
        for (int k = 0; k < lp.A.nnz(); ++k) {
            if (lp.A.val[k] == 0.0) continue;
            add_entry(lp.A.row_idx[k], lp.A.col_idx[k], lp.A.val[k]);
        }
        rowmag_.assign(m0_, 0.0);
        for (int i = 0; i < m0_; ++i) {
            if (std::isfinite(rL_[i])) rowmag_[i] = std::max(rowmag_[i], std::fabs(rL_[i]));
            if (std::isfinite(rU_[i])) rowmag_[i] = std::max(rowmag_[i], std::fabs(rU_[i]));
        }
    }

    // Integer bounds rounded inward; false if an integer domain is empty.
    bool round_integer_bounds() {
        for (int j = 0; j < n0_; ++j) if (isInt_[j]) {
            if (std::isfinite(l_[j])) l_[j] = std::ceil(l_[j] - 1e-9);
            if (std::isfinite(u_[j])) u_[j] = std::floor(u_[j] + 1e-9);
            if (l_[j] > u_[j]) return false;
        }
        return true;
    }

    std::string run(int max_passes) {
        for (passes_ = 0; passes_ < max_passes; ++passes_) {
            bool changed = false;
            for (int i = 0; i < m0_ && status_.empty(); ++i) if (rowAct_[i]) changed |= row_rules(i);
            for (int j = 0; j < n0_ && status_.empty(); ++j) if (colAct_[j]) changed |= col_rules(j);
            if (!status_.empty()) return status_;
            if (!changed) break;
        }
        return "ok";
    }

    void build(const RangedLP& orig, PresolveResult& res) const {
        std::vector<int> newr(m0_, -1), newc(n0_, -1);
        for (int i = 0; i < m0_; ++i) if (rowAct_[i]) { newr[i] = (int)res.row_ids.size(); res.row_ids.push_back(i); }
        for (int j = 0; j < n0_; ++j) if (colAct_[j]) { newc[j] = (int)res.col_ids.size(); res.col_ids.push_back(j); }
        RangedLP& R = res.reduced;
        R.name = orig.name + " (presolved)";
        R.original_sense = orig.original_sense;
        R.obj_offset = offset_;
        for (int i : res.row_ids) { R.row_names.push_back(orig.row_names[i]); R.rL.push_back(rL_[i]); R.rU.push_back(rU_[i]); }
        for (int j : res.col_ids) { R.col_names.push_back(orig.col_names[j]); R.c.push_back(c_[j]); R.l.push_back(l_[j]); R.u.push_back(u_[j]); }
        R.A.rows = (int)res.row_ids.size(); R.A.cols = (int)res.col_ids.size();
        for (int j : res.col_ids)
            for (int e : colE_[j]) {
                const Entry& E = pool_[e];
                if (!E.alive) continue;
                R.A.row_idx.push_back(newr[E.r]); R.A.col_idx.push_back(newc[j]); R.A.val.push_back(E.v);
            }
        if (orig.Q.nnz() > 0) {                   // Q columns are never removed
            R.Q.rows = R.Q.cols = (int)res.col_ids.size();
            for (int k = 0; k < orig.Q.nnz(); ++k) {
                R.Q.row_idx.push_back(newc[orig.Q.row_idx[k]]);
                R.Q.col_idx.push_back(newc[orig.Q.col_idx[k]]);
                R.Q.val.push_back(orig.Q.val[k]);
            }
        }
        if (!orig.integer.empty())
            for (int j : res.col_ids) R.integer.push_back(orig.integer[j]);
    }

    std::vector<Step> steps;
    int passes() const { return passes_; }

private:
    // ------------------------------------------------------------ storage
    void add_entry(int r, int c, double v) {
        pool_.push_back({r, c, v, true});
        aok_[r] = 0;
        int e = (int)pool_.size() - 1;
        rowE_[r].push_back(e); colE_[c].push_back(e);
        rowLen_[r]++; colLen_[c]++;
    }
    void kill(int e) {
        Entry& E = pool_[e];
        if (!E.alive) return;
        E.alive = false;
        aok_[E.r] = 0;
        rowLen_[E.r]--; colLen_[E.c]--;
    }
    // Alive entry ids of a row / column (compacting the list in place).
    const std::vector<int>& row(int i) { compact(rowE_[i], rowLen_[i]); return rowE_[i]; }
    const std::vector<int>& col(int j) { compact(colE_[j], colLen_[j]); return colE_[j]; }
    void compact(std::vector<int>& list, int len) {
        if ((int)list.size() == len) return;
        list.erase(std::remove_if(list.begin(), list.end(), [&](int e) { return !pool_[e].alive; }), list.end());
    }
    int find(int r, int c) {
        const auto& L = rowLen_[r] <= colLen_[c] ? row(r) : col(c);
        for (int e : L) if (pool_[e].r == r && pool_[e].c == c) return e;
        return -1;
    }
    void remove_row(int i) { for (int e : row(i)) kill(e); rowE_[i].clear(); rowAct_[i] = 0; }
    void remove_col(int j) { for (int e : col(j)) kill(e); colE_[j].clear(); colAct_[j] = 0; }
    double rtol(int i) const { return 1e-9 * (1.0 + rowmag_[i]); }
    bool eliminable(int j) const { return !inQ_[j] && !(mip_ && isInt_[j]); }

    // Fix column j at v: shift its rows, move its cost into the offset.
    void fix_col(int j, double v, const char* kind) {
        Step s; s.kind = kind; s.col_id = j; s.value = v; s.cost = c_[j];
        for (int e : col(j)) {
            const Entry& E = pool_[e];
            s.column.emplace_back(E.r, E.v);
            shift_row(E.r, E.v * v);
        }
        offset_ += c_[j] * v;
        steps.push_back(std::move(s));
        remove_col(j);
    }
    void shift_row(int r, double d) {
        if (d == 0.0) return;
        rL_[r] -= d; rU_[r] -= d;
        rowmag_[r] += std::fabs(d);
    }

    void set_bounds(int j, double lo, double hi) {
        for (int e : col(j)) aok_[pool_[e].r] = 0;
        l_[j] = lo; u_[j] = hi;
    }

    // Activity range of row i (sums of finite parts, counts of infinite
    // ones), cached until an entry of the row or a bound of one of its
    // variables changes.
    void activity(int i, double& mn, double& mx, int& ninf_mn, int& ninf_mx) {
        if (aok_[i]) { mn = amin_[i]; mx = amax_[i]; ninf_mn = anmin_[i]; ninf_mx = anmax_[i]; return; }
        mn = mx = 0.0; ninf_mn = ninf_mx = 0;
        for (int e : row(i)) {
            const Entry& E = pool_[e];
            double lo = l_[E.c], hi = u_[E.c], a = E.v;
            if (a > 0) {
                if (std::isfinite(lo)) mn += a * lo; else ++ninf_mn;
                if (std::isfinite(hi)) mx += a * hi; else ++ninf_mx;
            } else {
                if (std::isfinite(hi)) mn += a * hi; else ++ninf_mn;
                if (std::isfinite(lo)) mx += a * lo; else ++ninf_mx;
            }
        }
        amin_[i] = mn; amax_[i] = mx; anmin_[i] = ninf_mn; anmax_[i] = ninf_mx; aok_[i] = 1;
    }

    // ------------------------------------------------------------ row rules
    bool row_rules(int i) {
        const int len = rowLen_[i];
        const double tol = rtol(i);
        if (len == 0) {
            if (!(rL_[i] <= tol && rU_[i] >= -tol)) { status_ = "infeasible"; return false; }
            Step s; s.kind = "R1"; s.row_id = i; steps.push_back(s);
            rowAct_[i] = 0;
            return true;
        }
        if (len == 1) return singleton_row(i);

        double mn, mx; int nmn, nmx;
        activity(i, mn, mx, nmn, nmx);
        const double itol = 1e-6 * (1.0 + rowmag_[i] + (nmn == 0 ? std::fabs(mn) : 0.0) + (nmx == 0 ? std::fabs(mx) : 0.0));
        if ((nmn == 0 && mn > rU_[i] + itol) || (nmx == 0 && mx < rL_[i] - itol)) { status_ = "infeasible"; return false; }
        const bool lo_ok = !std::isfinite(rL_[i]) || (nmn == 0 && mn >= rL_[i] - tol);
        const bool hi_ok = !std::isfinite(rU_[i]) || (nmx == 0 && mx <= rU_[i] + tol);
        if (lo_ok && hi_ok) {                                   // redundant row
            Step s; s.kind = "RR"; s.row_id = i; steps.push_back(s);
            remove_row(i);
            return true;
        }
        if (std::isfinite(rU_[i]) && nmn == 0 && std::fabs(mn - rU_[i]) <= tol) return forcing_row(i, true);
        if (std::isfinite(rL_[i]) && nmx == 0 && std::fabs(mx - rL_[i]) <= tol) return forcing_row(i, false);
        if (rL_[i] == rU_[i]) {
            if (len == 2 && doubleton(i)) return true;
            if (len <= 40) return aggregate(i);
        }
        return false;
    }

    bool singleton_row(int i) {
        const int e = row(i)[0];
        const int j = pool_[e].c;
        const double a = pool_[e].v;
        double lo_row = (a > 0 ? rL_[i] : rU_[i]) / a, hi_row = (a > 0 ? rU_[i] : rL_[i]) / a;
        if (!std::isfinite(rL_[i]) && a > 0) lo_row = -INFV;
        if (!std::isfinite(rU_[i]) && a > 0) hi_row = INFV;
        if (!std::isfinite(rU_[i]) && a < 0) lo_row = -INFV;
        if (!std::isfinite(rL_[i]) && a < 0) hi_row = INFV;
        double nlo = std::max(l_[j], lo_row), nhi = std::min(u_[j], hi_row);
        const double btol = 1e-9 * (1.0 + rowmag_[i] / std::fabs(a));
        if (isInt_[j]) {
            if (std::isfinite(nlo)) nlo = std::ceil(nlo - std::max(1e-9, btol));
            if (std::isfinite(nhi)) nhi = std::floor(nhi + std::max(1e-9, btol));
        }
        if (nlo - nhi > std::max(1e-7, 100.0 * btol)) { status_ = "infeasible"; return false; }
        if (nlo > nhi) nlo = nhi = 0.5 * (nlo + nhi);
        Step s; s.kind = "R4"; s.row_id = i; s.col_id = j; s.a_ij = a;
        s.lo_from_row = lo_row; s.hi_from_row = hi_row; s.lo_before = l_[j]; s.hi_before = u_[j];
        steps.push_back(s);
        set_bounds(j, nlo, nhi);
        remove_row(i);
        return true;
    }

    // Row activity pinned at a bound: every variable sits at the bound that
    // produces it. upper: min activity == rU.
    bool forcing_row(int i, bool upper) {
        for (int e : row(i)) if (inQ_[pool_[e].c]) return false;
        Step s; s.kind = "FR"; s.row_id = i; s.value = upper ? 1.0 : -1.0;
        std::vector<int> cols;
        for (int e : row(i)) {
            const Entry& E = pool_[e];
            ForcedVar f;
            f.col = E.c; f.a = E.v; f.cost = c_[E.c];
            f.value = ((E.v > 0) == upper) ? l_[E.c] : u_[E.c];
            s.forced.push_back(f);
            cols.push_back(E.c);
        }
        for (size_t k = 0; k < cols.size(); ++k) {
            int j = cols[k]; double v = s.forced[k].value;
            for (int e : col(j)) {
                const Entry& E = pool_[e];
                if (E.r == i) continue;
                s.forced[k].others.emplace_back(E.r, E.v);
                shift_row(E.r, E.v * v);
            }
            offset_ += c_[j] * v;
            remove_col(j);
        }
        rowAct_[i] = 0; rowE_[i].clear();
        steps.push_back(std::move(s));
        return true;
    }

    // a_j x_j + a_k x_k = b: substitute x_k = (b - a_j x_j) / a_k everywhere.
    bool doubleton(int i) {
        const auto& R = row(i);
        int e1 = R[0], e2 = R[1];
        const double amax = std::max(std::fabs(pool_[e1].v), std::fabs(pool_[e2].v));
        auto ok = [&](int e) {
            int k = pool_[e].c;
            return eliminable(k) && std::fabs(pool_[e].v) >= 1e-2 * amax && colLen_[k] <= 30;
        };
        int ek = -1, ej = -1;
        if (ok(e1) && (!ok(e2) || colLen_[pool_[e1].c] <= colLen_[pool_[e2].c])) { ek = e1; ej = e2; }
        else if (ok(e2)) { ek = e2; ej = e1; }
        else return false;
        const int k = pool_[ek].c, j = pool_[ej].c;
        const double ak = pool_[ek].v, aj = pool_[ej].v, b = rL_[i];

        // Bounds of x_k carried over to x_j.
        auto xj_of = [&](double xk) {
            if (!std::isfinite(xk)) return (-ak / aj > 0 ? xk : -xk);
            return (b - ak * xk) / aj;
        };
        double v1 = xj_of(l_[k]), v2 = xj_of(u_[k]);
        double lo_k = std::min(v1, v2), hi_k = std::max(v1, v2);
        double nlo = std::max(l_[j], lo_k), nhi = std::min(u_[j], hi_k);
        bool lo_from = lo_k > l_[j], hi_from = hi_k < u_[j];
        if (isInt_[j]) {
            if (std::isfinite(nlo)) nlo = std::ceil(nlo - 1e-9);
            if (std::isfinite(nhi)) nhi = std::floor(nhi + 1e-9);
        }
        const double btol = 1e-9 * (1.0 + std::fabs(nlo < INFV ? nlo : 0.0) + std::fabs(nhi > -INFV ? nhi : 0.0));
        if (nlo - nhi > std::max(1e-7, 100.0 * btol)) { status_ = "infeasible"; return false; }
        if (nlo > nhi) nlo = nhi = 0.5 * (nlo + nhi);

        Step s; s.kind = "DT"; s.row_id = i; s.col_id = k; s.col2 = j;
        s.a_ij = aj; s.a2 = ak; s.value = b; s.cost = c_[k];
        s.lo_before = l_[k]; s.hi_before = u_[k]; s.lo_from = lo_from; s.hi_from = hi_from;
        const double ratio = aj / ak;
        std::vector<std::pair<int,double>> kcol;
        for (int e : col(k)) if (pool_[e].r != i) kcol.emplace_back(pool_[e].r, pool_[e].v);
        s.column = kcol;
        steps.push_back(std::move(s));

        c_[j] -= c_[k] * ratio;
        offset_ += c_[k] * b / ak;
        remove_row(i);
        remove_col(k);
        for (auto [r, ark] : kcol) {
            shift_row(r, ark * b / ak);
            const double delta = -ark * ratio;
            int e = find(r, j);
            if (e >= 0) {
                double nv = pool_[e].v + delta;
                if (std::fabs(nv) <= 1e-12 * (std::fabs(pool_[e].v) + std::fabs(delta))) kill(e);
                else { pool_[e].v = nv; aok_[r] = 0; }
            } else {
                add_entry(r, j, delta);
            }
        }
        set_bounds(j, nlo, nhi);
        return true;
    }

    // ------------------------------------------------------------ column rules
    bool col_rules(int j) {
        if (inQ_[j]) return false;
        if (std::isfinite(l_[j]) && u_[j] - l_[j] < ZERO_TOL) { fix_col(j, l_[j], "R3"); return true; }
        const int len = colLen_[j];
        if (len == 0) {
            double v;
            if (c_[j] > ZERO_TOL) { if (!std::isfinite(l_[j])) { status_ = "unbounded"; return false; } v = l_[j]; }
            else if (c_[j] < -ZERO_TOL) { if (!std::isfinite(u_[j])) { status_ = "unbounded"; return false; } v = u_[j]; }
            else v = std::isfinite(l_[j]) ? l_[j] : (std::isfinite(u_[j]) ? u_[j] : 0.0);
            fix_col(j, v, "R2");
            return true;
        }
        // Dual fixing: lowering x_j never violates a row and never raises the
        // cost -> x_j = l_j (and symmetrically for u_j).
        bool down_free = true, up_free = true;
        for (int e : col(j)) {
            const Entry& E = pool_[e];
            if (E.v > 0) { down_free &= !std::isfinite(rL_[E.r]); up_free &= !std::isfinite(rU_[E.r]); }
            else { down_free &= !std::isfinite(rU_[E.r]); up_free &= !std::isfinite(rL_[E.r]); }
        }
        if (down_free && c_[j] >= 0 && std::isfinite(l_[j])) { fix_col(j, l_[j], "R3"); return true; }
        if (up_free && c_[j] <= 0 && std::isfinite(u_[j])) { fix_col(j, u_[j], "R3"); return true; }
        if (len == 1 && eliminable(j)) return free_col_singleton(j);
        return false;
    }

    // Range of x_j implied by row r (both of its bounds) and the bounds of
    // the row's other variables.
    void row_implied(int r, int j, double a, double& ilo, double& ihi) {
        double smin, smax; int nmin, nmax;
        activity(r, smin, smax, nmin, nmax);
        // take x_j's own term back out
        const double lo = l_[j], hi = u_[j];
        if (a > 0) {
            if (std::isfinite(lo)) smin -= a * lo; else --nmin;
            if (std::isfinite(hi)) smax -= a * hi; else --nmax;
        } else {
            if (std::isfinite(hi)) smin -= a * hi; else --nmin;
            if (std::isfinite(lo)) smax -= a * lo; else --nmax;
        }
        // a x_j in [rL - smax, rU - smin]
        const double lo_ax = (std::isfinite(rL_[r]) && !nmax) ? rL_[r] - smax : -INFV;
        const double hi_ax = (std::isfinite(rU_[r]) && !nmin) ? rU_[r] - smin : INFV;
        if (a > 0) { ilo = lo_ax / a; ihi = hi_ax / a; }
        else { ilo = hi_ax / a; ihi = lo_ax / a; }
        if (std::isnan(ilo)) ilo = -INFV;
        if (std::isnan(ihi)) ihi = INFV;
    }

    // x_j is implied free if each of its finite bounds is implied by one of
    // its rows (any row: the bounds are then redundant and can be dropped).
    // Each such drop is checked against the current problem only (x_j's own
    // bounds excluded), so it leaves the feasible set unchanged; a sequence of
    // drops therefore cannot become circular -- a variable whose bounds were
    // dropped has already left the problem and supports no later check.
    bool implied_free(int i, int j, double a) {
        bool lo_ok = !std::isfinite(l_[j]), hi_ok = !std::isfinite(u_[j]);
        auto check = [&](int r, double ar) {
            double ilo, ihi;
            row_implied(r, j, ar, ilo, ihi);
            const double tol = 1e-9 * (1.0 + rowmag_[r]) / std::fabs(ar);
            lo_ok |= ilo >= l_[j] - tol;
            hi_ok |= ihi <= u_[j] + tol;
        };
        check(i, a);
        if (lo_ok && hi_ok) return true;
        for (int f : col(j)) {
            if (pool_[f].r == i) continue;
            check(pool_[f].r, pool_[f].v);
            if (lo_ok && hi_ok) return true;
        }
        return false;
    }

    // Column singleton in an equality row whose bounds the row implies.
    bool free_col_singleton(int j) {
        const int e = col(j)[0];
        const int i = pool_[e].r;
        if (!(rL_[i] == rU_[i])) return false;
        return substitute(i, j);
    }

    // Aggregator: in an equality row, pick an implied-free variable with a
    // short column and substitute it out of every other row.
    bool aggregate(int i) {
        const auto& R = row(i);
        double amax = 0.0;
        for (int e : R) amax = std::max(amax, std::fabs(pool_[e].v));
        int best = -1, best_len = 1 << 30;
        for (int e : R) {
            const Entry& E = pool_[e];
            const int len = colLen_[E.c];
            if (!eliminable(E.c) || len >= best_len || len > 40) continue;
            if (std::fabs(E.v) < 0.1 * amax) continue;
            if ((long long)(rowLen_[i] - 1) * (len - 1) > 400) continue;
            if (!implied_free(i, E.c, E.v)) continue;
            best = E.c; best_len = len;
        }
        return best >= 0 && substitute(i, best);
    }

    // x_j = (b - sum_{k != j} a_ik x_k) / a_ij into every other row of j;
    // row i and column j both go. Postsolve: y_i makes z_j = 0, and every
    // other reduced cost is unchanged by the substitution.
    bool substitute(int i, int j) {
        const int ej = find(i, j);
        const double a = pool_[ej].v;
        double amax = 0.0;
        for (int f : row(i)) amax = std::max(amax, std::fabs(pool_[f].v));
        if (std::fabs(a) < 1e-2 * amax) return false;
        if (!implied_free(i, j, a)) return false;
        const double b = rL_[i];
        Step s; s.kind = "FCS"; s.row_id = i; s.col_id = j; s.a_ij = a; s.value = b; s.cost = c_[j];
        for (int f : row(i)) if (pool_[f].c != j) s.row.emplace_back(pool_[f].c, pool_[f].v);
        for (int f : col(j)) if (pool_[f].r != i) s.column.emplace_back(pool_[f].r, pool_[f].v);
        offset_ += c_[j] * b / a;
        for (auto [k, ak] : s.row) c_[k] -= c_[j] * ak / a;
        remove_row(i);
        remove_col(j);
        for (auto [r, arj] : s.column) {
            const double f = arj / a;
            shift_row(r, f * b);
            for (auto [k, ak] : s.row) {
                const double delta = -f * ak;
                int e = find(r, k);
                if (e >= 0) {
                    double nv = pool_[e].v + delta;
                    if (std::fabs(nv) <= 1e-12 * (std::fabs(pool_[e].v) + std::fabs(delta))) kill(e);
                    else { pool_[e].v = nv; aok_[r] = 0; }
                } else {
                    add_entry(r, k, delta);
                }
            }
        }
        steps.push_back(std::move(s));
        return true;
    }

    bool mip_;
    int m0_ = 0, n0_ = 0, passes_ = 0;
    std::vector<double> c_, rL_, rU_, l_, u_, rowmag_;
    double offset_ = 0.0;
    std::vector<char> rowAct_, colAct_, inQ_, isInt_, aok_;
    std::vector<double> amin_, amax_;
    std::vector<int> anmin_, anmax_;
    std::vector<Entry> pool_;
    std::vector<std::vector<int>> rowE_, colE_;
    std::vector<int> rowLen_, colLen_;
    std::string status_;
};

// MILP coefficient tightening (MILP Step 2.1) on the reduced problem. For a
// one-sided row a^T x <= b and a binary x_j: if the row cannot bind when x_j
// takes its "loose" value, the coefficient (and b) can move by the slack d
// without removing any integer point:
//   a_j > 0, x_j = 0 loose: d = b - (maxact - a_j) > 0  ->  a_j -= d, b -= d
//   a_j < 0, x_j = 1 loose: d = b - (maxact + a_j) > 0  ->  a_j += d
// Integer points are unchanged, so postsolve needs no record; the LP
// relaxation (and hence the MILP bound) gets tighter.
void tighten_coefficients(RangedLP& R) {
    if (R.integer.empty() || R.A.rows == 0) return;
    CSR A = to_csr(R.A);
    std::vector<char> binary(R.A.cols, 0);
    for (int j = 0; j < R.A.cols; ++j) binary[j] = R.integer[j] && R.l[j] == 0.0 && R.u[j] == 1.0;
    std::vector<double> newval = A.data;
    for (int i = 0; i < R.A.rows; ++i) {
        bool le = std::isfinite(R.rU[i]) && !std::isfinite(R.rL[i]);
        bool ge = std::isfinite(R.rL[i]) && !std::isfinite(R.rU[i]);
        if (!le && !ge) continue;
        const double sg = le ? 1.0 : -1.0;
        double b = le ? R.rU[i] : -R.rL[i];
        double maxact = 0.0;
        bool finite = true;
        for (int p = A.indptr[i]; p < A.indptr[i + 1]; ++p) {
            int j = A.indices[p]; double a = sg * A.data[p];
            double t = a > 0 ? a * R.u[j] : a * R.l[j];
            if (!std::isfinite(t)) { finite = false; break; }
            maxact += t;
        }
        if (!finite || maxact <= b + 1e-9) continue;
        for (int p = A.indptr[i]; p < A.indptr[i + 1]; ++p) {
            int j = A.indices[p];
            if (!binary[j]) continue;
            double a = sg * newval[p];
            const double tol = 1e-9 * (1.0 + std::fabs(b));
            if (a > 0) {
                double d = b - (maxact - a);
                if (d > tol && d < a) { a -= d; b -= d; maxact -= d; }
            } else if (a < 0) {
                double d = b - (maxact + a);
                if (d > tol && d < -a) { a += d; }
            }
            newval[p] = sg * a;
        }
        if (le) R.rU[i] = b; else R.rL[i] = -b;
    }
    SparseMatrix T; T.rows = R.A.rows; T.cols = R.A.cols;
    for (int i = 0; i < R.A.rows; ++i)
        for (int p = A.indptr[i]; p < A.indptr[i + 1]; ++p)
            if (newval[p] != 0.0) { T.row_idx.push_back(i); T.col_idx.push_back(A.indices[p]); T.val.push_back(newval[p]); }
    R.A = T;
}

} // namespace

PresolveResult presolve(const RangedLP& ranged, int max_iterations, bool mip_mode) {
    PresolveResult result;
    Presolver P(ranged, mip_mode);
    if (!P.round_integer_bounds()) {
        result.status = "infeasible";
        for (int i = 0; i < ranged.m(); ++i) result.row_ids.push_back(i);
        for (int j = 0; j < ranged.n(); ++j) result.col_ids.push_back(j);
        return result;
    }
    std::string st = P.run(max_iterations);
    result.passes = P.passes();
    if (st != "ok") {
        result.status = st;
        for (int i = 0; i < ranged.m(); ++i) result.row_ids.push_back(i);
        for (int j = 0; j < ranged.n(); ++j) result.col_ids.push_back(j);
        return result;
    }
    P.build(ranged, result);
    result.steps = std::move(P.steps);
    if (mip_mode) tighten_coefficients(result.reduced);
    result.has_reduced = true;
    result.status = "ok";
    return result;
}

void postsolve(int n0, int m0, const std::vector<int>& row_ids, const std::vector<int>& col_ids,
               const std::vector<double>& x_reduced, const std::vector<double>& y_reduced,
               const std::vector<double>& z_reduced, const std::vector<Step>& steps,
               std::vector<double>& x, std::vector<double>& y, std::vector<double>& z) {
    x.assign(n0, std::nan(""));
    y.assign(m0, std::nan(""));
    z.assign(n0, std::nan(""));
    for (size_t k = 0; k < col_ids.size(); ++k) { x[col_ids[k]] = x_reduced[k]; z[col_ids[k]] = z_reduced[k]; }
    for (size_t k = 0; k < row_ids.size(); ++k) y[row_ids[k]] = y_reduced[k];

    // z is kept equal to the reduced costs of the problem as it stood before
    // the step being undone (costs and coefficients of that stage).
    for (auto it = steps.rbegin(); it != steps.rend(); ++it) {
        const Step& p = *it;
        if (p.kind == "R1" || p.kind == "RR") {
            y[p.row_id] = 0.0;
        } else if (p.kind == "R2" || p.kind == "R3") {
            x[p.col_id] = p.value;
            double zj = p.cost;
            for (auto [r, a] : p.column) zj -= a * y[r];
            z[p.col_id] = zj;
        } else if (p.kind == "R4") {
            // The row supplied a bound of x_j. If x_j sits on that bound (the
            // sign of z_j tells which side is active), the multiplier moves to
            // the row. (Deciding by |x_j - bound| < tol breaks for large x_j.)
            int j = p.col_id, r = p.row_id;
            bool row_defines_lo = p.lo_from_row > p.lo_before;
            bool row_defines_hi = p.hi_from_row < p.hi_before;
            if ((z[j] > 0 && row_defines_lo) || (z[j] < 0 && row_defines_hi)) { y[r] = z[j] / p.a_ij; z[j] = 0.0; }
            else y[r] = 0.0;
        } else if (p.kind == "FCS") {
            double s = p.value;
            for (auto [k, a] : p.row) s -= a * x[k];
            x[p.col_id] = s / p.a_ij;
            double d = p.cost;
            for (auto [r, a] : p.column) d -= a * y[r];
            y[p.row_id] = d / p.a_ij;
            z[p.col_id] = 0.0;
        } else if (p.kind == "DT") {
            // a_j x_j + a_k x_k = b, x_k eliminated. D = reduced cost of x_k
            // without row i. Either x_k is off its bounds (z_k = 0) or x_j sits
            // on a bound inherited from x_k, and then z_j moves over to x_k.
            const int k = p.col_id, j = p.col2, i = p.row_id;
            const double aj = p.a_ij, ak = p.a2;
            x[k] = (p.value - aj * x[j]) / ak;
            double D = p.cost;
            for (auto [r, a] : p.column) D -= a * y[r];
            const bool on_inherited = (z[j] > 0 && p.lo_from) || (z[j] < 0 && p.hi_from);
            if (on_inherited) {
                double zk = -(ak / aj) * z[j];
                y[i] = (D - zk) / ak;
                z[k] = zk;
                z[j] = 0.0;
            } else {
                y[i] = D / ak;
                z[k] = 0.0;
            }
        } else if (p.kind == "FR") {
            // Every variable of the row sits at a bound; pick the row multiplier
            // that leaves each of their reduced costs with the right sign.
            const bool upper = p.value > 0;
            std::vector<double> D(p.forced.size());
            double yi = 0.0;
            for (size_t k = 0; k < p.forced.size(); ++k) {
                const ForcedVar& f = p.forced[k];
                double d = f.cost;
                for (auto [r, a] : f.others) d -= a * y[r];
                D[k] = d;
                double t = d / f.a;
                yi = upper ? std::min(yi, t) : std::max(yi, t);
            }
            y[p.row_id] = yi;
            for (size_t k = 0; k < p.forced.size(); ++k) {
                const ForcedVar& f = p.forced[k];
                x[f.col] = f.value;
                z[f.col] = D[k] - f.a * yi;
            }
        }
    }
}
