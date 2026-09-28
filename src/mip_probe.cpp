#include "mip_probe.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <tuple>

namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();
double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
}

// ============================================================ propagation

Propagator::Propagator(const RangedLP& lp, const std::vector<char>& is_int)
    : m_(lp.m()), n_(lp.n()), rL_(lp.rL), rU_(lp.rU), is_int_(is_int) {
    const SparseMatrix& A = lp.A;
    rp_.assign(m_ + 1, 0); cp_.assign(n_ + 1, 0);
    for (int k = 0; k < A.nnz(); ++k) { rp_[A.row_idx[k] + 1]++; cp_[A.col_idx[k] + 1]++; }
    for (int i = 0; i < m_; ++i) rp_[i + 1] += rp_[i];
    for (int j = 0; j < n_; ++j) cp_[j + 1] += cp_[j];
    ri_.assign(A.nnz(), 0); rv_.assign(A.nnz(), 0.0); ci_.assign(A.nnz(), 0);
    std::vector<int> rc(rp_.begin(), rp_.end() - 1), cc(cp_.begin(), cp_.end() - 1);
    for (int k = 0; k < A.nnz(); ++k) {
        int i = A.row_idx[k], j = A.col_idx[k];
        int p = rc[i]++; ri_[p] = j; rv_[p] = A.val[k];
        ci_[cc[j]++] = i;
    }
    is_int_.resize(n_, 0);
}

bool Propagator::propagate(std::vector<double>& lo, std::vector<double>& hi, const std::vector<int>& seed,
                           std::vector<std::tuple<int,double,double>>* log, long long work_limit) {
    std::vector<int> queue;
    std::vector<char> queued(m_, 0);
    auto push_col_rows = [&](int j) {
        for (int p = cp_[j]; p < cp_[j + 1]; ++p) {
            int i = ci_[p];
            if (!queued[i]) { queued[i] = 1; queue.push_back(i); }
        }
    };
    if (seed.empty()) { for (int i = 0; i < m_; ++i) { queued[i] = 1; queue.push_back(i); } }
    else for (int j : seed) push_col_rows(j);

    long long work = 0;
    size_t head = 0;
    auto set_bound = [&](int k, double nlo, double nhi) -> bool {
        if (log) log->emplace_back(k, lo[k], hi[k]);
        lo[k] = nlo; hi[k] = nhi;
        if (nlo > nhi + 1e-9 * (1.0 + std::fabs(nlo))) return false;
        if (nlo > nhi) lo[k] = hi[k] = 0.5 * (nlo + nhi);
        push_col_rows(k);
        return true;
    };
    while (head < queue.size()) {
        if (work > work_limit) break;
        int i = queue[head++];
        queued[i] = 0;
        double mn = 0, mx = 0; int nmn = 0, nmx = 0;
        for (int p = rp_[i]; p < rp_[i + 1]; ++p) {
            int j = ri_[p]; double a = rv_[p];
            if (a > 0) {
                if (std::isfinite(lo[j])) mn += a * lo[j]; else ++nmn;
                if (std::isfinite(hi[j])) mx += a * hi[j]; else ++nmx;
            } else {
                if (std::isfinite(hi[j])) mn += a * hi[j]; else ++nmn;
                if (std::isfinite(lo[j])) mx += a * lo[j]; else ++nmx;
            }
        }
        work += rp_[i + 1] - rp_[i];
        const double scale = 1.0 + std::fabs(std::isfinite(rU_[i]) ? rU_[i] : 0.0) + std::fabs(std::isfinite(rL_[i]) ? rL_[i] : 0.0);
        if ((nmn == 0 && mn > rU_[i] + 1e-6 * (scale + std::fabs(mn))) ||
            (nmx == 0 && mx < rL_[i] - 1e-6 * (scale + std::fabs(mx)))) return false;
        if (nmn > 1 && nmx > 1) continue;                                   // nothing can be derived
        for (int p = rp_[i]; p < rp_[i + 1]; ++p) {
            int k = ri_[p]; double a = rv_[p];
            // activity of the row without x_k
            double mn_k, mx_k; int nmn_k = nmn, nmx_k = nmx;
            if (a > 0) {
                mn_k = std::isfinite(lo[k]) ? mn - a * lo[k] : (--nmn_k, mn);
                mx_k = std::isfinite(hi[k]) ? mx - a * hi[k] : (--nmx_k, mx);
            } else {
                mn_k = std::isfinite(hi[k]) ? mn - a * hi[k] : (--nmn_k, mn);
                mx_k = std::isfinite(lo[k]) ? mx - a * lo[k] : (--nmx_k, mx);
            }
            double nlo = lo[k], nhi = hi[k];
            // a x_k <= rU - mn_k   and   a x_k >= rL - mx_k
            if (std::isfinite(rU_[i]) && nmn_k == 0) {
                double v = (rU_[i] - mn_k) / a;
                if (a > 0) nhi = std::min(nhi, v); else nlo = std::max(nlo, v);
            }
            if (std::isfinite(rL_[i]) && nmx_k == 0) {
                double v = (rL_[i] - mx_k) / a;
                if (a > 0) nlo = std::max(nlo, v); else nhi = std::min(nhi, v);
            }
            if (is_int_[k]) {
                if (std::isfinite(nlo)) nlo = std::ceil(nlo - 1e-6);
                if (std::isfinite(nhi)) nhi = std::floor(nhi + 1e-6);
            }
            // accept only real progress: integers any change, continuous
            // at least 1e-3 of the range (avoids endless tiny steps)
            double range = std::isfinite(hi[k]) && std::isfinite(lo[k]) ? hi[k] - lo[k] : kInf;
            double step = is_int_[k] ? 0.5 : std::max(1e-6 * (1.0 + std::fabs(lo[k] + hi[k])), std::isfinite(range) ? 1e-3 * range : 1e-6);
            bool up = nlo > lo[k] + step, down = nhi < hi[k] - step;
            if (!up && !down) continue;
            if (!set_bound(k, up ? nlo : lo[k], down ? nhi : hi[k])) return false;
        }
    }
    return true;
}

// ================================================================ probing

ProbeResult probe(const RangedLP& lp, const std::vector<char>& is_int, std::vector<double>& lo,
                  std::vector<double>& hi, double seconds) {
    ProbeResult res;
    const double t_end = now_s() + seconds;
    const int n = lp.n();
    Propagator P(lp, is_int);
    std::vector<std::tuple<int,double,double>> log;
    if (!P.propagate(lo, hi, {}, nullptr)) { res.infeasible = true; return res; }

    std::vector<int> nnz(n, 0);
    for (int k = 0; k < lp.A.nnz(); ++k) nnz[lp.A.col_idx[k]]++;
    std::vector<int> cand;
    for (int j = 0; j < n; ++j)
        if (j < (int)is_int.size() && is_int[j] && lo[j] == 0.0 && hi[j] == 1.0) cand.push_back(j);
    std::sort(cand.begin(), cand.end(), [&](int a, int b) { return nnz[a] > nnz[b]; });

    std::vector<double> lo0, hi0, lo1, hi1;
    std::vector<int> changed0, changed1;
    for (int j : cand) {
        if (now_s() > t_end) break;
        if (lo[j] == hi[j]) continue;
        ++res.probed;
        // side v: run on the global bounds, remember the result, undo.
        auto side = [&](int v, std::vector<int>& changed, std::vector<double>& slo, std::vector<double>& shi) {
            log.clear();
            log.emplace_back(j, lo[j], hi[j]);
            lo[j] = hi[j] = v;
            bool ok = P.propagate(lo, hi, {j}, &log, 200000);
            changed.clear();
            if (ok) {
                std::vector<char> seen;
                for (const auto& [k, a, b] : log) changed.push_back(k);
                std::sort(changed.begin(), changed.end());
                changed.erase(std::unique(changed.begin(), changed.end()), changed.end());
                slo.resize(changed.size()); shi.resize(changed.size());
                for (size_t t = 0; t < changed.size(); ++t) { slo[t] = lo[changed[t]]; shi[t] = hi[changed[t]]; }
            }
            for (auto it = log.rbegin(); it != log.rend(); ++it) { lo[std::get<0>(*it)] = std::get<1>(*it); hi[std::get<0>(*it)] = std::get<2>(*it); }
            return ok;
        };
        bool ok0 = side(0, changed0, lo0, hi0);
        bool ok1 = side(1, changed1, lo1, hi1);
        if (!ok0 && !ok1) { res.infeasible = true; return res; }
        if (!ok0 || !ok1) {
            // one side infeasible: the other side's bounds hold globally
            const auto& ch = ok0 ? changed0 : changed1;
            const auto& sl = ok0 ? lo0 : lo1;
            const auto& sh = ok0 ? hi0 : hi1;
            for (size_t t = 0; t < ch.size(); ++t) { lo[ch[t]] = sl[t]; hi[ch[t]] = sh[t]; }
            ++res.fixed;
            if (!P.propagate(lo, hi, ch, nullptr)) { res.infeasible = true; return res; }
            continue;
        }
        // bounds implied by both sides hold globally
        size_t a = 0, b = 0;
        while (a < changed0.size() && b < changed1.size()) {
            if (changed0[a] < changed1[b]) { ++a; continue; }
            if (changed1[b] < changed0[a]) { ++b; continue; }
            int k = changed0[a];
            if (k != j) {
                double nl = std::min(lo0[a], lo1[b]), nh = std::max(hi0[a], hi1[b]);
                if (nl > lo[k] || nh < hi[k]) { lo[k] = std::max(lo[k], nl); hi[k] = std::min(hi[k], nh); ++res.tightened; }
            }
            ++a; ++b;
        }
        // implications beyond the global bounds
        for (int v = 0; v < 2; ++v) {
            const auto& ch = v ? changed1 : changed0;
            const auto& sl = v ? lo1 : lo0;
            const auto& sh = v ? hi1 : hi0;
            for (size_t t = 0; t < ch.size() && res.impl.size() < 400000; ++t) {
                int k = ch[t];
                if (k == j) continue;
                if (sh[t] < hi[k] - 1e-9 * (1.0 + std::fabs(hi[k]))) res.impl.push_back({j, v, k, true, sh[t]});
                if (sl[t] > lo[k] + 1e-9 * (1.0 + std::fabs(lo[k]))) res.impl.push_back({j, v, k, false, sl[t]});
            }
        }
    }
    return res;
}

// ============================================================ clique table

void CliqueTable::add_clique(std::vector<int> lits) {
    std::sort(lits.begin(), lits.end());
    lits.erase(std::unique(lits.begin(), lits.end()), lits.end());
    if (lits.size() < 2) return;
    int id = (int)cliques_.size();
    for (int l : lits) {
        if ((int)of_lit_.size() <= l) of_lit_.resize(l + 1);
        of_lit_[l].push_back(id);
    }
    cliques_.push_back(std::move(lits));
}

// Rows over binaries: after complementing negative coefficients the row is
// sum a_k l_k <= b with a_k > 0; any two items with a_i + a_j > b are a
// conflicting pair. With items sorted by decreasing a, the longest prefix
// whose two smallest members still conflict is a clique.
void CliqueTable::build_from_rows(const RangedLP& lp, const std::vector<char>& is_int,
                                  const std::vector<double>& lo, const std::vector<double>& hi) {
    CSR A = to_csr(lp.A);
    for (int i = 0; i < lp.m(); ++i) {
        for (int side = 0; side < 2; ++side) {
            double b = side == 0 ? lp.rU[i] : -lp.rL[i];
            if (!std::isfinite(b)) continue;
            const double sg = side == 0 ? 1.0 : -1.0;
            std::vector<std::pair<double,int>> items;
            bool ok = true;
            for (int p = A.indptr[i]; p < A.indptr[i + 1]; ++p) {
                int j = A.indices[p]; double a = sg * A.data[p];
                bool bin = j < (int)is_int.size() && is_int[j] && lo[j] >= 0.0 && hi[j] <= 1.0;
                if (!bin) {
                    // a non-binary term only loosens the row by its minimum
                    double mn = a > 0 ? a * lo[j] : a * hi[j];
                    if (!std::isfinite(mn)) { ok = false; break; }
                    b -= mn;
                    continue;
                }
                if (lo[j] == hi[j]) { b -= a * lo[j]; continue; }
                if (a > 0) items.push_back({a, 2 * j});
                else if (a < 0) { items.push_back({-a, 2 * j + 1}); b -= a; }
            }
            if (!ok || items.size() < 2) continue;
            std::sort(items.begin(), items.end(), [](const auto& x, const auto& y) { return x.first > y.first; });
            size_t t = 1;
            while (t < items.size() && items[t - 1].first + items[t].first > b + 1e-9) ++t;
            if (t >= 2) {
                std::vector<int> lits;
                for (size_t k = 0; k < t; ++k) lits.push_back(items[k].second);
                add_clique(std::move(lits));
            }
        }
    }
}

// x_j = v  =>  binary x_k fixed to w : literals (j=v) and (k=1-w) conflict.
void CliqueTable::add_implications(const std::vector<Implication>& impl, const std::vector<char>& is_binary) {
    // Pairwise edges only while the table stays small: on set-partitioning
    // models the rows already give the big cliques, and hundreds of
    // thousands of two-literal cliques only slow the separation down.
    for (const auto& im : impl) {
        if (cliques_.size() >= 50000) break;
        if (im.col >= (int)is_binary.size() || !is_binary[im.col]) continue;
        int w;
        if (im.upper && im.bound < 0.5) w = 0;          // x_k <= 0
        else if (!im.upper && im.bound > 0.5) w = 1;    // x_k >= 1
        else continue;
        int lj = 2 * im.bin + (im.val == 1 ? 0 : 1);    // literal that is 1 when x_j = v
        int lk = 2 * im.col + (w == 0 ? 0 : 1);         // literal that is 1 when x_k = 1 - w
        add_clique({lj, lk});
    }
}

bool CliqueTable::adjacent(int a, int b) const {
    if (a >= (int)of_lit_.size() || b >= (int)of_lit_.size()) return false;
    const auto& A = of_lit_[a].size() <= of_lit_[b].size() ? of_lit_[a] : of_lit_[b];
    int other = of_lit_[a].size() <= of_lit_[b].size() ? b : a;
    for (int c : A) if (std::binary_search(cliques_[c].begin(), cliques_[c].end(), other)) return true;
    return false;
}

int CliqueTable::separate(const std::vector<double>& x, std::vector<DualSimplex::Row>& out, int max_cuts) const {
    auto val = [&](int l) { double v = x[l / 2]; return (l & 1) ? 1.0 - v : v; };
    std::vector<int> lits;
    for (int l = 0; l < (int)of_lit_.size(); ++l)
        if (!of_lit_[l].empty() && val(l) > 1e-6) lits.push_back(l);
    std::sort(lits.begin(), lits.end(), [&](int a, int b) { return val(a) > val(b); });
    int made = 0;
    std::vector<char> used(of_lit_.size(), 0);
    for (int start : lits) {
        if (made >= max_cuts) break;
        if (used[start] || val(start) > 1.0 - 1e-6) continue;
        // candidates: literals sharing a clique with start, by decreasing value
        std::vector<int> cand;
        for (int c : of_lit_[start]) for (int l : cliques_[c]) if (l != start && val(l) > 1e-6) cand.push_back(l);
        std::sort(cand.begin(), cand.end());
        cand.erase(std::unique(cand.begin(), cand.end()), cand.end());
        std::sort(cand.begin(), cand.end(), [&](int a, int b) { return val(a) > val(b); });
        std::vector<int> C{start};
        double sum = val(start);
        for (int l : cand) {
            if ((l ^ 1) == start) continue;
            bool ok = true;
            for (int c : C) if (!adjacent(l, c) || (l ^ 1) == c) { ok = false; break; }
            if (ok) { C.push_back(l); sum += val(l); }
        }
        if (C.size() < 2 || sum <= 1.0 + 1e-4) continue;
        // sum over C of literals <= 1   ->   sum x_pos - sum x_neg <= 1 - |neg|
        DualSimplex::Row row;
        double rhs = 1.0;
        for (int l : C) {
            if (l & 1) { row.entries.emplace_back(l / 2, -1.0); rhs -= 1.0; }
            else row.entries.emplace_back(l / 2, 1.0);
            used[l] = 1;
        }
        row.lo = -kInf; row.hi = rhs;
        out.push_back(std::move(row));
        ++made;
    }
    return made;
}
