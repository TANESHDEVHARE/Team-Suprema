#include "qp_ipm.hpp"
#include "sparse_ldl.hpp"
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <queue>

namespace {

double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}
constexpr double kInf = std::numeric_limits<double>::infinity();

double inf_norm(const std::vector<double>& v) {
    double m = 0.0;
    for (double x : v) m = std::max(m, std::fabs(x));
    return m;
}

// Largest step in (0, 1] keeping s + a*ds >= 0 for every entry with mask set.
double max_step(const std::vector<double>& s, const std::vector<double>& ds, const std::vector<char>& mask) {
    double a = 1.0;
    for (size_t k = 0; k < s.size(); ++k)
        if (mask[k] && ds[k] < 0.0) a = std::min(a, -s[k] / ds[k]);
    return a;
}

} // namespace

IpmResult solve_ipm(const RangedLP& lp, const IpmOptions& opt) {
    const double t0 = now_s();
    IpmResult res;
    const int n = lp.n(), m = lp.m(), N = n + m;
    const CSR A = to_csr(lp.A), AT = to_csc_as_transposed_csr(lp.A);
    const auto& c = lp.c;

    // ---- Q: full symmetric CSR for products, lower entries for K ----
    SparseMatrix Qfull; Qfull.rows = Qfull.cols = n;
    for (int k = 0; k < lp.Q.nnz(); ++k) {
        int i = lp.Q.row_idx[k], j = lp.Q.col_idx[k]; double v = lp.Q.val[k];
        Qfull.row_idx.push_back(i); Qfull.col_idx.push_back(j); Qfull.val.push_back(v);
        if (i != j) { Qfull.row_idx.push_back(j); Qfull.col_idx.push_back(i); Qfull.val.push_back(v); }
    }
    const CSR Q = to_csr(Qfull);

    // ---- bounds (fixed variables are widened to a hair-thin box) ----
    std::vector<double> l = lp.l, u = lp.u;
    std::vector<char> hl(n), hu(n);
    for (int j = 0; j < n; ++j) {
        if (std::isfinite(l[j]) && std::isfinite(u[j]) && u[j] - l[j] < 1e-9 * (1.0 + std::fabs(l[j]))) {
            double v = 0.5 * (l[j] + u[j]), e = 1e-8 * (1.0 + std::fabs(v));
            l[j] = v - e; u[j] = v + e;
        }
        hl[j] = std::isfinite(l[j]); hu[j] = std::isfinite(u[j]);
    }
    std::vector<char> eq(m), rl(m), ru(m);
    std::vector<double> b(m, 0.0);
    for (int i = 0; i < m; ++i) {
        eq[i] = std::isfinite(lp.rL[i]) && lp.rL[i] == lp.rU[i];
        if (eq[i]) { b[i] = lp.rL[i]; continue; }
        rl[i] = std::isfinite(lp.rL[i]); ru[i] = std::isfinite(lp.rU[i]);
    }

    // ---- pattern of K (lower triangle CSC) and value positions ----
    SymMatrix K; K.n = N;
    std::vector<int> dpos(N), qpos, apos(A.indices.size());
    {
        std::vector<std::vector<std::pair<int,int>>> cols(N);   // (row, tag)
        // tags: -1 diag, 0.. Q lower entry index, -(2+p) A entry p (in AT/CSC order)
        for (int j = 0; j < N; ++j) cols[j].push_back({j, -1});
        for (int k = 0; k < lp.Q.nnz(); ++k) {
            int i = lp.Q.row_idx[k], j = lp.Q.col_idx[k];
            if (i != j) cols[j].push_back({i, k});
        }
        for (int j = 0; j < n; ++j)
            for (int p = AT.indptr[j]; p < AT.indptr[j + 1]; ++p) cols[j].push_back({n + AT.indices[p], -(2 + p)});
        K.colptr.assign(N + 1, 0);
        for (int j = 0; j < N; ++j) K.colptr[j + 1] = K.colptr[j] + (int)cols[j].size();
        K.rowidx.resize(K.colptr[N]); K.val.assign(K.colptr[N], 0.0);
        qpos.assign(lp.Q.nnz(), -1);
        for (int j = 0; j < N; ++j) {
            int q = K.colptr[j];
            for (const auto& e : cols[j]) {
                K.rowidx[q] = e.first;
                if (e.second == -1) dpos[j] = q;
                else if (e.second >= 0) qpos[e.second] = q;
                else apos[-(e.second + 2)] = q;
                ++q;
            }
        }
    }
    std::vector<double> qdiag(n, 0.0);
    for (int k = 0; k < lp.Q.nnz(); ++k) if (lp.Q.row_idx[k] == lp.Q.col_idx[k]) qdiag[lp.Q.row_idx[k]] += lp.Q.val[k];
    SparseLDL ldl;
    {
        double ta = now_s();
        ldl.analyze(K);
        if (opt.verbose)
            std::printf("  ipm: n %d m %d  nnz(K) %zu  nnz(L) %lld  analyze %.3fs\n",
                        n, m, K.rowidx.size(), ldl.nnz_L(), now_s() - ta);
    }
    res.nnz_L = ldl.nnz_L();
    std::vector<signed char> sign(N, 1);
    for (int j = 0; j < n; ++j) sign[j] = -1;
    const double dp = 1e-8, dd = 1e-8;        // primal/dual regularization (removed again by refinement)

    // ---- starting point ----
    auto interior = [](double v, double lo, double hi, bool fl, bool fh) {
        double th = 1.0;
        if (fl && fh) th = std::min(1.0, 0.25 * (hi - lo));
        if (fl) v = std::max(v, lo + th);
        if (fh) v = std::min(v, hi - th);
        return v;
    };
    std::vector<double> x(n, 0.0), w(m, 0.0), lam(m, 0.0), zl(n, 0.0), zu(n, 0.0), yl(m, 0.0), yu(m, 0.0);
    for (int j = 0; j < n; ++j) {
        x[j] = interior(0.0, l[j], u[j], hl[j], hu[j]);
        if (hl[j]) zl[j] = 1.0;
        if (hu[j]) zu[j] = 1.0;
    }
    {
        auto ax = matvec(A, x);
        for (int i = 0; i < m; ++i) if (!eq[i]) {
            w[i] = interior(ax[i], lp.rL[i], lp.rU[i], rl[i], ru[i]);
            if (rl[i]) yl[i] = 1.0;
            if (ru[i]) yu[i] = 1.0;
            lam[i] = yl[i] - yu[i];
        }
    }

    // ---- linearly dependent equality rows ----
    // Kept, a dependent row makes K singular up to the regularization, and the
    // regularization then absorbs rounding-level inconsistency as an
    // ever-growing multiplier that corrupts the duality gap (seen on
    // FINNIS/BNL2). They are found by a sparse row-echelon reduction of the
    // equality rows (b carried along); a row that reduces to zero with a
    // consistent right-hand side is dropped (lambda_i = 0).
    int dependent_rows = 0;
    {
        std::vector<int> pivcol_owner(n, -1);            // column -> index of the echelon row pivoting on it
        std::vector<std::vector<std::pair<int,double>>> ech;
        std::vector<int> ech_piv;
        std::vector<double> ech_b, v(n, 0.0);
        std::vector<char> nz(n, 0);
        std::vector<int> touched;
        long long budget = 20LL * (long long)A.indices.size() + 100000, used = 0;
        for (int i = 0; i < m && used < budget; ++i) {
            if (!eq[i]) continue;
            touched.clear();
            double vmax = 0.0, vb = b[i];
            std::priority_queue<int, std::vector<int>, std::greater<int>> heap;
            auto touch = [&](int j) {
                if (!nz[j]) { nz[j] = 1; touched.push_back(j); if (pivcol_owner[j] >= 0) heap.push(pivcol_owner[j]); }
            };
            for (int p = A.indptr[i]; p < A.indptr[i + 1]; ++p) {
                int j = A.indices[p];
                v[j] += A.data[p]; vmax = std::max(vmax, std::fabs(A.data[p]));
                touch(j);
            }
            int last = -1;
            while (!heap.empty()) {
                int k = heap.top(); heap.pop();
                if (k == last) continue;
                last = k;
                double vp = v[ech_piv[k]];
                if (vp == 0.0) continue;
                double f = vp / ech[k].front().second;       // front() holds the pivot entry
                for (const auto& e : ech[k]) { v[e.first] -= f * e.second; touch(e.first); }
                v[ech_piv[k]] = 0.0;
                vb -= f * ech_b[k];
                used += (long long)ech[k].size();
            }
            double rmax = 0.0; int pc = -1;
            for (int j : touched) if (std::fabs(v[j]) > rmax) { rmax = std::fabs(v[j]); pc = j; }
            if (rmax <= 1e-9 * vmax) {
                if (std::fabs(vb) <= 1e-9 * (1.0 + std::fabs(b[i]))) {   // dependent and consistent
                    eq[i] = 0; rl[i] = ru[i] = 0; ++dependent_rows;
                }
            } else {
                std::vector<std::pair<int,double>> row;
                row.emplace_back(pc, v[pc]);
                for (int j : touched) if (j != pc && std::fabs(v[j]) > 1e-14 * rmax) row.emplace_back(j, v[j]);
                pivcol_owner[pc] = (int)ech.size();
                ech_piv.push_back(pc); ech_b.push_back(vb);
                used += (long long)row.size();
                ech.push_back(std::move(row));
            }
            for (int j : touched) { v[j] = 0.0; nz[j] = 0; }
        }
        if (opt.verbose && dependent_rows)
            std::printf("  ipm: %d linearly dependent equality rows dropped\n", dependent_rows);
    }

    // ---- Mehrotra starting point ----
    // Regularized least squares  [-(Q + I)  A^T ; A  E] [x ; lam] = [c ; r]
    // (E = I on inequality rows, ~0 on equalities; r = b or a row-bound
    // target), then z = Q x + c - A^T lam, then one uniform shift of every
    // slack and dual into the interior with Mehrotra's balancing term. Duals
    // start at the magnitude of the costs instead of at 1, which is what
    // keeps the first steps from being cut to nothing.
    {
        for (int j = 0; j < n; ++j) K.val[dpos[j]] = -(qdiag[j] + 1.0);
        for (int k = 0; k < lp.Q.nnz(); ++k) if (qpos[k] >= 0) K.val[qpos[k]] = -lp.Q.val[k];
        for (int p = 0; p < (int)apos.size(); ++p) K.val[apos[p]] = AT.data[p];
        for (int i = 0; i < m; ++i) K.val[dpos[n + i]] = eq[i] ? 1e-8 : 1.0;
        ldl.factor(K, sign, 1e-13);
        std::vector<double> r0(N);
        for (int j = 0; j < n; ++j) r0[j] = c[j];
        for (int i = 0; i < m; ++i) {
            double t = 0.0;
            if (eq[i]) t = b[i];
            else if (rl[i] && ru[i]) t = 0.5 * (lp.rL[i] + lp.rU[i]);
            else if (rl[i]) t = lp.rL[i];
            else if (ru[i]) t = lp.rU[i];
            r0[n + i] = t;
        }
        ldl.solve(r0);
        std::vector<double> x0(r0.begin(), r0.begin() + n), l0(r0.begin() + n, r0.end());
        auto qx0 = matvec(Q, x0), aty0 = matvec(AT, l0), ax0 = matvec(A, x0);
        std::vector<double> z0(n);
        for (int j = 0; j < n; ++j) z0[j] = qx0[j] + c[j] - aty0[j];
        // slacks and duals of every complementarity pair at the LS point
        double smin = kInf, dmin = kInf;
        std::vector<double> S, Dd;
        auto pair = [&](double s, double d) { S.push_back(s); Dd.push_back(d); smin = std::min(smin, s); dmin = std::min(dmin, d); };
        for (int j = 0; j < n; ++j) {
            if (hl[j]) pair(x0[j] - l[j], hu[j] ? std::max(z0[j], 0.0) : z0[j]);
            if (hu[j]) pair(u[j] - x0[j], hl[j] ? std::max(-z0[j], 0.0) : -z0[j]);
        }
        for (int i = 0; i < m; ++i) if (!eq[i]) {
            if (rl[i]) pair(ax0[i] - lp.rL[i], ru[i] ? std::max(l0[i], 0.0) : l0[i]);
            if (ru[i]) pair(lp.rU[i] - ax0[i], rl[i] ? std::max(-l0[i], 0.0) : -l0[i]);
        }
        double ds = S.empty() ? 1.0 : std::max(0.0, -1.5 * smin);
        double dd0 = Dd.empty() ? 1.0 : std::max(0.0, -1.5 * dmin);
        double prod = 0.0, sums = 0.0, sumd = 0.0;
        for (size_t k = 0; k < S.size(); ++k) {
            prod += (S[k] + ds) * (Dd[k] + dd0);
            sums += S[k] + ds; sumd += Dd[k] + dd0;
        }
        double dsh = ds + (sumd > 0 ? 0.5 * prod / sumd : 0.0);
        double ddh = dd0 + (sums > 0 ? 0.5 * prod / sums : 0.0);
        dsh = std::max(dsh, 1e-2); ddh = std::max(ddh, 1e-2);
        // x (and w) move into the interior by the slack shift
        for (int j = 0; j < n; ++j) {
            double v = x0[j];
            if (hl[j] && hu[j]) {
                double th = std::min(dsh, 0.5 * (u[j] - l[j]));
                v = std::min(std::max(v, l[j] + th), u[j] - th);
            } else if (hl[j]) v = l[j] + (x0[j] - l[j]) + dsh;
            else if (hu[j]) v = u[j] - (u[j] - x0[j]) - dsh;
            if (hl[j] && v <= l[j]) v = l[j] + 0.5 * dsh;
            if (hu[j] && v >= u[j]) v = u[j] - 0.5 * dsh;
            x[j] = v;
            zl[j] = hl[j] ? (hu[j] ? std::max(z0[j], 0.0) : z0[j]) + ddh : 0.0;
            zu[j] = hu[j] ? (hl[j] ? std::max(-z0[j], 0.0) : -z0[j]) + ddh : 0.0;
            if (hl[j]) zl[j] = std::max(zl[j], 1e-2 * ddh);
            if (hu[j]) zu[j] = std::max(zu[j], 1e-2 * ddh);
        }
        for (int i = 0; i < m; ++i) {
            if (eq[i]) { lam[i] = l0[i]; continue; }
            double v = ax0[i];
            if (rl[i] && ru[i]) { double th = std::min(dsh, 0.5 * (lp.rU[i] - lp.rL[i])); v = std::min(std::max(v, lp.rL[i] + th), lp.rU[i] - th); }
            else if (rl[i]) v = ax0[i] + dsh;
            else if (ru[i]) v = ax0[i] - dsh;
            if (rl[i] && v <= lp.rL[i]) v = lp.rL[i] + 0.5 * dsh;
            if (ru[i] && v >= lp.rU[i]) v = lp.rU[i] - 0.5 * dsh;
            w[i] = v;
            yl[i] = rl[i] ? std::max((ru[i] ? std::max(l0[i], 0.0) : l0[i]) + ddh, 1e-2 * ddh) : 0.0;
            yu[i] = ru[i] ? std::max((rl[i] ? std::max(-l0[i], 0.0) : -l0[i]) + ddh, 1e-2 * ddh) : 0.0;
            lam[i] = yl[i] - yu[i];
        }
    }

    const double bnorm = std::max({inf_norm(b), [&] { double v = 0; for (int i = 0; i < m; ++i) { if (rl[i]) v = std::max(v, std::fabs(lp.rL[i])); if (ru[i]) v = std::max(v, std::fabs(lp.rU[i])); } return v; }()});
    const double cnorm = inf_norm(c);
    int ncomp = 0;
    for (int j = 0; j < n; ++j) ncomp += hl[j] + hu[j];
    for (int i = 0; i < m; ++i) ncomp += rl[i] + ru[i];
    ncomp = std::max(ncomp, 1);

    std::vector<double> rd(n), rp(m), rw(m), sxl(n), sxu(n), swl(m), swu(m), Dx(n), Dw(m);
    std::vector<double> dx(n), dlam(m), dw(m), dzl(n), dzu(n), dyl(m), dyu(m);
    std::vector<double> ax_(n), rhs(N), sol(N), resid(N);
    std::vector<char> mxl(hl), mxu(hu), mwl(m), mwu(m);
    for (int i = 0; i < m; ++i) { mwl[i] = rl[i]; mwu[i] = ru[i]; }
    std::vector<char> mall_n(n, 1), mall_m(m, 1);

    double last_solve_res = 0.0;
    // K * v with the unregularized values (for iterative refinement)
    auto kmul = [&](const std::vector<double>& v, std::vector<double>& out) {
        std::fill(out.begin(), out.end(), 0.0);
        for (int j = 0; j < N; ++j)
            for (int p = K.colptr[j]; p < K.colptr[j + 1]; ++p) {
                int i = K.rowidx[p]; double val = K.val[p];
                if (i == j) val -= j < n ? -dp : dd;          // remove regularization
                out[i] += val * v[j];
                if (i != j) out[j] += val * v[i];
            }
    };

    // Newton solve for given complementarity right-hand sides.
    double rscale = 1.0;   // 0 for centrality correctors: residuals are not re-targeted
    auto newton = [&](const std::vector<double>& rcl, const std::vector<double>& rcu,
                      const std::vector<double>& rcwl, const std::vector<double>& rcwu) {
        std::vector<double> gw(m, 0.0);
        for (int j = 0; j < n; ++j) {
            double v = rscale * rd[j];
            if (hl[j]) v -= rcl[j] / sxl[j];
            if (hu[j]) v += rcu[j] / sxu[j];
            rhs[j] = v;
        }
        for (int i = 0; i < m; ++i) {
            if (eq[i]) { rhs[n + i] = -rscale * rp[i]; continue; }
            double g = -rscale * rw[i];
            if (rl[i]) g += rcwl[i] / swl[i];
            if (ru[i]) g -= rcwu[i] / swu[i];
            gw[i] = g;
            rhs[n + i] = -rscale * rp[i] + g / Dw[i];
        }
        sol = rhs;
        ldl.solve(sol);
        // Iterative refinement against the unregularized K (QP Step 6.3):
        // removes both rounding error and the bias of the regularization.
        double prev_rn = kInf;
        std::vector<double> best_sol = sol;
        for (int ref = 0; ref < 10; ++ref) {
            kmul(sol, resid);
            double rn = 0.0;
            for (int k = 0; k < N; ++k) { resid[k] = rhs[k] - resid[k]; rn = std::max(rn, std::fabs(resid[k])); }
            if (rn < prev_rn) { best_sol = sol; prev_rn = rn; }
            else break;                                   // stagnated: keep the best
            if (rn < 1e-14 * (1.0 + inf_norm(rhs))) break;
            ldl.solve(resid);
            for (int k = 0; k < N; ++k) sol[k] += resid[k];
        }
        sol = best_sol;
        last_solve_res = prev_rn / (1.0 + inf_norm(rhs));
        for (int j = 0; j < n; ++j) dx[j] = sol[j];
        for (int i = 0; i < m; ++i) dlam[i] = sol[n + i];
        for (int i = 0; i < m; ++i) dw[i] = eq[i] ? 0.0 : (gw[i] - dlam[i]) / Dw[i];
        for (int j = 0; j < n; ++j) {
            dzl[j] = hl[j] ? (rcl[j] - zl[j] * dx[j]) / sxl[j] : 0.0;
            dzu[j] = hu[j] ? (rcu[j] + zu[j] * dx[j]) / sxu[j] : 0.0;
        }
        for (int i = 0; i < m; ++i) {
            dyl[i] = rl[i] ? (rcwl[i] - yl[i] * dw[i]) / swl[i] : 0.0;
            dyu[i] = ru[i] ? (rcwu[i] + yu[i] * dw[i]) / swu[i] : 0.0;
        }
    };

    std::vector<double> rcl(n), rcu(n), rcwl(m), rcwu(m);
    int bad_steps = 0, last_nreg = 0, stall_its = 0;
    double best_worst = kInf;
    for (int it = 0; ; ++it) {
        // ---- residuals ----
        auto qx = matvec(Q, x);
        auto aty = matvec(AT, lam);
        auto axv = matvec(A, x);
        double xqx = 0.0, cx = 0.0;
        for (int j = 0; j < n; ++j) { xqx += x[j] * qx[j]; cx += c[j] * x[j]; }
        for (int j = 0; j < n; ++j) rd[j] = qx[j] + c[j] - aty[j] - zl[j] + zu[j];
        for (int i = 0; i < m; ++i) {
            rp[i] = eq[i] ? axv[i] - b[i] : axv[i] - w[i];
            rw[i] = eq[i] ? 0.0 : lam[i] - yl[i] + yu[i];
        }
        double mu = 0.0;
        for (int j = 0; j < n; ++j) {
            sxl[j] = hl[j] ? x[j] - l[j] : 1.0;
            sxu[j] = hu[j] ? u[j] - x[j] : 1.0;
            if (hl[j]) mu += sxl[j] * zl[j];
            if (hu[j]) mu += sxu[j] * zu[j];
        }
        for (int i = 0; i < m; ++i) {
            swl[i] = rl[i] ? w[i] - lp.rL[i] : 1.0;
            swu[i] = ru[i] ? lp.rU[i] - w[i] : 1.0;
            if (rl[i]) mu += swl[i] * yl[i];
            if (ru[i]) mu += swu[i] * yu[i];
        }
        mu /= ncomp;
        double pobj = 0.5 * xqx + cx, dobj = -0.5 * xqx;
        for (int j = 0; j < n; ++j) { if (hl[j]) dobj += l[j] * zl[j]; if (hu[j]) dobj -= u[j] * zu[j]; }
        for (int i = 0; i < m; ++i) {
            if (eq[i]) dobj += b[i] * lam[i];
            else { if (rl[i]) dobj += lp.rL[i] * yl[i]; if (ru[i]) dobj -= lp.rU[i] * yu[i]; }
        }
        res.pres = inf_norm(rp) / (1.0 + bnorm);
        res.dres = std::max(inf_norm(rd), inf_norm(rw)) / (1.0 + cnorm);
        res.gap = std::fabs(pobj - dobj) / (1.0 + std::fabs(pobj) + std::fabs(dobj));
        res.iterations = it;
        if (opt.verbose)
            std::printf("  ipm %3d  pobj % .10e  dobj % .10e  pres %.1e  dres %.1e  gap %.1e  mu %.1e\n",
                        it, pobj + lp.obj_offset, dobj + lp.obj_offset, res.pres, res.dres, res.gap, mu);
        if (res.pres <= opt.tol && res.dres <= opt.tol && res.gap <= opt.tol) { res.status = "optimal"; break; }
        // Stall: no residual improved for 15 iterations. If everything is
        // already within 1e-6, stop and say so ("near_optimal"); the caller's
        // verifier reports the exact residuals either way.
        {
            double worst = std::max({res.pres, res.dres, res.gap});
            if (worst < best_worst * 0.9) { best_worst = worst; stall_its = 0; }
            else if (++stall_its >= 15 && worst <= 1e-6) { res.status = "near_optimal"; break; }
        }
        if (it >= opt.max_iterations) { res.status = "iteration_limit"; break; }
        if (now_s() - t0 > opt.time_limit) { res.status = "time_limit"; break; }
        // divergence monitor (QP Step 5.4): exploding iterates with collapsing steps
        if (inf_norm(x) > 1e12 || inf_norm(lam) > 1e12 || bad_steps >= 8) { res.status = "infeasible_or_unbounded"; break; }

        // ---- assemble and factor K ----
        for (int j = 0; j < n; ++j) {
            Dx[j] = (hl[j] ? zl[j] / sxl[j] : 0.0) + (hu[j] ? zu[j] / sxu[j] : 0.0);
            K.val[dpos[j]] = -(qdiag[j] + Dx[j] + dp);
        }
        for (int k = 0; k < lp.Q.nnz(); ++k) if (qpos[k] >= 0) K.val[qpos[k]] = -lp.Q.val[k];
        for (int p = 0; p < (int)apos.size(); ++p) K.val[apos[p]] = AT.data[p];
        for (int i = 0; i < m; ++i) {
            if (eq[i]) { K.val[dpos[n + i]] = dd; continue; }
            Dw[i] = (rl[i] ? yl[i] / swl[i] : 0.0) + (ru[i] ? yu[i] / swu[i] : 0.0);
            Dw[i] = std::max(Dw[i], 1e-14);                       // free row: lambda pinned near 0
            K.val[dpos[n + i]] = 1.0 / Dw[i] + dd;
        }
        int nreg = ldl.factor(K, sign, 1e-10);
        res.regularized_pivots += nreg;
        last_nreg = nreg;

        // ---- predictor (affine scaling) ----
        for (int j = 0; j < n; ++j) { rcl[j] = hl[j] ? -sxl[j] * zl[j] : 0.0; rcu[j] = hu[j] ? -sxu[j] * zu[j] : 0.0; }
        for (int i = 0; i < m; ++i) { rcwl[i] = rl[i] ? -swl[i] * yl[i] : 0.0; rcwu[i] = ru[i] ? -swu[i] * yu[i] : 0.0; }
        newton(rcl, rcu, rcwl, rcwu);
        std::vector<double> ndx(n), ndw(m);
        for (int j = 0; j < n; ++j) ndx[j] = -dx[j];
        for (int i = 0; i < m; ++i) ndw[i] = -dw[i];
        double ap = std::min({max_step(sxl, dx, mxl), max_step(sxu, ndx, mxu), max_step(swl, dw, mwl), max_step(swu, ndw, mwu)});
        double ad = std::min({max_step(zl, dzl, mxl), max_step(zu, dzu, mxu), max_step(yl, dyl, mwl), max_step(yu, dyu, mwu)});
        double mu_aff = 0.0;
        for (int j = 0; j < n; ++j) {
            if (hl[j]) mu_aff += (sxl[j] + ap * dx[j]) * (zl[j] + ad * dzl[j]);
            if (hu[j]) mu_aff += (sxu[j] - ap * dx[j]) * (zu[j] + ad * dzu[j]);
        }
        for (int i = 0; i < m; ++i) {
            if (rl[i]) mu_aff += (swl[i] + ap * dw[i]) * (yl[i] + ad * dyl[i]);
            if (ru[i]) mu_aff += (swu[i] - ap * dw[i]) * (yu[i] + ad * dyu[i]);
        }
        mu_aff /= ncomp;
        double sigma = std::pow(std::max(mu_aff, 0.0) / std::max(mu, 1e-300), 3.0);
        sigma = std::min(sigma, 1.0);

        // ---- corrector ----
        for (int j = 0; j < n; ++j) {
            rcl[j] = hl[j] ? sigma * mu - sxl[j] * zl[j] - dx[j] * dzl[j] : 0.0;
            rcu[j] = hu[j] ? sigma * mu - sxu[j] * zu[j] + dx[j] * dzu[j] : 0.0;
        }
        for (int i = 0; i < m; ++i) {
            rcwl[i] = rl[i] ? sigma * mu - swl[i] * yl[i] - dw[i] * dyl[i] : 0.0;
            rcwu[i] = ru[i] ? sigma * mu - swu[i] * yu[i] + dw[i] * dyu[i] : 0.0;
        }
        newton(rcl, rcu, rcwl, rcwu);
        for (int j = 0; j < n; ++j) ndx[j] = -dx[j];
        for (int i = 0; i < m; ++i) ndw[i] = -dw[i];
        ap = std::min({max_step(sxl, dx, mxl), max_step(sxu, ndx, mxu), max_step(swl, dw, mwl), max_step(swu, ndw, mwu)});
        ad = std::min({max_step(zl, dzl, mxl), max_step(zu, dzu, mxu), max_step(yl, dyl, mwl), max_step(yu, dyu, mwu)});

        // ---- Gondzio multiple centrality correctors ----
        // Aim for a longer step a~ and push the complementarity products of
        // the trial point back into [0.1, 10] * sigma*mu; the correction is a
        // Newton solve with zero residuals on the same factorization. Kept
        // only if the step really grows.
        for (int kc = 0; kc < 3; ++kc) {
            const double a = std::min(ap, ad);
            if (a >= 0.99) break;
            const double at = std::min(1.0, 1.5 * a + 0.1);
            const double lo_t = 0.1 * sigma * mu, hi_t = 10.0 * sigma * mu;
            auto tgt = [&](double v) { double t = std::min(std::max(v, lo_t), hi_t); return std::max(t - v, -hi_t); };
            for (int j = 0; j < n; ++j) {
                rcl[j] = hl[j] ? tgt((sxl[j] + at * dx[j]) * (zl[j] + at * dzl[j])) : 0.0;
                rcu[j] = hu[j] ? tgt((sxu[j] - at * dx[j]) * (zu[j] + at * dzu[j])) : 0.0;
            }
            for (int i = 0; i < m; ++i) {
                rcwl[i] = rl[i] ? tgt((swl[i] + at * dw[i]) * (yl[i] + at * dyl[i])) : 0.0;
                rcwu[i] = ru[i] ? tgt((swu[i] - at * dw[i]) * (yu[i] + at * dyu[i])) : 0.0;
            }
            auto bdx = dx, bdlam = dlam, bdw = dw, bdzl = dzl, bdzu = dzu, bdyl = dyl, bdyu = dyu;
            rscale = 0.0;
            newton(rcl, rcu, rcwl, rcwu);
            rscale = 1.0;
            for (int j = 0; j < n; ++j) { dx[j] += bdx[j]; dzl[j] += bdzl[j]; dzu[j] += bdzu[j]; }
            for (int i = 0; i < m; ++i) { dlam[i] += bdlam[i]; dw[i] += bdw[i]; dyl[i] += bdyl[i]; dyu[i] += bdyu[i]; }
            for (int j = 0; j < n; ++j) ndx[j] = -dx[j];
            for (int i = 0; i < m; ++i) ndw[i] = -dw[i];
            double ap2 = std::min({max_step(sxl, dx, mxl), max_step(sxu, ndx, mxu), max_step(swl, dw, mwl), max_step(swu, ndw, mwu)});
            double ad2 = std::min({max_step(zl, dzl, mxl), max_step(zu, dzu, mxu), max_step(yl, dyl, mwl), max_step(yu, dyu, mwu)});
            if (std::min(ap2, ad2) >= a + 0.1 * (at - a)) { ap = ap2; ad = ad2; }
            else {
                dx = bdx; dlam = bdlam; dw = bdw; dzl = bdzl; dzu = bdzu; dyl = bdyl; dyu = bdyu;
                break;
            }
        }
        // fraction to the boundary: never exactly 1, or a slack lands on 0
        const double eta = std::min(0.9999, std::max(0.9, 1.0 - 10.0 * mu));
        ap = std::min(1.0, eta * ap);
        ad = std::min(1.0, eta * ad);
        // QP: the dual residual Qx + c - A'lam - z moves with BOTH steps
        // (alpha_p Q dx - alpha_d (A'dlam + dz)), so it only shrinks when the
        // steps are equal. LP keeps separate primal and dual steps.
        if (lp.Q.nnz() > 0) ap = ad = std::min(ap, ad);
        bad_steps = (ap < 1e-8 && ad < 1e-8) ? bad_steps + 1 : 0;
        if (opt.verbose >= 2) {
            std::printf("        step primal %.3e dual %.3e  sigma %.2e  |dx| %.2e |dlam| %.2e  reg %d  solve res %.1e\n",
                        ap, ad, sigma, inf_norm(dx), inf_norm(dlam), last_nreg, last_solve_res);
            if (opt.verbose >= 3) {
                // which pair blocks the step
                double bp = kInf; const char* what = "-"; int who = -1;
                for (int j = 0; j < n; ++j) {
                    if (hl[j] && dx[j] < 0 && -sxl[j] / dx[j] < bp) { bp = -sxl[j] / dx[j]; what = "x-l"; who = j; }
                    if (hu[j] && dx[j] > 0 && sxu[j] / dx[j] < bp) { bp = sxu[j] / dx[j]; what = "u-x"; who = j; }
                    if (hl[j] && dzl[j] < 0 && -zl[j] / dzl[j] < bp) { bp = -zl[j] / dzl[j]; what = "zl"; who = j; }
                    if (hu[j] && dzu[j] < 0 && -zu[j] / dzu[j] < bp) { bp = -zu[j] / dzu[j]; what = "zu"; who = j; }
                }
                for (int i = 0; i < m; ++i) {
                    if (rl[i] && dw[i] < 0 && -swl[i] / dw[i] < bp) { bp = -swl[i] / dw[i]; what = "w-rL"; who = i; }
                    if (ru[i] && dw[i] > 0 && swu[i] / dw[i] < bp) { bp = swu[i] / dw[i]; what = "rU-w"; who = i; }
                    if (rl[i] && dyl[i] < 0 && -yl[i] / dyl[i] < bp) { bp = -yl[i] / dyl[i]; what = "yl"; who = i; }
                    if (ru[i] && dyu[i] < 0 && -yu[i] / dyu[i] < bp) { bp = -yu[i] / dyu[i]; what = "yu"; who = i; }
                }
                std::printf("        blocking: %s index %d at %.3e\n", what, who, bp);
                int jm = 0, im = 0;
                for (int j = 0; j < n; ++j) if (std::fabs(dx[j]) > std::fabs(dx[jm])) jm = j;
                for (int i = 0; i < m; ++i) if (std::fabs(dlam[i]) > std::fabs(dlam[im])) im = i;
                std::printf("        max dx at %d: x %.3e [%g,%g] Dx %.3e colnnz %d rd %.3e | max dlam at %d eq %d [%g,%g] w %.3e Dw %.3e\n",
                            jm, x[jm], l[jm], u[jm], Dx[jm], AT.indptr[jm + 1] - AT.indptr[jm], rd[jm],
                            im, (int)eq[im], lp.rL[im], lp.rU[im], w[im], Dw[im]);
            }
        }

        for (int j = 0; j < n; ++j) { x[j] += ap * dx[j]; zl[j] += ad * dzl[j]; zu[j] += ad * dzu[j]; }
        for (int i = 0; i < m; ++i) {
            if (!eq[i]) w[i] += ap * dw[i];
            lam[i] += ad * dlam[i];
            yl[i] += ad * dyl[i]; yu[i] += ad * dyu[i];
        }
        // keep strictly interior (guards against rounding on the boundary)
        for (int j = 0; j < n; ++j) {
            if (hl[j] && x[j] - l[j] <= 0) x[j] = l[j] + 1e-14 * (1.0 + std::fabs(l[j]));
            if (hu[j] && u[j] - x[j] <= 0) x[j] = u[j] - 1e-14 * (1.0 + std::fabs(u[j]));
            if (hl[j]) zl[j] = std::max(zl[j], 1e-300);
            if (hu[j]) zu[j] = std::max(zu[j], 1e-300);
        }
        for (int i = 0; i < m; ++i) {
            if (eq[i]) continue;
            if (rl[i] && w[i] - lp.rL[i] <= 0) w[i] = lp.rL[i] + 1e-14 * (1.0 + std::fabs(lp.rL[i]));
            if (ru[i] && lp.rU[i] - w[i] <= 0) w[i] = lp.rU[i] - 1e-14 * (1.0 + std::fabs(lp.rU[i]));
            if (rl[i]) yl[i] = std::max(yl[i], 1e-300);
            if (ru[i]) yu[i] = std::max(yu[i], 1e-300);
        }
    }

    res.x = x;
    res.y = lam;
    res.z.resize(n);
    auto qx = matvec(Q, x);
    double obj = lp.obj_offset;
    for (int j = 0; j < n; ++j) { res.z[j] = zl[j] - zu[j]; obj += c[j] * x[j] + 0.5 * x[j] * qx[j]; }
    res.objective = obj;
    res.seconds = now_s() - t0;
    return res;
}
