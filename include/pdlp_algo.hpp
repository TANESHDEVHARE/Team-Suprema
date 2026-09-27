// pdlp_algo.hpp -- restarted PDLP (LP pipeline Step 3), written ONCE and run
// on either backend: CpuBackend (src/pdlp.cpp) or GpuBackend
// (gpu/pdlp_gpu_solve.cu). The CPU/GPU race in solve.cpp therefore compares
// identical mathematics on different hardware, never two drifting copies.
//
// Problem (the "PDLP form" from build_pdlp_form, already Ruiz + Pock-Chambolle
// scaled):  min c^T x  s.t.  A x = b (equality rows), A x >= b (the rest),
//           l <= x <= u.
//
// Algorithm (Applegate et al., "Practical large-scale linear programming using
// primal-dual hybrid gradient", NeurIPS 2021, with the KKT-based restarts of
// its later versions):
//   * PDHG step with primal weight w:  tau = eta / w, sigma = eta * w
//       x' = proj_X(x - tau (c - A^T y))
//       y' = proj_Y(y + sigma (b - A(2x' - x)))
//   * adaptive step size: accept iff eta <= eta_bar, where
//       eta_bar = (w |x'-x|^2 + |y'-y|^2 / w) / (2 |(y'-y)^T A (x'-x)|)
//     next eta = min((1-(k+1)^-0.3) eta_bar, (1+(k+1)^-0.6) eta)
//   * step-size-weighted ergodic average
//   * restarts to the better of {current, average} by KKT error: sufficient
//     decay (0.2), necessary decay (0.8) with no further progress, or an
//     artificial restart after 36% of all iterations
//   * primal weight update at each restart:
//       w = exp(0.5 log(|dy| / |dx|) + 0.5 log w)
// A x and A^T y are carried with the iterates, so every step costs exactly
// two SpMVs (A x' and A^T y').
//
// Termination and the KKT merit are judged on the UNSCALED problem with the
// independent verify() (include/verify.hpp), every `check_every` steps.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <atomic>
#include "pdlp.hpp"
#include "verify.hpp"

// Backend requirements (Vec is the backend's vector handle):
//   Vec vec_n(); Vec vec_m();                          zero-initialised vectors
//   void Ax(const Vec& x, Vec& out);  void ATy(const Vec& y, Vec& out);
//   void primal_step(const Vec& x, const Vec& aty, double tau, Vec& xn);
//   void dual_step(const Vec& y, const Vec& axn, const Vec& ax, double sigma, Vec& yn);
//   void step_stats(x, xn, y, yn, ax, axn, double& dx2, double& dy2, double& inter);
//   void axpy(double a, const Vec& x, Vec& y);         y += a x
//   void zero(Vec& v);  void swap(Vec& a, Vec& b);
//   void to_host(const Vec& v, std::vector<double>& h);
//   void from_host(const std::vector<double>& h, Vec& v);
template <class Backend>
PdlpResult pdlp_run(Backend& B, const RangedLP& unscaled_form, const RangedLP& scaled_form,
                    const std::vector<double>& Dr, const std::vector<double>& Dc,
                    int max_iterations, int check_every, double tol, std::atomic<bool>* stop_flag) {
    using Vec = typename Backend::Vec;
    const int n = scaled_form.n(), m = scaled_form.m();

    // Fast verifier inputs, built once.
    const CSR Au = to_csr(unscaled_form.A), AuT = to_csc_as_transposed_csr(unscaled_form.A);
    auto unscaled_kkt = [&](const std::vector<double>& xs, const std::vector<double>& ys,
                            std::vector<double>& xu, std::vector<double>& yu) {
        xu.resize(n); yu.resize(m);
        for (int j = 0; j < n; ++j) xu[j] = Dc[j] * xs[j];
        for (int i = 0; i < m; ++i) yu[i] = Dr[i] * ys[i];
        return verify(unscaled_form, Au, AuT, xu, yu);
    };
    auto merit = [](const KKTReport& r) { return std::sqrt(r.eps_P * r.eps_P + r.eps_D * r.eps_D + r.eps_G * r.eps_G); };

    // ---- initial point, step size, primal weight ----
    std::vector<double> hx(n), hy(m, 0.0);
    for (int j = 0; j < n; ++j) hx[j] = std::min(std::max(0.0, scaled_form.l[j]), scaled_form.u[j]);
    double amax = 0.0;
    for (double v : scaled_form.A.val) amax = std::max(amax, std::fabs(v));
    double eta = amax > 0 ? 1.0 / amax : 1.0;
    double cn = 0.0, bn = 0.0;
    for (double v : scaled_form.c) cn += v * v;
    for (double v : scaled_form.rL) if (std::isfinite(v)) bn += v * v;
    cn = std::sqrt(cn); bn = std::sqrt(bn);
    double omega = (cn > 1e-10 && bn > 1e-10) ? cn / bn : 1.0;

    Vec x = B.vec_n(), y = B.vec_m(), xn = B.vec_n(), yn = B.vec_m();
    Vec ax = B.vec_m(), axn = B.vec_m(), aty = B.vec_n();
    Vec xsum = B.vec_n(), ysum = B.vec_m();
    B.from_host(hx, x); B.from_host(hy, y);
    B.Ax(x, ax); B.ATy(y, aty);
    double wsum = 0.0;

    std::vector<double> hx_last = hx, hy_last = hy;          // last restart point (scaled)
    std::vector<double> hxs(n), hys(m), xu, yu, xu2, yu2;
    double merit_last_restart = std::numeric_limits<double>::infinity();
    double merit_prev_cand = std::numeric_limits<double>::infinity();
    int since_restart = 0, restarts = 0, accepted = 0;

    PdlpResult best;
    double best_merit = std::numeric_limits<double>::infinity();
    auto keep_best = [&](const std::vector<double>& xu_, const std::vector<double>& yu_, const KKTReport& r) {
        double mm = merit(r);
        if (mm < best_merit) {
            best_merit = mm; best.x = xu_; best.y = yu_;
            best.eps_P = r.eps_P; best.eps_D = r.eps_D; best.eps_G = r.eps_G;
        }
    };

    int attempts = 0;
    while (attempts < max_iterations) {
        if (stop_flag && stop_flag->load(std::memory_order_relaxed)) break;

        // ---- one PDHG step with adaptive step size ----
        while (attempts < max_iterations) {
            ++attempts;
            const double tau = eta / omega, sigma = eta * omega;
            B.primal_step(x, aty, tau, xn);
            B.Ax(xn, axn);
            B.dual_step(y, axn, ax, sigma, yn);
            double dx2, dy2, inter;
            B.step_stats(x, xn, y, yn, ax, axn, dx2, dy2, inter);
            const double num = omega * dx2 + dy2 / omega;
            const double eta_bar = std::fabs(inter) > 0 ? num / (2.0 * std::fabs(inter)) : std::numeric_limits<double>::infinity();
            // k counts from 1 in the paper, so the base is k+1 >= 2 (at 1 the
            // first factor is 0 and the step size would collapse to zero).
            const double k1 = accepted + 2.0;
            const double grow = (1.0 + std::pow(k1, -0.6)) * eta;
            const double eta_next = std::isfinite(eta_bar) ? std::min((1.0 - std::pow(k1, -0.3)) * eta_bar, grow) : grow;
            if (eta <= eta_bar) {
                B.swap(x, xn); B.swap(y, yn); B.swap(ax, axn);
                B.ATy(y, aty);
                B.axpy(eta, x, xsum); B.axpy(eta, y, ysum); wsum += eta;
                eta = eta_next;
                ++accepted; ++since_restart;
                break;
            }
            eta = eta_next;
        }

        if (accepted % check_every != 0 && attempts < max_iterations) continue;

        // ---- KKT check on the unscaled problem: current vs average ----
        B.to_host(x, hx); B.to_host(y, hy);
        B.to_host(xsum, hxs); B.to_host(ysum, hys);
        if (wsum > 0) { for (auto& v : hxs) v /= wsum; for (auto& v : hys) v /= wsum; }
        else { hxs = hx; hys = hy; }
        KKTReport rc = unscaled_kkt(hx, hy, xu, yu);
        KKTReport ra = unscaled_kkt(hxs, hys, xu2, yu2);
        keep_best(xu, yu, rc);
        keep_best(xu2, yu2, ra);
        const bool use_avg = merit(ra) < merit(rc);
        const KKTReport& rep = use_avg ? ra : rc;
        const double mc = merit(rep);
        if (std::max({rep.eps_P, rep.eps_D, rep.eps_G}) <= tol) {
            PdlpResult res;
            res.x = use_avg ? xu2 : xu; res.y = use_avg ? yu2 : yu;
            res.eps_P = rep.eps_P; res.eps_D = rep.eps_D; res.eps_G = rep.eps_G;
            res.iterations = attempts; res.restarts = restarts; res.converged = true;
            return res;
        }

        // ---- restart decision ----
        const bool sufficient = mc <= 0.2 * merit_last_restart;
        const bool necessary = mc <= 0.8 * merit_last_restart && mc > merit_prev_cand;
        const bool artificial = since_restart >= 0.36 * accepted;
        merit_prev_cand = mc;
        if (sufficient || necessary || artificial) {
            const std::vector<double>& cx = use_avg ? hxs : hx;
            const std::vector<double>& cy = use_avg ? hys : hy;
            double ddx = 0.0, ddy = 0.0;
            for (int j = 0; j < n; ++j) ddx += (cx[j] - hx_last[j]) * (cx[j] - hx_last[j]);
            for (int i = 0; i < m; ++i) ddy += (cy[i] - hy_last[i]) * (cy[i] - hy_last[i]);
            ddx = std::sqrt(ddx); ddy = std::sqrt(ddy);
            if (ddx > 1e-10 && ddy > 1e-10) omega = std::exp(0.5 * std::log(ddy / ddx) + 0.5 * std::log(omega));
            hx_last = cx; hy_last = cy;
            if (use_avg) { B.from_host(cx, x); B.from_host(cy, y); B.Ax(x, ax); B.ATy(y, aty); }
            B.zero(xsum); B.zero(ysum); wsum = 0.0;
            since_restart = 0; ++restarts;
            merit_last_restart = mc;
            merit_prev_cand = std::numeric_limits<double>::infinity();
        }
    }

    // Stopped (limit or another engine won): report the best point seen.
    if (!std::isfinite(best_merit)) {
        B.to_host(x, hx); B.to_host(y, hy);
        KKTReport r = unscaled_kkt(hx, hy, xu, yu);
        keep_best(xu, yu, r);
    }
    best.iterations = attempts; best.restarts = restarts; best.converged = false;
    return best;
}
