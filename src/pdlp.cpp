#include "pdlp.hpp"
#include "verify.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

std::pair<RangedLP, PdlpFormMeta> build_pdlp_form(const RangedLP& ranged) {
    int m = ranged.m(), n = ranged.n();
    const auto& rL = ranged.rL; const auto& rU = ranged.rU;
    const auto& l = ranged.l; const auto& u = ranged.u; const auto& c = ranged.c;

    std::vector<int> eq_orig, ineq_orig, free_orig;
    std::vector<double> eq_sign, ineq_sign, eq_b, ineq_b;
    std::vector<int> slack_of_eq;                    // parallel to eq_orig; -1 = no slack
    std::vector<std::pair<double,double>> slack_bounds;

    for (int i = 0; i < m; ++i) {
        double lo = rL[i], hi = rU[i];
        if (lo == hi) {
            eq_orig.push_back(i); eq_sign.push_back(1.0); eq_b.push_back(lo); slack_of_eq.push_back(-1);
        } else if (std::isfinite(lo) && std::isfinite(hi)) {
            eq_orig.push_back(i); eq_sign.push_back(1.0); eq_b.push_back(0.0);
            slack_bounds.emplace_back(lo, hi);
            slack_of_eq.push_back((int)slack_bounds.size() - 1);
        } else if (std::isfinite(hi)) {
            ineq_orig.push_back(i); ineq_sign.push_back(-1.0); ineq_b.push_back(-hi);
        } else if (std::isfinite(lo)) {
            ineq_orig.push_back(i); ineq_sign.push_back(1.0); ineq_b.push_back(lo);
        } else {
            free_orig.push_back(i);
        }
    }

    int n_slack = (int)slack_bounds.size();
    std::vector<int> row_perm = eq_orig; row_perm.insert(row_perm.end(), ineq_orig.begin(), ineq_orig.end());
    int m2 = (int)row_perm.size();
    int n2 = n + n_slack;

    std::vector<double> row_signs = eq_sign; row_signs.insert(row_signs.end(), ineq_sign.begin(), ineq_sign.end());

    // row -> new position map, so we can scan original A once and place entries.
    std::vector<int> pos_of_orig_row(m, -1);
    for (int pos = 0; pos < m2; ++pos) pos_of_orig_row[row_perm[pos]] = pos;

    SparseMatrix A2; A2.rows = m2; A2.cols = n2;
    for (int k = 0; k < ranged.A.nnz(); ++k) {
        int orow = ranged.A.row_idx[k];
        int pos = pos_of_orig_row[orow];
        if (pos < 0) continue; // a free row (both bounds infinite) contributes nothing
        double sign = row_signs[pos];
        A2.row_idx.push_back(pos); A2.col_idx.push_back(ranged.A.col_idx[k]); A2.val.push_back(sign * ranged.A.val[k]);
    }
    for (int pos = 0; pos < (int)eq_orig.size(); ++pos) {
        int k = slack_of_eq[pos];
        if (k >= 0) { A2.row_idx.push_back(pos); A2.col_idx.push_back(n + k); A2.val.push_back(-1.0); }
    }

    std::vector<double> b2 = eq_b; b2.insert(b2.end(), ineq_b.begin(), ineq_b.end());
    std::vector<double> rL2 = b2;
    std::vector<double> rU2 = eq_b;
    for (size_t k = 0; k < ineq_orig.size(); ++k) rU2.push_back(INF);

    std::vector<double> l2 = l, u2 = u, c2 = c;
    for (auto& sb : slack_bounds) { l2.push_back(sb.first); u2.push_back(sb.second); }
    c2.resize(n2, 0.0);

    RangedLP pdlp_form;
    pdlp_form.name = ranged.name + " (pdlp-form)";
    pdlp_form.col_names = ranged.col_names;
    for (int k = 0; k < n_slack; ++k) pdlp_form.col_names.push_back("__range_slack_" + std::to_string(k));
    pdlp_form.row_names.reserve(m2);
    for (int pos = 0; pos < m2; ++pos) pdlp_form.row_names.push_back(ranged.row_names[row_perm[pos]]);
    pdlp_form.c = c2; pdlp_form.A = A2; pdlp_form.rL = rL2; pdlp_form.rU = rU2; pdlp_form.l = l2; pdlp_form.u = u2;
    pdlp_form.obj_offset = ranged.obj_offset; pdlp_form.original_sense = ranged.original_sense;

    PdlpFormMeta meta;
    meta.n_orig = n; meta.m_orig = m; meta.n_slack = n_slack;
    meta.row_perm = row_perm; meta.row_signs = row_signs; meta.free_rows = free_orig;

    return { pdlp_form, meta };
}

void extract_solution(const PdlpFormMeta& meta, const std::vector<double>& x2, const std::vector<double>& y2,
                       std::vector<double>& x, std::vector<double>& y) {
    x.assign(x2.begin(), x2.begin() + meta.n_orig);
    y.assign(meta.m_orig, 0.0);
    for (size_t pos = 0; pos < meta.row_perm.size(); ++pos)
        y[meta.row_perm[pos]] = meta.row_signs[pos] * y2[pos];
}

static void unscale(const std::vector<double>& Dr, const std::vector<double>& Dc,
                     const std::vector<double>& xs, const std::vector<double>& ys,
                     std::vector<double>& x, std::vector<double>& y) {
    x.resize(xs.size()); for (size_t j = 0; j < xs.size(); ++j) x[j] = Dc[j] * xs[j];
    y.resize(ys.size()); for (size_t i = 0; i < ys.size(); ++i) y[i] = Dr[i] * ys[i];
}

PdlpResult solve_pdhg(const RangedLP& unscaled_form, const RangedLP& scaled_form,
                       const std::vector<double>& Dr, const std::vector<double>& Dc,
                       double eta, int max_iterations, int check_every, double tol,
                       std::atomic<bool>* stop_flag) {
    int m2 = scaled_form.m(), n2 = scaled_form.n();
    CSR A = to_csr(scaled_form.A);
    CSR AT = to_csc_as_transposed_csr(scaled_form.A);
    const auto& c = scaled_form.c; const auto& rL = scaled_form.rL; const auto& rU = scaled_form.rU;
    const auto& l = scaled_form.l; const auto& u = scaled_form.u;
    std::vector<char> is_ineq(m2);
    for (int i = 0; i < m2; ++i) is_ineq[i] = (rL[i] != rU[i]);
    const auto& b = rL;

    double tau = eta, sigma = eta;

    std::vector<double> x(n2, 0.0), y(m2, 0.0);
    for (int j = 0; j < n2; ++j) x[j] = std::min(std::max(0.0, l[j]), u[j]);
    std::vector<double> x_sum(n2, 0.0), y_sum(m2, 0.0);
    int n_since_restart = 0;
    double last_restart_score = std::numeric_limits<double>::infinity();
    int restarts = 0;

    for (int k = 1; k <= max_iterations; ++k) {
        if (stop_flag && stop_flag->load(std::memory_order_relaxed)) {
            // Another engine already produced a verified answer -- stop
            // racing and hand back whatever this engine has right now
            // (unconverged; the race harness discards it and keeps the
            // winner's result, so this value is never actually used, but
            // returning a well-formed result keeps the function's contract
            // simple regardless of who called it).
            std::vector<double> xu, yu; unscale(Dr, Dc, x, y, xu, yu);
            KKTReport rep = verify(unscaled_form, xu, yu);
            PdlpResult res; res.x = xu; res.y = yu; res.iterations = k; res.converged = false;
            res.eps_P = rep.eps_P; res.eps_D = rep.eps_D; res.eps_G = rep.eps_G; res.restarts = restarts;
            return res;
        }
        auto ATy = matvec_T(AT, y);
        std::vector<double> x_new(n2);
        for (int j = 0; j < n2; ++j) x_new[j] = std::min(std::max(x[j] - tau * (c[j] - ATy[j]), l[j]), u[j]);

        std::vector<double> two_xnew_minus_x(n2);
        for (int j = 0; j < n2; ++j) two_xnew_minus_x[j] = 2 * x_new[j] - x[j];
        auto Adx = matvec(A, two_xnew_minus_x);
        std::vector<double> y_new(m2);
        for (int i = 0; i < m2; ++i) {
            double yv = y[i] + sigma * (b[i] - Adx[i]);
            if (is_ineq[i]) yv = std::max(yv, 0.0);
            y_new[i] = yv;
        }
        x = x_new; y = y_new;
        for (int j = 0; j < n2; ++j) x_sum[j] += x[j];
        for (int i = 0; i < m2; ++i) y_sum[i] += y[i];
        n_since_restart++;

        if (k % check_every == 0) {
            std::vector<double> x_avg(n2), y_avg(m2);
            for (int j = 0; j < n2; ++j) x_avg[j] = x_sum[j] / n_since_restart;
            for (int i = 0; i < m2; ++i) y_avg[i] = y_sum[i] / n_since_restart;

            std::vector<double> xu_cur, yu_cur, xu_avg, yu_avg;
            unscale(Dr, Dc, x, y, xu_cur, yu_cur);
            unscale(Dr, Dc, x_avg, y_avg, xu_avg, yu_avg);
            KKTReport rep_cur = verify(unscaled_form, xu_cur, yu_cur);
            KKTReport rep_avg = verify(unscaled_form, xu_avg, yu_avg);
            double score_cur = rep_cur.eps_P + rep_cur.eps_D + rep_cur.eps_G;
            double score_avg = rep_avg.eps_P + rep_avg.eps_D + rep_avg.eps_G;

            bool use_avg = score_avg <= score_cur;
            const auto& cand_x = use_avg ? x_avg : x;
            const auto& cand_y = use_avg ? y_avg : y;
            double cand_score = use_avg ? score_avg : score_cur;
            const KKTReport& cand_rep = use_avg ? rep_avg : rep_cur;

            if (std::max({cand_rep.eps_P, cand_rep.eps_D, cand_rep.eps_G}) <= tol) {
                std::vector<double> xu, yu; unscale(Dr, Dc, cand_x, cand_y, xu, yu);
                PdlpResult res; res.x = xu; res.y = yu; res.iterations = k; res.converged = true;
                res.eps_P = cand_rep.eps_P; res.eps_D = cand_rep.eps_D; res.eps_G = cand_rep.eps_G; res.restarts = restarts;
                return res;
            }

            bool sufficient_decay = cand_score <= 0.2 * last_restart_score;
            bool artificial = n_since_restart >= std::max(64, (int)(0.36 * k));
            if (sufficient_decay || artificial) {
                x = cand_x; y = cand_y;
                std::fill(x_sum.begin(), x_sum.end(), 0.0);
                std::fill(y_sum.begin(), y_sum.end(), 0.0);
                n_since_restart = 0;
                last_restart_score = cand_score;
                restarts++;
            }
        }
    }

    std::vector<double> x_avg(n2), y_avg(m2);
    int denom = std::max(1, n_since_restart);
    for (int j = 0; j < n2; ++j) x_avg[j] = x_sum[j] / denom;
    for (int i = 0; i < m2; ++i) y_avg[i] = y_sum[i] / denom;
    std::vector<double> xu_cur, yu_cur, xu_avg, yu_avg;
    unscale(Dr, Dc, x, y, xu_cur, yu_cur);
    unscale(Dr, Dc, x_avg, y_avg, xu_avg, yu_avg);
    KKTReport rep_cur = verify(unscaled_form, xu_cur, yu_cur);
    KKTReport rep_avg = verify(unscaled_form, xu_avg, yu_avg);

    PdlpResult res;
    if ((rep_avg.eps_P + rep_avg.eps_D + rep_avg.eps_G) <= (rep_cur.eps_P + rep_cur.eps_D + rep_cur.eps_G)) {
        res.x = xu_avg; res.y = yu_avg; res.eps_P = rep_avg.eps_P; res.eps_D = rep_avg.eps_D; res.eps_G = rep_avg.eps_G;
    } else {
        res.x = xu_cur; res.y = yu_cur; res.eps_P = rep_cur.eps_P; res.eps_D = rep_cur.eps_D; res.eps_G = rep_cur.eps_G;
    }
    res.iterations = max_iterations; res.converged = false; res.restarts = restarts;
    return res;
}
