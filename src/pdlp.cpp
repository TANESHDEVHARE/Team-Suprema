#include "pdlp.hpp"
#include "verify.hpp"
#include "pdlp_algo.hpp"
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

// ------------------------------------------------------------ CPU backend --
// The PDLP algorithm itself lives in include/pdlp_algo.hpp and is shared with
// the CUDA backend (gpu/pdlp_gpu_solve.cu); this is just the arithmetic layer.
namespace {

struct CpuBackend {
    using Vec = std::vector<double>;
    int n, m;
    CSR A, AT;
    const std::vector<double> &c, &l, &u, &b;
    std::vector<char> is_ineq;

    explicit CpuBackend(const RangedLP& s)
        : n(s.n()), m(s.m()), A(to_csr(s.A)), AT(to_csc_as_transposed_csr(s.A)),
          c(s.c), l(s.l), u(s.u), b(s.rL), is_ineq(s.m()) {
        for (int i = 0; i < m; ++i) is_ineq[i] = s.rL[i] != s.rU[i];
    }
    Vec vec_n() const { return Vec(n, 0.0); }
    Vec vec_m() const { return Vec(m, 0.0); }

    static void spmv(const CSR& M, const Vec& v, Vec& out) {
        for (int i = 0; i < M.rows; ++i) {
            double s = 0.0;
            for (int p = M.indptr[i]; p < M.indptr[i + 1]; ++p) s += M.data[p] * v[M.indices[p]];
            out[i] = s;
        }
    }
    void Ax(const Vec& x, Vec& out) const { spmv(A, x, out); }
    void ATy(const Vec& y, Vec& out) const { spmv(AT, y, out); }

    void primal_step(const Vec& x, const Vec& aty, double tau, Vec& xn) const {
        for (int j = 0; j < n; ++j) xn[j] = std::min(std::max(x[j] - tau * (c[j] - aty[j]), l[j]), u[j]);
    }
    void dual_step(const Vec& y, const Vec& axn, const Vec& ax, double sigma, Vec& yn) const {
        for (int i = 0; i < m; ++i) {
            double v = y[i] + sigma * (b[i] - (2.0 * axn[i] - ax[i]));
            yn[i] = (is_ineq[i] && v < 0.0) ? 0.0 : v;
        }
    }
    void step_stats(const Vec& x, const Vec& xn, const Vec& y, const Vec& yn, const Vec& ax, const Vec& axn,
                    double& dx2, double& dy2, double& inter) const {
        dx2 = dy2 = inter = 0.0;
        for (int j = 0; j < n; ++j) { double d = xn[j] - x[j]; dx2 += d * d; }
        for (int i = 0; i < m; ++i) { double d = yn[i] - y[i]; dy2 += d * d; inter += d * (axn[i] - ax[i]); }
    }
    void axpy(double a, const Vec& x, Vec& y) const { for (size_t k = 0; k < x.size(); ++k) y[k] += a * x[k]; }
    void zero(Vec& v) const { std::fill(v.begin(), v.end(), 0.0); }
    void swap(Vec& a, Vec& b2) const { a.swap(b2); }
    void to_host(const Vec& v, std::vector<double>& h) const { h = v; }
    void from_host(const std::vector<double>& h, Vec& v) const { v = h; }
};

} // namespace

PdlpResult solve_pdhg(const RangedLP& unscaled_form, const RangedLP& scaled_form,
                       const std::vector<double>& Dr, const std::vector<double>& Dc,
                       double /*eta: step size is adaptive now*/, int max_iterations, int check_every, double tol,
                       std::atomic<bool>* stop_flag) {
    CpuBackend B(scaled_form);
    return pdlp_run(B, unscaled_form, scaled_form, Dr, Dc, max_iterations, check_every, tol, stop_flag);
}
