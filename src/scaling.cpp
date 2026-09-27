#include "scaling.hpp"
#include <cmath>

static std::vector<double> row_max_abs_all(const SparseMatrix& A) {
    std::vector<double> out(A.rows, 0.0);
    for (int k = 0; k < A.nnz(); ++k) out[A.row_idx[k]] = std::max(out[A.row_idx[k]], std::fabs(A.val[k]));
    return out;
}
static std::vector<double> col_max_abs_all(const SparseMatrix& A) {
    std::vector<double> out(A.cols, 0.0);
    for (int k = 0; k < A.nnz(); ++k) out[A.col_idx[k]] = std::max(out[A.col_idx[k]], std::fabs(A.val[k]));
    return out;
}
static std::vector<double> row_abs_sum_all(const SparseMatrix& A) {
    std::vector<double> out(A.rows, 0.0);
    for (int k = 0; k < A.nnz(); ++k) out[A.row_idx[k]] += std::fabs(A.val[k]);
    return out;
}
static std::vector<double> col_abs_sum_all(const SparseMatrix& A) {
    std::vector<double> out(A.cols, 0.0);
    for (int k = 0; k < A.nnz(); ++k) out[A.col_idx[k]] += std::fabs(A.val[k]);
    return out;
}

ScalingResult scale(const RangedLP& ranged, int ruiz_passes) {
    int m = ranged.m(), n = ranged.n();
    SparseMatrix A = ranged.A;
    std::vector<double> Dr(m, 1.0), Dc(n, 1.0);
    auto keep = [&](int j) { return !ranged.integer.empty() && ranged.integer[j]; };
    // QP: Q is scaled symmetrically (D Q D) and its column norms take part in
    // the column equilibration (QP pipeline Step 1.3).
    const bool qp = ranged.Q.nnz() > 0;
    SparseMatrix Q = ranged.Q;

    // ---- Ruiz equilibration: rows then columns, ruiz_passes times ----
    for (int pass = 0; pass < ruiz_passes; ++pass) {
        auto rmax = row_max_abs_all(A);
        std::vector<double> row_scale(m);
        for (int i = 0; i < m; ++i) row_scale[i] = rmax[i] > 0 ? 1.0 / std::sqrt(rmax[i]) : 1.0;
        A = scale_rows_cols(A, row_scale, std::vector<double>(n, 1.0));
        for (int i = 0; i < m; ++i) Dr[i] *= row_scale[i];

        auto cmax = col_max_abs_all(A);
        if (qp) for (int k = 0; k < Q.nnz(); ++k) {
            double v = std::fabs(Q.val[k]);
            cmax[Q.row_idx[k]] = std::max(cmax[Q.row_idx[k]], v);
            cmax[Q.col_idx[k]] = std::max(cmax[Q.col_idx[k]], v);
        }
        std::vector<double> col_scale(n);
        for (int j = 0; j < n; ++j) col_scale[j] = (cmax[j] > 0 && !keep(j)) ? 1.0 / std::sqrt(cmax[j]) : 1.0;
        A = scale_rows_cols(A, std::vector<double>(m, 1.0), col_scale);
        if (qp) for (int k = 0; k < Q.nnz(); ++k) Q.val[k] *= col_scale[Q.row_idx[k]] * col_scale[Q.col_idx[k]];
        for (int j = 0; j < n; ++j) Dc[j] *= col_scale[j];
    }

    // ---- Pock-Chambolle alpha=1 diagonal preconditioning (Theorem 9) ----
    auto row_sum = row_abs_sum_all(A);
    auto col_sum = col_abs_sum_all(A);
    std::vector<double> sigma(m), tau(n), Dr_pc(m), Dc_pc(n);
    for (int i = 0; i < m; ++i) { sigma[i] = row_sum[i] > 0 ? 1.0 / row_sum[i] : 1.0; Dr_pc[i] = std::sqrt(sigma[i]); }
    for (int j = 0; j < n; ++j) { tau[j] = col_sum[j] > 0 ? 1.0 / col_sum[j] : 1.0; Dc_pc[j] = keep(j) ? 1.0 : std::sqrt(tau[j]); }
    if (qp) { std::fill(Dr_pc.begin(), Dr_pc.end(), 1.0); std::fill(Dc_pc.begin(), Dc_pc.end(), 1.0); }   // PDHG-only step

    A = scale_rows_cols(A, Dr_pc, Dc_pc);
    for (int i = 0; i < m; ++i) Dr[i] *= Dr_pc[i];
    for (int j = 0; j < n; ++j) Dc[j] *= Dc_pc[j];

    RangedLP scaled;
    scaled.name = ranged.name + " (scaled)";
    scaled.col_names = ranged.col_names; scaled.row_names = ranged.row_names;
    scaled.c.resize(n); for (int j = 0; j < n; ++j) scaled.c[j] = Dc[j] * ranged.c[j];
    scaled.A = A;
    scaled.rL.resize(m); scaled.rU.resize(m);
    for (int i = 0; i < m; ++i) { scaled.rL[i] = Dr[i] * ranged.rL[i]; scaled.rU[i] = Dr[i] * ranged.rU[i]; }
    scaled.l.resize(n); scaled.u.resize(n);
    for (int j = 0; j < n; ++j) { scaled.l[j] = ranged.l[j] / Dc[j]; scaled.u[j] = ranged.u[j] / Dc[j]; }
    scaled.obj_offset = ranged.obj_offset;
    scaled.original_sense = ranged.original_sense;
    scaled.integer = ranged.integer;
    scaled.Q = Q;
    if (qp) for (int k = 0; k < Q.nnz(); ++k) scaled.Q.val[k] *= Dc_pc[Q.row_idx[k]] * Dc_pc[Q.col_idx[k]];

    return { scaled, Dr, Dc };
}
