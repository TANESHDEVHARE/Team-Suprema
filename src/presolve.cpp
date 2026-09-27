#include "presolve.hpp"
#include <cmath>
#include <algorithm>

static constexpr double ZERO_TOL = 1e-12;
static constexpr double FEAS_TOL = 1e-9;

PresolveResult presolve(const RangedLP& ranged, int max_iterations, bool mip_mode) {
    int n0 = ranged.n(), m0 = ranged.m();

    SparseMatrix A = ranged.A;
    std::vector<double> c = ranged.c;
    std::vector<double> rL = ranged.rL, rU = ranged.rU;
    std::vector<double> l = ranged.l, u = ranged.u;
    double obj_offset = ranged.obj_offset;
    // Integer columns: bounds are rounded inward (valid for every integer point).
    auto is_int = [&](int orig_col) { return !ranged.integer.empty() && ranged.integer[orig_col]; };
    // QP: columns that appear in Q cannot be removed by R2/R3 (both assume a
    // linear objective in that variable).
    std::vector<char> in_q(n0, 0);
    for (int k = 0; k < ranged.Q.nnz(); ++k) { in_q[ranged.Q.row_idx[k]] = 1; in_q[ranged.Q.col_idx[k]] = 1; }
    bool empty_int_domain = false;
    for (int j = 0; j < n0; ++j) if (is_int(j)) {
        if (std::isfinite(l[j])) l[j] = std::ceil(l[j] - 1e-9);
        if (std::isfinite(u[j])) u[j] = std::floor(u[j] + 1e-9);
        if (l[j] > u[j]) empty_int_domain = true;
    }

    std::vector<int> row_ids(m0), col_ids(n0);
    // Magnitude of each (original) row's right-hand side plus every shift
    // applied to it by fixed columns: the scale for relative feasibility
    // tolerances (absolute ones fail on models with large coefficients --
    // RENTACAR was declared infeasible by rounding noise).
    std::vector<double> rowmag(m0, 0.0);
    for (int i = 0; i < m0; ++i) {
        if (std::isfinite(rL[i])) rowmag[i] = std::max(rowmag[i], std::fabs(rL[i]));
        if (std::isfinite(rU[i])) rowmag[i] = std::max(rowmag[i], std::fabs(rU[i]));
    }
    for (int i = 0; i < m0; ++i) row_ids[i] = i;
    for (int j = 0; j < n0; ++j) col_ids[j] = j;

    PresolveResult result;
    if (empty_int_domain) { result.status = "infeasible"; result.row_ids = row_ids; result.col_ids = col_ids; return result; }

    for (int iter = 0; iter < max_iterations; ++iter) {
        int m = A.rows, n = A.cols;
        if (m == 0 || n == 0) break;

        CSR Acsr = to_csr(A);
        auto rownnz = row_nnz_counts(Acsr);

        // ---- R1: empty row ----
        std::vector<int> empty_rows;
        for (int i = 0; i < m; ++i) if (rownnz[i] == 0) empty_rows.push_back(i);
        if (!empty_rows.empty()) {
            std::vector<char> keep(m, 1);
            for (int r : empty_rows) {
                const double tol = FEAS_TOL * (1.0 + rowmag[row_ids[r]]);
                if (!(rL[r] <= tol && rU[r] >= -tol)) {
                    result.has_reduced = false; result.status = "infeasible";
                    result.row_ids = row_ids; result.col_ids = col_ids;
                    return result;
                }
                Step s; s.kind = "R1"; s.row_id = row_ids[r];
                result.steps.push_back(s);
                keep[r] = 0;
            }
            A = filter_rows(A, keep);
            std::vector<double> nrL, nrU; std::vector<int> nrids;
            for (int i = 0; i < m; ++i) if (keep[i]) { nrL.push_back(rL[i]); nrU.push_back(rU[i]); nrids.push_back(row_ids[i]); }
            rL = nrL; rU = nrU; row_ids = nrids;
            continue;
        }

        auto colnnz = col_nnz_counts(A);

        // ---- R2: empty column ----
        std::vector<int> empty_cols;
        for (int j = 0; j < n; ++j) if (colnnz[j] == 0 && !in_q[col_ids[j]]) empty_cols.push_back(j);
        if (!empty_cols.empty()) {
            std::vector<char> keep(n, 1);
            for (int j : empty_cols) {
                double cj = c[j];
                double val;
                if (cj > ZERO_TOL) {
                    if (!std::isfinite(l[j])) { result.status = "unbounded"; result.row_ids = row_ids; result.col_ids = col_ids; return result; }
                    val = l[j];
                } else if (cj < -ZERO_TOL) {
                    if (!std::isfinite(u[j])) { result.status = "unbounded"; result.row_ids = row_ids; result.col_ids = col_ids; return result; }
                    val = u[j];
                } else {
                    val = std::isfinite(l[j]) ? l[j] : (std::isfinite(u[j]) ? u[j] : 0.0);
                }
                obj_offset += cj * val;
                Step s; s.kind = "R2"; s.col_id = col_ids[j]; s.value = val; s.cost = cj;
                result.steps.push_back(s);
                keep[j] = 0;
            }
            A = filter_cols(A, keep);
            std::vector<double> nc, nl, nu; std::vector<int> ncids;
            for (int j = 0; j < n; ++j) if (keep[j]) { nc.push_back(c[j]); nl.push_back(l[j]); nu.push_back(u[j]); ncids.push_back(col_ids[j]); }
            c = nc; l = nl; u = nu; col_ids = ncids;
            continue;
        }

        // ---- R3: fixed variable ----
        std::vector<int> fixed;
        for (int j = 0; j < n; ++j) if (std::isfinite(l[j]) && (u[j] - l[j]) < ZERO_TOL && !in_q[col_ids[j]]) fixed.push_back(j);
        if (!fixed.empty()) {
            std::vector<char> keep(n, 1);
            for (int j : fixed) {
                double v = l[j];
                auto col_entries = get_col_entries(A, j);
                Step s; s.kind = "R3"; s.col_id = col_ids[j]; s.value = v; s.cost = c[j];
                obj_offset += c[j] * v;
                for (auto& [ri, va] : col_entries) {
                    rL[ri] -= va * v;
                    rowmag[row_ids[ri]] += std::fabs(va * v);
                    rU[ri] -= va * v;
                    s.column.emplace_back(row_ids[ri], va);
                }
                result.steps.push_back(s);
                keep[j] = 0;
            }
            A = filter_cols(A, keep);
            std::vector<double> nc, nl, nu; std::vector<int> ncids;
            for (int j = 0; j < n; ++j) if (keep[j]) { nc.push_back(c[j]); nl.push_back(l[j]); nu.push_back(u[j]); ncids.push_back(col_ids[j]); }
            c = nc; l = nl; u = nu; col_ids = ncids;
            continue;
        }

        // ---- R4: singleton row ----
        Acsr = to_csr(A);
        rownnz = row_nnz_counts(Acsr);
        std::vector<int> singleton_rows;
        for (int i = 0; i < m; ++i) if (rownnz[i] == 1) singleton_rows.push_back(i);
        if (!singleton_rows.empty()) {
            std::vector<char> keep(m, 1);
            for (int r : singleton_rows) {
                int start = Acsr.indptr[r];
                int j = Acsr.indices[start];
                double a_ij = Acsr.data[start];
                double lo_before = l[j], hi_before = u[j];
                double b1 = rL[r] / a_ij, b2 = rU[r] / a_ij;
                double lo_row = (a_ij > 0) ? b1 : b2;
                double hi_row = (a_ij > 0) ? b2 : b1;
                double new_lo = std::max(l[j], lo_row);
                double new_hi = std::min(u[j], hi_row);
                const double btol = 1e-9 * (1.0 + rowmag[row_ids[r]] / std::fabs(a_ij));
                if (is_int(col_ids[j])) {
                    if (std::isfinite(new_lo)) new_lo = std::ceil(new_lo - std::max(1e-9, btol));
                    if (std::isfinite(new_hi)) new_hi = std::floor(new_hi + std::max(1e-9, btol));
                }
                if (new_lo - new_hi > std::max(1e-7, 100.0 * btol)) {
                    result.status = "infeasible"; result.row_ids = row_ids; result.col_ids = col_ids; return result;
                }
                if (new_lo > new_hi) new_lo = new_hi = 0.5 * (new_lo + new_hi);   // rounding-level crossing
                l[j] = new_lo; u[j] = new_hi;
                Step s; s.kind = "R4"; s.row_id = row_ids[r]; s.col_id = col_ids[j]; s.a_ij = a_ij;
                s.lo_from_row = lo_row; s.hi_from_row = hi_row; s.lo_before = lo_before; s.hi_before = hi_before;
                result.steps.push_back(s);
                keep[r] = 0;
            }
            A = filter_rows(A, keep);
            std::vector<double> nrL, nrU; std::vector<int> nrids;
            for (int i = 0; i < m; ++i) if (keep[i]) { nrL.push_back(rL[i]); nrU.push_back(rU[i]); nrids.push_back(row_ids[i]); }
            rL = nrL; rU = nrU; row_ids = nrids;
            continue;
        }

        break; // fixpoint
    }

    // ---- MILP coefficient tightening (MILP Step 2.1) ----
    // For a one-sided row a^T x <= b and a binary x_j: if the row cannot bind
    // when x_j takes its "loose" value, the coefficient (and b) can move by
    // the slack d without removing any integer point:
    //   a_j > 0, x_j = 0 loose: d = b - (maxact - a_j) > 0  ->  a_j -= d, b -= d
    //   a_j < 0, x_j = 1 loose: d = b - (maxact + a_j) > 0  ->  a_j += d
    // Integer points are unchanged, so postsolve needs no record; the LP
    // relaxation (and hence the MILP bound) gets tighter.
    if (mip_mode && !ranged.integer.empty() && A.rows > 0) {
        CSR R = to_csr(A);
        std::vector<char> binary(A.cols, 0);
        for (int j = 0; j < A.cols; ++j) binary[j] = is_int(col_ids[j]) && l[j] == 0.0 && u[j] == 1.0;
        std::vector<double> newval = R.data;
        for (int i = 0; i < A.rows; ++i) {
            bool le = std::isfinite(rU[i]) && !std::isfinite(rL[i]);
            bool ge = std::isfinite(rL[i]) && !std::isfinite(rU[i]);
            if (!le && !ge) continue;
            const double sg = le ? 1.0 : -1.0;              // work on sg * row <= sg * bound
            double b = le ? rU[i] : -rL[i];
            double maxact = 0.0;
            bool finite = true;
            for (int p = R.indptr[i]; p < R.indptr[i + 1]; ++p) {
                int j = R.indices[p]; double a = sg * R.data[p];
                double t = a > 0 ? a * u[j] : a * l[j];
                if (!std::isfinite(t)) { finite = false; break; }
                maxact += t;
            }
            if (!finite || maxact <= b + 1e-9) continue;      // redundant row: leave it
            for (int p = R.indptr[i]; p < R.indptr[i + 1]; ++p) {
                int j = R.indices[p];
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
            if (le) rU[i] = b; else rL[i] = -b;
        }
        // write back (COO rebuilt from the CSR)
        SparseMatrix T; T.rows = A.rows; T.cols = A.cols;
        for (int i = 0; i < A.rows; ++i)
            for (int p = R.indptr[i]; p < R.indptr[i + 1]; ++p)
                if (newval[p] != 0.0) { T.row_idx.push_back(i); T.col_idx.push_back(R.indices[p]); T.val.push_back(newval[p]); }
        A = T;
    }

    RangedLP reduced;
    reduced.name = ranged.name + " (presolved)";
    reduced.col_names.reserve(col_ids.size());
    for (int j : col_ids) reduced.col_names.push_back(ranged.col_names[j]);
    reduced.row_names.reserve(row_ids.size());
    for (int i : row_ids) reduced.row_names.push_back(ranged.row_names[i]);
    reduced.c = c; reduced.A = A; reduced.rL = rL; reduced.rU = rU; reduced.l = l; reduced.u = u;
    reduced.obj_offset = obj_offset; reduced.original_sense = ranged.original_sense;
    if (ranged.Q.nnz() > 0) {                // remap Q to the reduced columns (Q columns are never removed)
        std::vector<int> newc(n0, -1);
        for (size_t k = 0; k < col_ids.size(); ++k) newc[col_ids[k]] = (int)k;
        reduced.Q.rows = reduced.Q.cols = (int)col_ids.size();
        for (int k = 0; k < ranged.Q.nnz(); ++k) {
            reduced.Q.row_idx.push_back(newc[ranged.Q.row_idx[k]]);
            reduced.Q.col_idx.push_back(newc[ranged.Q.col_idx[k]]);
            reduced.Q.val.push_back(ranged.Q.val[k]);
        }
    }
    if (!ranged.integer.empty()) {
        reduced.integer.resize(col_ids.size());
        for (size_t k = 0; k < col_ids.size(); ++k) reduced.integer[k] = ranged.integer[col_ids[k]];
    }

    result.has_reduced = true;
    result.reduced = reduced;
    result.row_ids = row_ids;
    result.col_ids = col_ids;
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

    for (auto it = steps.rbegin(); it != steps.rend(); ++it) {
        const Step& p = *it;
        if (p.kind == "R1") {
            y[p.row_id] = 0.0;
        } else if (p.kind == "R2") {
            x[p.col_id] = p.value; z[p.col_id] = p.cost;
        } else if (p.kind == "R3") {
            int j = p.col_id;
            x[j] = p.value;
            double total = p.cost;
            for (auto& [rid, coeff] : p.column) total -= coeff * y[rid];
            z[j] = total;
        } else if (p.kind == "R4") {
            // The reduced problem saw one bound per side: max(lo_before, lo_row)
            // and min(hi_before, hi_row). The sign of z_j says which side is
            // active (z > 0: lower, z < 0: upper); if this row is what defined
            // that side, the multiplier belongs to the row, not the column.
            // (Deciding by |x_j - bound| < tol instead breaks for large x_j.)
            int j = p.col_id, r = p.row_id; double a_ij = p.a_ij;
            bool row_defines_lo = p.lo_from_row > p.lo_before;
            bool row_defines_hi = p.hi_from_row < p.hi_before;
            if ((z[j] > 0 && row_defines_lo) || (z[j] < 0 && row_defines_hi)) { y[r] = z[j] / a_ij; z[j] = 0.0; }
            else { y[r] = 0.0; }
        }
    }
}
