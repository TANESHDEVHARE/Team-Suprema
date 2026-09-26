#include "presolve.hpp"
#include <cmath>
#include <algorithm>

static constexpr double ZERO_TOL = 1e-12;
static constexpr double FEAS_TOL = 1e-9;

PresolveResult presolve(const RangedLP& ranged, int max_iterations) {
    int n0 = ranged.n(), m0 = ranged.m();

    SparseMatrix A = ranged.A;
    std::vector<double> c = ranged.c;
    std::vector<double> rL = ranged.rL, rU = ranged.rU;
    std::vector<double> l = ranged.l, u = ranged.u;
    double obj_offset = ranged.obj_offset;

    std::vector<int> row_ids(m0), col_ids(n0);
    for (int i = 0; i < m0; ++i) row_ids[i] = i;
    for (int j = 0; j < n0; ++j) col_ids[j] = j;

    PresolveResult result;

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
                if (!(rL[r] <= FEAS_TOL && rU[r] >= -FEAS_TOL)) {
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
        for (int j = 0; j < n; ++j) if (colnnz[j] == 0) empty_cols.push_back(j);
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
        for (int j = 0; j < n; ++j) if (std::isfinite(l[j]) && (u[j] - l[j]) < ZERO_TOL) fixed.push_back(j);
        if (!fixed.empty()) {
            std::vector<char> keep(n, 1);
            for (int j : fixed) {
                double v = l[j];
                auto col_entries = get_col_entries(A, j);
                Step s; s.kind = "R3"; s.col_id = col_ids[j]; s.value = v; s.cost = c[j];
                obj_offset += c[j] * v;
                for (auto& [ri, va] : col_entries) {
                    rL[ri] -= va * v;
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
                if (new_lo - new_hi > 1e-7) {
                    result.status = "infeasible"; result.row_ids = row_ids; result.col_ids = col_ids; return result;
                }
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

    RangedLP reduced;
    reduced.name = ranged.name + " (presolved)";
    reduced.col_names.reserve(col_ids.size());
    for (int j : col_ids) reduced.col_names.push_back(ranged.col_names[j]);
    reduced.row_names.reserve(row_ids.size());
    for (int i : row_ids) reduced.row_names.push_back(ranged.row_names[i]);
    reduced.c = c; reduced.A = A; reduced.rL = rL; reduced.rU = rU; reduced.l = l; reduced.u = u;
    reduced.obj_offset = obj_offset; reduced.original_sense = ranged.original_sense;

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
            int j = p.col_id, r = p.row_id; double a_ij = p.a_ij;
            double xj = x[j];
            bool at_lo = p.lo_from_row > p.lo_before + 1e-9 && std::fabs(xj - p.lo_from_row) < 1e-6;
            bool at_hi = p.hi_from_row < p.hi_before - 1e-9 && std::fabs(xj - p.hi_from_row) < 1e-6;
            if (at_lo || at_hi) { y[r] = z[j] / a_ij; z[j] = 0.0; }
            else { y[r] = 0.0; }
        }
    }
}
