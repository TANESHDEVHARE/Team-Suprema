// sparse.hpp
// ===========
// A minimal sparse matrix (COO-backed, CSR/CSC built on demand) used by every
// module below. This mirrors what scipy.sparse gave the Python POC, kept
// deliberately simple (no external dependency) so this project builds with
// nothing but a C++17 compiler -- exactly the "port, don't rewrite the math"
// discipline: same operations, same semantics, just no SciPy underneath.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

struct SparseMatrix {
    int rows = 0, cols = 0;
    std::vector<int> row_idx;      // COO row indices
    std::vector<int> col_idx;      // COO col indices
    std::vector<double> val;       // COO values

    int nnz() const { return static_cast<int>(val.size()); }
};

// CSR view: row-major, indptr[rows+1], indices/data in row order.
struct CSR {
    int rows = 0, cols = 0;
    std::vector<int> indptr;
    std::vector<int> indices;
    std::vector<double> data;
};

inline CSR to_csr(const SparseMatrix& A) {
    CSR c;
    c.rows = A.rows; c.cols = A.cols;
    c.indptr.assign(A.rows + 1, 0);
    for (int r : A.row_idx) c.indptr[r + 1]++;
    for (int i = 0; i < A.rows; ++i) c.indptr[i + 1] += c.indptr[i];
    c.indices.assign(A.nnz(), 0);
    c.data.assign(A.nnz(), 0.0);
    std::vector<int> cursor(c.indptr.begin(), c.indptr.end() - 1);
    for (int k = 0; k < A.nnz(); ++k) {
        int r = A.row_idx[k];
        int pos = cursor[r]++;
        c.indices[pos] = A.col_idx[k];
        c.data[pos] = A.val[k];
    }
    return c;
}

// CSC is just the CSR of the transpose.
inline CSR to_csc_as_transposed_csr(const SparseMatrix& A) {
    SparseMatrix T;
    T.rows = A.cols; T.cols = A.rows;
    T.row_idx = A.col_idx;
    T.col_idx = A.row_idx;
    T.val = A.val;
    return to_csr(T);
}

// y = A * x
inline std::vector<double> matvec(const CSR& A, const std::vector<double>& x) {
    std::vector<double> y(A.rows, 0.0);
    for (int i = 0; i < A.rows; ++i)
        for (int p = A.indptr[i]; p < A.indptr[i + 1]; ++p)
            y[i] += A.data[p] * x[A.indices[p]];
    return y;
}

// x = A^T * y, given AT = CSR of A's transpose (precomputed once, PDLP-style).
inline std::vector<double> matvec_T(const CSR& AT, const std::vector<double>& y) {
    return matvec(AT, y); // AT is just A^T stored as an ordinary CSR
}

inline std::vector<int> row_nnz_counts(const CSR& A) {
    std::vector<int> out(A.rows);
    for (int i = 0; i < A.rows; ++i) out[i] = A.indptr[i + 1] - A.indptr[i];
    return out;
}

inline std::vector<int> col_nnz_counts(const SparseMatrix& A) {
    std::vector<int> out(A.cols, 0);
    for (int c : A.col_idx) out[c]++;
    return out;
}

// Drop rows not in keep[]; remap indices; caller tracks row_ids separately.
inline SparseMatrix filter_rows(const SparseMatrix& A, const std::vector<char>& keep) {
    std::vector<int> remap(A.rows, -1);
    int nr = 0;
    for (int i = 0; i < A.rows; ++i) if (keep[i]) remap[i] = nr++;
    SparseMatrix out; out.rows = nr; out.cols = A.cols;
    for (int k = 0; k < A.nnz(); ++k) {
        int nr2 = remap[A.row_idx[k]];
        if (nr2 >= 0) { out.row_idx.push_back(nr2); out.col_idx.push_back(A.col_idx[k]); out.val.push_back(A.val[k]); }
    }
    return out;
}

inline SparseMatrix filter_cols(const SparseMatrix& A, const std::vector<char>& keep) {
    std::vector<int> remap(A.cols, -1);
    int nc = 0;
    for (int j = 0; j < A.cols; ++j) if (keep[j]) remap[j] = nc++;
    SparseMatrix out; out.rows = A.rows; out.cols = nc;
    for (int k = 0; k < A.nnz(); ++k) {
        int nc2 = remap[A.col_idx[k]];
        if (nc2 >= 0) { out.row_idx.push_back(A.row_idx[k]); out.col_idx.push_back(nc2); out.val.push_back(A.val[k]); }
    }
    return out;
}

// All (row, value) entries of column j -- O(nnz) scan, fine at POC scale
// (mirrors the Python POC's own scope: correctness first, not peak speed).
inline std::vector<std::pair<int,double>> get_col_entries(const SparseMatrix& A, int j) {
    std::vector<std::pair<int,double>> out;
    for (int k = 0; k < A.nnz(); ++k) if (A.col_idx[k] == j) out.emplace_back(A.row_idx[k], A.val[k]);
    return out;
}

inline double row_max_abs(const CSR& A, int i) {
    double m = 0.0;
    for (int p = A.indptr[i]; p < A.indptr[i + 1]; ++p) m = std::max(m, std::fabs(A.data[p]));
    return m;
}

inline SparseMatrix scale_rows_cols(const SparseMatrix& A, const std::vector<double>& row_scale,
                                     const std::vector<double>& col_scale) {
    SparseMatrix out = A;
    for (int k = 0; k < A.nnz(); ++k) out.val[k] = A.val[k] * row_scale[A.row_idx[k]] * col_scale[A.col_idx[k]];
    return out;
}

constexpr double INF = std::numeric_limits<double>::infinity();
