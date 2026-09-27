// sparse_ldl.hpp -- sparse symmetric LDL^T factorization for the interior-
// point method (QP pipeline Steps 5-6), from scratch:
//
//   ordering:  minimum degree on the quotient graph (elements absorb the
//              cliques of eliminated nodes, as in AMD; degrees are the usual
//              approximate external degrees). No METIS (Math/PIPELINE_NOTES.md).
//   symbolic:  elimination tree + column counts, computed once per pattern.
//   numeric:   supernodal left-looking LDL^T: columns with nested patterns
//              form dense blocks; each block is assembled, updated by its
//              descendants with dense kernels, then factored densely.
//   stability: the IPM's augmented matrix is quasidefinite: the first block
//              of pivots must be negative, the second positive. A pivot with
//              the wrong sign or magnitude below `reg` is replaced by
//              sign * max(|d|, reg) -- dynamic regularization that enforces
//              the target inertia (QP Step 6.1-6.2) -- and counted.
#pragma once
#include <vector>

struct SymMatrix {                 // lower triangle (i >= j) in CSC, n x n
    int n = 0;
    std::vector<int> colptr, rowidx;
    std::vector<double> val;
};

class SparseLDL {
public:
    // Pattern of K (lower triangle, CSC). Computes the ordering and the
    // symbolic factorization; call once per pattern.
    void analyze(const SymMatrix& K);
    // Numeric factorization. sign[i] = -1 or +1 is the expected sign of
    // pivot i (original numbering). Returns the number of regularized pivots.
    int factor(const SymMatrix& K, const std::vector<signed char>& sign, double reg);
    // Solve K x = b in place (original numbering).
    void solve(std::vector<double>& b) const;

    long long nnz_L() const { return nnzL_; }
    // Original indices of the pivots regularized by the last factor() call.
    const std::vector<int>& regularized() const { return regularized_; }
    int n() const { return n_; }

private:
    int n_ = 0;
    long long nnzL_ = 0;
    std::vector<int> perm_, iperm_;          // perm_[k] = original index of the k-th pivot
    std::vector<int> parent_;                // elimination tree (permuted)
    std::vector<double> D_;
    std::vector<int> regularized_;
    // K permuted: upper by columns (etree) and lower by columns (assembly);
    // *map_ holds each entry's position in K.val.
    std::vector<int> Cp_, Ci_, map_, Lcp_, Lci_, Lmap_;
    // supernodes: columns [sn_start_[s], sn_start_[s+1]), rows below the
    // diagonal block sn_rows_[sn_rowptr_[s] .. sn_rowptr_[s+1]), values as a
    // dense column-major (w + r) x w block at sn_val_[sn_valptr_[s]].
    std::vector<int> sn_start_, sn_of_, sn_rowptr_, sn_rows_;
    std::vector<long long> sn_valptr_;
    std::vector<double> sn_val_;
};

// Minimum-degree ordering of the symmetric pattern (lower triangle CSC).
std::vector<int> min_degree_order(const SymMatrix& K);
