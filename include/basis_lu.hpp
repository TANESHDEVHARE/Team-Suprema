// basis_lu.hpp -- sparse LU factorization of a simplex basis matrix B
// (LP pipeline, Step 5.1 and 5.3).
//
//   Factorization:  Markowitz pivot selection with a threshold test
//                   |b_rc| >= u * max_i |b_ic|  (column threshold, u = 0.1),
//                   so every L multiplier is bounded by 1/u.
//   Singular bases: columns that cannot be pivoted are reported back and
//                   replaced by the logical (slack) column of an unpivoted
//                   row -- the caller repairs its basis accordingly.
//   Updates:        Forrest-Tomlin. After a basis change the replaced U
//                   column is swapped for the "spike" L^{-1}a_q, its U-index
//                   moves to the end of the pivot order, and the row it
//                   leaves behind is eliminated with one row eta.
//
// Representation:  B^{-1} = U^{-1} R_k ... R_1 L^{-1}
//   L^{-1}: column etas from the elimination, in original row space.
//   R_i   : Forrest-Tomlin row etas.
//   U     : indexed by "U-index" t. Row t lives on original row row_of[t],
//           column t belongs to basis slot slot_of[t]. `ord` is the current
//           triangular order (U(s,t) != 0 only if s precedes t in ord).
//
// All solves use dense work vectors of length m. That is O(m) per solve,
// which is fine at Netlib / mid-size scale; hypersparse solves are a later
// optimisation, not a correctness issue.
#pragma once
#include <vector>
#include <functional>
#include <utility>

class BasisLU {
public:
    // get_col(slot, rows, vals) must fill the sparse column of basis slot.
    using ColumnFn = std::function<void(int slot, std::vector<int>& rows, std::vector<double>& vals)>;

    // Factorize the m x m basis. Returns the list of (slot, row) pairs whose
    // column was singular and has been replaced by the unit column e_row.
    std::vector<std::pair<int,int>> factorize(int m, const ColumnFn& get_col);

    // B x = a.  `rhs` is in row space (length m) and is overwritten; the
    // result is returned in slot space in `x`. If `spike` is non-null it
    // receives R L^{-1} a (row space), which update() needs.
    void ftran(std::vector<double>& rhs, std::vector<double>& x, std::vector<double>* spike = nullptr) const;

    // B^T y = e.  `e` is in slot space (length m) and is overwritten; the
    // result y is returned in row space.
    void btran(std::vector<double>& e, std::vector<double>& y) const;

    // Replace the column in basis slot `slot` using the spike returned by
    // ftran() for the entering column. `alpha_pivot` (the pivot element of
    // the FTRANed entering column, optional) enables the determinant check
    // |new U diagonal| == |alpha_pivot * old diagonal|. Returns false if the
    // update is numerically unacceptable (caller must refactorize).
    bool update(int slot, const std::vector<double>& spike, double alpha_pivot = 0.0);

    int num_updates() const { return static_cast<int>(r_etas_.size()); }
    int m() const { return m_; }
    long long fill() const;   // nnz(L) + nnz(U), for diagnostics

    double pivot_threshold = 0.1;     // Markowitz threshold u
    double singular_tol = 1e-11;      // |pivot| below this => singular

private:
    struct Eta { int pivot_row; std::vector<std::pair<int,double>> entries; };

    int m_ = 0;
    std::vector<Eta> l_etas_;                     // L^{-1} column etas, in application order
    std::vector<std::vector<std::pair<int,double>>> lrow_;  // same etas, row-wise (BTRAN)
    std::vector<int> lrow_order_;                 // pivot rows in elimination order
    std::vector<Eta> r_etas_;                     // Forrest-Tomlin row etas, in application order
    std::vector<std::vector<std::pair<int,double>>> ucol_;  // ucol_[t]: (U-index s, value), s before t
    std::vector<std::vector<std::pair<int,double>>> urow_;  // urow_[s]: (U-index t, value), t after s
    std::vector<double> diag_;
    std::vector<int> row_of_, slot_of_, t_of_row_, t_of_slot_;
    std::vector<int> ord_, pos_;                  // pivot order and its inverse
    mutable std::vector<double> work_;
};
