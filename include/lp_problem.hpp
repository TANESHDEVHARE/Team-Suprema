// lp_problem.hpp -- port of the Python POC's lp_problem.py
#pragma once
#include <string>
#include <vector>
#include <map>
#include "sparse.hpp"

struct LPProblem {
    std::string name;
    std::string sense;                         // "min" or "max"
    std::vector<std::string> col_names;
    std::vector<std::string> row_names;
    std::vector<char> row_types;                // 'L','G','E' per row
    std::vector<double> row_rhs;
    std::map<std::string, double> ranges;       // row_name -> range value
    std::string obj_name;
    std::vector<double> obj;
    SparseMatrix A;
    std::vector<double> lo, hi;
    double obj_constant = 0.0;                  // from an RHS entry on the objective row (= -rhs)
    std::vector<char> integer;                  // 1 if column j is integer (INTORG marker or BV bound)
    SparseMatrix Q;                             // QP: lower triangle of Q (QUADOBJ / QMATRIX), objective 1/2 x'Qx

    int m() const { return static_cast<int>(row_names.size()); }
    int n() const { return static_cast<int>(col_names.size()); }
};
