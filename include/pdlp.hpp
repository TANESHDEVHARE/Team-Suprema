// pdlp.hpp -- the GPU-designed engine (port of pdlp.py). Every inner-loop
// op is a sparse matrix-vector product or an elementwise clamp -- no
// factorization, no row-to-row sequential dependency. That is exactly why
// this is the module gpu/pdlp_spmv_kernel.cu targets: the loop below is
// written as plain array operations on purpose, so the CUDA port is a
// mechanical translation of THIS code, not a rewrite.
#pragma once
#include "ranged_lp.hpp"
#include <atomic>

struct PdlpFormMeta {
    int n_orig = 0, m_orig = 0, n_slack = 0;
    std::vector<int> row_perm;
    std::vector<double> row_signs;
    std::vector<int> free_rows;
};

struct PdlpResult {
    std::vector<double> x, y;
    int iterations = 0;
    bool converged = false;
    double eps_P = 0, eps_D = 0, eps_G = 0;
    int restarts = 0;
};

std::pair<RangedLP, PdlpFormMeta> build_pdlp_form(const RangedLP& ranged);
void extract_solution(const PdlpFormMeta& meta, const std::vector<double>& x2, const std::vector<double>& y2,
                       std::vector<double>& x, std::vector<double>& y);

// stop_flag: checked once per outer iteration (relaxed load, cheap). When set
// by another thread, this engine breaks out immediately and returns its best
// candidate so far with converged=false -- this is what lets solve_mps_race()
// (src/solve.cpp) stop the loser the instant the winner is verified, instead
// of racing to completion and discarding the slower result.
PdlpResult solve_pdhg(const RangedLP& unscaled_form, const RangedLP& scaled_form,
                       const std::vector<double>& Dr, const std::vector<double>& Dc,
                       double eta = 0.99, int max_iterations = 200000, int check_every = 64, double tol = 1e-6,
                       std::atomic<bool>* stop_flag = nullptr);
