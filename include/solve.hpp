// solve.hpp -- wires the whole pipeline together (port of solve.py):
//   MPS -> Presolve -> PDLP-form + Scaling -> PDHG -> Unscale -> Postsolve
//        -> KKT verification
#pragma once
#include <string>
#include <map>
#include "ranged_lp.hpp"

struct Solution {
    std::string status;   // "optimal" | "infeasible" | "unbounded" | "iteration_limit"
    double objective = 0;
    bool has_solution = false;
    std::map<std::string,double> x, y;
    int iterations = 0, restarts = 0;
    double eps_P = 0, eps_D = 0, eps_G = 0;
    // "cpu", "gpu", or "n/a" (box-shortcut path never runs an engine at all).
    // Only ever "gpu" when this binary was built with SOVEREIGN_WITH_CUDA
    // AND the GPU engine actually won the race on the machine that ran it.
    std::string engine_used = "n/a";
};

Solution solve_mps(const std::string& path, double tol = 1e-6, int max_iterations = 200000, double eta = 0.99);
