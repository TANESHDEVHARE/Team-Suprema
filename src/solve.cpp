#include "solve.hpp"
#include "mps_reader.hpp"
#include "presolve.hpp"
#include "scaling.hpp"
#include "pdlp.hpp"
#include "pdlp_gpu.hpp"
#include "verify.hpp"
#include <cmath>
#include <limits>
#include <atomic>
#include <iostream>
#ifdef SOVEREIGN_WITH_CUDA
#include <thread>
#include <exception>
#endif

// run_pdlp_stage(): the ONE place solve_mps() decides how PdlpResult gets
// produced. Built without CUDA, this is a direct call to the CPU engine --
// byte-for-byte what solve_mps() always did. Built WITH CUDA (CMake found
// nvcc on this machine), it races solve_pdhg() (CPU) against
// solve_pdhg_gpu() (GPU) on two std::threads sharing one atomic stop flag:
// whichever engine reaches a converged, KKT-verified answer first flips the
// flag, the other engine notices it on its very next iteration and returns
// early, and that early, discarded result is simply never used. This is the
// literal implementation of the "first verified answer wins" box in the
// project's architecture diagram -- absent here on a non-CUDA build, present
// and real on a CUDA-capable one, with no other code path change needed.
static PdlpResult run_pdlp_stage(const RangedLP& pdlp_form, const RangedLP& scaled,
                                  const std::vector<double>& Dr, const std::vector<double>& Dc,
                                  double eta, int max_iterations, double tol, std::string& engine_used) {
#ifndef SOVEREIGN_WITH_CUDA
    engine_used = "cpu";
    return solve_pdhg(pdlp_form, scaled, Dr, Dc, eta, max_iterations, 64, tol);
#else
    std::atomic<bool> stop_flag{false};
    std::atomic<int> winner{-1};   // -1 = none yet, 0 = cpu, 1 = gpu
    PdlpResult cpu_result, gpu_result;
    bool cpu_ok = false, gpu_ok = false;
    std::string cpu_error, gpu_error;

    // IMPORTANT: an exception that escapes a std::thread's function body
    // does NOT propagate to whoever join()s it -- the C++ runtime calls
    // std::terminate() immediately, which is a silent, message-less abort()
    // (exactly the "Debug Error! abort() has been called" dialog with no
    // diagnostic text). Every engine call below is therefore wrapped in its
    // own try/catch: a failure is recorded and printed, not left to crash
    // the whole process, and if the OTHER engine still succeeded, its
    // result is used and the run completes normally instead of dying.
    std::thread cpu_thread([&]() {
        try {
            PdlpResult r = solve_pdhg(pdlp_form, scaled, Dr, Dc, eta, max_iterations, 64, tol, &stop_flag);
            cpu_result = r; cpu_ok = true;
            if (r.converged) {
                int expected = -1;
                if (winner.compare_exchange_strong(expected, 0)) stop_flag.store(true, std::memory_order_relaxed);
            }
        } catch (const std::exception& e) {
            cpu_error = e.what();
            std::cerr << "[sovereign_cpp] CPU engine failed: " << cpu_error << "\n";
        } catch (...) {
            cpu_error = "unknown exception";
            std::cerr << "[sovereign_cpp] CPU engine failed with a non-std::exception.\n";
        }
    });
    std::thread gpu_thread([&]() {
        try {
            PdlpResult r = solve_pdhg_gpu(pdlp_form, scaled, Dr, Dc, eta, max_iterations, 64, tol, &stop_flag);
            gpu_result = r; gpu_ok = true;
            if (r.converged) {
                int expected = -1;
                if (winner.compare_exchange_strong(expected, 1)) stop_flag.store(true, std::memory_order_relaxed);
            }
        } catch (const std::exception& e) {
            gpu_error = e.what();
            std::cerr << "[sovereign_cpp] GPU engine failed: " << gpu_error << "\n";
        } catch (...) {
            gpu_error = "unknown exception";
            std::cerr << "[sovereign_cpp] GPU engine failed with a non-std::exception.\n";
        }
    });
    cpu_thread.join();
    gpu_thread.join();

    int w = winner.load();
    if (w == 0) { engine_used = "cpu"; return cpu_result; }
    if (w == 1) { engine_used = "gpu"; return gpu_result; }

    // Neither engine reported a converged, verified answer. If exactly one
    // of them ran without throwing, use its (possibly unconverged) result
    // rather than crash -- this is what lets the CLI keep working even
    // while a GPU bug is still being diagnosed.
    if (cpu_ok && !gpu_ok) { engine_used = "cpu (gpu failed: " + gpu_error + ")"; return cpu_result; }
    if (gpu_ok && !cpu_ok) { engine_used = "gpu (cpu failed: " + cpu_error + ")"; return gpu_result; }
    if (!cpu_ok && !gpu_ok) {
        throw std::runtime_error("both engines failed -- cpu: " + cpu_error + " | gpu: " + gpu_error);
    }
    // Both ran and both returned (neither converged, neither threw) --
    // report whichever has the smaller total KKT residual, same tie-break
    // each engine already uses internally at its own tail.
    double cpu_score = cpu_result.eps_P + cpu_result.eps_D + cpu_result.eps_G;
    double gpu_score = gpu_result.eps_P + gpu_result.eps_D + gpu_result.eps_G;
    if (cpu_score <= gpu_score) { engine_used = "cpu"; return cpu_result; }
    engine_used = "gpu"; return gpu_result;
#endif
}

Solution solve_mps(const std::string& path, double tol, int max_iterations, double eta) {
    LPProblem problem = read_mps(path);
    RangedLP ranged = to_ranged_lp(problem);
    int n0 = ranged.n(), m0 = ranged.m();

    PresolveResult pres = presolve(ranged);
    if (pres.status == "infeasible") { Solution s; s.status = "infeasible"; s.eps_P = s.eps_D = s.eps_G = INF; return s; }
    if (pres.status == "unbounded") { Solution s; s.status = "unbounded"; s.eps_P = s.eps_D = s.eps_G = INF; return s; }

    RangedLP& reduced = pres.reduced;

    if (reduced.n() == 0 || reduced.m() == 0) {
        // pure box problem left after presolve -- push each variable to its
        // improving finite bound directly (see solve.py's own comment).
        int n_r = reduced.n(), m_r = reduced.m();
        std::vector<double> x_r(n_r, 0.0);
        for (int j = 0; j < n_r; ++j) {
            double cj = reduced.c[j];
            if (cj > 0) {
                if (!std::isfinite(reduced.l[j])) { Solution s; s.status = "unbounded"; s.eps_P=s.eps_D=s.eps_G=INF; return s; }
                x_r[j] = reduced.l[j];
            } else if (cj < 0) {
                if (!std::isfinite(reduced.u[j])) { Solution s; s.status = "unbounded"; s.eps_P=s.eps_D=s.eps_G=INF; return s; }
                x_r[j] = reduced.u[j];
            } else {
                x_r[j] = std::isfinite(reduced.l[j]) ? reduced.l[j] : (std::isfinite(reduced.u[j]) ? reduced.u[j] : 0.0);
            }
        }
        std::vector<double> y_r(m_r, 0.0);
        std::vector<double> z_r(n_r);
        for (int j = 0; j < n_r; ++j) z_r[j] = reduced.c[j]; // A is empty here, so A^T y = 0

        std::vector<double> x_orig, y_orig, z_orig;
        postsolve(n0, m0, pres.row_ids, pres.col_ids, x_r, y_r, z_r, pres.steps, x_orig, y_orig, z_orig);
        KKTReport final = verify(ranged, x_orig, y_orig);

        Solution s; s.status = "optimal"; s.has_solution = true;
        s.objective = report_objective(ranged, final.objective);
        for (size_t j = 0; j < problem.col_names.size(); ++j) s.x[problem.col_names[j]] = x_orig[j];
        for (size_t i = 0; i < problem.row_names.size(); ++i) s.y[problem.row_names[i]] = y_orig[i];
        s.iterations = 0; s.restarts = 0; s.eps_P = final.eps_P; s.eps_D = final.eps_D; s.eps_G = final.eps_G;
        return s;
    }

    auto [pdlp_form, meta] = build_pdlp_form(reduced);
    ScalingResult sr = scale(pdlp_form);
    std::string engine_used;
    PdlpResult result = run_pdlp_stage(pdlp_form, sr.scaled, sr.Dr, sr.Dc, eta, max_iterations, tol, engine_used);

    std::vector<double> x_reduced, y_reduced;
    extract_solution(meta, result.x, result.y, x_reduced, y_reduced);
    CSR ATcsr = to_csc_as_transposed_csr(reduced.A);
    auto ATy = matvec_T(ATcsr, y_reduced);
    std::vector<double> z_reduced(reduced.n());
    for (int j = 0; j < reduced.n(); ++j) z_reduced[j] = reduced.c[j] - ATy[j];

    std::vector<double> x_orig, y_orig, z_orig;
    postsolve(n0, m0, pres.row_ids, pres.col_ids, x_reduced, y_reduced, z_reduced, pres.steps, x_orig, y_orig, z_orig);

    KKTReport final = verify(ranged, x_orig, y_orig);
    Solution s;
    s.status = result.converged ? "optimal" : "iteration_limit";
    s.has_solution = true;
    s.objective = report_objective(ranged, final.objective);
    for (size_t j = 0; j < problem.col_names.size(); ++j) s.x[problem.col_names[j]] = x_orig[j];
    for (size_t i = 0; i < problem.row_names.size(); ++i) s.y[problem.row_names[i]] = y_orig[i];
    s.iterations = result.iterations; s.restarts = result.restarts;
    s.eps_P = final.eps_P; s.eps_D = final.eps_D; s.eps_G = final.eps_G;
    s.engine_used = engine_used;
    return s;
}
