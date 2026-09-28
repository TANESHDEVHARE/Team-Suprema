#include "solve.hpp"
#include "mps_reader.hpp"
#include "presolve.hpp"
#include "scaling.hpp"
#include "pdlp.hpp"
#include "pdlp_gpu.hpp"
#include "verify.hpp"
#include "simplex.hpp"
#include "mip.hpp"
#include "qp_ipm.hpp"
#include <cmath>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <limits>
#include <atomic>
#include <iostream>
#include <thread>
#include <exception>
#include <mutex>
#include <condition_variable>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <stdexcept>

// Helper engines of the concurrent portfolio run below normal priority so the
// dual simplex (usually the winner) keeps its core. Declared directly rather
// than through <windows.h>, whose macros clash with ordinary identifiers.
#ifdef _WIN32
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread();
extern "C" __declspec(dllimport) int __stdcall SetThreadPriority(void* thread, int priority);
static void lower_thread_priority() { SetThreadPriority(GetCurrentThread(), -1); }   // THREAD_PRIORITY_BELOW_NORMAL
#else
static void lower_thread_priority() {}
#endif

// MPS parse time of the last solve_mps* call on this thread, so the CLI can
// report solve time on the same footing as HiGHS's run() (which excludes it).
static thread_local double g_last_read_seconds = 0;
double last_read_seconds() { return g_last_read_seconds; }

// The reported status follows the independent verifier's worst relative
// residual on the original problem: "optimal" up to the engine's tolerance,
// "near_optimal" up to 1e-4, and "inaccurate" beyond that -- an answer the
// verifier does not certify is never labelled as (nearly) optimal.
static std::string certified_status(const std::string& engine_status, double eps, double opt_tol) {
    if (engine_status != "optimal" && engine_status != "near_optimal") return engine_status;
    if (eps <= opt_tol) return engine_status;
    return eps <= 1e-4 ? "near_optimal" : "inaccurate";
}

static LPProblem timed_read_mps(const std::string& path) {
    auto t0 = std::chrono::steady_clock::now();
    LPProblem p = read_mps(path);
    g_last_read_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return p;
}

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
    LPProblem problem = timed_read_mps(path);
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
    // PDLP stops on its own KKT test on the reduced PDLP form; the verifier
    // below judges the original problem, whose normalization differs. If the
    // original-space residuals miss `tol`, rerun with a 10x tighter internal
    // tolerance (at most twice) instead of reporting an unverified optimum.
    PdlpResult result;
    std::vector<double> x_orig, y_orig, z_orig;
    KKTReport final;
    int total_iterations = 0;
    double inner_tol = tol;
    for (int attempt = 0; attempt < 3; ++attempt, inner_tol *= 0.1) {
        result = run_pdlp_stage(pdlp_form, sr.scaled, sr.Dr, sr.Dc, eta, max_iterations, inner_tol, engine_used);
        total_iterations += result.iterations;
        std::vector<double> x_reduced, y_reduced;
        extract_solution(meta, result.x, result.y, x_reduced, y_reduced);
        CSR ATcsr = to_csc_as_transposed_csr(reduced.A);
        auto ATy = matvec_T(ATcsr, y_reduced);
        std::vector<double> z_reduced(reduced.n());
        for (int j = 0; j < reduced.n(); ++j) z_reduced[j] = reduced.c[j] - ATy[j];
        postsolve(n0, m0, pres.row_ids, pres.col_ids, x_reduced, y_reduced, z_reduced, pres.steps, x_orig, y_orig, z_orig);
        final = verify(ranged, x_orig, y_orig);
        if (!result.converged || std::max({final.eps_P, final.eps_D, final.eps_G}) <= tol) break;
    }
    result.iterations = total_iterations;

    Solution s;
    s.status = result.converged ? "optimal" : "iteration_limit";
    s.status = certified_status(s.status, std::max({final.eps_P, final.eps_D, final.eps_G}), 10.0 * tol);
    s.has_solution = true;
    s.objective = report_objective(ranged, final.objective);
    for (size_t j = 0; j < problem.col_names.size(); ++j) s.x[problem.col_names[j]] = x_orig[j];
    for (size_t i = 0; i < problem.row_names.size(); ++i) s.y[problem.row_names[i]] = y_orig[i];
    s.iterations = result.iterations; s.restarts = result.restarts;
    s.eps_P = final.eps_P; s.eps_D = final.eps_D; s.eps_G = final.eps_G;
    s.engine_used = engine_used;
    return s;
}

Solution solve_mps_simplex(const std::string& path, const SimplexOptions& opt, bool use_presolve,
                           bool crossover, double pdlp_tol, int pdlp_max_iterations, bool race) {
    LPProblem problem = timed_read_mps(path);
    RangedLP ranged = to_ranged_lp(problem);
    int n0 = ranged.n(), m0 = ranged.m();

    Solution s;
    s.engine_used = "simplex";
    s.eps_P = s.eps_D = s.eps_G = INF;

    const auto tpre = std::chrono::steady_clock::now();
    PresolveResult pres = presolve(ranged, use_presolve ? 500 : 0);
    if (opt.verbose)
        std::printf("  presolve: %d x %d -> %d x %d in %d passes, %zu steps, %.4fs\n", m0, n0,
                    (int)pres.row_ids.size(), (int)pres.col_ids.size(), pres.passes, pres.steps.size(),
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - tpre).count());
    if (pres.status == "infeasible" || pres.status == "unbounded") { s.status = pres.status; return s; }
    RangedLP& reduced = pres.reduced;

    std::vector<double> x_r(reduced.n(), 0.0), y_r(reduced.m(), 0.0);
    if (reduced.n() > 0) {
        // Scale for the simplex; integer columns will need to be excluded
        // from column scaling once MILP arrives (see the MILP pipeline notes).
        ScalingResult sr;
        if (opt.scaling) sr = scale(reduced);
        else { sr.scaled = reduced; sr.Dr.assign(reduced.m(), 1.0); sr.Dc.assign(reduced.n(), 1.0); }
        DualSimplex spx, spx_b;
        SimplexOptions sopt = opt;
        std::string st;
        bool used_b = false;
        if (race) {
            // "First verified answer wins" (architecture diagram): a cold dual
            // simplex on the CPU races PDLP (on the GPU when built with CUDA)
            // followed by crossover and a warm simplex. The first definitive
            // answer (optimal / infeasible / unbounded) stops the other.
            std::atomic<bool> stop{false};
            std::atomic<int> winner{-1};
            std::string st_a, st_b, eng_b = "cpu";
            auto definitive = [](const std::string& x) {
                return x == "optimal" || x == "infeasible" || x == "unbounded" || x == "dual_infeasible";
            };
            auto claim = [&](int who) { int e = -1; if (winner.compare_exchange_strong(e, who)) stop.store(true); };
            auto tp = std::chrono::steady_clock::now();
            std::thread ta([&]() {
                spx.load(sr.scaled);
                SimplexOptions o = opt; o.stop_flag = &stop;
                st_a = spx.solve(o);
                if (definitive(st_a)) claim(0);
            });
            std::thread tb([&]() {
                try {
                    auto [pf, meta] = build_pdlp_form(reduced);
                    ScalingResult psr = scale(pf);
                    PdlpResult pr;
#ifdef SOVEREIGN_WITH_CUDA
                    eng_b = "gpu";
                    pr = solve_pdhg_gpu(pf, psr.scaled, psr.Dr, psr.Dc, 0.99, pdlp_max_iterations, 64, pdlp_tol, &stop);
#else
                    pr = solve_pdhg(pf, psr.scaled, psr.Dr, psr.Dc, 0.99, pdlp_max_iterations, 64, pdlp_tol, &stop);
#endif
                    s.pdlp_iterations = pr.iterations;
                    s.pdlp_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
                    if (stop.load()) { st_b = "stopped"; return; }
                    std::vector<double> xp, yp;
                    extract_solution(meta, pr.x, pr.y, xp, yp);
                    for (int j = 0; j < reduced.n(); ++j) xp[j] /= sr.Dc[j];
                    for (int i = 0; i < reduced.m(); ++i) yp[i] /= sr.Dr[i];
                    spx_b.load(sr.scaled);
                    spx_b.crossover_start(xp, yp);
                    SimplexOptions o = opt; o.stop_flag = &stop; o.shift_instead_of_phase1 = true;
                    st_b = spx_b.solve(o);
                    if (definitive(st_b)) claim(1);
                } catch (const std::exception& e) {
                    st_b = std::string("error: ") + e.what();
                }
            });
            ta.join(); tb.join();
            used_b = winner.load() == 1;
            st = used_b ? st_b : st_a;
            s.engine_used = used_b ? "pdlp(" + eng_b + ")+crossover+simplex (won the race)"
                                   : "dual simplex (won the race vs pdlp(" + eng_b + ")+crossover)";
        } else {
            spx.load(sr.scaled);
            if (crossover) {
                // LP Step 3: PDLP warm start on its own form/scaling, mapped back
                // to the reduced space, then into the simplex's scaled space.
                auto tp = std::chrono::steady_clock::now();
                auto [pdlp_form, meta] = build_pdlp_form(reduced);
                ScalingResult psr = scale(pdlp_form);
                std::string eng;
                PdlpResult pr = run_pdlp_stage(pdlp_form, psr.scaled, psr.Dr, psr.Dc, 0.99, pdlp_max_iterations, pdlp_tol, eng);
                std::vector<double> xp, yp;
                extract_solution(meta, pr.x, pr.y, xp, yp);
                for (int j = 0; j < reduced.n(); ++j) xp[j] /= sr.Dc[j];
                for (int i = 0; i < reduced.m(); ++i) yp[i] /= sr.Dr[i];
                s.pdlp_iterations = pr.iterations;
                s.pdlp_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - tp).count();
                // LP Step 4: crossover to a basis; Step 5 finishes from it.
                int nb = spx.crossover_start(xp, yp);
                sopt.shift_instead_of_phase1 = true;
                s.engine_used = "pdlp(" + eng + ")+crossover+simplex";
                if (opt.verbose)
                    std::printf("  pdlp: %d its, eps P/D/G %.1e %.1e %.1e, %.3fs; crossover: %d structurals basic\n",
                                pr.iterations, pr.eps_P, pr.eps_D, pr.eps_G, s.pdlp_seconds, nb);
            }
            st = spx.solve(sopt);
        }
        DualSimplex& W = used_b ? spx_b : spx;
        s.simplex_seconds = W.stats().seconds;
        s.iterations = W.stats().iterations;
        if (st == "dual_infeasible") {
            // Dual phase 1 found no dual feasible basis. That verdict rests on
            // tolerances, so it is not trusted: the primal simplex continues
            // from this basis and either reaches the optimum, returns an
            // unbounded ray (a real proof), or proves infeasibility.
            SimplexOptions fo = sopt;
            fo.stop_flag = nullptr;
            std::string ps = W.solve_primal(fo);
            if (opt.verbose) std::printf("  simplex: dual phase 1 reported dual infeasible; primal simplex -> %s\n", ps.c_str());
            st = ps;
            s.iterations += W.stats().iterations;
        }
        if (st != "optimal" && st != "iteration_limit" && st != "time_limit") { s.status = st; return s; }
        s.status = st;
        auto xs = W.primal();
        auto ys = W.row_duals();
        for (int j = 0; j < reduced.n(); ++j) x_r[j] = sr.Dc[j] * xs[j];
        for (int i = 0; i < reduced.m(); ++i) y_r[i] = sr.Dr[i] * ys[i];
        if (opt.verbose) {
            KKTReport ks = verify(sr.scaled, xs, ys);
            KKTReport kr = verify(reduced, x_r, y_r);
            std::printf("  kkt scaled:  eps_P %.2e eps_D %.2e eps_G %.2e\n", ks.eps_P, ks.eps_D, ks.eps_G);
            std::printf("  kkt reduced: eps_P %.2e eps_D %.2e eps_G %.2e  (n %d m %d of %d x %d)\n",
                        kr.eps_P, kr.eps_D, kr.eps_G, reduced.n(), reduced.m(), n0, m0);
        }
    } else {
        s.status = "optimal";
    }

    CSR ATcsr = to_csc_as_transposed_csr(reduced.A);
    auto ATy = matvec_T(ATcsr, y_r);
    std::vector<double> z_r(reduced.n());
    for (int j = 0; j < reduced.n(); ++j) z_r[j] = reduced.c[j] - ATy[j];

    std::vector<double> x_orig, y_orig, z_orig;
    postsolve(n0, m0, pres.row_ids, pres.col_ids, x_r, y_r, z_r, pres.steps, x_orig, y_orig, z_orig);

    KKTReport final = verify(ranged, x_orig, y_orig);
    if (opt.verbose) {
        // Worst original-space violations, to tell presolve/postsolve error
        // apart from simplex tolerance.
        CSR Ao = to_csr(ranged.A);
        auto ax = matvec(Ao, x_orig);
        double wr = 0, wb = 0; int ir = -1, jb = -1;
        for (int i = 0; i < m0; ++i) {
            double v = std::max(ranged.rL[i] - ax[i], ax[i] - ranged.rU[i]);
            if (v > wr) { wr = v; ir = i; }
        }
        for (int j = 0; j < n0; ++j) {
            double v = std::max(ranged.l[j] - x_orig[j], x_orig[j] - ranged.u[j]);
            if (v > wb) { wb = v; jb = j; }
        }
        std::printf("  worst row violation %.3e (%s)  worst bound violation %.3e (%s)\n", wr,
                    ir >= 0 ? ranged.row_names[ir].c_str() : "-", wb, jb >= 0 ? ranged.col_names[jb].c_str() : "-");
        if (ir >= 0) {
            bool kept = std::find(pres.row_ids.begin(), pres.row_ids.end(), ir) != pres.row_ids.end();
            std::printf("    row %s: [%g, %g] activity %.12g, %s by presolve\n", ranged.row_names[ir].c_str(),
                        ranged.rL[ir], ranged.rU[ir], ax[ir], kept ? "kept" : "removed");
            if (kept) {
                int k = (int)(std::find(pres.row_ids.begin(), pres.row_ids.end(), ir) - pres.row_ids.begin());
                auto axr = matvec(to_csr(reduced.A), x_r);
                std::printf("    reduced row: [%g, %g] activity %.12g\n", reduced.rL[k], reduced.rU[k], axr[k]);
                // entries of the original row whose column presolve removed
                for (int p = Ao.indptr[ir]; p < Ao.indptr[ir + 1]; ++p) {
                    int j = Ao.indices[p];
                    if (std::find(pres.col_ids.begin(), pres.col_ids.end(), j) == pres.col_ids.end())
                        std::printf("      removed col %s a=%g x=%.12g\n", ranged.col_names[j].c_str(), Ao.data[p], x_orig[j]);
                }
            }
            for (const auto& st : pres.steps) {
                bool touches = st.row_id == ir;
                for (const auto& e : st.column) touches |= e.first == ir;
                if (touches) std::printf("    step %s row %d col %d value %g\n", st.kind.c_str(), st.row_id, st.col_id, st.value);
            }
        }
    }
    s.has_solution = true;
    s.objective = report_objective(ranged, final.objective);
    for (size_t j = 0; j < problem.col_names.size(); ++j) s.x[problem.col_names[j]] = x_orig[j];
    for (size_t i = 0; i < problem.row_names.size(); ++i) s.y[problem.row_names[i]] = y_orig[i];
    s.eps_P = final.eps_P; s.eps_D = final.eps_D; s.eps_G = final.eps_G;
    // The reported status follows the independent verifier on the original
    // problem, not the engine's own (scaled, presolved) view.
    s.status = certified_status(s.status, std::max({s.eps_P, s.eps_D, s.eps_G}), 1e-6);
    return s;
}

// ============================================== concurrent LP portfolio ==
//
// One presolve and one scaling, then up to four engines on their own threads:
//   0 dual simplex          1 primal simplex
//   2 interior point -> crossover -> simplex polish
//   3 PDLP (GPU when built with CUDA) -> crossover -> simplex polish
// Each engine's answer is postsolved and checked by the independent verifier
// on the original problem. The first certified optimum (or a definitive
// infeasible / unbounded proof) wins and cancels the others; an engine whose
// "optimal" fails verification simply does not claim, and the rest go on.
Solution solve_mps_concurrent(const std::string& path, const SimplexOptions& opt, bool use_presolve,
                              double pdlp_tol, unsigned engines) {
    LPProblem problem = timed_read_mps(path);
    RangedLP ranged = to_ranged_lp(problem);
    const int n0 = ranged.n(), m0 = ranged.m();
    PresolveResult pres = presolve(ranged, use_presolve ? 500 : 0);
    if (pres.status == "infeasible" || pres.status == "unbounded" || pres.reduced.n() == 0)
        return solve_mps_simplex(path, opt, use_presolve);          // decided by presolve alone
    const RangedLP& reduced = pres.reduced;

    ScalingResult sr;
    if (opt.scaling) sr = scale(reduced);
    else { sr.scaled = reduced; sr.Dr.assign(reduced.m(), 1.0); sr.Dc.assign(reduced.n(), 1.0); }
    const CSR ATred = to_csc_as_transposed_csr(reduced.A);

    struct Outcome {
        std::string st = "not started";
        std::vector<double> x_orig, y_orig;
        KKTReport k;
        int iterations = 0;
        double seconds = 0;
        bool has_point = false;
    };
    const char* names[4] = {"dual simplex", "primal simplex", "ipm+crossover",
#ifdef SOVEREIGN_WITH_CUDA
                            "pdlp(gpu)+crossover"
#else
                            "pdlp(cpu)+crossover"
#endif
    };
    Outcome out[4];
    std::atomic<bool> stop{false};
    std::atomic<int> winner{-1};
    std::mutex log_mu;
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    std::mutex wait_mu;
    std::condition_variable wait_cv;
    auto claim = [&](int who) {
        int e = -1;
        if (winner.compare_exchange_strong(e, who)) {
            { std::lock_guard<std::mutex> g(wait_mu); stop.store(true); }
            wait_cv.notify_all();
        }
    };
    // Staggered start: an engine waits `secs` and runs only if nobody has won
    // by then, so easy LPs never pay for starting the other engines (or for
    // creating a CUDA context).
    auto start_after = [&](double secs) {
        std::unique_lock<std::mutex> lk(wait_mu);
        return !wait_cv.wait_for(lk, std::chrono::duration<double>(secs), [&]() { return stop.load(); });
    };
    const long long nnzA = (long long)reduced.A.val.size();

    // xs, ys live in the simplex's scaled space.
    auto finish = [&](int who, const std::string& st, const std::vector<double>& xs, const std::vector<double>& ys, int its) {
        Outcome& o = out[who];
        o.st = st; o.iterations = its; o.seconds = elapsed();
        if (st == "infeasible" || st == "unbounded") claim(who);
        else if (st == "optimal") {
            std::vector<double> x_r(reduced.n()), y_r(reduced.m()), z_r(reduced.n()), z_orig;
            for (int j = 0; j < reduced.n(); ++j) x_r[j] = sr.Dc[j] * xs[j];
            for (int i = 0; i < reduced.m(); ++i) y_r[i] = sr.Dr[i] * ys[i];
            auto ATy = matvec_T(ATred, y_r);
            for (int j = 0; j < reduced.n(); ++j) z_r[j] = reduced.c[j] - ATy[j];
            postsolve(n0, m0, pres.row_ids, pres.col_ids, x_r, y_r, z_r, pres.steps, o.x_orig, o.y_orig, z_orig);
            o.k = verify(ranged, o.x_orig, o.y_orig);
            o.has_point = true;
            if (std::max({o.k.eps_P, o.k.eps_D, o.k.eps_G}) <= 1e-6) claim(who);
        }
        if (opt.verbose) {
            std::lock_guard<std::mutex> g(log_mu);
            std::printf("  concurrent: %-20s -> %-16s %8.3fs  %d its%s\n", names[who], st.c_str(), o.seconds, its,
                        o.has_point ? (std::max({o.k.eps_P, o.k.eps_D, o.k.eps_G}) <= 1e-6 ? "  (verified)" : "  (NOT verified)") : "");
        }
    };
    auto guarded = [&](int who, auto&& body) {
        return std::thread([&, who, body]() {
            lower_thread_priority();
            try { body(); }
            catch (const std::exception& e) { finish(who, std::string("error: ") + e.what(), {}, {}, 0); }
        });
    };

    SimplexOptions so = opt;
    so.stop_flag = &stop;
    std::vector<std::thread> threads;
    // Tiny LPs: the dual simplex finishes before a helper could start, so do
    // not even create the threads (explicit engine choices are honoured).
    if (engines == 0xFu && nnzA < opt.concurrent_min_nnz) engines = 1u;
    if (engines & 2u) threads.push_back(guarded(1, [&]() {
        if (engines != 2u && !start_after(opt.primal_delay)) return;
        DualSimplex spx; spx.load(sr.scaled);
        std::string st = spx.solve_primal(so);
        finish(1, st, spx.primal(), spx.row_duals(), spx.stats().iterations);
    }));
    if (engines & 4u) threads.push_back(guarded(2, [&]() {
        if (!start_after(opt.concurrent_delay)) return;
        IpmOptions io;
        io.stop_flag = &stop;
        io.time_limit = opt.time_limit;
        // A converging IPM needs a few dozen iterations; one still running
        // after 80 is stalling and only slows the dual simplex, so it stops.
        io.max_iterations = 80;
        IpmResult r = solve_ipm(sr.scaled, io);
        if (r.status != "optimal" && r.status != "near_optimal") { finish(2, "ipm " + r.status, {}, {}, r.iterations); return; }
        DualSimplex spx; spx.load(sr.scaled);
        spx.crossover_start(r.x, r.y);
        SimplexOptions o = so; o.shift_instead_of_phase1 = true;
        std::string st = spx.solve(o);
        finish(2, st, spx.primal(), spx.row_duals(), r.iterations + spx.stats().iterations);
    }));
    // First-order PDLP pays off only on large LPs (and its GPU start-up costs
    // ~0.1 s), so it joins late and only above a size threshold.
    if ((engines & 8u) && (nnzA >= opt.pdlp_min_nnz || engines == 8u)) threads.push_back(guarded(3, [&]() {
        if (engines != 8u && !start_after(opt.pdlp_delay)) return;
        auto [pf, meta] = build_pdlp_form(reduced);
        ScalingResult psr = scale(pf);
#ifdef SOVEREIGN_WITH_CUDA
        PdlpResult pr = solve_pdhg_gpu(pf, psr.scaled, psr.Dr, psr.Dc, 0.99, 200000, 64, pdlp_tol, &stop);
#else
        PdlpResult pr = solve_pdhg(pf, psr.scaled, psr.Dr, psr.Dc, 0.99, 200000, 64, pdlp_tol, &stop);
#endif
        if (stop.load()) { finish(3, "stopped", {}, {}, pr.iterations); return; }
        std::vector<double> xp, yp;
        extract_solution(meta, pr.x, pr.y, xp, yp);
        for (int j = 0; j < reduced.n(); ++j) xp[j] /= sr.Dc[j];
        for (int i = 0; i < reduced.m(); ++i) yp[i] /= sr.Dr[i];
        DualSimplex spx; spx.load(sr.scaled);
        spx.crossover_start(xp, yp);
        SimplexOptions o = so; o.shift_instead_of_phase1 = true;
        std::string st = spx.solve(o);
        finish(3, st, spx.primal(), spx.row_duals(), pr.iterations + spx.stats().iterations);
    }));
    // Portfolio-wide deadline: when the time limit passes, every engine is
    // told to stop. Without it an engine with no time limit of its own (PDLP
    // runs to an iteration cap) kept the solve alive after the others quit.
    bool all_done = false, timed_out = false;
    std::thread deadline;
    if (std::isfinite(opt.time_limit) && opt.time_limit < 1e20)
        deadline = std::thread([&]() {
            std::unique_lock<std::mutex> lk(wait_mu);
            if (!wait_cv.wait_for(lk, std::chrono::duration<double>(opt.time_limit), [&]() { return stop.load() || all_done; })) {
                timed_out = true;
                stop.store(true);
            }
            wait_cv.notify_all();                        // release helpers still waiting to start
        });
    // The dual simplex runs on the calling thread (no spawn, warm caches):
    // on easy LPs it finishes before any helper has started.
    if (engines & 1u) {
        try {
            DualSimplex spx; spx.load(sr.scaled);
            std::string st = spx.solve(so);
            int its = spx.stats().iterations;
            if (st == "dual_infeasible") { st = spx.solve_primal(so); its += spx.stats().iterations; }   // see solve_mps_simplex
            finish(0, st, spx.primal(), spx.row_duals(), its);
        } catch (const std::exception& e) { finish(0, std::string("error: ") + e.what(), {}, {}, 0); }
    }
    for (auto& t : threads) t.join();
    if (deadline.joinable()) {
        { std::lock_guard<std::mutex> g(wait_mu); all_done = true; }
        wait_cv.notify_all();
        deadline.join();
    }

    // Winner, or else the best verified-looking point any engine produced.
    int w = winner.load();
    if (w < 0) {
        double best = INF;
        for (int e = 0; e < 4; ++e)
            if (out[e].has_point) {
                double eps = std::max({out[e].k.eps_P, out[e].k.eps_D, out[e].k.eps_G});
                if (eps < best) { best = eps; w = e; }
            }
    }
    Solution s;
    s.eps_P = s.eps_D = s.eps_G = INF;
    std::string summary;
    for (int e = 0; e < 4; ++e)
        if (engines & (1u << e)) {
            if (!summary.empty()) summary += " | ";
            summary += std::string(names[e]) + ": " + out[e].st;
        }
    if (w < 0) {
        // No engine produced a point: report the most informative status.
        s.status = out[0].st != "not run" ? out[0].st : out[1].st;
        if (s.status == "dual_infeasible") s.status = "unbounded_or_infeasible";
        if (timed_out && s.status == "stopped") s.status = "time_limit";   // stopped by the deadline, not by a winner
        s.engine_used = "concurrent (" + summary + ")";
        return s;
    }
    const Outcome& o = out[w];
    s.engine_used = std::string(names[w]) + " won the concurrent race (" + summary + ")";
    s.iterations = o.iterations;
    s.simplex_seconds = o.seconds;
    s.status = o.st;
    if (timed_out && s.status == "stopped") s.status = "time_limit";
    if (o.has_point) {
        s.has_solution = true;
        s.objective = report_objective(ranged, o.k.objective);
        for (size_t j = 0; j < problem.col_names.size(); ++j) s.x[problem.col_names[j]] = o.x_orig[j];
        for (size_t i = 0; i < problem.row_names.size(); ++i) s.y[problem.row_names[i]] = o.y_orig[i];
        s.eps_P = o.k.eps_P; s.eps_D = o.k.eps_D; s.eps_G = o.k.eps_G;
        s.status = certified_status(s.status, std::max({s.eps_P, s.eps_D, s.eps_G}), 1e-6);
    }
    return s;
}

Solution check_solution(const std::string& path, const std::string& solution_file) {
    LPProblem problem = timed_read_mps(path);
    RangedLP ranged = to_ranged_lp(problem);
    std::unordered_map<std::string, int> col, row;
    for (int j = 0; j < ranged.n(); ++j) col[ranged.col_names[j]] = j;
    for (int i = 0; i < ranged.m(); ++i) row[ranged.row_names[i]] = i;
    std::vector<double> x(ranged.n(), 0.0), y(ranged.m(), 0.0);
    std::ifstream in(solution_file);
    if (!in) throw std::runtime_error("cannot open solution file: " + solution_file);
    std::string line, section = "PRIMAL";
    bool has_duals = false;
    while (std::getline(in, line)) {
        std::istringstream ls(line);
        std::string name; double v;
        if (!(ls >> name)) continue;
        if (name[0] == '#') continue;                                   // comment line
        if (name == "PRIMAL" || name == "DUAL") { section = name; has_duals |= name == "DUAL"; continue; }
        if (!(ls >> v)) throw std::runtime_error("bad solution line: " + line);
        auto& idx = section == "PRIMAL" ? col : row;
        auto it = idx.find(name);
        if (it == idx.end()) throw std::runtime_error("unknown " + section + " name in solution file: " + name);
        (section == "PRIMAL" ? x : y)[it->second] = v;
    }
    // Duals in the file follow the model's own sense; the verifier works on
    // the minimization form (to_ranged_lp negates c for "max").
    if (problem.sense == "max") for (double& v : y) v = -v;
    KKTReport k = verify(ranged, x, y);
    Solution s;
    s.status = "checked";
    s.engine_used = "none (external solution)";
    s.has_solution = true;
    s.objective = report_objective(ranged, k.objective);
    s.eps_P = k.eps_P;
    s.eps_D = has_duals ? k.eps_D : NAN;
    s.eps_G = has_duals ? k.eps_G : NAN;
    for (int j = 0; j < ranged.n(); ++j)
        if (!problem.integer.empty() && problem.integer[j])
            s.max_int_violation = std::max(s.max_int_violation, std::fabs(x[j] - std::round(x[j])));
    return s;
}

Solution solve_mps_mip(const std::string& path, const MipOptions& opt, bool use_presolve) {
    LPProblem problem = timed_read_mps(path);
    RangedLP ranged = to_ranged_lp(problem);
    int n0 = ranged.n(), m0 = ranged.m();

    Solution s;
    s.is_mip = true;
    s.engine_used = "branch-and-cut";
    s.eps_P = s.eps_D = s.eps_G = INF;

    PresolveResult pres = presolve(ranged, use_presolve ? 500 : 0, use_presolve);
    if (pres.status == "infeasible" || pres.status == "unbounded") { s.status = pres.status; return s; }
    RangedLP& reduced = pres.reduced;

    std::vector<double> x_r(reduced.n(), 0.0), y_r(reduced.m(), 0.0);
    if (reduced.n() > 0) {
        ScalingResult sr = scale(reduced);          // integer columns keep Dc = 1
        MipResult mr = solve_mip(sr.scaled, opt);
        s.status = mr.status;
        s.nodes = mr.nodes;
        s.iterations = (int)std::min<long long>(mr.lp_iterations, 2000000000LL);
        s.cuts = mr.cuts_added;
        s.root_lp = report_objective(ranged, mr.root_lp);
        s.root_after_cuts = report_objective(ranged, mr.root_after_cuts);
        s.mip_bound = report_objective(ranged, mr.bound);
        s.mip_gap = mr.gap;
        s.simplex_seconds = mr.seconds;
        if (!mr.has_solution) return s;
        for (int j = 0; j < reduced.n(); ++j) x_r[j] = sr.Dc[j] * mr.x[j];
    } else {
        s.status = "optimal";
    }

    std::vector<double> z_r(reduced.c), x_orig, y_orig, z_orig;
    postsolve(n0, m0, pres.row_ids, pres.col_ids, x_r, y_r, z_r, pres.steps, x_orig, y_orig, z_orig);
    for (auto& v : y_orig) if (std::isnan(v)) v = 0.0;

    // MILP Step 12.4: feasibility + bounds + integrality on the original data.
    KKTReport final = verify(ranged, x_orig, y_orig);
    double iv = 0.0;
    for (int j = 0; j < n0; ++j)
        if (!ranged.integer.empty() && ranged.integer[j]) iv = std::max(iv, std::fabs(x_orig[j] - std::round(x_orig[j])));
    s.max_int_violation = iv;
    s.has_solution = true;
    s.objective = report_objective(ranged, final.objective);
    for (size_t j = 0; j < problem.col_names.size(); ++j) s.x[problem.col_names[j]] = x_orig[j];
    s.eps_P = final.eps_P;
    s.eps_D = 0; s.eps_G = 0;     // no dual certificate for a MILP (see MILP Step 12.4)
    return s;
}

Solution solve_mps_ipm(const std::string& path, const IpmOptions& opt, bool use_presolve) {
    LPProblem problem = timed_read_mps(path);
    RangedLP ranged = to_ranged_lp(problem);
    int n0 = ranged.n(), m0 = ranged.m();
    const bool qp = ranged.Q.nnz() > 0;

    Solution s;
    s.engine_used = qp ? "ipm (convex QP)" : "ipm";
    s.eps_P = s.eps_D = s.eps_G = INF;

    PresolveResult pres = presolve(ranged, use_presolve ? 500 : 0);
    if (pres.status == "infeasible" || pres.status == "unbounded") { s.status = pres.status; return s; }
    RangedLP& reduced = pres.reduced;

    std::vector<double> x_r(reduced.n(), 0.0), y_r(reduced.m(), 0.0);
    if (reduced.n() > 0) {
        ScalingResult sr = scale(reduced);
        IpmResult r = solve_ipm(sr.scaled, opt);
        s.status = r.status;
        s.iterations = r.iterations;
        s.simplex_seconds = r.seconds;
        if (r.status != "optimal" && r.status != "near_optimal" && r.status != "iteration_limit" && r.status != "time_limit") return s;
        for (int j = 0; j < reduced.n(); ++j) x_r[j] = sr.Dc[j] * r.x[j];
        for (int i = 0; i < reduced.m(); ++i) y_r[i] = sr.Dr[i] * r.y[i];
    } else {
        s.status = "optimal";
    }

    // reduced costs z = Q x + c - A^T y (postsolve needs them for its dual recovery)
    CSR ATcsr = to_csc_as_transposed_csr(reduced.A);
    auto ATy = matvec_T(ATcsr, y_r);
    std::vector<double> z_r(reduced.n());
    std::vector<double> qx_r(reduced.n(), 0.0);
    for (int k = 0; k < reduced.Q.nnz(); ++k) {
        int i = reduced.Q.row_idx[k], j = reduced.Q.col_idx[k]; double v = reduced.Q.val[k];
        qx_r[i] += v * x_r[j];
        if (i != j) qx_r[j] += v * x_r[i];
    }
    for (int j = 0; j < reduced.n(); ++j) z_r[j] = qx_r[j] + reduced.c[j] - ATy[j];

    std::vector<double> x_orig, y_orig, z_orig;
    postsolve(n0, m0, pres.row_ids, pres.col_ids, x_r, y_r, z_r, pres.steps, x_orig, y_orig, z_orig);

    // Verify on the original problem. For QP, the LP verifier applied to the
    // linearized cost c + Qx checks exactly: primal feasibility, dual
    // feasibility of z = Qx + c - A^T y, and the QP duality gap.
    RangedLP lin = ranged;
    std::vector<double> qx(n0, 0.0);
    for (int k = 0; k < ranged.Q.nnz(); ++k) {
        int i = ranged.Q.row_idx[k], j = ranged.Q.col_idx[k]; double v = ranged.Q.val[k];
        qx[i] += v * x_orig[j];
        if (i != j) qx[j] += v * x_orig[i];
    }
    double half_xqx = 0.0;
    for (int j = 0; j < n0; ++j) { lin.c[j] += qx[j]; half_xqx += 0.5 * x_orig[j] * qx[j]; }
    KKTReport final = verify(lin, x_orig, y_orig);
    s.has_solution = true;
    s.objective = report_objective(ranged, final.objective - half_xqx);   // c'x + offset + 1/2 x'Qx
    for (size_t j = 0; j < problem.col_names.size(); ++j) s.x[problem.col_names[j]] = x_orig[j];
    for (size_t i = 0; i < problem.row_names.size(); ++i) s.y[problem.row_names[i]] = y_orig[i];
    s.eps_P = final.eps_P; s.eps_D = final.eps_D; s.eps_G = final.eps_G;
    // The reported status follows the independent verifier on the original
    // problem, not the engine's own (scaled, presolved) view.
    s.status = certified_status(s.status, std::max({s.eps_P, s.eps_D, s.eps_G}), 1e-6);
    return s;
}
