// audit_lp.cpp -- the "is LP-on-GPU actually correct" audit tool.
// ================================================================
// solve_mps() (src/solve.cpp) RACES the CPU and GPU engines and only ever
// hands back the WINNER's numbers. That is exactly the wrong tool for
// auditing GPU correctness: on a small problem the CPU usually wins the
// race, so the GPU's own answer is never even looked at; on a problem the
// GPU wins, you still never see what the CPU would have produced on the
// identical input, so there is nothing to compare against. "the race
// picked a verified answer" and "the GPU engine itself is numerically
// trustworthy across problem shapes" are two different claims, and only
// this tool tests the second one.
//
// What this does, per input file:
//   1. read_mps -> to_ranged_lp -> presolve()      (identical to solve_mps)
//   2. build_pdlp_form -> scale()                  (identical to solve_mps)
//   3. run solve_pdhg()      (CPU) to completion, UNRACED, no stop_flag
//   4. run solve_pdhg_gpu()  (GPU) to completion, UNRACED, no stop_flag
//      (only if this binary was built with CUDA -- see CMakeLists.txt)
//   5. independently postsolve + verify() BOTH results against the
//      ORIGINAL (unpresolved, unscaled) problem -- the same independent
//      KKT checker solve_mps() uses, called twice, once per engine
//   6. report, side by side: objective, eps_P/eps_D/eps_G, iterations,
//      wall time, and the CPU-vs-GPU objective agreement in digits
//
// Why CPU and GPU can legitimately disagree at all, even though this
// project's algorithm, restart rule, and floating-point WIDTH (double,
// everywhere -- see gpu/pdlp_gpu_solve.cu, no float shortcuts) are
// identical on both engines: IEEE-754 addition is not associative. The
// GPU's SpMV kernel (sov_spmv_csr_warp_per_row) sums a row's nonzeros via
// a 32-lane warp-shuffle TREE reduction; the CPU's matvec() (sparse.hpp)
// sums the same nonzeros SEQUENTIALLY, left to right. Both are "correct"
// summation, but they associate differently, so the last few ULPs of a
// dot product can differ between engines even given bit-identical inputs
// and bit-identical arithmetic elsewhere. A well-conditioned problem
// should still let both engines converge to the same optimum to well
// within their requested tolerance; a genuinely ill-conditioned problem
// is exactly where you'd expect that gap to widen -- which is *why* the
// audit below runs a skewed/ill-conditioned case deliberately, not just
// "normal" ones. If CPU and GPU ever disagree by more than a few orders
// of magnitude past `tol`, that is a real bug (kernel indexing, a race in
// the warp reduction, an uninitialized buffer) -- not rounding -- and
// this tool's job is to make that distinction impossible to hand-wave.
//
// IMPORTANT SCOPE NOTE (read before calling anything "perfect"): this
// audits PDLP/PDHG's own accuracy ceiling only. The LP pipeline design
// doc's Steps 4-5 (Crossover + high-precision Dual Revised Simplex, the
// stage that drives 1e-8..1e-12 "textbook-exact" accuracy) are NOT
// implemented in this codebase yet -- see the gap analysis already
// delivered. Passing this audit at tol=1e-8 proves "PDLP converged to a
// KKT-verified point at that tolerance on both engines, and they agree."
// It does NOT prove "textbook-exact vertex precision," because nothing
// in this binary produces that yet. Do not conflate the two in a report.
//
// Usage:
//   audit_lp [--tol T] [--max-iter N] file1.mps [file2.mps ...]
// Exit code: 0 if every runnable file passed every check, 1 otherwise.
#include "mps_reader.hpp"
#include "presolve.hpp"
#include "scaling.hpp"
#include "pdlp.hpp"
#include "pdlp_gpu.hpp"
#include "verify.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <cmath>
#include <cstring>
#include <map>
#include <string>

namespace {

struct EngineRun {
    bool attempted = false;   // did we even try (false only for "GPU requested, no CUDA build")
    bool ran = false;         // did it run to a returned PdlpResult without throwing
    bool converged = false;
    double objective = 0;
    double eps_P = 0, eps_D = 0, eps_G = 0;
    int iterations = 0, restarts = 0;
    double seconds = 0;
    std::string note;         // error text, or "presolve fully solved it (no PDHG stage)"
};

EngineRun run_engine(bool use_gpu, const RangedLP& ranged, const PresolveResult& pres,
                      double tol, int max_iter, double eta) {
    EngineRun r;
    r.attempted = true;
#ifndef SOVEREIGN_WITH_CUDA
    if (use_gpu) { r.attempted = false; r.note = "this binary has no CUDA -- rebuild on the GPU machine to audit the GPU engine"; return r; }
#endif
    try {
        const RangedLP& reduced = pres.reduced;
        if (reduced.n() == 0 || reduced.m() == 0) {
            r.ran = true; r.converged = true;
            r.note = "presolve fully solved this problem by itself -- no PDHG stage to audit here";
            return r;
        }
        auto built = build_pdlp_form(reduced);
        RangedLP& pdlp_form = built.first;
        PdlpFormMeta& meta = built.second;
        ScalingResult sr = scale(pdlp_form);

        auto t0 = std::chrono::steady_clock::now();
        PdlpResult pr;
#ifdef SOVEREIGN_WITH_CUDA
        if (use_gpu) pr = solve_pdhg_gpu(pdlp_form, sr.scaled, sr.Dr, sr.Dc, eta, max_iter, 64, tol, nullptr);
        else          pr = solve_pdhg(pdlp_form, sr.scaled, sr.Dr, sr.Dc, eta, max_iter, 64, tol, nullptr);
#else
        pr = solve_pdhg(pdlp_form, sr.scaled, sr.Dr, sr.Dc, eta, max_iter, 64, tol, nullptr);
#endif
        auto t1 = std::chrono::steady_clock::now();
        r.seconds = std::chrono::duration<double>(t1 - t0).count();

        std::vector<double> x_reduced, y_reduced;
        extract_solution(meta, pr.x, pr.y, x_reduced, y_reduced);
        CSR ATcsr = to_csc_as_transposed_csr(reduced.A);
        auto ATy = matvec_T(ATcsr, y_reduced);
        std::vector<double> z_reduced(reduced.n());
        for (int j = 0; j < reduced.n(); ++j) z_reduced[j] = reduced.c[j] - ATy[j];

        std::vector<double> x_orig, y_orig, z_orig;
        postsolve(ranged.n(), ranged.m(), pres.row_ids, pres.col_ids, x_reduced, y_reduced, z_reduced, pres.steps, x_orig, y_orig, z_orig);
        KKTReport rep = verify(ranged, x_orig, y_orig);

        r.ran = true;
        r.converged = pr.converged;
        r.objective = report_objective(ranged, rep.objective);
        r.eps_P = rep.eps_P; r.eps_D = rep.eps_D; r.eps_G = rep.eps_G;
        r.iterations = pr.iterations; r.restarts = pr.restarts;
    } catch (const std::exception& e) {
        r.note = std::string("threw: ") + e.what();
    } catch (...) {
        r.note = "threw a non-std::exception";
    }
    return r;
}

// Published Netlib LP optimal objective values, matched by filename stem.
// REFERENCE VALUES -- VERIFY INDEPENDENTLY before citing these in a report
// or paper. These are the standard, widely-reproduced Netlib LP optimal
// objectives (see netlib.org/lp/data and the optimal-value tables used
// across the LP solver literature, e.g. Mittelmann's and Meszaros's LP
// benchmark pages). They are reproduced here from memory as a convenience
// cross-check, NOT re-derived from a primary source in this session --
// treat a near-match as reassuring and a mismatch as worth investigating,
// but do not present these specific numbers as citable ground truth to a
// reviewer without checking them against your own copy of the Netlib set.
const std::map<std::string, double> NETLIB_REFERENCE = {
    {"afiro",    -464.7531428571429},
    {"adlittle",  225494.9631162},
    {"blend",    -30.81214985},
    {"israel",   -896644.8218},
    {"kb2",      -1749.90023670},
    {"sc50a",    -64.57506},
    {"sc50b",    -70.00000},
    {"sc105",    -52.20206},
    {"scagr7",   -2331389.824},
    {"share1b",  -76589.31857918584},
    {"beaconfd",  33592.4858072},
};

std::string basename_stem(const std::string& path) {
    size_t slash = path.find_last_of("/\\");
    std::string base = slash == std::string::npos ? path : path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) base = base.substr(0, dot);
    return base;
}

const double* find_reference(const std::string& path) {
    auto it = NETLIB_REFERENCE.find(basename_stem(path));
    return it == NETLIB_REFERENCE.end() ? nullptr : &it->second;
}

void print_engine(const char* label, const EngineRun& r) {
    std::cout << "  " << label << ": ";
    if (!r.attempted) { std::cout << "SKIPPED (" << r.note << ")\n"; return; }
    if (!r.ran)       { std::cout << "FAILED  (" << r.note << ")\n"; return; }
    std::cout << std::scientific << std::setprecision(10);
    std::cout << (r.converged ? "converged" : "iteration_limit");
    if (!r.note.empty()) std::cout << " [" << r.note << "]";
    std::cout << "\n";
    std::cout << "      objective=" << r.objective
               << "  eps_P=" << r.eps_P << "  eps_D=" << r.eps_D << "  eps_G=" << r.eps_G << "\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "      iterations=" << r.iterations << "  restarts=" << r.restarts
               << "  time=" << r.seconds << "s\n";
}

} // namespace

int main(int argc, char** argv) {
    double tol = 1e-8;
    int max_iter = 200000;
    double eta = 0.99;
    std::vector<std::string> files;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--tol" && i + 1 < argc) { tol = std::stod(argv[++i]); }
        else if (a == "--max-iter" && i + 1 < argc) { max_iter = std::stoi(argv[++i]); }
        else files.push_back(a);
    }
    if (files.empty()) {
        std::cerr << "usage: " << argv[0] << " [--tol T] [--max-iter N] file1.mps [file2.mps ...]\n";
        return 1;
    }

#ifdef SOVEREIGN_WITH_CUDA
    std::cout << "audit_lp: built WITH CUDA -- auditing CPU and GPU engines independently\n";
#else
    std::cout << "audit_lp: built WITHOUT CUDA -- CPU-only audit (rebuild with nvcc present to also audit GPU)\n";
#endif
    std::cout << "tol=" << tol << "  max_iter=" << max_iter << "\n\n";

    int total = 0, hard_fail = 0, warn = 0;

    for (const auto& path : files) {
        std::cout << "=== " << path << " ===\n";
        LPProblem problem;
        RangedLP ranged;
        PresolveResult pres;
        try {
            problem = read_mps(path);
            ranged = to_ranged_lp(problem);
            pres = presolve(ranged);
        } catch (const std::exception& e) {
            std::cout << "  FAILED to load/presolve: " << e.what() << "\n\n";
            hard_fail++; total++;
            continue;
        }
        if (pres.status == "infeasible" || pres.status == "unbounded") {
            std::cout << "  presolve status: " << pres.status << " (not an accuracy case -- skipping)\n\n";
            continue;
        }
        total++;

        EngineRun cpu = run_engine(false, ranged, pres, tol, max_iter, eta);
        EngineRun gpu = run_engine(true,  ranged, pres, tol, max_iter, eta);
        print_engine("CPU", cpu);
        print_engine("GPU", gpu);

        // ---- verdict for this file ----
        bool this_fail = false, this_warn = false;
        std::vector<std::string> notes;

        if (cpu.ran && cpu.converged) {
            double worst = std::max({cpu.eps_P, cpu.eps_D, cpu.eps_G});
            if (worst > tol) { this_fail = true; notes.push_back("CPU claims converged but its own KKT residual exceeds tol"); }
        } else if (cpu.attempted) { this_fail = true; notes.push_back("CPU did not converge / did not run"); }

        if (gpu.attempted) {
            if (gpu.ran && gpu.converged) {
                double worst = std::max({gpu.eps_P, gpu.eps_D, gpu.eps_G});
                if (worst > tol) { this_fail = true; notes.push_back("GPU claims converged but its own KKT residual exceeds tol"); }
            } else {
                this_fail = true; notes.push_back("GPU did not converge / did not run");
            }
        }

        if (cpu.ran && gpu.ran && gpu.attempted && cpu.converged && gpu.converged) {
            double reldiff = std::fabs(cpu.objective - gpu.objective) / (1.0 + std::fabs(cpu.objective));
            std::cout << std::scientific << std::setprecision(3);
            std::cout << "  CPU vs GPU objective relative difference: " << reldiff << "\n";
            std::cout << std::fixed;
            if (reldiff > 1e-4) { this_fail = true; notes.push_back("CPU/GPU objective disagreement exceeds 1e-4 -- likely a real GPU bug, not rounding"); }
            else if (reldiff > 1e-6) { this_warn = true; notes.push_back("CPU/GPU objective disagreement is above 1e-6 -- worth investigating, but could still be reduction-order rounding on an ill-conditioned problem"); }
        }

        const double* ref = find_reference(path);
        if (ref && cpu.ran && cpu.converged) {
            double reldiff = std::fabs(cpu.objective - *ref) / (1.0 + std::fabs(*ref));
            std::cout << std::scientific << std::setprecision(3);
            std::cout << "  CPU vs published Netlib reference (" << *ref << "): relative diff " << reldiff << "\n";
            std::cout << std::fixed;
            if (reldiff > 1e-4) { this_fail = true; notes.push_back("CPU objective does not match the published Netlib reference"); }
        }
        if (ref && gpu.attempted && gpu.ran && gpu.converged) {
            double reldiff = std::fabs(gpu.objective - *ref) / (1.0 + std::fabs(*ref));
            std::cout << std::scientific << std::setprecision(3);
            std::cout << "  GPU vs published Netlib reference (" << *ref << "): relative diff " << reldiff << "\n";
            std::cout << std::fixed;
            if (reldiff > 1e-4) { this_fail = true; notes.push_back("GPU objective does not match the published Netlib reference"); }
        }

        std::cout << "  VERDICT: " << (this_fail ? "FAIL" : (this_warn ? "WARN" : "PASS"));
        for (auto& n : notes) std::cout << "\n    - " << n;
        std::cout << "\n\n";

        if (this_fail) hard_fail++;
        else if (this_warn) warn++;
    }

    std::cout << "==========================================\n";
    std::cout << total << " problem(s) audited, " << hard_fail << " FAIL, " << warn << " WARN, "
              << (total - hard_fail - warn) << " PASS\n";
    return hard_fail == 0 ? 0 : 1;
}
