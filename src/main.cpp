// main.cpp -- the CLI:
//   ./sovereign_solve problem.mps [time_limit]                      (auto: MILP -> branch-and-cut,
//                                                                     QP -> interior point, LP -> concurrent portfolio)
//   ./sovereign_solve --concurrent problem.mps [time_limit_seconds]  (dual + primal simplex, IPM+crossover,
//                                                                     PDLP(GPU)+crossover; first verified answer wins)
//   ./sovereign_solve --pdlp problem.mps [tol] [max_iterations]     (PDLP only, CPU/GPU race)
//   ./sovereign_solve --simplex problem.mps [time_limit_seconds]     (dual simplex)
//   ./sovereign_solve --crossover problem.mps [time_limit_seconds]   (PDLP -> crossover -> simplex)
//   ./sovereign_solve --mip problem.mps [time_limit_seconds]         (branch-and-cut)
//   ./sovereign_solve --ipm problem.mps [time_limit_seconds]         (interior point: LP and convex QP)
//   ./sovereign_solve --auto problem.mps [time_limit_seconds]        (race: dual simplex vs PDLP(GPU)+crossover)
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include "solve.hpp"
#include "simplex.hpp"
#include "mip.hpp"
#include "qp_ipm.hpp"
#include "mps_reader.hpp"
#include "report_io.hpp"
#include "alloc_stats.hpp"
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <thread>

int main(int argc, char** argv) {
    std::vector<std::string> args;
    bool use_simplex = false, use_presolve = true, stats_only = false, use_crossover = false, use_mip = false;
    MipOptions mopt;
    mopt.threads = std::max(1u, std::min(8u, std::thread::hardware_concurrency()));   // --threads overrides
    bool use_ipm = false, use_race = false, use_pdlp = false, use_concurrent = false;
    unsigned engines = 0xFu;
    double pdlp_tol = 1e-4;
    int verbose = 0;
    std::string check_file;            // --check=<solution.txt>: verify an external solution
    std::string json_file, sol_file, csv_prefix;   // machine-readable outputs (include/report_io.hpp)
    SimplexOptions sflags;             // robustness switches for ablation runs
    for (int k = 1; k < argc; ++k) {
        std::string a = argv[k];
        if (a == "--simplex") use_simplex = true;
        else if (a == "--no-presolve") use_presolve = false;
        else if (a == "--stats") stats_only = true;
        else if (a.rfind("--check=", 0) == 0) check_file = a.substr(8);
        else if (a.rfind("--json=", 0) == 0) json_file = a.substr(7);
        else if (a.rfind("--sol=", 0) == 0) sol_file = a.substr(6);
        else if (a.rfind("--csv=", 0) == 0) csv_prefix = a.substr(6);
        else if (a == "--live") std::setvbuf(stdout, nullptr, _IONBF, 0);   // log lines reach a pipe immediately (UI)
        else if (a == "--crossover") { use_simplex = true; use_crossover = true; }
        else if (a.rfind("--pdlp-tol=", 0) == 0) pdlp_tol = std::stod(a.substr(11));
        else if (a == "--mip") use_mip = true;
        else if (a == "--ipm") use_ipm = true;
        else if (a == "--auto") { use_simplex = true; use_race = true; }
        else if (a == "--concurrent") { use_simplex = true; use_concurrent = true; }
        else if (a.rfind("--engines=", 0) == 0) {   // e.g. --engines=dual,primal,ipm,pdlp
            use_simplex = true; use_concurrent = true; engines = 0;
            std::string list = a.substr(10) + ",";
            for (size_t p0 = 0, p1; (p1 = list.find(',', p0)) != std::string::npos; p0 = p1 + 1) {
                std::string e = list.substr(p0, p1 - p0);
                if (e == "dual") engines |= 1u; else if (e == "primal") engines |= 2u;
                else if (e == "ipm") engines |= 4u; else if (e == "pdlp") engines |= 8u;
                else if (!e.empty()) { std::cerr << "unknown engine: " << e << "\n"; return 2; }
            }
        }
        else if (a == "--pdlp") use_pdlp = true;
        else if (a.rfind("--gap=", 0) == 0) mopt.rel_gap = std::stod(a.substr(6));
        else if (a.rfind("--threads=", 0) == 0) mopt.threads = std::stoi(a.substr(10));
        else if (a.rfind("--nodes=", 0) == 0) mopt.node_limit = std::stoll(a.substr(8));
        else if (a == "--no-cuts") mopt.cuts = false;
        else if (a == "--no-heuristics") mopt.heuristics = false;
        else if (a == "--no-dse") sflags.steepest_edge = false;
        else if (a == "--no-harris") sflags.harris_bfrt = false;
        else if (a == "--no-perturb") sflags.perturb_costs = false;
        else if (a == "--no-bland") sflags.bland_fallback = false;
        else if (a == "--no-scaling") sflags.scaling = false;
        else if (a.rfind("--refactor=", 0) == 0) sflags.refactor_frequency = std::stoi(a.substr(11));
        else if (a == "--textbook") {  // all simplex robustness features off
            sflags.steepest_edge = sflags.harris_bfrt = sflags.perturb_costs = sflags.bland_fallback = sflags.scaling = false;
        }
        else if (a == "-v") verbose = 1;
        else if (a == "-vv") verbose = 2;
        else if (a == "-vvv") verbose = 3;
        else args.push_back(a);
    }
    if (args.empty()) {
        std::cerr << "usage: " << argv[0] << " <problem.mps> [time_limit_seconds]   (engine chosen by problem type)\n"
                  << "       " << argv[0] << " --pdlp <problem.mps> [tol] [max_iterations]\n"
                  << "       " << argv[0] << " --auto <problem.mps> [time_limit_seconds]   (LP: simplex vs PDLP+crossover race)\n"
                  << "       " << argv[0] << " --concurrent|--engines=dual,primal,ipm,pdlp <problem.mps> [time_limit_seconds]   (LP portfolio)\n"
                  << "       " << argv[0] << " --ipm <problem.mps|.qps> [time_limit_seconds]\n"
                  << "       " << argv[0] << " --simplex [--no-presolve] [--textbook | --no-dse --no-harris --no-perturb --no-bland --no-scaling] [-v|-vv] <problem.mps> [time_limit_seconds]\n"
                  << "       " << argv[0] << " --crossover [--pdlp-tol=1e-4] [-v] <problem.mps> [time_limit_seconds]\n"
                  << "       " << argv[0] << " --mip [--gap=1e-4] [--threads=N] [--nodes=N] [--no-cuts] [--no-heuristics] [-v]"
                  << " <problem.mps> [time_limit_seconds]\n";
        return 1;
    }
    std::string path = args[0];

    // Every failure path (a bad file, a CUDA error, anything) now prints a
    // readable message and exits cleanly instead of hitting an uncaught
    // exception -- which on MSVC's Debug runtime shows as a message-less
    // "abort() has been called" dialog. Never demo this without this net.
    try {
        if (stats_only) {
            LPProblem p = read_mps(path);
            int nrhs = 0, nlo = 0, nhi = 0, nobj = 0;
            for (double v : p.row_rhs) nrhs += v != 0.0;
            for (double v : p.lo) nlo += std::isfinite(v) && v != 0.0;
            for (double v : p.hi) nhi += std::isfinite(v);
            for (double v : p.obj) nobj += v != 0.0;
            std::cout << p.name << ": rows " << p.m() << " cols " << p.n() << " nnz " << p.A.nnz()
                      << " obj_nz " << nobj << " rhs_nz " << p.row_rhs.size() << "/" << nrhs
                      << " ranges " << p.ranges.size() << " lo!=0 " << nlo << " finite_hi " << nhi << "\n";
            return 0;
        }
        if (!check_file.empty()) {
            Solution c = check_solution(path, check_file);
            std::cout << std::setprecision(10) << "Problem:    " << path << "\nSolution:   " << check_file
                      << "\nObjective:  " << c.objective << std::scientific
                      << "\neps_P:      " << c.eps_P << "\neps_D:      " << c.eps_D << "\neps_G:      " << c.eps_G
                      << "\nmax integrality violation: " << c.max_int_violation << "\n";
            return 0;
        }
        // No engine flag: pick by problem type. Integer markers -> branch-and-cut,
        // a quadratic objective -> interior point, a plain LP -> dual simplex. The old positional "[tol] [iters]" form
        // (and --pdlp) still select the PDLP-only path.
        if (!use_ipm && !use_mip && !use_simplex && !use_pdlp && args.size() <= 2) {
            LPProblem p = read_mps(path);
            bool has_int = false;
            for (char f : p.integer) has_int |= f != 0;
            if (has_int) use_mip = true;
            else if (p.Q.nnz() > 0) use_ipm = true;
            else { use_simplex = true; use_concurrent = true; }   // LP: concurrent portfolio (dual simplex first)
        }
        auto t0 = std::chrono::steady_clock::now();
        Solution sol;
        if (use_ipm) {
            IpmOptions iopt;
            iopt.verbose = verbose;
            if (args.size() > 1) iopt.time_limit = std::stod(args[1]);
            sol = solve_mps_ipm(path, iopt, use_presolve);
        } else if (use_mip) {
            mopt.verbose = verbose;
            if (args.size() > 1) mopt.time_limit = std::stod(args[1]);
            sol = solve_mps_mip(path, mopt, use_presolve);
        } else if (use_simplex) {
            SimplexOptions opt = sflags;
            opt.verbose = verbose;
            if (args.size() > 1) opt.time_limit = std::stod(args[1]);
            sol = use_concurrent ? solve_mps_concurrent(path, opt, use_presolve, pdlp_tol, engines)
                                 : solve_mps_simplex(path, opt, use_presolve, use_crossover, pdlp_tol, 200000, use_race);
        } else {
            double tol = args.size() > 1 ? std::stod(args[1]) : 1e-6;
            int max_iter = args.size() > 2 ? std::stoi(args[2]) : 500000; // see include/solve.hpp's comment on this default
            sol = solve_mps(path, tol, max_iter);
        }
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Problem:    " << path << "\n";
        std::cout << "Status:     " << sol.status << "\n";
        if (sol.is_mip) {
            std::cout << std::setprecision(10);
            if (sol.has_solution) std::cout << "Objective:  " << sol.objective << "\n";
            std::cout << "Bound:      " << sol.mip_bound << "\n";
            if (sol.has_solution) std::cout << "Gap:        " << 100.0 * sol.mip_gap << " %\n";
            std::cout << "Root LP:    " << sol.root_lp << "  (after " << sol.cuts << " cuts: " << sol.root_after_cuts << ")\n";
            std::cout << "Nodes:      " << sol.nodes << "   LP iterations: " << sol.iterations << "\n";
            if (sol.has_solution) {
                std::cout << std::scientific;
                std::cout << "eps_P:      " << sol.eps_P << "   max integrality violation: " << sol.max_int_violation << "\n";
                std::cout << std::fixed;
            }
        } else if (sol.has_solution) {
            std::cout << std::setprecision(10);
            std::cout << "Objective:  " << sol.objective << "\n";
            std::cout << "Engine:     " << sol.engine_used << "\n";
            if (use_crossover)
                std::cout << "Iterations: " << sol.iterations << " simplex after " << sol.pdlp_iterations
                          << " PDLP (" << sol.pdlp_seconds << " s PDLP, " << sol.simplex_seconds << " s simplex)\n";
            else if (use_simplex || use_ipm) std::cout << "Iterations: " << sol.iterations << "\n";
            else std::cout << "Iterations: " << sol.iterations << "  (restarts: " << sol.restarts << ")\n";
            std::cout << std::scientific;
            std::cout << "eps_P:      " << sol.eps_P << "\n";
            std::cout << "eps_D:      " << sol.eps_D << "\n";
            std::cout << "eps_G:      " << sol.eps_G << "\n";
            std::cout << std::fixed;
        }
        std::cout << "Time:       " << secs << " s\n";
        std::cout << "Read time:  " << last_read_seconds() << " s\n";
        std::cout << "Solve time: " << secs - last_read_seconds() << " s   (presolve + solve + postsolve + verification)\n";
        if (!json_file.empty() || !sol_file.empty() || !csv_prefix.empty()) {
            RunInfo info;
            info.model_path = path;
            info.total_seconds = secs;
            info.read_seconds = last_read_seconds();
            info.threads = use_mip ? mopt.threads : 1;
            info.have_resources = true;
            info.alloc = alloc_stats();       // before the writers allocate their own tables
            info.proc = process_stats();
            info.mode = use_mip ? "branch-and-cut" : use_ipm ? "interior point" : use_concurrent ? "concurrent LP portfolio"
                      : use_race ? "simplex vs PDLP race" : use_crossover ? "PDLP + crossover" : use_simplex ? "dual simplex" : "PDLP";
            if (!json_file.empty()) write_result_json(json_file, sol, info);
            if (!sol_file.empty()) write_solution_file(sol_file, sol, info);
            if (!csv_prefix.empty()) write_solution_csv(csv_prefix, sol, info);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sovereign_solve: error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "sovereign_solve: error: unknown failure (non-std::exception)\n";
        return 1;
    }
}
