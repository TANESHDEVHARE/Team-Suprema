// run_checks.cpp -- the same correctness checks the Python POC's pytest
// suite ran (Wyndor Glass textbook LP + real Netlib AFIRO against its
// published optimum), now against the C++ port, proving the port is
// faithful and not just "it compiles."
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "solve.hpp"
#include "simplex.hpp"
#include "mip.hpp"
#include "qp_ipm.hpp"
#include <algorithm>

static int failures = 0;

static void check(bool cond, const std::string& msg) {
    if (!cond) { std::cerr << "  FAIL: " << msg << "\n"; failures++; }
    else std::cout << "  ok:   " << msg << "\n";
}

int main() {
  try {
    std::cout << "[1] AFIRO (real Netlib LP, 27 rows x 32 cols)\n";
    {
        Solution s = solve_mps("sample_problems/afiro.mps", 1e-8, 200000);
        check(s.status == "optimal", "status == optimal");
        check(s.objective > -465.0 && s.objective < -464.0, "objective in published range (-465,-464): got " + std::to_string(s.objective));
        check(s.eps_P < 1e-5 && s.eps_D < 1e-5 && s.eps_G < 1e-5, "independent verifier accepts the solution");
        std::cout << "  engine: " << s.engine_used << "\n";
    }

    std::cout << "[2] BLEND (real Netlib LP)\n";
    {
        Solution s = solve_mps("sample_problems/blend.mps", 1e-8, 200000);
        check(s.status == "optimal", "status == optimal");
        check(s.eps_P < 1e-5 && s.eps_D < 1e-5 && s.eps_G < 1e-5, "independent verifier accepts the solution");
        std::cout << "  engine: " << s.engine_used << "\n";
    }

    std::cout << "[3] SC50A (real Netlib LP)\n";
    {
        Solution s = solve_mps("sample_problems/sc50a.mps", 1e-8, 200000);
        check(s.status == "optimal", "status == optimal");
        check(s.eps_P < 1e-5 && s.eps_D < 1e-5 && s.eps_G < 1e-5, "independent verifier accepts the solution");
        std::cout << "  engine: " << s.engine_used << "\n";
    }

    // [3b-3e] added after tools/audit_lp.cpp's audit found these 4 real
    // Netlib fixtures were sitting in sample_problems/ completely untested
    // by this suite -- 3 of them (kb2, sc105, scagr7) were also being
    // mis-reported as "iteration_limit" at the OLD 200000-iteration default,
    // purely from too tight a budget (see include/solve.hpp's comment on
    // why the default is now 500000). They are genuine, if slower,
    // convergent cases -- worth guarding against a real regression.
    std::cout << "[3b] SC50B (real Netlib LP)\n";
    {
        Solution s = solve_mps("sample_problems/sc50b.mps", 1e-6, 500000);
        check(s.status == "optimal", "status == optimal");
        check(s.eps_P < 1e-5 && s.eps_D < 1e-5 && s.eps_G < 1e-5, "independent verifier accepts the solution");
        std::cout << "  engine: " << s.engine_used << "\n";
    }
    std::cout << "[3c] KB2 (real Netlib LP)\n";
    {
        Solution s = solve_mps("sample_problems/kb2.mps", 1e-6, 500000);
        check(s.status == "optimal", "status == optimal");
        check(s.eps_P < 1e-5 && s.eps_D < 1e-5 && s.eps_G < 1e-5, "independent verifier accepts the solution");
        std::cout << "  engine: " << s.engine_used << "\n";
    }
    std::cout << "[3d] SC105 (real Netlib LP)\n";
    {
        Solution s = solve_mps("sample_problems/sc105.mps", 1e-6, 500000);
        check(s.status == "optimal", "status == optimal");
        check(s.eps_P < 1e-5 && s.eps_D < 1e-5 && s.eps_G < 1e-5, "independent verifier accepts the solution");
        std::cout << "  engine: " << s.engine_used << "\n";
    }
    std::cout << "[3e] SCAGR7 (real Netlib LP)\n";
    {
        Solution s = solve_mps("sample_problems/scagr7.mps", 1e-6, 500000);
        check(s.status == "optimal", "status == optimal");
        check(s.eps_P < 1e-5 && s.eps_D < 1e-5 && s.eps_G < 1e-5, "independent verifier accepts the solution");
        std::cout << "  engine: " << s.engine_used << "\n";
    }
    // [3f-3g] SHARE1B and BEACONFD did not converge with the old fixed-step
    // PDHG (eps plateaued around 1e-3..1e-5 even after 1e6 iterations). With
    // the adaptive step size, primal weight and KKT restarts of
    // include/pdlp_algo.hpp they converge in ~20k-40k iterations -- added at
    // the same 1e-6 tolerance as the others, not a loosened one.
    for (const char* f : {"share1b", "beaconfd"}) {
        std::cout << "[3f] " << f << " (real Netlib LP, PDLP)\n";
        Solution s = solve_mps(std::string("sample_problems/") + f + ".mps", 1e-6, 500000);
        check(s.status == "optimal", "status == optimal");
        check(s.eps_P < 1e-5 && s.eps_D < 1e-5 && s.eps_G < 1e-5, "independent verifier accepts the solution");
    }

    std::cout << "[4] RANGES section (tiny_range.mps: min x1+2x2 s.t. 10<=x1+x2<=14)\n";
    {
        Solution s = solve_mps("sample_problems/tiny_range.mps", 1e-9, 50000);
        check(s.status == "optimal", "status == optimal");
        check(std::fabs(s.objective - 10.0) < 1e-4, "objective == 10 (cheapest feasible sum at the row's lower bound)");
    }

    std::cout << "[5] RHS with omitted vector-name (tiny_rhs_no_vectorname.mps)\n";
    {
        Solution s = solve_mps("sample_problems/tiny_rhs_no_vectorname.mps", 1e-9, 50000);
        check(s.status == "optimal", "status == optimal");
        check(std::fabs(s.objective - (-5.0)) < 1e-6, "objective == -5 (solved by presolve + box shortcut alone)");
    }

    // [6] Dual revised simplex (src/simplex.cpp) on every sample problem,
    // including share1b and beaconfd, which PDLP does not converge on.
    // Reference optima: Netlib / cross-checked with HiGHS 1.15. The simplex
    // returns a vertex, so the verifier gate here is 1e-9, not 1e-5.
    std::cout << "[6] Dual simplex on all sample problems\n";
    {
        struct Case { const char* file; double ref; };
        const Case cases[] = {
            {"afiro", -464.75314286}, {"adlittle", 225494.96316}, {"blend", -30.812149846},
            {"sc50a", -64.575077059}, {"sc50b", -70.0}, {"sc105", -52.202061212},
            {"kb2", -1749.9001299}, {"scagr7", -2331389.8243}, {"israel", -896644.82186},
            {"share1b", -76589.318579}, {"beaconfd", 33592.485807},
            {"tiny_range", 10.0}, {"tiny_rhs_no_vectorname", -5.0},
        };
        SimplexOptions opt;
        for (const auto& c : cases) {
            Solution s = solve_mps_simplex(std::string("sample_problems/") + c.file + ".mps", opt);
            double rel = std::fabs(s.objective - c.ref) / std::max(1.0, std::fabs(c.ref));
            check(s.status == "optimal" && rel < 1e-8 && s.eps_P < 1e-9 && s.eps_D < 1e-9 && s.eps_G < 1e-9,
                  std::string(c.file) + ": " + s.status + ", objective " + std::to_string(s.objective) +
                  " (rel err " + std::to_string(rel) + "), " + std::to_string(s.iterations) + " pivots");
        }
    }

    // [7] Interior-point method on LPs (same sample set, IPM path) and on
    // convex QPs from the Maros-Meszaros set (published optima).
    std::cout << "[7] Interior point: LP samples + Maros-Meszaros QPs\n";
    {
        struct Case { const char* file; double ref; };
        const Case cases[] = {
            {"afiro.mps", -464.75314286}, {"blend.mps", -30.812149846}, {"sc105.mps", -52.202061212},
            {"share1b.mps", -76589.318579}, {"beaconfd.mps", 33592.485807},
            {"qafiro.qps", -1.5907818}, {"hs21.qps", -99.96}, {"hs118.qps", 664.82045}, {"qpcblend.qps", -7.8425409e-03},
        };
        IpmOptions opt;
        for (const auto& c : cases) {
            Solution s = solve_mps_ipm(std::string("sample_problems/") + c.file, opt);
            double rel = std::fabs(s.objective - c.ref) / std::max(1.0, std::fabs(c.ref));
            check(s.status == "optimal" && rel < 1e-6 && s.eps_P < 1e-6 && s.eps_D < 1e-6 && s.eps_G < 1e-6,
                  std::string(c.file) + ": " + s.status + ", objective " + std::to_string(s.objective) +
                  ", " + std::to_string(s.iterations) + " iterations");
        }
    }

    // [8] Branch-and-cut on MIPLIB 3 instances (published optima), 1 and 4
    // threads; the solution must be feasible and integral on the original data.
    std::cout << "[8] Branch-and-cut: MIPLIB 3 samples\n";
    {
        struct Case { const char* file; double ref; };
        const Case cases[] = { {"p0033.mps", 3089}, {"flugpl.mps", 1201500}, {"egout.mps", 568.1007}, {"lseu.mps", 1120} };
        for (int threads : {1, 4}) {
            MipOptions opt;
            opt.threads = threads; opt.verbose = 0; opt.time_limit = 60;
            for (const auto& c : cases) {
                Solution s = solve_mps_mip(std::string("sample_problems/") + c.file, opt);
                double rel = std::fabs(s.objective - c.ref) / std::max(1.0, std::fabs(c.ref));
                check(s.status == "optimal" && rel < 1e-4 && s.eps_P < 1e-6 && s.max_int_violation < 1e-6,
                      std::string(c.file) + " (" + std::to_string(threads) + " threads): " + s.status + ", objective " +
                      std::to_string(s.objective) + ", " + std::to_string(s.nodes) + " nodes");
            }
        }
    }

    // [9] Refinery planning model (tools/gen_refinery.py, 12 crudes x 12
    // periods): the MILP (cargo and unit on/off binaries) and its LP-relaxed
    // convex QP variant (smooth-throughput penalty). References: HiGHS 1.15.
    std::cout << "[9] Refinery planning model (MILP and QP)\n";
    {
        MipOptions mo; mo.threads = 4; mo.verbose = 0; mo.time_limit = 60;
        Solution s = solve_mps_mip("sample_problems/refinery_12x12.mps", mo);
        check(s.status == "optimal" && std::fabs(s.objective - (-9920.606295)) < 1e-3 && s.eps_P < 1e-6 &&
              s.max_int_violation < 1e-6, "refinery MILP: " + s.status + ", margin " + std::to_string(-s.objective));
        IpmOptions io;
        Solution q = solve_mps_ipm("sample_problems/refinery_12x12_qp.mps", io);
        check(q.status == "optimal" && std::fabs(q.objective - (-13195.169642)) < 1e-3 && q.eps_G < 1e-6,
              "refinery QP: " + q.status + ", objective " + std::to_string(q.objective));
    }

    std::cout << "\n" << (failures == 0 ? "ALL CHECKS PASSED" : std::to_string(failures) + " CHECK(S) FAILED") << "\n";
    return failures == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::cerr << "run_checks: uncaught exception: " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "run_checks: uncaught non-std::exception\n";
    return 1;
  }
}
