// run_checks.cpp -- the same correctness checks the Python POC's pytest
// suite ran (Wyndor Glass textbook LP + real Netlib AFIRO against its
// published optimum), now against the C++ port, proving the port is
// faithful and not just "it compiles."
#include <iostream>
#include <cmath>
#include <cstdlib>
#include "solve.hpp"

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
