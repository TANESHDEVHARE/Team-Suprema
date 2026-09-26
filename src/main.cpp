// main.cpp -- the CLI: ./sovereign_solve problem.mps
#include <iostream>
#include <iomanip>
#include <chrono>
#include "solve.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <problem.mps> [tol] [max_iterations]\n";
        return 1;
    }
    std::string path = argv[1];
    double tol = argc > 2 ? std::stod(argv[2]) : 1e-6;
    int max_iter = argc > 3 ? std::stoi(argv[3]) : 200000;

    // Every failure path (a bad file, a CUDA error, anything) now prints a
    // readable message and exits cleanly instead of hitting an uncaught
    // exception -- which on MSVC's Debug runtime shows as a message-less
    // "abort() has been called" dialog. Never demo this without this net.
    try {
        auto t0 = std::chrono::steady_clock::now();
        Solution sol = solve_mps(path, tol, max_iter);
        auto t1 = std::chrono::steady_clock::now();
        double secs = std::chrono::duration<double>(t1 - t0).count();

        std::cout << "Problem:    " << path << "\n";
        std::cout << "Status:     " << sol.status << "\n";
        if (sol.has_solution) {
            std::cout << std::setprecision(10);
            std::cout << "Objective:  " << sol.objective << "\n";
            std::cout << "Engine:     " << sol.engine_used << "\n";
            std::cout << "Iterations: " << sol.iterations << "  (restarts: " << sol.restarts << ")\n";
            std::cout << std::scientific;
            std::cout << "eps_P:      " << sol.eps_P << "\n";
            std::cout << "eps_D:      " << sol.eps_D << "\n";
            std::cout << "eps_G:      " << sol.eps_G << "\n";
            std::cout << std::fixed;
        }
        std::cout << "Time:       " << secs << " s\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "sovereign_solve: error: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "sovereign_solve: error: unknown failure (non-std::exception)\n";
        return 1;
    }
}
