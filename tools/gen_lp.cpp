// gen_lp.cpp -- generates a large, guaranteed feasible & bounded LP as an
// MPS file, to give the CPU/GPU race something with enough parallel work
// per iteration to actually show a GPU advantage. AFIRO/BLEND/SC50A (tens
// of variables) are correctness fixtures, not performance benchmarks --
// PDLP's real GPU speedup only shows up at large nnz (Applegate et al.
// 2021; Lu & Yang 2023), so this generates a *balanced transportation LP*:
//
//   minimize   sum_{i,j} cost_ij * x_ij
//   subject to sum_j x_ij = supply_i   for each supplier i   (S rows)
//              sum_i x_ij = demand_j   for each demander j   (D rows)
//              x_ij >= 0                                     (default MPS bound)
//
// This structure is a standard large-scale LP stress test: n = S*D
// variables, m = S+D rows, exactly 2 nonzeros per column (nnz = 2*S*D) --
// sparse, and *always* feasible and bounded by construction as long as
// sum(supply) == sum(demand) (enforced exactly below, including the
// rounding remainder), so there's no risk of generating a file that PDLP
// can't solve for reasons unrelated to performance.
//
// Usage: gen_lp <S> <D> <outfile.mps> [seed] [cost_max]
//   n = S*D variables, m = S+D rows, nnz = 2*S*D constraint entries
//   Example sizes (n, nnz):
//     gen_lp 300 300 medium.mps    ->   90,000 vars,    180,000 nnz
//     gen_lp 800 800 large.mps     ->  640,000 vars,  1,280,000 nnz
//     gen_lp 1500 1500 huge.mps    -> 2,250,000 vars,  4,500,000 nnz
//
// cost_max (default 100): the top of the uniform cost range [1, cost_max].
// Left at the default this is a well-scaled, "easy" stress case -- good for
// a speed benchmark, not for testing numerical robustness. Pushing cost_max
// much higher (e.g. 1000000) spreads the objective coefficients across
// several orders of magnitude *before* the pipeline's own Ruiz/Pock-
// Chambolle scaling gets a chance to fix it, which is exactly the kind of
// ill-conditioning the problem statement asks the solver to be robust
// against. Use this to generate a deliberately harder audit case, e.g.:
//     gen_lp 200 200 illcond.mps 42 1000000
// then feed illcond.mps to audit_lp alongside the well-scaled cases -- a
// widening CPU-vs-GPU gap here (vs. a tight one on the default-scale case)
// is the expected signature of floating-point reduction-order sensitivity,
// not necessarily a bug (see audit_lp.cpp's header comment).
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>
#include <random>
#include <string>
#include <fstream>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <S suppliers> <D demanders> <outfile.mps> [seed]\n";
        std::cerr << "  writes a balanced transportation LP: n=S*D vars, m=S+D rows, nnz=2*S*D\n";
        return 1;
    }
    long S = std::atol(argv[1]);
    long D = std::atol(argv[2]);
    std::string outfile = argv[3];
    unsigned seed = argc > 4 ? (unsigned)std::atol(argv[4]) : 42u;
    long cost_max = argc > 5 ? std::atol(argv[5]) : 100;
    if (S < 2 || D < 2) { std::cerr << "S and D must be >= 2\n"; return 1; }
    if (cost_max < 2) { std::cerr << "cost_max must be >= 2\n"; return 1; }

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<long> supply_dist(50, 150);
    std::uniform_real_distribution<double> weight_dist(0.5, 1.5);

    // Supplies: random positive integers.
    std::vector<long> supply(S);
    long total_supply = 0;
    for (long i = 0; i < S; ++i) { supply[i] = supply_dist(rng); total_supply += supply[i]; }

    // Demands: random positive weights rescaled to sum EXACTLY to
    // total_supply (fix the rounding remainder on the last entry) so the
    // problem is exactly balanced -- guaranteed feasible, no slack needed.
    std::vector<double> w(D);
    double wsum = 0.0;
    for (long j = 0; j < D; ++j) { w[j] = weight_dist(rng); wsum += w[j]; }
    std::vector<long> demand(D);
    long assigned = 0;
    for (long j = 0; j < D - 1; ++j) {
        long dj = (long)((double)total_supply * w[j] / wsum);
        if (dj < 1) dj = 1;
        demand[j] = dj;
        assigned += dj;
    }
    demand[D - 1] = total_supply - assigned;
    if (demand[D - 1] < 1) { std::cerr << "generated an infeasible split -- try a different seed\n"; return 1; }

    std::ofstream f(outfile, std::ios::binary);
    if (!f) { std::cerr << "cannot open " << outfile << " for writing\n"; return 1; }

    long n = S * D;
    long m = S + D;
    long nnz = 2 * n;
    std::cerr << "generating: S=" << S << " D=" << D << " -> n=" << n
              << " vars, m=" << m << " rows, nnz=" << nnz << " constraint entries\n";

    f << "NAME TRANSPORT_" << S << "x" << D << "\n";
    f << "ROWS\n";
    f << " N  COST\n";
    for (long i = 0; i < S; ++i) f << " E  S" << i << "\n";
    for (long j = 0; j < D; ++j) f << " E  D" << j << "\n";

    f << "COLUMNS\n";
    std::uniform_int_distribution<long> cdist(1, cost_max);
    const long FLUSH_EVERY = 200000;
    std::string chunk;
    chunk.reserve(64 * 1024 * 1024);
    long since_flush = 0;
    for (long i = 0; i < S; ++i) {
        for (long j = 0; j < D; ++j) {
            long cost = cdist(rng);
            chunk += "    x_" + std::to_string(i) + "_" + std::to_string(j)
                   + "  COST  " + std::to_string(cost)
                   + "  S" + std::to_string(i) + "  1\n";
            chunk += "    x_" + std::to_string(i) + "_" + std::to_string(j)
                   + "  D" + std::to_string(j) + "  1\n";
            if (++since_flush >= FLUSH_EVERY) {
                f << chunk;
                chunk.clear();
                since_flush = 0;
            }
        }
    }
    if (!chunk.empty()) f << chunk;

    f << "RHS\n";
    for (long i = 0; i < S; ++i) f << "    RHS  S" << i << "  " << supply[i] << "\n";
    for (long j = 0; j < D; ++j) f << "    RHS  D" << j << "  " << demand[j] << "\n";
    f << "ENDATA\n";
    f.close();

    std::cerr << "wrote " << outfile << "\n";
    return 0;
}
