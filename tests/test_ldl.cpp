// test_ldl.cpp -- unit test for SparseLDL: random sparse quasidefinite
// matrices K = [-(M + a I)  B^T ; B  b I] (M symmetric PSD-ish), which is the
// structure of the interior-point augmented system. Checks solve residuals,
// that no pivot needed regularization, and that the ordering is a permutation.
#include <iostream>
#include <random>
#include <cmath>
#include <map>
#include <string>
#include <cstdio>
#include <algorithm>
#include "sparse_ldl.hpp"

static int failures = 0;
static void check(bool ok, const std::string& msg) { if (!ok) { std::cerr << "  FAIL: " << msg << "\n"; ++failures; } }

int main() {
    std::mt19937 rng(7);
    for (int trial = 0; trial < 15; ++trial) {
        const int n1 = 40 + 30 * trial, n2 = 20 + 25 * trial, n = n1 + n2;
        std::map<std::pair<int,int>, double> ent;   // lower triangle (i >= j)
        std::uniform_int_distribution<int> r1(0, n1 - 1), r2(0, n2 - 1);
        std::uniform_real_distribution<double> rv(-1, 1);
        // M = G^T G style: add random symmetric pairs, then a diagonal shift
        for (int k = 0; k < 2 * n1; ++k) {
            int i = r1(rng), j = r1(rng);
            if (i < j) std::swap(i, j);
            double v = rv(rng);
            ent[{i, j}] -= std::fabs(v) * (i == j ? 1.0 : 0.3);
        }
        for (int i = 0; i < n1; ++i) ent[{i, i}] -= 2.0 + std::fabs(rv(rng));
        for (int k = 0; k < 3 * n2; ++k) ent[{n1 + r2(rng), r1(rng)}] += rv(rng);
        for (int i = 0; i < n2; ++i) ent[{n1 + i, n1 + i}] += 1e-2 + std::fabs(rv(rng));

        SymMatrix K; K.n = n;
        K.colptr.assign(n + 1, 0);
        for (auto& e : ent) K.colptr[e.first.second + 1]++;
        for (int j = 0; j < n; ++j) K.colptr[j + 1] += K.colptr[j];
        K.rowidx.resize(ent.size()); K.val.resize(ent.size());
        std::vector<int> nx(K.colptr.begin(), K.colptr.end() - 1);
        for (auto& e : ent) { int q = nx[e.first.second]++; K.rowidx[q] = e.first.first; K.val[q] = e.second; }

        auto perm = min_degree_order(K);
        std::vector<int> seen(n, 0);
        for (int p : perm) seen[p]++;
        check((int)perm.size() == n && std::all_of(seen.begin(), seen.end(), [](int c) { return c == 1; }),
              "ordering is not a permutation");

        SparseLDL ldl;
        ldl.analyze(K);
        std::vector<signed char> sign(n, 1);
        for (int i = 0; i < n1; ++i) sign[i] = -1;
        int reg = ldl.factor(K, sign, 1e-12);
        check(reg == 0, "trial " + std::to_string(trial) + ": " + std::to_string(reg) + " pivots regularized");

        std::vector<double> x(n), b(n, 0.0);
        for (auto& v : x) v = rv(rng);
        for (auto& e : ent) {
            int i = e.first.first, j = e.first.second;
            b[i] += e.second * x[j];
            if (i != j) b[j] += e.second * x[i];
        }
        ldl.solve(b);
        double err = 0, xm = 0;
        for (int i = 0; i < n; ++i) { err = std::max(err, std::fabs(b[i] - x[i])); xm = std::max(xm, std::fabs(x[i])); }
        char buf[64]; std::snprintf(buf, sizeof buf, "%.2e", err / xm);
        check(err / xm < 1e-9, "trial " + std::to_string(trial) + " solve error " + buf);
    }
    std::cout << (failures == 0 ? "LDL TESTS PASSED" : std::to_string(failures) + " LDL TEST(S) FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
