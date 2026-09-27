// test_lu.cpp -- unit test for BasisLU: factorize random sparse bases,
// check FTRAN/BTRAN residuals, then apply a long sequence of Forrest-Tomlin
// column replacements and re-check against the explicitly updated matrix.
// Also checks singular-basis repair.
#include <iostream>
#include <random>
#include <cmath>
#include <algorithm>
#include <string>
#include <cstdio>
#include "basis_lu.hpp"

using Col = std::vector<std::pair<int,double>>;

static int failures = 0;
static std::string sci(double v) { char b[32]; std::snprintf(b, sizeof b, "%.2e", v); return b; }
static void check(bool ok, const std::string& msg) {
    if (!ok) { std::cerr << "  FAIL: " << msg << "\n"; ++failures; }
}

static double residual_ftran(const std::vector<Col>& B, const std::vector<double>& a, const std::vector<double>& x) {
    int m = (int)B.size();
    std::vector<double> r = a;
    for (int s = 0; s < m; ++s) for (auto& e : B[s]) r[e.first] -= e.second * x[s];
    double mx = 0; for (double v : r) mx = std::max(mx, std::fabs(v));
    return mx;
}
static double residual_btran(const std::vector<Col>& B, const std::vector<double>& e, const std::vector<double>& y) {
    double mx = 0;
    for (size_t s = 0; s < B.size(); ++s) {
        double v = -e[s];
        for (auto& en : B[s]) v += en.second * y[en.first];
        mx = std::max(mx, std::fabs(v));
    }
    return mx;
}

static Col random_col(std::mt19937& rng, int m, int nnz) {
    std::uniform_int_distribution<int> ri(0, m - 1);
    std::uniform_real_distribution<double> rv(-10, 10);
    Col c;
    std::vector<char> used(m, 0);
    for (int k = 0; k < nnz; ++k) { int i = ri(rng); if (!used[i]) { used[i] = 1; c.emplace_back(i, rv(rng)); } }
    return c;
}

int main() {
    std::mt19937 rng(12345);
    for (int trial = 0; trial < 20; ++trial) {
        int m = 30 + trial * 15;
        std::vector<Col> B(m);
        // diagonal plus random off-diagonals: nonsingular with high probability
        for (int s = 0; s < m; ++s) {
            B[s] = random_col(rng, m, 3);
            bool has = false;
            for (auto& e : B[s]) if (e.first == s) { e.second += 20; has = true; }
            if (!has) B[s].emplace_back(s, 20.0 + s % 7);
        }
        // shuffle columns so the diagonal is not handed to the LU for free
        std::shuffle(B.begin(), B.end(), rng);

        BasisLU lu;
        auto fn = [&](int s, std::vector<int>& r, std::vector<double>& v) {
            r.clear(); v.clear(); for (auto& e : B[s]) { r.push_back(e.first); v.push_back(e.second); }
        };
        auto rep = lu.factorize(m, fn);
        check(rep.empty(), "nonsingular basis reported singular");

        std::uniform_real_distribution<double> rv(-1, 1);
        std::uniform_int_distribution<int> rs(0, m - 1);
        for (int upd = 0; upd <= 60; ++upd) {
            std::vector<double> a(m), x, e(m), y;
            for (auto& v : a) v = rv(rng);
            for (auto& v : e) v = rv(rng);
            std::vector<double> a0 = a, e0 = e;
            lu.ftran(a, x);
            lu.btran(e, y);
            // Residuals relative to the solution size: random column swaps can
            // make B ill-conditioned, which inflates |x| but not the backward error.
            double xmax = 1, ymax = 1;
            for (double v : x) xmax = std::max(xmax, std::fabs(v));
            for (double v : y) ymax = std::max(ymax, std::fabs(v));
            double rf = residual_ftran(B, a0, x) / xmax, rb = residual_btran(B, e0, y) / ymax;
            if (rf > 1e-10 || rb > 1e-10) {
                // Diagnose: does a fresh factorization of the same B do better?
                BasisLU fresh; fresh.factorize(m, fn);
                std::vector<double> a2 = a0, x2, e2 = e0, y2;
                fresh.ftran(a2, x2); fresh.btran(e2, y2);
                double xm2 = 1, ym2 = 1;
                for (double v : x2) xm2 = std::max(xm2, std::fabs(v));
                for (double v : y2) ym2 = std::max(ym2, std::fabs(v));
                std::cerr << "    fresh factor: ftran " << sci(residual_ftran(B, a0, x2) / xm2)
                          << " btran " << sci(residual_btran(B, e0, y2) / ym2) << "  |x|max " << sci(xm2) << "\n";
                check(false, "trial " + std::to_string(trial) + " after " + std::to_string(upd) +
                             " updates: ftran res " + sci(rf) + " btran res " + sci(rb));
                break;
            }
            // replace a random slot with a new random column
            int slot = rs(rng);
            Col nc = random_col(rng, m, 4);
            // make sure the replacement keeps B nonsingular: add weight on the row
            // the old column's diagonal lived on in the current solve
            std::vector<double> dense(m, 0.0), alpha, spike;
            for (auto& en : nc) dense[en.first] = en.second;
            lu.ftran(dense, alpha, &spike);
            // Accept only swaps a simplex ratio test could produce: pivot not tiny
            // relative to the rest of the transformed column.
            double amax = 0; for (double v : alpha) amax = std::max(amax, std::fabs(v));
            if (std::fabs(alpha[slot]) < 0.1 * amax) continue;
            B[slot] = nc;
            if (!lu.update(slot, spike)) { check(false, "update rejected"); break; }
        }
    }

    // ---- singular basis: two identical columns + one empty column ----
    {
        int m = 5;
        std::vector<Col> B = { {{0,1.0},{1,2.0}}, {{0,1.0},{1,2.0}}, {{2,3.0}}, {}, {{4,1.0},{3,1.0}} };
        BasisLU lu;
        auto fn = [&](int s, std::vector<int>& r, std::vector<double>& v) {
            r.clear(); v.clear(); for (auto& e : B[s]) { r.push_back(e.first); v.push_back(e.second); }
        };
        auto rep = lu.factorize(m, fn);
        check(rep.size() == 2, "singular repair should replace 2 columns, got " + std::to_string(rep.size()));
        for (auto& p : rep) B[p.first] = { {p.second, 1.0} };
        std::vector<double> a = {1, 2, 3, 4, 5}, a0 = a, x;
        lu.ftran(a, x);
        check(residual_ftran(B, a0, x) < 1e-12, "repaired basis solve");
    }

    std::cout << (failures == 0 ? "LU TESTS PASSED" : std::to_string(failures) + " LU TEST(S) FAILED") << "\n";
    return failures == 0 ? 0 : 1;
}
