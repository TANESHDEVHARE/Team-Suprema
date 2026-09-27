#include "verify.hpp"
#include <cmath>
#include <algorithm>
#include <limits>

static double finite_inf_norm(const std::vector<double>& v) {
    double m = 0.0; bool any = false;
    for (double x : v) if (std::isfinite(x)) { m = std::max(m, std::fabs(x)); any = true; }
    return any ? m : 0.0;
}
static double vec_norm2(const std::vector<double>& a, const std::vector<double>& b) {
    double s = 0.0;
    for (double v : a) s += v * v;
    for (double v : b) s += v * v;
    return std::sqrt(s);
}

KKTReport verify(const RangedLP& ranged, const std::vector<double>& x, const std::vector<double>& y) {
    CSR Acsr = to_csr(ranged.A);
    CSR ATcsr = to_csc_as_transposed_csr(ranged.A);
    return verify(ranged, Acsr, ATcsr, x, y);
}

KKTReport verify(const RangedLP& ranged, const CSR& Acsr, const CSR& ATcsr,
                 const std::vector<double>& x, const std::vector<double>& y) {
    const auto& c = ranged.c; const auto& rL = ranged.rL; const auto& rU = ranged.rU;
    const auto& l = ranged.l; const auto& u = ranged.u;
    int m = ranged.m(), n = ranged.n();

    auto a = matvec(Acsr, x);
    auto Aty = matvec_T(ATcsr, y);
    std::vector<double> z(n);
    for (int j = 0; j < n; ++j) z[j] = c[j] - Aty[j];

    // ---- eps_P ----
    std::vector<double> row_viol(m), bound_viol(n);
    for (int i = 0; i < m; ++i) row_viol[i] = std::max(rL[i] - a[i], 0.0) + std::max(a[i] - rU[i], 0.0);
    for (int j = 0; j < n; ++j) bound_viol[j] = std::max(l[j] - x[j], 0.0) + std::max(x[j] - u[j], 0.0);
    double denom_p = 1.0 + std::max(finite_inf_norm(rL), finite_inf_norm(rU));
    double eps_P = vec_norm2(row_viol, bound_viol) / denom_p;

    // ---- eps_D ----
    std::vector<double> row_sign_viol(m), col_sign_viol(n);
    for (int i = 0; i < m; ++i) {
        double v = 0.0;
        if (!std::isfinite(rL[i]) && y[i] > 0) v += y[i];
        if (!std::isfinite(rU[i]) && y[i] < 0) v += -y[i];
        row_sign_viol[i] = v;
    }
    for (int j = 0; j < n; ++j) {
        double v = 0.0;
        if (!std::isfinite(l[j]) && z[j] > 0) v += z[j];
        if (!std::isfinite(u[j]) && z[j] < 0) v += -z[j];
        col_sign_viol[j] = v;
    }
    double denom_d = 1.0 + finite_inf_norm(c);
    double eps_D = vec_norm2(row_sign_viol, col_sign_viol) / denom_d;

    // ---- dual objective D(y,z) ----
    std::vector<double> y_for_D(m), z_for_D(n);
    for (int i = 0; i < m; ++i) {
        double yv = y[i];
        if (!std::isfinite(rL[i]) && yv > 0) yv = 0.0;
        if (!std::isfinite(rU[i]) && yv < 0) yv = 0.0;
        y_for_D[i] = yv;
    }
    for (int j = 0; j < n; ++j) {
        double zv = z[j];
        if (!std::isfinite(l[j]) && zv > 0) zv = 0.0;
        if (!std::isfinite(u[j]) && zv < 0) zv = 0.0;
        z_for_D[j] = zv;
    }
    double D = 0.0;
    for (int i = 0; i < m; ++i) {
        double y_pos = std::max(y_for_D[i], 0.0), y_neg = std::max(-y_for_D[i], 0.0);
        if (y_pos > 0 && std::isfinite(rL[i])) D += rL[i] * y_pos;
        if (y_neg > 0 && std::isfinite(rU[i])) D += -rU[i] * y_neg;
        if (y_pos > 0 && !std::isfinite(rL[i])) D = -std::numeric_limits<double>::infinity();
        if (y_neg > 0 && !std::isfinite(rU[i])) D = -std::numeric_limits<double>::infinity();
    }
    for (int j = 0; j < n; ++j) {
        double z_pos = std::max(z_for_D[j], 0.0), z_neg = std::max(-z_for_D[j], 0.0);
        if (z_pos > 0 && std::isfinite(l[j])) D += l[j] * z_pos;
        if (z_neg > 0 && std::isfinite(u[j])) D += -u[j] * z_neg;
        if (z_pos > 0 && !std::isfinite(l[j])) D = -std::numeric_limits<double>::infinity();
        if (z_neg > 0 && !std::isfinite(u[j])) D = -std::numeric_limits<double>::infinity();
    }

    double cx = 0.0; for (int j = 0; j < n; ++j) cx += c[j] * x[j];
    double objective = cx + ranged.obj_offset;

    double eps_G;
    if (!std::isfinite(D)) eps_G = std::numeric_limits<double>::infinity();
    else eps_G = std::fabs(cx - D) / (1.0 + std::fabs(cx) + std::fabs(D));

    KKTReport rep;
    rep.eps_P = eps_P; rep.eps_D = eps_D; rep.eps_G = eps_G;
    rep.objective = objective;
    rep.dual_objective = std::isfinite(D) ? D + ranged.obj_offset : D;
    rep.max_row_violation = row_viol.empty() ? 0.0 : *std::max_element(row_viol.begin(), row_viol.end());
    rep.max_bound_violation = bound_viol.empty() ? 0.0 : *std::max_element(bound_viol.begin(), bound_viol.end());
    return rep;
}
