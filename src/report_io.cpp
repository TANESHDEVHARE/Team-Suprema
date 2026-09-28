#include "report_io.hpp"
#include "mps_reader.hpp"
#include "ranged_lp.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace {

// Everything the tables need, recomputed from the model file and the
// solution (the solver's internal state is not needed).
struct Tables {
    LPProblem p;
    RangedLP r;
    bool flip = false, have_x = false, have_duals = false;
    std::vector<double> x, y, act, z;      // y, z in the model's own sense
};

Tables build(const Solution& s, const RunInfo& info) {
    Tables T;
    T.p = read_mps(info.model_path);
    T.r = to_ranged_lp(T.p);
    const int n = T.r.n(), m = T.r.m();
    T.flip = T.p.sense == "max";
    T.x.assign(n, NAN);
    T.have_x = s.has_solution && !s.x.empty();
    if (T.have_x)
        for (int j = 0; j < n; ++j) { auto it = s.x.find(T.r.col_names[j]); if (it != s.x.end()) T.x[j] = it->second; }
    T.have_duals = T.have_x && !s.is_mip && !s.y.empty();
    std::vector<double> ymin(m, 0.0);
    if (T.have_duals)
        for (int i = 0; i < m; ++i) { auto it = s.y.find(T.r.row_names[i]); if (it != s.y.end()) ymin[i] = it->second; }
    T.act.assign(m, 0.0);
    std::vector<double> aty(n, 0.0), qx(n, 0.0);
    for (int k = 0; k < T.r.A.nnz(); ++k) {
        int i = T.r.A.row_idx[k], j = T.r.A.col_idx[k]; double a = T.r.A.val[k];
        if (T.have_x) T.act[i] += a * T.x[j];
        aty[j] += a * ymin[i];
    }
    for (int k = 0; k < T.r.Q.nnz(); ++k) {
        int i = T.r.Q.row_idx[k], j = T.r.Q.col_idx[k]; double v = T.r.Q.val[k];
        qx[i] += v * T.x[j];
        if (i != j) qx[j] += v * T.x[i];
    }
    T.y.assign(m, NAN); T.z.assign(n, NAN);
    if (T.have_duals) {
        const double sg = T.flip ? -1.0 : 1.0;
        for (int i = 0; i < m; ++i) T.y[i] = sg * ymin[i];
        for (int j = 0; j < n; ++j) T.z[j] = sg * (T.r.c[j] + qx[j] - aty[j]);
    }
    if (!T.have_x) std::fill(T.act.begin(), T.act.end(), NAN);
    return T;
}

std::string num(double v) {
    if (std::isnan(v)) return "null";
    if (std::isinf(v)) return v > 0 ? "\"Infinity\"" : "\"-Infinity\"";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.15g", v);
    return buf;
}
std::string csvnum(double v) {
    if (std::isnan(v)) return "";
    if (std::isinf(v)) return v > 0 ? "inf" : "-inf";
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.15g", v);
    return buf;
}
std::string jstr(const std::string& s) {
    std::string o = "\"";
    for (char ch : s) {
        if (ch == '"' || ch == '\\') { o += '\\'; o += ch; }
        else if ((unsigned char)ch < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", ch); o += b; }
        else o += ch;
    }
    return o + "\"";
}
std::string csvstr(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string o = "\"";
    for (char ch : s) { if (ch == '"') o += '"'; o += ch; }
    return o + "\"";
}

double tol_of(double v) { return 1e-7 * (1.0 + std::fabs(v)); }

// Where a variable sits: at a bound, fixed, or strictly between.
std::string var_position(double x, double lo, double hi) {
    if (std::isnan(x)) return "";
    if (lo == hi) return "fixed";
    if (std::isfinite(lo) && std::fabs(x - lo) <= tol_of(lo)) return "at lower";
    if (std::isfinite(hi) && std::fabs(x - hi) <= tol_of(hi)) return "at upper";
    return "between";
}
std::string row_position(double a, double lo, double hi) {
    if (std::isnan(a)) return "";
    bool at_lo = std::isfinite(lo) && std::fabs(a - lo) <= tol_of(lo);
    bool at_hi = std::isfinite(hi) && std::fabs(a - hi) <= tol_of(hi);
    if (lo == hi) return "equality";
    if (at_lo) return "binding (lower)";
    if (at_hi) return "binding (upper)";
    return "slack";
}
double row_slack(double a, double lo, double hi) {
    if (std::isnan(a)) return NAN;
    double s = INFINITY;
    if (std::isfinite(lo)) s = std::min(s, a - lo);
    if (std::isfinite(hi)) s = std::min(s, hi - a);
    return s;
}
const char* row_type(char t, bool ranged) { return ranged ? "R" : (t == 'L' ? "L" : t == 'G' ? "G" : "E"); }

} // namespace

void write_result_json(const std::string& out_path, const Solution& s, const RunInfo& info, size_t max_rows) {
    Tables T = build(s, info);
    const RangedLP& r = T.r;
    const int n = r.n(), m = r.m();
    int nint = 0, nbin = 0;
    for (int j = 0; j < n; ++j) if (!r.integer.empty() && r.integer[j]) {
        ++nint;
        if (r.l[j] >= 0.0 && r.u[j] <= 1.0) ++nbin;
    }
    int nL = 0, nG = 0, nE = 0, nR = 0;
    for (int i = 0; i < m; ++i) {
        bool ranged = T.p.ranges.count(T.p.row_names[i]) > 0;
        if (ranged) ++nR; else if (T.p.row_types[i] == 'L') ++nL; else if (T.p.row_types[i] == 'G') ++nG; else ++nE;
    }
    std::ofstream f(out_path);
    if (!f) throw std::runtime_error("cannot write " + out_path);
    f << "{\n";
    f << " \"problem\": {\"name\": " << jstr(T.p.name) << ", \"file\": " << jstr(info.model_path)
      << ", \"sense\": " << jstr(T.p.sense.empty() ? "min" : T.p.sense)
      << ", \"rows\": " << m << ", \"cols\": " << n << ", \"nonzeros\": " << r.A.nnz()
      << ", \"integer_cols\": " << nint << ", \"binary_cols\": " << nbin << ", \"quadratic_nonzeros\": " << r.Q.nnz()
      << ", \"row_types\": {\"L\": " << nL << ", \"G\": " << nG << ", \"E\": " << nE << ", \"R\": " << nR << "}"
      << ", \"kind\": " << jstr(nint ? "MILP" : (r.Q.nnz() ? "QP" : "LP")) << "},\n";
    f << " \"status\": " << jstr(s.status) << ",\n";
    f << " \"has_solution\": " << (s.has_solution ? "true" : "false") << ",\n";
    f << " \"objective\": " << (s.has_solution ? num(s.objective) : "null") << ",\n";
    f << " \"engine\": " << jstr(s.engine_used) << ",\n";
    f << " \"mode\": " << jstr(info.mode) << ",\n";
    f << " \"iterations\": " << s.iterations << ",\n";
    f << " \"residuals\": {\"primal\": " << num(s.eps_P) << ", \"dual\": " << num(s.is_mip ? NAN : s.eps_D)
      << ", \"gap\": " << num(s.is_mip ? NAN : s.eps_G) << "},\n";
    f << " \"times\": {\"total\": " << num(info.total_seconds) << ", \"read\": " << num(info.read_seconds)
      << ", \"solve\": " << num(info.total_seconds - info.read_seconds) << "},\n";
    if (info.have_resources) {
        f << " \"resources\": {\"cpu_user_seconds\": " << num(info.proc.cpu_user) << ", \"cpu_kernel_seconds\": " << num(info.proc.cpu_kernel)
          << ", \"peak_working_set_bytes\": " << info.proc.peak_working_set << ", \"peak_private_bytes\": " << info.proc.peak_private
          << ", \"heap_allocations\": " << info.alloc.allocations << ", \"heap_bytes_allocated\": " << info.alloc.bytes_allocated
          << ", \"heap_peak_live_bytes\": " << info.alloc.peak_live_bytes << ", \"heap_live_bytes_at_end\": " << info.alloc.live_bytes << "},\n";
    }
    if (s.is_mip) {
        f << " \"mip\": {\"bound\": " << num(s.mip_bound) << ", \"gap\": " << num(s.has_solution ? s.mip_gap : NAN)
          << ", \"nodes\": " << s.nodes << ", \"root_lp\": " << num(s.root_lp) << ", \"root_after_cuts\": " << num(s.root_after_cuts)
          << ", \"cuts\": " << s.cuts << ", \"max_integrality_violation\": " << num(s.max_int_violation)
          << ", \"threads\": " << info.threads << "},\n";
    }
    // variables
    const size_t nv = std::min((size_t)n, max_rows), nr = std::min((size_t)m, max_rows);
    f << " \"variables_total\": " << n << ", \"variables_truncated\": " << (nv < (size_t)n ? "true" : "false") << ",\n";
    f << " \"variables\": [\n";
    for (size_t j = 0; j < nv; ++j) {
        const double cost = T.p.obj[j];
        f << "  {\"name\": " << jstr(r.col_names[j]) << ", \"value\": " << num(T.x[j])
          << ", \"lower\": " << num(r.l[j]) << ", \"upper\": " << num(r.u[j]) << ", \"cost\": " << num(cost)
          << ", \"reduced_cost\": " << num(T.z[j]) << ", \"integer\": " << ((!r.integer.empty() && r.integer[j]) ? "true" : "false")
          << ", \"position\": " << jstr(var_position(T.x[j], r.l[j], r.u[j])) << "}" << (j + 1 < nv ? ",\n" : "\n");
    }
    f << " ],\n";
    f << " \"constraints_total\": " << m << ", \"constraints_truncated\": " << (nr < (size_t)m ? "true" : "false") << ",\n";
    f << " \"constraints\": [\n";
    for (size_t i = 0; i < nr; ++i) {
        bool ranged = T.p.ranges.count(T.p.row_names[i]) > 0;
        f << "  {\"name\": " << jstr(r.row_names[i]) << ", \"type\": " << jstr(row_type(T.p.row_types[i], ranged))
          << ", \"activity\": " << num(T.act[i]) << ", \"lower\": " << num(r.rL[i]) << ", \"upper\": " << num(r.rU[i])
          << ", \"slack\": " << num(row_slack(T.act[i], r.rL[i], r.rU[i])) << ", \"dual\": " << num(T.y[i])
          << ", \"position\": " << jstr(row_position(T.act[i], r.rL[i], r.rU[i])) << "}" << (i + 1 < nr ? ",\n" : "\n");
    }
    f << " ]\n}\n";
}

void write_solution_file(const std::string& out_path, const Solution& s, const RunInfo& info) {
    Tables T = build(s, info);
    std::ofstream f(out_path);
    if (!f) throw std::runtime_error("cannot write " + out_path);
    char buf[64];
    f << "# Sovereign solver solution\n# model: " << info.model_path << "\n# status: " << s.status << "\n";
    if (s.has_solution) { std::snprintf(buf, sizeof buf, "%.15g", s.objective); f << "# objective: " << buf << "\n"; }
    f << "# format: 'name value' per column; after the line DUAL, 'name dual' per row (model's objective sense)\n";
    if (!T.have_x) return;
    for (int j = 0; j < T.r.n(); ++j) { std::snprintf(buf, sizeof buf, "%.17g", T.x[j]); f << T.r.col_names[j] << " " << buf << "\n"; }
    if (T.have_duals) {
        f << "DUAL\n";
        for (int i = 0; i < T.r.m(); ++i) { std::snprintf(buf, sizeof buf, "%.17g", T.y[i]); f << T.r.row_names[i] << " " << buf << "\n"; }
    }
}

void write_solution_csv(const std::string& prefix, const Solution& s, const RunInfo& info) {
    Tables T = build(s, info);
    const RangedLP& r = T.r;
    {
        std::ofstream f(prefix + "_variables.csv");
        if (!f) throw std::runtime_error("cannot write " + prefix + "_variables.csv");
        f << "name,value,lower,upper,cost,reduced_cost,integer,position\n";
        for (int j = 0; j < r.n(); ++j)
            f << csvstr(r.col_names[j]) << "," << csvnum(T.x[j]) << "," << csvnum(r.l[j]) << "," << csvnum(r.u[j]) << ","
              << csvnum(T.p.obj[j]) << "," << csvnum(T.z[j]) << "," << ((!r.integer.empty() && r.integer[j]) ? 1 : 0) << ","
              << var_position(T.x[j], r.l[j], r.u[j]) << "\n";
    }
    {
        std::ofstream f(prefix + "_constraints.csv");
        if (!f) throw std::runtime_error("cannot write " + prefix + "_constraints.csv");
        f << "name,type,activity,lower,upper,slack,dual,position\n";
        for (int i = 0; i < r.m(); ++i) {
            bool ranged = T.p.ranges.count(T.p.row_names[i]) > 0;
            f << csvstr(r.row_names[i]) << "," << row_type(T.p.row_types[i], ranged) << "," << csvnum(T.act[i]) << ","
              << csvnum(r.rL[i]) << "," << csvnum(r.rU[i]) << "," << csvnum(row_slack(T.act[i], r.rL[i], r.rU[i])) << ","
              << csvnum(T.y[i]) << "," << row_position(T.act[i], r.rL[i], r.rU[i]) << "\n";
        }
    }
}
