// mps_reader.cpp -- direct port of the Python POC's mps_reader.py.
// MPS parser (free format, fixed-format fallback): NAME, OBJSENSE, ROWS, COLUMNS (INTORG/INTEND
// recorded in LPProblem::integer), RHS (objective-row RHS = -constant), RANGES, BOUNDS, ENDATA.
#include "mps_reader.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <algorithm>

static const std::set<std::string> SECTION_KEYWORDS = {
    "NAME","OBJSENSE","ROWS","COLUMNS","RHS","RANGES","BOUNDS","ENDATA","QUADOBJ","QMATRIX","QSECTION"
};

static std::vector<std::string> split_ws(const std::string& s) {
    std::istringstream iss(s);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

// Fixed-format MPS: fields at columns 2-3, 5-12, 15-22, 25-36, 40-47, 50-61
// (1-based). Names may contain spaces, so fields are cut by position.
static std::vector<std::string> fixed_fields(const std::string& line) {
    static const int start[6] = {1, 4, 14, 24, 39, 49};
    static const int len[6]   = {2, 8, 8, 12, 8, 12};
    std::vector<std::string> f(6);
    for (int k = 0; k < 6; ++k) {
        if ((int)line.size() <= start[k]) break;
        std::string s = line.substr(start[k], len[k]);
        size_t a = s.find_first_not_of(" \t"), b = s.find_last_not_of(" \t");
        f[k] = a == std::string::npos ? "" : s.substr(a, b - a + 1);
    }
    return f;
}

static bool is_number(const std::string& s) {
    if (s.empty()) return false;
    char* end = nullptr;
    std::strtod(s.c_str(), &end);
    return end && *end == '\0';
}

static LPProblem parse_mps(const std::string& path, bool fixed_format);

// Free format first (the common case, and the only one that allows long
// names); if that fails to parse, the file is re-read as fixed format.
LPProblem read_mps(const std::string& path) {
    try {
        return parse_mps(path, false);
    } catch (const std::exception& free_err) {
        try {
            return parse_mps(path, true);
        } catch (const std::exception&) {
            throw std::runtime_error(std::string("MPS parse failed: ") + free_err.what());
        }
    }
}

static LPProblem parse_mps(const std::string& path, bool fixed_format) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open MPS file: " + path);
    double obj_constant = 0.0;
    std::unordered_set<std::string> integer_cols;

    std::string name = "";
    std::string sense = "min";
    std::string section = "";

    std::vector<std::string> row_order;
    std::unordered_map<std::string,char> row_type;
    std::string obj_row_name;
    bool has_obj_row = false;
    std::unordered_set<std::string> free_rows;

    std::vector<std::string> col_order;
    std::unordered_map<std::string,int> col_index;
    bool int_marker_active = false;

    struct Entry { std::string row, col; double val; };
    std::vector<Entry> entries;
    std::unordered_map<std::string,double> obj_coeffs;

    std::unordered_map<std::string,double> rhs;
    std::unordered_map<std::string,double> ranges;

    std::unordered_map<std::string,double> bound_lo, bound_hi;
    struct QEntry { std::string a, b; double v; bool full; };
    std::vector<QEntry> quad;
    std::unordered_set<std::string> bound_explicit_lo;

    std::string raw_line;
    while (std::getline(f, raw_line)) {
        if (!raw_line.empty() && raw_line.back() == '\r') raw_line.pop_back();
        std::string trimmed_check = raw_line;
        // find first non-space char
        size_t firstc = raw_line.find_first_not_of(" \t");
        if (firstc == std::string::npos) continue;          // blank line
        if (raw_line[firstc] == '*') continue;               // comment

        auto tokens_all = split_ws(raw_line);
        std::string first_token = tokens_all.empty() ? "" : tokens_all[0];
        bool starts_col1 = (raw_line[0] != ' ' && raw_line[0] != '\t');

        if (SECTION_KEYWORDS.count(first_token) && (starts_col1 || first_token == "ENDATA")) {
            if (first_token == "NAME") {
                name = tokens_all.size() > 1 ? tokens_all[1] : "UNNAMED";
                section = "";
                continue;
            }
            if (first_token == "ENDATA") break;
            section = first_token;
            continue;
        }

        std::vector<std::string> tokens = tokens_all;
        if (fixed_format && section != "OBJSENSE") {
            auto fl = fixed_fields(raw_line);
            tokens.clear();
            if (section == "ROWS") tokens = {fl[0], fl[1]};
            else if (section == "BOUNDS") {
                tokens = {fl[0], fl[1].empty() ? std::string("BND") : fl[1], fl[2]};
                if (!fl[3].empty()) tokens.push_back(fl[3]);
            } else {
                // COLUMNS: name, row, value[, row, value]; RHS/RANGES: set, row, value[, ...]
                std::string lead = fl[1];
                if (lead.empty() && section != "COLUMNS") lead = "RHS";
                tokens = {lead, fl[2], fl[3]};
                if (!fl[4].empty()) { tokens.push_back(fl[4]); tokens.push_back(fl[5]); }
            }
        }

        if (section == "OBJSENSE") {
            std::string s = tokens[0];
            for (auto& ch : s) ch = tolower(ch);
            sense = s;
            continue;
        }

        if (section == "ROWS") {
            // A free-format row line has exactly two tokens; more means names
            // contain spaces, i.e. this is really a fixed-format file.
            if (!fixed_format && tokens.size() != 2) throw std::runtime_error("ROWS line is not free format: " + raw_line);
            char rtype = toupper(tokens[0][0]);
            std::string rname = tokens[1];
            row_type[rname] = rtype;
            if (rtype == 'N') {
                if (!has_obj_row) { obj_row_name = rname; has_obj_row = true; }
                else free_rows.insert(rname);
            } else {
                row_order.push_back(rname);
            }
            continue;
        }

        if (section == "COLUMNS") {
            // Markers are usually quoted: MARK0000 'MARKER' 'INTORG'. Check the
            // raw free-format tokens, since fixed-format fields cut them apart.
            auto unq = [](std::string t) { t.erase(std::remove(t.begin(), t.end(), '\''), t.end()); return t; };
            bool is_marker = false;
            for (auto& t : tokens_all) if (unq(t) == "MARKER") is_marker = true;
            if (is_marker) {
                for (auto& t : tokens_all) {
                    if (unq(t) == "INTORG") int_marker_active = true;
                    if (unq(t) == "INTEND") int_marker_active = false;
                }
                continue;
            }
            std::string cname = tokens[0];
            if (!col_index.count(cname)) { col_index[cname] = (int)col_order.size(); col_order.push_back(cname); }
            if (int_marker_active) integer_cols.insert(cname);
            for (size_t i = 1; i + 1 < tokens.size(); i += 2) {
                const std::string& rname = tokens[i];
                double val = std::stod(tokens[i+1]);
                if (rname == obj_row_name) obj_coeffs[cname] += val;
                else if (free_rows.count(rname)) continue;
                else entries.push_back({rname, cname, val});
            }
            continue;
        }

        if (section == "RHS") {
            size_t start = (tokens.size() % 2 == 1) ? 1 : 0;
            for (size_t i = start; i + 1 < tokens.size(); i += 2) {
                // An RHS on the objective row is minus the objective constant.
                if (tokens[i] == obj_row_name) obj_constant = -std::stod(tokens[i+1]);
                else rhs[tokens[i]] = std::stod(tokens[i+1]);
            }
            continue;
        }

        if (section == "RANGES") {
            size_t start = (tokens.size() % 2 == 1) ? 1 : 0;
            for (size_t i = start; i + 1 < tokens.size(); i += 2)
                ranges[tokens[i]] = std::stod(tokens[i+1]);
            continue;
        }

        if (section == "QUADOBJ" || section == "QMATRIX" || section == "QSECTION") {
            // QUADOBJ/QSECTION: lower triangle, each off-diagonal once.
            // QMATRIX: the full symmetric matrix; keep i >= j only.
            if (tokens.size() < 3) continue;
            quad.push_back({tokens[0], tokens[1], std::stod(tokens[2]), section == "QMATRIX"});
            continue;
        }

        if (section == "BOUNDS") {
            std::string btype = tokens[0];
            for (auto& ch : btype) ch = toupper(ch);
            // The bound-set name is optional: "UP BND X 4" and "UP X 4" are both legal.
            bool needs_val = btype == "UP" || btype == "LO" || btype == "FX" || btype == "LI" || btype == "UI";
            bool has_set;
            if (needs_val) has_set = tokens.size() >= 4;
            else if (btype == "BV") has_set = tokens.size() >= 4 ||
                     (tokens.size() == 3 && !(col_index.count(tokens[1]) && is_number(tokens[2])));
            else has_set = tokens.size() >= 3;
            size_t ci = has_set ? 2 : 1;
            if (tokens.size() <= ci) throw std::runtime_error("malformed BOUNDS line: " + raw_line);
            std::string cname = tokens[ci];
            if (!col_index.count(cname)) { col_index[cname] = (int)col_order.size(); col_order.push_back(cname); }
            bool has_val = tokens.size() > ci + 1;
            double val = has_val ? std::stod(tokens[ci + 1]) : 0.0;

            if (btype == "UP") { bound_hi[cname] = val; }
            else if (btype == "LI") { bound_lo[cname] = val; bound_explicit_lo.insert(cname); integer_cols.insert(cname); }
            else if (btype == "UI") { bound_hi[cname] = val; integer_cols.insert(cname); }
            else if (btype == "LO") { bound_lo[cname] = val; bound_explicit_lo.insert(cname); }
            else if (btype == "FX") { bound_lo[cname] = val; bound_hi[cname] = val; bound_explicit_lo.insert(cname); }
            else if (btype == "FR") { bound_lo[cname] = -INF; bound_hi[cname] = INF; bound_explicit_lo.insert(cname); }
            else if (btype == "MI") { bound_lo[cname] = -INF; bound_explicit_lo.insert(cname); }
            else if (btype == "PL") { bound_hi[cname] = INF; }
            else if (btype == "BV") { bound_lo[cname] = 0.0; bound_hi[cname] = 1.0; bound_explicit_lo.insert(cname); integer_cols.insert(cname); }
            else throw std::runtime_error("Unsupported BOUNDS type '" + btype + "' in MPS file");
            continue;
        }
    }

    if (!has_obj_row) throw std::runtime_error("No objective (N) row found in MPS file");

    int n = (int)col_order.size();
    int m = (int)row_order.size();
    std::unordered_map<std::string,int> row_idx;
    for (int i = 0; i < m; ++i) row_idx[row_order[i]] = i;

    SparseMatrix A; A.rows = m; A.cols = n;
    for (auto& e : entries) {
        // Explicit zeros carry no information and break structural rules
        // (a "singleton" row with a zero coefficient -- seen in RENTACAR).
        if (e.val == 0.0) continue;
        auto it = row_idx.find(e.row);
        if (it == row_idx.end()) continue; // stray row, ignore (matches Python's safety check)
        A.row_idx.push_back(it->second);
        A.col_idx.push_back(col_index[e.col]);
        A.val.push_back(e.val);
    }

    std::vector<double> obj(n, 0.0);
    for (auto& kv : obj_coeffs) obj[col_index[kv.first]] = kv.second;

    std::vector<double> row_rhs(m, 0.0);
    for (auto& kv : rhs) { auto it = row_idx.find(kv.first); if (it != row_idx.end()) row_rhs[it->second] = kv.second; }

    std::map<std::string,double> ranges_by_name;
    for (auto& kv : ranges) if (row_idx.count(kv.first)) ranges_by_name[kv.first] = kv.second;

    std::vector<double> lo(n, 0.0), hi(n, INF);
    for (auto& kv : col_index) {
        const std::string& cname = kv.first; int i = kv.second;
        if (bound_lo.count(cname)) lo[i] = bound_lo[cname];
        if (bound_hi.count(cname)) hi[i] = bound_hi[cname];
        if (bound_hi.count(cname) && bound_hi[cname] < 0 && !bound_explicit_lo.count(cname)) lo[i] = -INF;
    }

    std::vector<char> row_types_list(m);
    for (int i = 0; i < m; ++i) row_types_list[i] = row_type[row_order[i]];

    LPProblem p;
    p.name = name.empty() ? "UNNAMED" : name;
    p.sense = sense;
    p.col_names = col_order;
    p.row_names = row_order;
    p.row_types = row_types_list;
    p.row_rhs = row_rhs;
    p.ranges = ranges_by_name;
    p.obj_name = obj_row_name;
    p.obj = obj;
    p.A = A;
    p.lo = lo; p.hi = hi;
    p.obj_constant = obj_constant;
    p.Q.rows = p.Q.cols = n;
    for (const auto& q : quad) {
        auto ia = col_index.find(q.a), ib = col_index.find(q.b);
        if (ia == col_index.end() || ib == col_index.end()) throw std::runtime_error("QUADOBJ refers to unknown column");
        int i = ia->second, j = ib->second;
        if (q.full && i < j) continue;
        if (i < j) std::swap(i, j);
        p.Q.row_idx.push_back(i); p.Q.col_idx.push_back(j); p.Q.val.push_back(q.v);
    }
    p.integer.assign(n, 0);
    for (const auto& cname : integer_cols) p.integer[col_index[cname]] = 1;
    return p;
}
