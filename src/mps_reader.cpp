// mps_reader.cpp -- direct port of the Python POC's mps_reader.py.
// Free-format MPS parser: NAME, OBJSENSE, ROWS, COLUMNS (INTORG/INTEND
// noted, integrality not enforced -- LP only), RHS, RANGES, BOUNDS, ENDATA.
#include "mps_reader.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <stdexcept>
#include <cmath>
#include <limits>

static const std::set<std::string> SECTION_KEYWORDS = {
    "NAME","OBJSENSE","ROWS","COLUMNS","RHS","RANGES","BOUNDS","ENDATA"
};

static std::vector<std::string> split_ws(const std::string& s) {
    std::istringstream iss(s);
    std::vector<std::string> out;
    std::string tok;
    while (iss >> tok) out.push_back(tok);
    return out;
}

LPProblem read_mps(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open MPS file: " + path);

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

        if (section == "OBJSENSE") {
            std::string s = tokens[0];
            for (auto& ch : s) ch = tolower(ch);
            sense = s;
            continue;
        }

        if (section == "ROWS") {
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
            bool is_marker = false;
            for (auto& t : tokens) if (t == "MARKER") is_marker = true;
            if (is_marker) {
                for (auto& t : tokens) {
                    if (t == "INTORG") int_marker_active = true;
                    if (t == "INTEND") int_marker_active = false;
                }
                continue;
            }
            std::string cname = tokens[0];
            if (!col_index.count(cname)) { col_index[cname] = (int)col_order.size(); col_order.push_back(cname); }
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
            for (size_t i = start; i + 1 < tokens.size(); i += 2)
                rhs[tokens[i]] = std::stod(tokens[i+1]);
            continue;
        }

        if (section == "RANGES") {
            size_t start = (tokens.size() % 2 == 1) ? 1 : 0;
            for (size_t i = start; i + 1 < tokens.size(); i += 2)
                ranges[tokens[i]] = std::stod(tokens[i+1]);
            continue;
        }

        if (section == "BOUNDS") {
            std::string btype = tokens[0];
            for (auto& ch : btype) ch = toupper(ch);
            std::string cname = tokens[2];
            if (!col_index.count(cname)) { col_index[cname] = (int)col_order.size(); col_order.push_back(cname); }
            bool has_val = tokens.size() > 3;
            double val = has_val ? std::stod(tokens[3]) : 0.0;

            if (btype == "UP") { bound_hi[cname] = val; }
            else if (btype == "LO") { bound_lo[cname] = val; bound_explicit_lo.insert(cname); }
            else if (btype == "FX") { bound_lo[cname] = val; bound_hi[cname] = val; bound_explicit_lo.insert(cname); }
            else if (btype == "FR") { bound_lo[cname] = -INF; bound_hi[cname] = INF; bound_explicit_lo.insert(cname); }
            else if (btype == "MI") { bound_lo[cname] = -INF; bound_explicit_lo.insert(cname); }
            else if (btype == "PL") { bound_hi[cname] = INF; }
            else if (btype == "BV") { bound_lo[cname] = 0.0; bound_hi[cname] = 1.0; bound_explicit_lo.insert(cname); }
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
    return p;
}
