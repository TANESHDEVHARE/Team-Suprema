// report_io.hpp -- machine-readable results for the CLI and the local UI.
//
//   --json=<file>  full result: problem statistics, status, objective,
//                  verifier residuals, timings, engine, MILP bound/gap, and a
//                  per-variable / per-constraint table (values, bounds, costs,
//                  reduced costs, activities, slacks, duals). Tables are capped
//                  at `max_rows` entries each (flagged "truncated"); the CSV
//                  files always hold everything.
//   --sol=<file>   "name value" lines, then "DUAL" and row duals -- the same
//                  format --check reads, so any result can be re-verified.
//   --csv=<prefix> <prefix>_variables.csv and <prefix>_constraints.csv.
//
// Duals and reduced costs are reported in the model's own objective sense
// (for a "max" model they are the negatives of the minimization-form ones).
#pragma once
#include <string>
#include "solve.hpp"
#include "alloc_stats.hpp"

struct RunInfo {
    std::string model_path;
    std::string mode;              // engine choice as requested
    double total_seconds = 0, read_seconds = 0;
    int threads = 1;
    bool have_resources = false;   // filled by the CLI at the end of the run
    AllocStats alloc;
    ProcessStats proc;
};

void write_result_json(const std::string& out_path, const Solution& s, const RunInfo& info, size_t max_rows = 20000);
void write_solution_file(const std::string& out_path, const Solution& s, const RunInfo& info);
void write_solution_csv(const std::string& prefix, const Solution& s, const RunInfo& info);
