"""Numerical robustness demonstration.

    python tools/robustness.py [--classes ill,degenerate,infeasible,weak,large] [--highs --python PY]
                               [--time 120] [--large-time 600] [--textbook-time 30] [--out FILE]
    (or: sovereign robust ...)

Models live in bench_data/robustness/<class>/ (see ROBUSTNESS.md for their
sources). Every model is solved three ways:

  ours      the sovereign solver, default settings
  simple    the same solver with its robustness machinery switched off -- the
            "simpler implementation": textbook simplex for LPs (Dantzig pricing,
            plain ratio test, no bound flipping, no perturbation, no anti-cycling,
            no scaling), and branch-and-bound without cuts for MILPs
  highs     HiGHS 1.x (optional, --highs), its answer run through our verifier

and judged against a reference optimum (HiGHS's, or the original model's for
the rescaled ones; infeasible models must be proven infeasible):
  correct   status and objective match the reference (1e-6 relative for LP,
            the gap tolerance for MILP)
  unreferenced  finished, but no reference optimum is known (not counted as correct)
  wrong     claims optimal/infeasible but contradicts the reference
  failed    no answer (time limit, numerical failure, error)
Each class also reports what makes its models hard: coefficient range
(ill-conditioned), share of degenerate pivots (degenerate), root LP gap
(weak relaxations), size (large).
"""
import argparse, json, math, os, re, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver_runner as sr
import miplib2017 as m17

DATA = os.path.join(sr.ROOT, "bench_data", "robustness")
CLASSES = {
    "ill": "Ill-conditioned constraint matrices",
    "degenerate": "Degeneracy",
    "infeasible": "Infeasibility detection",
    "weak": "Weak LP relaxations (MILP)",
    "large": "Large scale",
}
WHY = {
    "ill": "Netlib models with rows and columns rescaled by 10^-4..10^4 (optimum unchanged) and PDE-derived LPs; coefficient ratios up to 1e23",
    "degenerate": "Netlib's degenerate / stalling models and large degenerate LPs (quadratic assignment bounds, set covering)",
    "infeasible": "the Netlib infeasible set: the only correct answer is a proof of infeasibility",
    "weak": "MIPLIB fixed-charge, network design and big-M models whose LP bound is far from the integer optimum",
    "large": "Kennington LPs and a 1,000,000-variable transportation LP",
}


def coef_ratio(path):
    """max|a_ij| / min|a_ij| over the COLUMNS section (quick scan)."""
    lo, hi, sec = math.inf, 0.0, None
    with open(path, errors="replace") as f:
        for line in f:
            if not line.strip() or line[0] == "*":
                continue
            if not line[0].isspace():
                sec = line.split()[0]
                continue
            if sec != "COLUMNS" or "MARKER" in line:
                continue
            p = line.split()
            for k in range(2, len(p), 2):
                try:
                    v = abs(float(p[k]))
                except ValueError:
                    continue
                if v > 0:
                    lo, hi = min(lo, v), max(hi, v)
    return hi / lo if hi > 0 and lo < math.inf else None


def run_ours(exe, path, flags, tlim, mem_mb, out_dir, tag):
    js = os.path.join(out_dir, tag + ".json")
    if os.path.isfile(js):
        os.remove(js)
    args = [exe, "-v", "--mem-limit=%d" % mem_mb, "--json=" + js] + flags + [path, str(tlim)]
    log_path = os.path.join(out_dir, tag + ".log")
    t0 = time.time()
    with open(log_path, "w") as log:
        p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        peak, why = m17.watch(p, (mem_mb + 1024) * 1048576, t0 + tlim + 120)
    text = open(log_path, errors="replace").read()
    r = {"time": time.time() - t0, "peak_mem": peak}
    if why or "memory_limit" in text:
        r["status"] = why or "memory_limit"
        return r
    m = re.search(r"Status:\s+(\S+)", text)
    r["status"] = m.group(1) if m else "error"
    if os.path.isfile(js):
        R = json.load(open(js))
        r.update(status=R["status"], obj=R["objective"] if R["has_solution"] else None, time=R["times"]["solve"],
                 eps=max(v for v in R["residuals"].values() if isinstance(v, (int, float))) if R["has_solution"] else None,
                 engine=R["engine"], kind=R["problem"]["kind"], rows=R["problem"]["rows"], cols=R["problem"]["cols"],
                 nnz=R["problem"]["nonzeros"])
        if R.get("mip"):
            r.update(root_lp=R["mip"]["root_lp"], root_after_cuts=R["mip"]["root_after_cuts"], nodes=R["mip"]["nodes"],
                     bound=R["mip"]["bound"], gap=R["mip"]["gap"])
    d = [(int(a), int(b)) for a, b in re.findall(r"simplex: \S+\s+its (\d+) .*?degenerate (\d+)", text)]
    if d:
        its, deg = sum(a for a, _ in d), sum(b for _, b in d)
        r["degenerate_pct"] = 100.0 * deg / its if its else 0.0
    return r


def judge(r, ref, kind, gap_tol):
    """correct / wrong / failed against the reference (kind: 'opt' value or 'inf')."""
    if not r:
        return "-"
    st = m17.norm_status(r.get("status"))
    if ref is None:                                   # nothing to compare with: never counted as correct
        return "unreferenced" if st in ("optimal", "infeasible") else "failed"
    rkind, val = ref
    if st == "infeasible":
        return "correct" if rkind == "inf" else "wrong"
    if st == "optimal":
        if rkind == "inf":
            return "wrong"
        obj = r.get("obj")
        if obj is None or val is None:
            return "failed"
        tol = (gap_tol * 2 if kind == "MILP" else 1e-6) * max(1.0, abs(val))
        return "correct" if abs(obj - val) <= tol else "wrong"
    return "failed"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe")
    ap.add_argument("--classes", default="ill,degenerate,infeasible,weak,large")
    ap.add_argument("--time", type=float, default=120, help="time limit per model (s)")
    ap.add_argument("--large-time", type=float, default=600, help="time limit for the large class (s)")
    ap.add_argument("--textbook-time", type=float, default=30, help="time limit for the simpler implementation (s)")
    ap.add_argument("--milp-time", type=float, default=60, help="time limit for MILPs (s)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gap", type=float, default=1e-4)
    ap.add_argument("--mem-mb", type=int, default=6000)
    ap.add_argument("--highs", action="store_true")
    ap.add_argument("--python", default=sys.executable, help="interpreter with highspy")
    ap.add_argument("--refs", default=os.path.join(DATA, "references.json"), help="reference optima (written with --highs)")
    ap.add_argument("--only", help="comma list of model names")
    ap.add_argument("--out", default=os.path.join(sr.ROOT, "bench_results", "robustness.json"))
    ap.add_argument("--rejudge", action="store_true", help="re-score the saved --out results against --refs without solving")
    a = ap.parse_args()
    refs = json.load(open(a.refs)) if os.path.isfile(a.refs) else {}
    prev = json.load(open(a.out)) if os.path.isfile(a.out) else None
    if a.rejudge:
        if not prev:
            sys.exit("no saved results in " + a.out)
        for rec in prev["results"]:
            c = rec["class"]
            ref = ["inf", None] if c == "infeasible" else refs.get(rec["name"])
            rec["ref"] = ref
            kind = rec["ours"].get("kind", "MILP" if c == "weak" else "LP")
            for s in ("ours", "simple", "highs", "highs_nodrop"):
                if s in rec:
                    rec[s]["verdict"] = judge(rec[s], tuple(ref) if ref else None, kind, a.gap)
        classes = [c for c in CLASSES if any(r["class"] == c for r in prev["results"])]
        print_summary(prev["results"], classes, prev, a.out)
        return
    exe = sr.find_exe(a.exe)
    classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    only = set(a.only.split(",")) if a.only else None
    ratios = {}
    rf = os.path.join(DATA, "ill", "coefficient_ranges.txt")
    if os.path.isfile(rf):
        for line in open(rf):
            m = re.match(r"(\S+): coefficient range .* \(ratio (\S+)\)", line)
            if m:
                ratios[m.group(1) + "_s4"] = float(m.group(2))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    results = []
    for c in classes:
        folder = os.path.join(DATA, c)
        models = sorted(f[:-4] for f in os.listdir(folder) if f.endswith(".mps"))
        if only:
            models = [m for m in models if m in only]
        print("\n== %s (%d models): %s" % (CLASSES[c], len(models), WHY[c]), flush=True)
        for name in models:
            path = os.path.join(folder, name + ".mps")
            out_dir = os.path.join(DATA, "runs", c, name)
            os.makedirs(out_dir, exist_ok=True)
            tl = a.large_time if c == "large" else (a.milp_time if c == "weak" else a.time)
            rec = {"class": c, "name": name}
            rec["ours"] = run_ours(exe, path, ["--threads=%d" % a.threads, "--gap=%g" % a.gap], tl, a.mem_mb, out_dir, "ours")
            kind = rec["ours"].get("kind", "MILP" if c == "weak" else "LP")
            if kind == "MILP":
                simple = ["--mip", "--no-cuts", "--threads=%d" % a.threads, "--gap=%g" % a.gap]
                stl = tl
            else:
                simple = ["--simplex", "--textbook"]
                stl = min(tl, a.textbook_time)
            rec["simple"] = run_ours(exe, path, simple, stl, a.mem_mb, out_dir, "simple")
            rec["simple"]["time_limit"] = stl
            if a.highs:
                ns = argparse.Namespace(time=tl, gap=a.gap, mem_mb=a.mem_mb)
                h = m17.run_highs(a.python, exe, path, ns, out_dir)
                h["status"] = m17.norm_status(h.get("status"))
                rec["highs"] = h
                if c == "ill":                                # fairness: HiGHS without its small-coefficient drop
                    h2 = m17.run_highs(a.python, exe, path, ns, out_dir, small=1e-12, tag="highs_nodrop")
                    h2["status"] = m17.norm_status(h2.get("status"))
                    rec["highs_nodrop"] = h2
                if name not in refs and not name.endswith("_s4") and c != "infeasible":     # rescaled models keep the original's optimum
                    if h["status"] == "optimal" and h.get("obj") is not None:
                        refs[name] = ["opt", h["obj"]]
                    elif h["status"] == "infeasible":
                        refs[name] = ["inf", None]
            # the infeasible set shares names with Netlib models (greenbea): always "must be proven infeasible"
            ref = ["inf", None] if c == "infeasible" else refs.get(name)
            rec["ref"] = ref
            for s in ("ours", "simple", "highs", "highs_nodrop"):
                if s in rec:
                    rec[s]["verdict"] = judge(rec[s], tuple(ref) if ref else None, kind, a.gap)
            if c == "ill":
                rec["ratio"] = ratios.get(name) or coef_ratio(path)
            if kind == "MILP" and ref and ref[0] == "opt" and rec["ours"].get("root_lp") is not None:
                v = ref[1]
                rec["root_gap_pct"] = 100 * abs(v - rec["ours"]["root_lp"]) / max(1.0, abs(v))
                ra = rec["ours"].get("root_after_cuts")
                if ra is not None and v != rec["ours"]["root_lp"]:
                    rec["closed_pct"] = 100 * (ra - rec["ours"]["root_lp"]) / (v - rec["ours"]["root_lp"])
            results.append(rec)
            fmt = lambda s: "" if s not in rec else "%-8s %-15s %7.1fs" % (rec[s]["verdict"], m17.norm_status(rec[s].get("status"))[:15], min(rec[s].get("time") or 0, 9999))
            extra = ""
            if c == "ill" and rec.get("ratio"):
                extra = "ratio %.0e" % rec["ratio"]
            elif rec["ours"].get("degenerate_pct") is not None and c == "degenerate":
                extra = "degenerate %.0f%%" % rec["ours"]["degenerate_pct"]
            elif rec.get("root_gap_pct") is not None:
                extra = "root gap %.1f%%" % rec["root_gap_pct"]
            elif c == "large" and rec["ours"].get("cols"):
                extra = "%s cols" % format(rec["ours"]["cols"], ",")
            print("  %-22s %-18s | ours %s | simple %s | highs %s%s" % (name, extra, fmt("ours"), fmt("simple"), fmt("highs"),
                  (" | highs(drop off) " + fmt("highs_nodrop")) if "highs_nodrop" in rec else ""), flush=True)
            json.dump({"settings": vars(a), "results": results}, open(a.out, "w"), indent=1)
    if a.highs:
        json.dump(refs, open(a.refs, "w"), indent=1)
    if only and prev:                                 # --only re-runs a few models: merge into the saved results
        fresh = {(r["class"], r["name"]): r for r in results}
        results = [fresh.pop((r["class"], r["name"]), r) for r in prev["results"]] + list(fresh.values())
        classes = [c for c in CLASSES if any(r["class"] == c for r in results)]
    print_summary(results, classes, {"settings": vars(a)}, a.out)


def print_summary(results, classes, doc, out):
    print("\n== Summary (correct / models)")
    summary = {}
    for c in classes:
        rows = [r for r in results if r["class"] == c]
        line = {}
        for s in ("ours", "simple", "highs", "highs_nodrop"):
            got = [r[s]["verdict"] for r in rows if s in r]
            if got:
                line[s] = {"correct": got.count("correct"), "wrong": got.count("wrong"), "failed": got.count("failed"), "n": len(got)}
        summary[c] = line
        print("  %-38s %s" % (CLASSES[c], "   ".join("%s %d/%d (wrong %d)" % (s, v["correct"], v["n"], v["wrong"]) for s, v in line.items())))
    doc["summary"] = summary
    doc["results"] = results
    json.dump(doc, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
