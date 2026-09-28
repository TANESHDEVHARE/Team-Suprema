"""MIPLIB 2017 benchmark: the sovereign solver against HiGHS.

    python tools/miplib2017.py [--status easy] [--max-nnz 100000] [--time 60] [--threads 4]
                               [--mem-mb 3000] [--highs] [--names a,b,c] [--out bench_results/miplib2017.json]

Instance list, metadata and reference values come from miplib.zib.de (the
240-instance benchmark set, its status/size table, and miplib2017-v35.solu).
Each run is a separate process under a memory watchdog; our solver also gets
--mem-limit (a heap ceiling). A result is judged against the published
reference:
  OK     proven optimal and within the gap tolerance of the known optimum,
         or proven infeasible when MIPLIB says infeasible
  LIMIT  stopped at the time limit (final gap reported), never contradicting
         the reference
  WRONG  a proven claim that contradicts the reference (an "optimum" off the
         known optimum, a bound above it, a solution better than it)
  FAIL   memory limit, crash, or no result
Summary metrics follow MIPLIB practice: number solved and the shifted
geometric mean of the solve time (shift 10 s, unsolved counted at the limit).
"""
import argparse, gzip, json, math, os, re, subprocess, sys, time, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver_runner as sr

BASE = "https://miplib.zib.de"
DATA = os.path.join(sr.ROOT, "bench_data", "miplib2017")


# ------------------------------------------------------------------ data
def fetch(url, dest):
    with urllib.request.urlopen(url, timeout=300) as r, open(dest + ".part", "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    os.replace(dest + ".part", dest)


def load_meta():
    os.makedirs(DATA, exist_ok=True)
    meta_json = os.path.join(DATA, "benchmark_meta.json")
    if not os.path.isfile(meta_json):
        page = os.path.join(DATA, "tag_benchmark.html")
        if not os.path.isfile(page):
            fetch(BASE + "/tag_benchmark.html", page)
        import html
        t = open(page, encoding="utf-8", errors="replace").read()
        hdr, rows = None, []
        for r in re.findall(r"<tr[^>]*>(.*?)</tr>", t, re.S):
            cells = [html.unescape(re.sub("<[^>]+>", "", c)).strip() for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S)]
            if cells and cells[0] == "Instance":
                hdr = cells
                continue
            if not hdr or len(cells) != len(hdr):
                continue
            d = dict(zip(hdr, cells))
            for k in ("Variables", "Binaries", "Integers", "Continuous", "Constraints", "Nonz."):
                d[k] = int(d[k])
            rows.append(d)
        json.dump(rows, open(meta_json, "w"), indent=0)
    solu = os.path.join(DATA, "miplib2017-v35.solu")
    if not os.path.isfile(solu):
        fetch(BASE + "/downloads/miplib2017-v35.solu", solu)
    ref = {}
    for line in open(solu):
        p = line.split()
        if len(p) >= 2:
            kind = p[0].strip("=")
            ref[p[1]] = (kind, float(p[2]) if len(p) > 2 else None)
    return json.load(open(meta_json)), ref


def instance_path(name):
    mps = os.path.join(DATA, name + ".mps")
    if not os.path.isfile(mps):
        gz = mps + ".gz"
        if not os.path.isfile(gz):
            fetch("%s/WebData/instances/%s.mps.gz" % (BASE, name), gz)
        with gzip.open(gz, "rb") as fi, open(mps + ".part", "wb") as fo:
            while True:
                chunk = fi.read(1 << 22)
                if not chunk:
                    break
                fo.write(chunk)
        os.replace(mps + ".part", mps)
        os.remove(gz)
    return mps


# ------------------------------------------------------------------ runs
def watch(p, mem_bytes, deadline):
    """Wait for p; kill it past the deadline or the memory ceiling."""
    h = sr._open(p.pid)
    peak, why = 0, None
    try:
        while p.poll() is None:
            s = sr._sample(h) if h else None
            if s:
                peak = max(peak, s[1])
                if mem_bytes and max(s[1], s[2]) > mem_bytes:
                    why = "memory_limit"
            if time.time() > deadline:
                why = why or "hard_timeout"
            if why:
                p.kill()
                break
            time.sleep(0.2)
    finally:
        if h:
            sr._close(h)
    p.wait()
    return peak, why


def run_ours(exe, path, a, out_dir):
    js = os.path.join(out_dir, "result.json")
    if os.path.isfile(js):
        os.remove(js)
    args = [exe, "--mip", "--threads=%d" % a.threads, "--gap=%g" % a.gap, "--mem-limit=%d" % a.mem_mb, "--json=" + js, path, str(a.time)]
    t0 = time.time()
    with open(os.path.join(out_dir, "ours.log"), "w") as log:
        p = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT)
        peak, why = watch(p, (a.mem_mb + 1024) * 1048576, t0 + a.time + 120)
    wall = time.time() - t0
    text = open(os.path.join(out_dir, "ours.log"), errors="replace").read()
    if why or "memory_limit" in text:
        return {"status": why or "memory_limit", "time": wall, "peak_mem": peak}
    if not os.path.isfile(js):
        m = re.search(r"error: (.*)", text)
        return {"status": "error", "error": (m.group(1) if m else text[-300:]).strip(), "time": wall, "peak_mem": peak}
    R = json.load(open(js))
    mip = R.get("mip") or {}
    return {"status": R["status"], "obj": R["objective"] if R["has_solution"] else None, "bound": mip.get("bound"),
            "gap": mip.get("gap"), "nodes": mip.get("nodes"), "time": R["times"]["solve"], "wall": wall,
            "peak_mem": (R.get("resources") or {}).get("peak_working_set_bytes", peak),
            "eps_P": R["residuals"]["primal"], "int_viol": mip.get("max_integrality_violation"),
            "root_lp": mip.get("root_lp"), "root_after_cuts": mip.get("root_after_cuts")}


HIGHS_CHILD = r"""
import json, sys, time, highspy
path, tl, sol = sys.argv[1], float(sys.argv[2]), sys.argv[3]
h = highspy.Highs(); h.setOptionValue("output_flag", False); h.setOptionValue("time_limit", tl)
h.setOptionValue("mip_rel_gap", float(sys.argv[4]))
if len(sys.argv) > 5: h.setOptionValue("small_matrix_value", float(sys.argv[5]))   # default 1e-9 drops tiny coefficients
h.readModel(path); t0 = time.time(); h.run(); t = time.time() - t0
st = h.modelStatusToString(h.getModelStatus()); i = h.getInfo()
has = i.primal_solution_status == 2
if has:
    lp, s = h.getLp(), h.getSolution()
    with open(sol, "w") as f:
        for nm, v in zip(lp.col_names_, s.col_value): f.write("%s %r\n" % (nm, v))
print(json.dumps({"status": st, "obj": i.objective_function_value if has else None, "bound": i.mip_dual_bound,
                  "gap": i.mip_gap, "nodes": i.mip_node_count, "time": t, "has": has}))
"""


def run_highs(py, exe, path, a, out_dir, small=None, tag="highs"):
    sol = os.path.join(out_dir, tag + ".sol")
    if os.path.isfile(sol):
        os.remove(sol)
    t0 = time.time()
    out_path = os.path.join(out_dir, tag + ".out")
    with open(out_path, "w") as out:
        extra = [repr(float(small))] if small is not None else []
        p = subprocess.Popen([py, "-c", HIGHS_CHILD, path, str(a.time), sol, str(a.gap)] + extra, stdout=out, stderr=subprocess.STDOUT)
        peak, why = watch(p, (a.mem_mb + 1024) * 1048576, t0 + a.time + 120)
    if why:
        return {"status": why, "time": time.time() - t0, "peak_mem": peak}
    text = open(out_path, errors="replace").read().strip().splitlines()
    try:
        R = json.loads(text[-1])
    except Exception:
        return {"status": "error", "error": "\n".join(text[-3:]), "time": time.time() - t0, "peak_mem": peak}
    R["peak_mem"] = peak
    if R.get("has") and os.path.isfile(sol):                 # HiGHS's solution through our verifier
        try:
            ck = sr.check(exe, path, sol)
            R["eps_P"], R["int_viol"] = ck["eps_P"], ck["int_violation"]
        except Exception as e:
            R["check_error"] = str(e)[:200]
    return R


# ------------------------------------------------------------------ judging
def norm_status(s):
    s = (s or "").lower()
    return {"optimal": "optimal", "infeasible": "infeasible", "time limit reached": "time_limit",
            "time_limit": "time_limit", "node_limit": "node_limit"}.get(s, s)


def verdict(r, ref, gap_tol):
    st = norm_status(r.get("status"))
    kind, val = ref if ref else ("unkn", None)
    tol = lambda v: max(gap_tol, 1e-6) * max(1.0, abs(v)) + 1e-6
    obj, bnd = r.get("obj"), r.get("bound")
    if st == "optimal":
        if kind == "inf":
            return "WRONG"
        if kind == "opt" and obj is not None and abs(obj - val) > 2 * tol(val):
            return "WRONG"
        return "OK"
    if st == "infeasible":
        return "OK" if kind == "inf" else ("WRONG" if kind in ("opt", "best") else "OK")
    if st in ("time_limit", "node_limit"):
        if kind == "opt":
            if obj is not None and obj < val - tol(val):
                return "WRONG"                     # "better" than the proven optimum: infeasible point
            if bnd is not None and math.isfinite(bnd) and bnd > val + tol(val):
                return "WRONG"                     # claimed bound cuts off the optimum
        return "LIMIT"
    return "FAIL"


def final_gap(r):
    obj, bnd = r.get("obj"), r.get("bound")
    if obj is None or bnd is None or not math.isfinite(bnd):
        return None
    return abs(obj - bnd) / max(1e-10, abs(obj), abs(bnd))


def sgm(times, shift=10.0):
    return math.exp(sum(math.log(t + shift) for t in times) / len(times)) - shift if times else None


def summarize(results, solvers, tlim):
    out = {}
    for s in solvers:
        rows = [r[s] for r in results if s in r]
        v = [x.get("verdict") for x in rows]
        solved = [x for x in rows if x.get("verdict") == "OK"]
        times = [min(x.get("time") or tlim, tlim) if x.get("verdict") == "OK" else tlim for x in rows]
        gaps = [final_gap(x) for x in rows if x.get("verdict") == "LIMIT" and final_gap(x) is not None]
        out[s] = {"n": len(rows), "OK": v.count("OK"), "LIMIT": v.count("LIMIT"), "WRONG": v.count("WRONG"),
                  "FAIL": v.count("FAIL"), "sgm_time": sgm(times), "mean_gap_at_limit": (sum(gaps) / len(gaps)) if gaps else None,
                  "with_incumbent_at_limit": sum(1 for x in rows if x.get("verdict") == "LIMIT" and x.get("obj") is not None)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe")
    ap.add_argument("--status", default="easy", help="MIPLIB status filter (easy, hard, open, any)")
    ap.add_argument("--max-nnz", type=int, default=100000)
    ap.add_argument("--names", help="comma list; overrides the filters")
    ap.add_argument("--dir", help="run every .mps in this folder instead (e.g. MIPLIB 3), judged with --refs-json")
    ap.add_argument("--refs-json", help="{name: [\"opt\"|\"inf\", value]} reference values for --dir")
    ap.add_argument("--limit", type=int, default=0, help="at most this many instances (smallest first)")
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gap", type=float, default=1e-4, help="relative gap for both solvers")
    ap.add_argument("--mem-mb", type=int, default=3000)
    ap.add_argument("--highs", action="store_true", help="also run HiGHS (needs highspy in --python)")
    ap.add_argument("--python", default=sys.executable, help="interpreter with highspy for the HiGHS runs")
    ap.add_argument("--skip-ours", action="store_true")
    ap.add_argument("--out", default=os.path.join(sr.ROOT, "bench_results", "miplib2017.json"))
    ap.add_argument("--resume", action="store_true", help="keep results already in --out")
    a = ap.parse_args()

    exe = sr.find_exe(a.exe)
    if a.dir:
        ref = {k: tuple(v) for k, v in json.load(open(a.refs_json)).items()} if a.refs_json else {}
        meta = [{"Instance": f[:-4], "Status": "-", "Constraints": 0, "Variables": 0, "Binaries": 0, "Integers": 0,
                 "Nonz.": os.path.getsize(os.path.join(a.dir, f))} for f in os.listdir(a.dir) if f.endswith(".mps")]
        a.status, a.max_nnz = "any", 1 << 62
    else:
        meta, ref = load_meta()
    if a.names:
        wanted = [n.strip() for n in a.names.split(",") if n.strip()]
        pick = [d for d in meta if d["Instance"] in wanted]
    else:
        pick = [d for d in meta if (a.status == "any" or d["Status"] == a.status) and d["Nonz."] <= a.max_nnz]
    pick.sort(key=lambda d: d["Nonz."])
    if a.limit:
        pick = pick[:a.limit]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    prev = {}
    if a.resume and os.path.isfile(a.out):
        prev = {r["name"]: r for r in json.load(open(a.out))["results"]}
    solvers = ([] if a.skip_ours else ["ours"]) + (["highs"] if a.highs else [])
    print("MIPLIB 2017 benchmark set: %d instances (status %s, nnz <= %d), %gs limit, ours %d threads, gap %g"
          % (len(pick), a.status, a.max_nnz, a.time, a.threads, a.gap), flush=True)
    results = []
    for k, d in enumerate(pick):
        name = d["Instance"]
        rec = prev.get(name, {"name": name, "status_miplib": d["Status"], "rows": d["Constraints"], "cols": d["Variables"],
                              "nnz": d["Nonz."], "binaries": d["Binaries"], "integers": d["Integers"], "ref": ref.get(name)})
        try:
            path = os.path.join(a.dir, name + ".mps") if a.dir else instance_path(name)
        except Exception as e:
            print("%3d/%d %-28s download failed: %s" % (k + 1, len(pick), name, e), flush=True)
            continue
        out_dir = os.path.join(DATA, "runs", name)
        os.makedirs(out_dir, exist_ok=True)
        for s in solvers:
            if s in rec and a.resume:
                continue
            r = run_ours(exe, path, a, out_dir) if s == "ours" else run_highs(a.python, exe, path, a, out_dir)
            r["verdict"] = verdict(r, rec["ref"], a.gap)
            rec[s] = r
        results.append(rec)
        def cell(s):
            r = rec.get(s)
            if not r:
                return ""
            g = final_gap(r)
            return "%-5s %-12s %7.1fs %s" % (r["verdict"], norm_status(r.get("status"))[:12], min(r.get("time") or 0, 9999),
                                           ("gap %.2f%%" % (100 * g)) if (r["verdict"] == "LIMIT" and g is not None) else "")
        print("%3d/%d %-28s nnz %8d | ours %-40s | highs %s" % (k + 1, len(pick), name, d["Nonz."], cell("ours"), cell("highs")), flush=True)
        json.dump({"settings": vars(a), "results": results}, open(a.out, "w"), indent=1)
    summ = summarize(results, solvers, a.time)
    json.dump({"settings": vars(a), "summary": summ, "results": results}, open(a.out, "w"), indent=1)
    print()
    for s in solvers:
        x = summ[s]
        print("%-6s solved %d/%d  limit %d (%d with incumbent, mean gap %s)  wrong %d  fail %d  sgm time %.1fs" % (
            s, x["OK"], x["n"], x["LIMIT"], x["with_incumbent_at_limit"],
            "%.2f%%" % (100 * x["mean_gap_at_limit"]) if x["mean_gap_at_limit"] is not None else "-",
            x["WRONG"], x["FAIL"], x["sgm_time"] or 0))


if __name__ == "__main__":
    main()
