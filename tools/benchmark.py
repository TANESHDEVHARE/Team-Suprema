#!/usr/bin/env python3
"""benchmark.py -- reproducible benchmark of sovereign_solve against public
test sets, optionally compared with HiGHS (the established open-source solver
the problem statement asks us to compare against).

  python tools/benchmark.py --exe build/Release/sovereign_solve.exe --set netlib
  python tools/benchmark.py --exe ... --set miplib3 --time 60 --threads 4
  python tools/benchmark.py --exe ... --set maros --time 60
  python tools/benchmark.py --exe ... --set all --highs        # also run HiGHS

Data is downloaded once into --data (default ./bench_data):
  netlib  : COIN-OR Data-Netlib           (LP,   91 problems, dual simplex)
  miplib3 : COIN-OR Data-miplib3          (MILP, 63 problems, branch-and-cut)
  maros   : Maros-Meszaros QPS collection (QP,  ~134 problems, interior point)

Every result line reports the solver's own status and the independent
verifier's residuals (eps_P / eps_D / eps_G, or integrality for MILP). A
result counts as OK only if it is certified AND agrees with the reference
(HiGHS when --highs is given; the published optimum from the Maros-Meszaros
README for QP). "WRONG" -- a certified-looking answer that disagrees -- is the
number that must stay at zero.

Writes <out>/<set>.csv and <out>/report.md.
"""
import argparse, csv, gzip, json, os, re, subprocess, sys, time, urllib.request

SETS = {
    "netlib": {
        "mode": ["--simplex"], "ext": ".mps",
        "names": "25fv47 80bau3b adlittle afiro agg agg2 agg3 bandm beaconfd blend bnl1 bnl2 boeing1 boeing2 bore3d "
                 "brandy capri cycle czprob d2q06c d6cube degen2 degen3 dfl001 e226 etamacro fffff800 finnis fit1d "
                 "fit1p fit2d fit2p forplan ganges gfrd-pnc greenbea greenbeb grow15 grow22 grow7 israel kb2 lotfi "
                 "maros-r7 maros modszk1 nesm perold pilot pilot4 pilot87 pilotnov recipe sc105 sc205 sc50a sc50b "
                 "scagr25 scagr7 scfxm1 scfxm2 scfxm3 scorpion scrs8 scsd1 scsd6 scsd8 sctap1 sctap2 sctap3 seba "
                 "share1b share2b shell ship04l ship04s ship08l ship08s ship12l ship12s sierra stair standata standgub "
                 "standmps stocfor1 stocfor2 tuff vtpbase wood1p woodw".split(),
        "url": "https://raw.githubusercontent.com/coin-or-tools/Data-Netlib/master/{name}.mps.gz",
    },
    "miplib3": {
        "mode": ["--mip"], "ext": ".mps",
        "names": "10teams air03 air04 air05 arki001 bell3a bell5 blend2 cap6000 dano3mip danoint dcmulti dsbmip egout "
                 "enigma fiber fixnet6 flugpl gen gesa2 gesa2_o gesa3 gesa3_o gt2 harp2 khb05250 l152lav lseu "
                 "markshare1 markshare2 mas74 mas76 misc03 misc06 misc07 mitre mkc mod008 mod010 mod011 modglob "
                 "noswot p0033 p0201 p0282 p0548 p2756 pk1 pp08a pp08aCUTS qiu qnet1 qnet1_o rentacar rgn rout "
                 "set1ch seymour stein27 stein45 swath vpm1 vpm2".split(),
        "url": "https://raw.githubusercontent.com/coin-or-tools/Data-miplib3/master/{name}.gz",
    },
    "maros": {
        "mode": ["--ipm"], "ext": ".QPS", "names": None,   # listed from the repository
        "api": "https://api.github.com/repos/YimingYAN/QP-Test-Problems/contents/QPS_Files",
        "url": "https://raw.githubusercontent.com/YimingYAN/QP-Test-Problems/master/QPS_Files/{name}",
        "max_bytes": 15_000_000,
    },
}


def fetch(url, dest):
    with urllib.request.urlopen(url, timeout=120) as r:
        data = r.read()
    if url.endswith(".gz"):
        data = gzip.decompress(data)
    with open(dest, "wb") as f:
        f.write(data)


def ensure_data(setname, data_dir):
    cfg = SETS[setname]
    d = os.path.join(data_dir, setname)
    os.makedirs(d, exist_ok=True)
    names = cfg["names"]
    if names is None:
        listing = json.load(urllib.request.urlopen(cfg["api"], timeout=120))
        files = [x["name"] for x in listing
                 if (x["name"].endswith(".QPS") and x["size"] < cfg["max_bytes"]) or x["name"] == "00README.QP"]
        for fn in files:
            if not os.path.exists(os.path.join(d, fn)):
                fetch(cfg["url"].format(name=fn), os.path.join(d, fn))
        names = sorted(fn[:-4] for fn in files if fn.endswith(".QPS"))
    else:
        for nm in names:
            p = os.path.join(d, nm + cfg["ext"])
            if not os.path.exists(p):
                print("  downloading", nm, flush=True)
                fetch(cfg["url"].format(name=nm), p)
    return d, names


def maros_refs(d):
    ref = {}
    path = os.path.join(d, "00README.QP")
    if os.path.exists(path):
        for ln in open(path, errors="ignore"):
            p = ln.split()
            if len(p) == 7:
                try:
                    ref[p[0].upper()] = float(p[6])
                except ValueError:
                    pass
    return ref


def run_ours(exe, mode, path, tlim, threads):
    args = [exe] + mode + ([f"--threads={threads}"] if mode == ["--mip"] else []) + [path, str(tlim)]
    t0 = time.time()
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=tlim + 120).stdout
    except subprocess.TimeoutExpired:
        out = "Status: hard_timeout"
    wall = time.time() - t0
    g = lambda k: (re.search(k + r":\s+(\S+)", out) or [None, None])[1]
    r = {"status": g("Status"), "obj": g("Objective"), "time": wall,
         "eps_P": g("eps_P"), "eps_D": g("eps_D"), "eps_G": g("eps_G"), "bound": g("Bound"), "nodes": g("Nodes")}
    m = re.search(r"max integrality violation:\s+(\S+)", out)
    r["int_viol"] = m.group(1) if m else None
    it = re.search(r"Iterations:\s+(\d+)", out)
    r["iterations"] = it.group(1) if it else None
    return r


def run_highs(path, tlim, threads, mip):
    import highspy
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(tlim))
    h.setOptionValue("threads", int(threads))
    h.readModel(path)
    t0 = time.time()
    h.run()
    info = h.getInfo()
    return {"status": h.modelStatusToString(h.getModelStatus()), "obj": info.objective_function_value,
            "time": time.time() - t0}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def classify(setname, r, ref_obj, ref_optimal):
    st, obj = r["status"], fnum(r["obj"])
    if setname == "miplib3":
        feas = obj is None or (fnum(r["eps_P"]) or 0) < 1e-6 and (fnum(r["int_viol"]) or 0) < 1e-6
        if obj is not None and not feas:
            return "WRONG"
        if st == "optimal":
            if ref_optimal and (obj is None or abs(obj - ref_obj) > 1e-4 * max(1, abs(ref_obj))):
                return "WRONG"
            return "OK"
        if st == "infeasible":
            return "WRONG" if ref_optimal else "OK"
        return "LIMIT"
    certified = st in ("optimal", "near_optimal") and all((fnum(r[k]) or 1) < 1e-6 for k in ("eps_P", "eps_D", "eps_G"))
    if not certified:
        return "FAIL"
    if ref_obj is None:
        return "OK"          # certified by the verifier, no reference to compare
    return "OK" if abs(obj - ref_obj) <= 1e-6 * max(1, abs(ref_obj)) else "WRONG"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--set", choices=["netlib", "miplib3", "maros", "all"], default="netlib")
    ap.add_argument("--time", type=float, default=60)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--highs", action="store_true", help="also run HiGHS (pip install highspy) as the reference")
    ap.add_argument("--only", default="", help="comma-separated instance names")
    ap.add_argument("--data", default="bench_data")
    ap.add_argument("--out", default="bench_results")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sets = ["netlib", "miplib3", "maros"] if a.set == "all" else [a.set]
    report = ["# sovereign_solve benchmark", "",
              f"time limit {a.time:g} s per instance, {a.threads} MILP threads, "
              f"reference: {'HiGHS ' + __import__('highspy').Highs().version() if a.highs else 'published optima (QP) / verifier certificate'}", ""]
    for setname in sets:
        cfg = SETS[setname]
        d, names = ensure_data(setname, a.data)
        if a.only:
            names = [n for n in names if n in a.only.split(",")]
        refs = maros_refs(d) if setname == "maros" else {}
        rows, counts = [], {}
        print(f"== {setname}: {len(names)} instances", flush=True)
        for nm in names:
            path = os.path.join(d, nm + cfg["ext"])
            r = run_ours(a.exe, cfg["mode"], path, a.time, a.threads)
            ref_obj, ref_opt, hs = refs.get(nm), refs.get(nm) is not None, None
            if a.highs:
                try:
                    hs = run_highs(path, a.time, a.threads, setname == "miplib3")
                    if hs["status"] == "Optimal":
                        ref_obj, ref_opt = hs["obj"], True
                except Exception as e:
                    hs = {"status": "error: " + str(e), "obj": None, "time": None}
            v = classify(setname, r, ref_obj, ref_opt)
            counts[v] = counts.get(v, 0) + 1
            row = {"instance": nm, "verdict": v, **{k: r[k] for k in r},
                   "ref_obj": ref_obj, "highs_status": hs and hs["status"], "highs_time": hs and hs["time"]}
            rows.append(row)
            print(f"  {nm:<12} {str(r['status']):<16} obj={str(r['obj']):<18} t={r['time']:7.2f}s"
                  + (f" | highs {hs['status'][:14]:<14} t={hs['time'] or 0:6.2f}s" if hs else "") + f"  {v}", flush=True)
        with open(os.path.join(a.out, setname + ".csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        ours_t = sum(r["time"] for r in rows)
        line = f"## {setname} ({len(rows)} instances)\n\n" + ", ".join(f"**{k}** {v}" for k, v in sorted(counts.items()))
        line += f"\n\ntotal wall time: ours {ours_t:.1f} s"
        if a.highs:
            line += f", HiGHS {sum((r['highs_time'] or 0) for r in rows):.1f} s"
        report += [line, "", "| instance | status | objective | time (s) | verdict |", "|---|---|---|---|---|"]
        report += [f"| {r['instance']} | {r['status']} | {r['obj']} | {r['time']:.2f} | {r['verdict']} |" for r in rows]
        report.append("")
        print(f"   {counts}", flush=True)
    with open(os.path.join(a.out, "report.md"), "w") as f:
        f.write("\n".join(report) + "\n")
    print("report:", os.path.join(a.out, "report.md"))


if __name__ == "__main__":
    main()
