"""Command-line showcase for the sovereign solver.

    sovereign info
    sovereign solve   <model.mps|.qps> [--engine E] [--time S] [--threads N] [--gap PCT] [--no-presolve] [--quiet]
    sovereign compare <model.mps> [--engines dual,concurrent,ipm,pdlp] [--highs] [--time S]
    sovereign scale   [--family transport|refinery_lp|refinery_milp|refinery_qp] [--sizes 50,100,200] [--engine E]
    sovereign verify  <model.mps> <solution.sol>
    sovereign robust  [--classes ill,degenerate,infeasible,weak,large] [--highs]

Every solve runs on this machine's CPU and GPU. Reports show the verified
status, the independent verifier's residuals, time, CPU cores, memory, heap
allocations, GPU use, time per engine phase, and the solution. Results and
solution files are kept under cli_runs/. Python standard library only
(highspy is used only for the optional --highs reference in `compare`).
"""
import argparse, math, os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver_runner as sr

# ------------------------------------------------------------------ terminal
COLOR = False
W = 92


def setup_terminal(no_color):
    global COLOR, W
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    if os.name == "nt":
        os.system("")                       # enables ANSI sequences in the Windows console
    COLOR = sys.stdout.isatty() and not no_color and not os.environ.get("NO_COLOR")
    try:
        W = max(78, min(120, os.get_terminal_size().columns - 1))
    except OSError:
        W = 92


def c(text, code):
    return "\033[%sm%s\033[0m" % (code, text) if COLOR else str(text)


bold = lambda t: c(t, "1")
dim = lambda t: c(t, "2")
green = lambda t: c(t, "32")
yellow = lambda t: c(t, "33")
red = lambda t: c(t, "31")
cyan = lambda t: c(t, "36")
ANSI = re.compile(r"\033\[[0-9;]*m")
vis = lambda s: len(ANSI.sub("", str(s)))


def rule(title=""):
    if not title:
        print(dim("─" * W))
        return
    print("\n" + bold(cyan("── " + title + " ")) + cyan("─" * max(3, W - vis(title) - 4)))


def table(headers, rows, align=None):
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    cols = list(zip(*([headers] + rows))) if rows else [[h] for h in headers]
    widths = [max(vis(x) for x in col) for col in cols]
    def line(cells, style=lambda s: s):
        out = []
        for cell, w, a in zip(cells, widths, align):
            pad = " " * (w - vis(cell))
            out.append((str(cell) + pad) if a == "l" else (pad + str(cell)))
        return "  " + style("  ".join(out))
    print(line(headers, dim))
    print("  " + dim("  ".join("─" * w for w in widths)))
    for r in rows:
        print(line(r))


def kv(pairs, key_width=26):
    for k, v in pairs:
        print("  " + dim(k.ljust(key_width)) + " " + str(v))


def bar(v, vmax, width=34):
    if not vmax or v is None:
        return ""
    blocks = " ▏▎▍▌▋▊▉█"
    x = max(0.0, min(1.0, v / vmax)) * width
    full = int(x)
    return "█" * full + (blocks[int((x - full) * 8)] if full < width else "")


def spark(values, width=60):
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return dim("(too few samples)")
    if len(vals) > width:                           # resample to the width
        step = len(vals) / width
        vals = [max(vals[int(i * step):max(int(i * step) + 1, int((i + 1) * step))]) for i in range(width)]
    lo, hi = 0.0, max(vals) or 1.0
    ticks = "▁▂▃▄▅▆▇█"
    return "".join(ticks[min(7, int((v - lo) / (hi - lo) * 7.999))] for v in vals)


def fnum(v, p=10):
    if v is None or v == "":
        return "–"
    if v == "Infinity":
        return "∞"
    if v == "-Infinity":
        return "−∞"
    if isinstance(v, (int, float)):
        if v == 0:
            return "0"
        if abs(v) >= 1e10 or abs(v) < 1e-5:
            return "%.4e" % v
        return ("%." + str(p) + "g") % v
    return str(v)


def ftime(t):
    if t is None:
        return "–"
    if t < 0.01:
        return "%.2f ms" % (t * 1000)
    if t < 60:
        return "%.3f s" % t if t < 1 else "%.2f s" % t
    return "%.1f min" % (t / 60)


def fbytes(b):
    if b is None:
        return "–"
    for unit in ("B", "KB", "MB", "GB"):
        if b < 1024 or unit == "GB":
            return ("%.0f %s" if unit == "B" else "%.1f %s") % (b, unit)
        b /= 1024.0


def fcount(n):
    return "–" if n is None else format(int(round(n)), ",")


def fres(v, tol=1e-6):
    if isinstance(v, str):                          # "Infinity" in the JSON: nothing was measured
        v = None
    if v is None:
        return "–", ""
    mark = green("✓ pass") if v <= tol else (yellow("~ ≤ 1e-4") if v <= 1e-4 else red("✗ fail"))
    return ("%.1e" % v if v else "0"), mark


STATUS = {
    "optimal": (green, "✓ VERIFIED OPTIMAL", "The independent verifier checked the answer on the original model: constraints and bounds hold, and optimality is certified (duality gap, or the MILP gap tolerance)."),
    "near_optimal": (yellow, "~ NEAR-OPTIMAL", "Verifier residuals are small (≤ 1e-4) but above the optimality tolerance."),
    "inaccurate": (yellow, "! INACCURATE", "The engine finished, but the verifier measures residuals above 1e-4. Not certified."),
    "infeasible": (red, "✗ INFEASIBLE", "No point satisfies all constraints and bounds; the engine proved it."),
    "unbounded": (red, "✗ UNBOUNDED", "The objective improves without limit; the engine found an unbounded ray."),
    "unbounded_or_infeasible": (red, "✗ UNBOUNDED OR INFEASIBLE", "No finite optimum exists; which case applies was not decided."),
    "time_limit": (yellow, "◷ TIME LIMIT", "Stopped at the time limit; best solution and proven bound shown."),
    "node_limit": (yellow, "◷ NODE LIMIT", "Stopped at the node limit; best solution and proven bound shown."),
    "iteration_limit": (yellow, "◷ ITERATION LIMIT", "Stopped at the iteration limit before convergence."),
}


def status_word(st):
    col, word, _ = STATUS.get(st, (yellow, (st or "–").upper(), ""))
    return col(word)


# ------------------------------------------------------------------ helpers
def out_dir_for(model, tag):
    stem = os.path.splitext(os.path.basename(model))[0]
    d = os.path.join(sr.ROOT, "cli_runs", "%s_%s_%s" % (time.strftime("%Y%m%d-%H%M%S"), stem, tag))
    os.makedirs(d, exist_ok=True)
    return d


def header(exe, gpu, subtitle):
    print()
    print(bold("SOVEREIGN SOLVER") + dim("  ·  LP · MILP · convex QP  ·  from-scratch engines, independently verified"))
    print(dim("%s  ·  %d CPU threads  ·  %s" % (os.path.basename(exe), os.cpu_count() or 0, gpu or "no NVIDIA GPU detected")))
    if subtitle:
        print(subtitle)
    rule()


def model_line(exe, path):
    st = sr.model_stats(exe, path)
    kv([("Model", "%s  (%s)" % (os.path.basename(path), fbytes(os.path.getsize(path)))),
        ("Size", "%s rows × %s columns, %s nonzeros" % (fcount(st.get("rows")), fcount(st.get("cols")), fcount(st.get("nnz"))))])
    return st


def run_solver(exe, gpu, model, opts, out_dir, live=True, label=""):
    """Solve with sampling; prints the log (live) or a one-line progress."""
    args = sr.build_args(exe, os.path.abspath(model), out_dir, opts)
    state = {"last": ""}
    def on_line(line):
        if live:
            print(dim("  │ ") + dim(line))
        elif line.strip():
            state["last"] = line.strip()
    run = sr.Run(args, out_dir, on_line=on_line, sample_gpu=gpu is not None).start()
    try:
        while run.state == "running":
            if not live and COLOR:
                msg = "  %s %s  %s" % (cyan("…"), label, dim(state["last"][: max(10, W - vis(label) - 20)]))
                sys.stdout.write("\r\033[K" + msg + dim("  %.1fs" % run.elapsed))
                sys.stdout.flush()
            time.sleep(0.2)
    except KeyboardInterrupt:
        run.cancel()
        print("\n" + yellow("  cancelled"))
    run.wait()
    if not live and COLOR:
        sys.stdout.write("\r\033[K")
    with open(os.path.join(out_dir, "solver.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(run.log) + "\n")
    return run


# ------------------------------------------------------------------ solve
def report(run, out_dir, show_vars):
    R = run.result
    if not R:
        rule("Result")
        print("  " + red("The run did not produce a result: ") + (run.error or run.state))
        return
    col, word, meaning = STATUS.get(R["status"], (yellow, R["status"].upper(), ""))
    P, mip = R["problem"], R.get("mip")
    M = sr.metrics(run)

    rule("Result")
    print("  " + bold(col(word)))
    print("  " + dim(meaning))
    print()
    kv([("Objective", bold(fnum(R["objective"], 12)) if R["has_solution"] else "–"),
        ("Sense / type", "%s · %s" % ("maximize" if P["sense"] == "max" else "minimize", P["kind"])),
        ("Engine", R["engine"]),
        ("Solve time", "%s  (read %s, total %s)" % (bold(ftime(R["times"]["solve"])), ftime(R["times"]["read"]), ftime(R["times"]["total"])))])
    if mip:
        kv([("Proven bound", fnum(mip["bound"], 12)),
            ("Gap", ("%.4f %%" % (mip["gap"] * 100)) if R["has_solution"] and mip["gap"] is not None else "–"),
            ("Nodes", fcount(mip["nodes"])),
            ("Root LP → after cuts", "%s → %s  (%d cuts)" % (fnum(mip["root_lp"], 10), fnum(mip["root_after_cuts"], 10), mip["cuts"]))])
    else:
        kv([("Iterations", fcount(R["iterations"]))])

    rule("Independent verification  (original, unscaled, unpresolved model)")
    if not R["has_solution"]:
        print("  " + dim("No solution to verify: the engine reports %s (a proof from the engine, not a point)." %
                         R["status"].replace("_", " ")))
    else:
        rows = [["Primal feasibility (rows + bounds)", *fres(R["residuals"]["primal"])]]
        if mip:
            rows.append(["Integrality (max violation)", *fres(mip["max_integrality_violation"])])
        else:
            rows.append(["Dual feasibility", *fres(R["residuals"]["dual"])])
            rows.append(["Duality gap", *fres(R["residuals"]["gap"])])
        table(["Check", "Residual", "Result"], rows, ["l", "r", "l"])

    rule("CPU and time")
    kv([("Wall / CPU time", "%s wall · %s CPU (user + kernel, all threads)" % (ftime(M["wall"]), ftime(M["cpu"]))),
        ("Cores busy", "%s average · %s peak · of %d hardware threads" % (
            "%.2f" % M["avg_cores"] if M["avg_cores"] is not None else "–",
            "%.1f" % M["peak_cores"] if M["peak_cores"] is not None else "–", os.cpu_count() or 0))])
    print("  " + dim("cores over time".ljust(26)) + " " + cyan(spark([s["cores"] for s in run.samples])))

    rule("Memory")
    kv([("Peak memory (OS)", "%s working set · %s private" % (bold(fbytes(M["peak_ws"])), fbytes(M["peak_priv"]))),
        ("Peak live heap", fbytes(M["heap_peak"]) + dim("   (solver's own counting allocator, exact)")),
        ("Heap allocations", "%s calls · %s requested in total%s" % (
            fcount(M["allocs"]), fbytes(M["alloc_bytes"]),
            ("  · %s/s" % fcount(M["allocs"] / M["wall"])) if M["allocs"] and M["wall"] else ""))])
    print("  " + dim("working set over time".ljust(26)) + " " + cyan(spark([s["ws"] for s in run.samples])))

    rule("GPU")
    if run.gpu:
        kv([("Peak utilization", "%d %%  (whole device)" % M["gpu_util"]),
            ("Device memory used", "+%.0f MB over the run" % M["gpu_mem"]),
            ("Engine used the GPU", green("yes") if M["gpu_engine"] else "no")])
        print("  " + dim("utilization over time".ljust(26)) + " " + cyan(spark([g["util"] for g in run.gpu])))
    else:
        print("  " + dim("no GPU samples (no NVIDIA GPU, or the run was too short)"))

    ph = sr.phases(run.log)
    if ph:
        rule("Where the time went  (engine timers)")
        mx = ph[0][1]
        for k, v in ph[:12]:
            print("  " + k.ljust(30) + " " + cyan(bar(v, mx).ljust(34)) + " " + ftime(v))
    stc = sr.structure(run.log)
    if stc:
        rule("Structure  (space drivers)")
        kv([(k, v) for k, v in [("Presolve rows × cols", stc.get("presolve")), ("LU fill (nnz L + U)", stc.get("lu_fill") and fcount(stc["lu_fill"])),
                                ("IPM factor nnz(L)", stc.get("ldl_nnz") and fcount(stc["ldl_nnz"])), ("MILP probing", stc.get("probing"))] if v])

    if R["has_solution"] and show_vars > 0:
        nz = [v for v in R["variables"] if isinstance(v["value"], (int, float)) and abs(v["value"]) > 1e-9]
        rule("Solution  (%s of %s variables nonzero%s)" % (fcount(len(nz)), fcount(R["variables_total"]),
                                                          ", table truncated" if R["variables_truncated"] else ""))
        nz.sort(key=lambda v: -abs(v["value"]))
        rows = [[v["name"], fnum(v["value"], 10), fnum(v["lower"], 6), fnum(v["upper"], 6),
                 "–" if v["reduced_cost"] is None else fnum(v["reduced_cost"], 6), "int" if v["integer"] else "cont"] for v in nz[:show_vars]]
        table(["Variable", "Value", "Lower", "Upper", "Reduced cost", "Type"], rows, ["l", "r", "r", "r", "r", "l"])
        if len(nz) > show_vars:
            print("  " + dim("… %s more nonzero variables in the CSV (--show-vars N to print more)" % fcount(len(nz) - show_vars)))
        binding = [r for r in R["constraints"] if r["position"] and r["position"] != "slack"]
        print("  " + dim("%s of %s constraints binding (equalities included)" % (fcount(len(binding)), fcount(R["constraints_total"]))))

    rule("Files")
    files = [("solution.sol", "solution (re-check: sovereign verify <model> <this file>)"),
             ("solution_variables.csv", "all variables"), ("solution_constraints.csv", "all constraints")] if R["has_solution"] else []
    for name, what in files + [("result.json", "full result and resource counters"), ("solver.log", "solver log")]:
        p = os.path.join(out_dir, name)
        if os.path.isfile(p):
            print("  " + p + dim("   " + what))
    print()


def cmd_solve(a):
    exe, gpu = sr.find_exe(a.exe), sr.gpu_name()
    header(exe, gpu, "")
    model_line(exe, a.model)
    kv([("Engine", "%s — %s" % (a.engine, sr.ENGINES[a.engine][1]))])
    out_dir = out_dir_for(a.model, a.engine)
    rule("Solver log" if not a.quiet else "Solving")
    opts = {"engine": a.engine, "time_limit": a.time, "threads": a.threads, "gap": a.gap / 100.0,
            "presolve": not a.no_presolve, "pdlp_tol": a.pdlp_tol}
    run = run_solver(exe, gpu, a.model, opts, out_dir, live=not a.quiet, label=a.engine)
    report(run, out_dir, a.show_vars)
    return 0 if run.result else 1


# ------------------------------------------------------------------ compare
def run_highs(exe, model, tl, out_dir):
    try:
        import highspy
    except ImportError:
        return {"error": "highspy not installed (pip install highspy)"}
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    h.setOptionValue("time_limit", float(tl))
    h.setOptionValue("threads", 1)
    h.readModel(os.path.abspath(model))
    t0 = time.time()
    h.run()
    t = time.time() - t0
    st = h.modelStatusToString(h.getModelStatus())
    info = h.getInfo()
    res = {"status": st, "time": t, "objective": None, "iterations": info.simplex_iteration_count + info.ipm_iteration_count,
           "nodes": getattr(info, "mip_node_count", None)}
    if st == "Optimal":
        res["objective"] = info.objective_function_value
        lp, sol = h.getLp(), h.getSolution()
        sp = os.path.join(out_dir, "highs_solution.sol")
        with open(sp, "w", encoding="utf-8") as f:
            for nm, v in zip(lp.col_names_, sol.col_value):
                f.write("%s %r\n" % (nm, v))
            if sol.dual_valid:
                f.write("DUAL\n")
                for nm, v in zip(lp.row_names_, sol.row_dual):
                    f.write("%s %r\n" % (nm, v))
        try:
            res["check"] = sr.check(exe, os.path.abspath(model), sp)
        except Exception as e:
            res["check_error"] = str(e)
    return res


def cmd_compare(a):
    exe, gpu = sr.find_exe(a.exe), sr.gpu_name()
    engines = [e.strip() for e in a.engines.split(",") if e.strip()]
    for e in engines:
        if e not in sr.ENGINES:
            sys.exit("unknown engine %s (choose from %s)" % (e, ", ".join(sr.ENGINES)))
    header(exe, gpu, "")
    model_line(exe, a.model)
    base_dir = out_dir_for(a.model, "compare")
    runs = []
    rule("Running %d engine%s%s" % (len(engines), "s" if len(engines) > 1 else "", " + HiGHS reference" if a.highs else ""))
    for k, e in enumerate(engines):
        d = os.path.join(base_dir, e)
        os.makedirs(d, exist_ok=True)
        opts = {"engine": e, "time_limit": a.time, "threads": a.threads, "gap": a.gap / 100.0, "presolve": not a.no_presolve,
                "pdlp_tol": a.pdlp_tol}
        run = run_solver(exe, gpu, a.model, opts, d, live=False, label="[%d/%d] %s" % (k + 1, len(engines), e))
        R = run.result
        print("  %s %-11s %s %s" % (dim("[%d/%d]" % (k + 1, len(engines))), e, status_word(R["status"]) if R else red("error"),
                                    dim(ftime(R["times"]["solve"])) if R else dim(run.error or "")))
        runs.append((e, run))
    hr = None
    if a.highs:
        print("  %s %-11s " % (dim("[ref]"), "HiGHS"), end="", flush=True)
        hr = run_highs(exe, a.model, a.time, base_dir)
        print(dim(hr.get("error")) if "error" in hr else "%s %s" % (hr["status"], dim(ftime(hr["time"]))))

    # reference objective: HiGHS if it is optimal, else the best verified one of ours
    ref = hr.get("objective") if hr and hr.get("objective") is not None else None
    ours = [r.result["objective"] for _, r in runs if r.result and r.result["status"] == "optimal" and r.result["has_solution"]]
    if ref is None and ours:
        ref = ours[0]
    rel = lambda v: "–" if v is None or ref is None else ("%.1e" % (abs(v - ref) / max(1.0, abs(ref))))

    rule("Side by side")
    rows = []
    for e, run in runs:
        R, M = run.result, sr.metrics(run)
        if not R:
            rows.append([e, red("error"), "–", "–", "–", "–", "–", "–", "–", "–", "–", "–"])
            continue
        its = ("%s nodes" % fcount(R["mip"]["nodes"])) if R.get("mip") else fcount(R["iterations"])
        rows.append([e, status_word(R["status"]), fnum(R["objective"], 10) if R["has_solution"] else "–",
                     rel(R["objective"] if R["has_solution"] else None), ftime(R["times"]["solve"]), ftime(M["cpu"]),
                     "%.2f" % M["avg_cores"] if M["avg_cores"] is not None else "–", fbytes(M["peak_ws"]), fbytes(M["heap_peak"]),
                     fcount(M["allocs"]), ("%d%%" % M["gpu_util"]) if M["gpu_util"] is not None else "–", its])
    if hr and "error" not in hr:
        its = ("%s nodes" % fcount(hr["nodes"])) if (hr.get("nodes") or 0) > 0 else fcount(hr["iterations"])
        rows.append([dim("HiGHS (ref)"), hr["status"], fnum(hr["objective"], 10), rel(hr["objective"]), ftime(hr["time"]),
                     "–", "–", "–", "–", "–", "–", its])
    table(["Engine", "Status", "Objective", "Δ rel", "Solve", "CPU", "Cores", "Peak mem", "Peak heap", "Allocs", "GPU", "Its / nodes"],
          rows, ["l", "l", "r", "r", "r", "r", "r", "r", "r", "r", "r", "r"])
    print("  " + dim("Δ rel: relative difference from %s. Resource columns are measured for our engines only (HiGHS runs in-process here)."
                     % ("HiGHS's optimum" if hr and hr.get("objective") is not None else "the first verified optimum")))

    rule("Solve time")
    times = [(e, r.result["times"]["solve"]) for e, r in runs if r.result] + ([("HiGHS (ref)", hr["time"])] if hr and "error" not in hr else [])
    mx = max((t for _, t in times), default=0)
    for e, t in times:
        print("  " + e.ljust(14) + " " + cyan(bar(t, mx, 40).ljust(40)) + " " + ftime(t))
    rule("Peak memory")
    mems = [(e, sr.metrics(r)["peak_ws"]) for e, r in runs if r.result]
    mx = max((m or 0 for _, m in mems), default=0)
    for e, m in mems:
        print("  " + e.ljust(14) + " " + cyan(bar(m or 0, mx, 40).ljust(40)) + " " + fbytes(m))

    if hr and hr.get("check"):
        ck = hr["check"]
        rule("HiGHS's answer, checked by our independent verifier")
        rows = [["Primal feasibility", *fres(ck["eps_P"])]]
        if ck.get("eps_D") is not None:
            rows += [["Dual feasibility", *fres(ck["eps_D"])], ["Duality gap", *fres(ck["eps_G"])]]
        table(["Check", "Residual", "Result"], rows, ["l", "r", "l"])

    verified = [(e, r) for e, r in runs if r.result and r.result["status"] == "optimal"]
    rule("Verdict")
    if verified:
        best = min(verified, key=lambda er: er[1].result["times"]["solve"])
        print("  Fastest verified engine: %s in %s" % (bold(best[0]), ftime(best[1].result["times"]["solve"])))
        objs = [r.result["objective"] for _, r in verified if r.result["has_solution"]]
        if len(objs) > 1:
            spread = max(abs(o - objs[0]) / max(1.0, abs(objs[0])) for o in objs)
            print("  All verified engines agree on the objective to %.1e (relative)." % spread)
    else:
        print("  " + yellow("No engine reached a verified optimum within the limits."))
    print("  " + dim("Results: " + base_dir))
    print()
    return 0


# ------------------------------------------------------------------ scale
def ascii_loglog(points, key, fitres, height=14, width=58, ylabel="seconds", yfmt=ftime):
    pts = [(q["nnz"], q[key]) for q in points if q.get(key) and q["nnz"] > 0]
    if len(pts) < 2:
        return
    lx = [math.log10(x) for x, _ in pts]
    ly = [math.log10(y) for _, y in pts]
    x0, x1, y0, y1 = min(lx), max(lx), min(ly), max(ly)
    if x1 == x0:
        x1 += 1
    if y1 == y0:
        y1 += 1
    grid = [[" "] * width for _ in range(height)]
    col = lambda x: min(width - 1, max(0, int(round((x - x0) / (x1 - x0) * (width - 1)))))
    row = lambda y: min(height - 1, max(0, height - 1 - int(round((y - y0) / (y1 - y0) * (height - 1)))))
    if fitres:
        k, cc, _ = fitres
        for i in range(width):
            x = x0 + (x1 - x0) * i / (width - 1)
            r = row(cc + k * x)
            if 0 <= r < height:
                grid[r][i] = "·"
    for x, y in zip(lx, ly):
        grid[row(y)][col(x)] = "●"
    top, bot = yfmt(10 ** y1), yfmt(10 ** y0)
    lw = max(len(top), len(bot))
    for i, g in enumerate(grid):
        lab = top if i == 0 else (bot if i == height - 1 else "")
        print("  " + dim(lab.rjust(lw)) + " " + dim("│") + cyan("".join(g)))
    print("  " + " " * lw + " " + dim("└" + "─" * width))
    left, right = fcount(10 ** x0), fcount(10 ** x1)
    print("  " + " " * lw + "  " + dim(left + " " * max(1, width - len(left) - len(right)) + right))
    print("  " + " " * lw + "  " + dim("nonzeros (log scale) · ● measured · fitted line · (%s, log scale)" % ylabel))


def cmd_scale(a):
    exe, gpu = sr.find_exe(a.exe), sr.gpu_name()
    if a.family not in sr.FAMILIES:
        sys.exit("unknown family (choose from %s)" % ", ".join(sr.FAMILIES))
    engine = a.engine or sr.FAMILIES[a.family][1]
    sizes = [int(s) for s in re.split(r"[\s,;]+", a.sizes) if s.strip()]
    header(exe, gpu, "")
    kv([("Study", "empirical time and space complexity"), ("Family", sr.FAMILIES[a.family][0]),
        ("Sizes", ", ".join(map(str, sizes))), ("Engine", "%s — %s" % (engine, sr.ENGINES[engine][1]))])
    base_dir = out_dir_for(a.family, "scale")
    rule("Running")
    points = []
    for size in sizes:
        try:
            model = sr.generate(exe, a.family, size, os.path.join(base_dir, "models"))
        except Exception as e:
            print("  " + red("size %d: generator failed: %s" % (size, e)))
            break
        d = os.path.join(base_dir, "size_%d" % size)
        os.makedirs(d, exist_ok=True)
        opts = {"engine": engine, "time_limit": a.time, "threads": a.threads, "gap": a.gap / 100.0, "presolve": True}
        run = run_solver(exe, gpu, model, opts, d, live=False, label="size %d" % size)
        R = run.result
        if not R:
            print("  size %-6d %s" % (size, red(run.error or run.state)))
            continue
        M = sr.metrics(run)
        points.append({"size": size, "rows": R["problem"]["rows"], "cols": R["problem"]["cols"], "nnz": R["problem"]["nonzeros"],
                       "status": R["status"], "time": R["times"]["solve"], "cpu": M["cpu"], "mem": M["peak_ws"],
                       "heap": M["heap_peak"], "allocs": M["allocs"]})
        print("  size %-6d %s nonzeros  %s  %s  %s" % (size, fcount(points[-1]["nnz"]).rjust(11), status_word(R["status"]),
                                                      ftime(points[-1]["time"]).rjust(10), fbytes(points[-1]["mem"]).rjust(9)))
    if not points:
        return 1
    rule("Measurements")
    table(["Size", "Rows", "Columns", "Nonzeros", "Status", "Solve time", "CPU time", "Peak memory", "Peak heap", "Allocations"],
          [[q["size"], fcount(q["rows"]), fcount(q["cols"]), fcount(q["nnz"]), status_word(q["status"]), ftime(q["time"]),
            ftime(q["cpu"]), fbytes(q["mem"]), fbytes(q["heap"]), fcount(q["allocs"])] for q in points])
    ft, fm, fh = sr.fit_power(points, "time"), sr.fit_power(points, "mem"), sr.fit_power(points, "heap")
    rule("Fitted growth  (least squares on log–log, verified runs only)")
    fmt = lambda f: ("%s  (R² %.3f)" % (bold("nnz^%.2f" % f[0]), f[2])) if f else dim("needs ≥ 2 verified sizes")
    kv([("Solve time ∝", fmt(ft)), ("Peak memory (OS) ∝", fmt(fm)), ("Peak live heap ∝", fmt(fh))])
    print("  " + dim("Empirical: this family, engine and size range on this machine — not a worst-case bound"))
    print("  " + dim("(simplex and branch-and-cut are exponential in the worst case). k ≈ 1 means linear growth."))
    rule("Solve time vs nonzeros")
    ascii_loglog(points, "time", ft)
    rule("Peak memory vs nonzeros")
    ascii_loglog(points, "mem", fm, ylabel="bytes", yfmt=fbytes)
    print("\n  " + dim("Results: " + base_dir) + "\n")
    return 0


# ------------------------------------------------------------------ verify / info
def cmd_verify(a):
    exe = sr.find_exe(a.exe)
    header(exe, sr.gpu_name(), "")
    model_line(exe, a.model)
    kv([("Solution file", a.solution)])
    with open(a.model, errors="replace") as f:           # integer markers present?
        a.model_is_mip = any("MARKER" in line and "INTORG" in line for line in f)
    ck = sr.check(exe, os.path.abspath(a.model), os.path.abspath(a.solution))
    rule("Independent verification")
    kv([("Objective of this solution", bold(fnum(ck["objective"], 12)))])
    rows = [["Primal feasibility (rows + bounds)", *fres(ck["eps_P"])]]
    if ck.get("int_violation") is not None and a.model_is_mip:
        rows.append(["Integrality (max violation)", *fres(ck["int_violation"])])
    if ck.get("eps_D") is not None:
        rows += [["Dual feasibility", *fres(ck["eps_D"])], ["Duality gap", *fres(ck["eps_G"])]]
    else:
        rows.append(["Dual feasibility / gap", "–", dim("no duals in the file")])
    table(["Check", "Residual", "Result"], rows, ["l", "r", "l"])
    print()
    return 0


def cmd_info(a):
    exe = sr.find_exe(a.exe)
    header(exe, sr.gpu_name(), "")
    kv([("Solver binary", exe)])
    rule("Engines")
    for k, (_, d) in sr.ENGINES.items():
        print("  " + bold(k.ljust(12)) + d)
    rule("Scaling families")
    for k, (d, e) in sr.FAMILIES.items():
        print("  " + bold(k.ljust(14)) + d + dim("  (engine: %s)" % e))
    print()
    return 0


def main():
    ap = argparse.ArgumentParser(prog="sovereign", description="Command-line showcase for the sovereign LP / MILP / QP solver.")
    ap.add_argument("--exe", help="path to sovereign_solve (default: build/Release)")
    ap.add_argument("--no-color", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def solver_opts(p, engine_default="auto"):
        p.add_argument("--time", type=float, default=60, help="time limit in seconds (default 60)")
        p.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 4), help="MILP threads")
        p.add_argument("--gap", type=float, default=0.01, help="MILP relative gap in percent (default 0.01)")
        p.add_argument("--no-presolve", action="store_true")
        p.add_argument("--pdlp-tol", type=float, default=1e-6, help="PDLP tolerance (engine pdlp)")

    p = sub.add_parser("solve", help="solve a model and print a full report")
    p.add_argument("model")
    p.add_argument("--engine", default="auto", choices=list(sr.ENGINES))
    p.add_argument("--quiet", action="store_true", help="progress line instead of the full solver log")
    p.add_argument("--show-vars", type=int, default=15, help="nonzero variables to print (default 15)")
    solver_opts(p)

    p = sub.add_parser("compare", help="run several engines on one model side by side")
    p.add_argument("model")
    p.add_argument("--engines", default="dual,concurrent,ipm", help="comma list (default dual,concurrent,ipm)")
    p.add_argument("--highs", action="store_true", help="add HiGHS as a reference (needs highspy)")
    solver_opts(p)

    p = sub.add_parser("scale", help="empirical complexity: solve a model family at growing sizes")
    p.add_argument("--family", default="transport", choices=list(sr.FAMILIES))
    p.add_argument("--sizes", default="50,100,200,400")
    p.add_argument("--engine", choices=list(sr.ENGINES))
    solver_opts(p)

    p = sub.add_parser("verify", help="check any solver's solution file against a model")
    p.add_argument("model")
    p.add_argument("solution")

    sub.add_parser("info", help="solver binary, hardware, engines")

    p = sub.add_parser("robust", help="numerical robustness demonstration (see tools/robustness.py)")
    # its options (--classes, --highs, --time, ...) are passed through to tools/robustness.py

    a, rest = ap.parse_known_args()
    if rest and a.cmd != "robust":
        ap.error("unrecognized arguments: " + " ".join(rest))
    a.rest = rest
    setup_terminal(a.no_color)
    if a.cmd == "robust":
        import subprocess
        args = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "robustness.py")]
        if a.exe:
            args += ["--exe", a.exe]
        return subprocess.call(args + a.rest)
    try:
        return {"solve": cmd_solve, "compare": cmd_compare, "scale": cmd_scale, "verify": cmd_verify, "info": cmd_info}[a.cmd](a)
    except (FileNotFoundError, ValueError) as e:
        print(red("error: ") + str(e))
        return 2


if __name__ == "__main__":
    sys.exit(main())
