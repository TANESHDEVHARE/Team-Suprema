"""Shared runner for the command-line showcase (tools/sovereign.py) and the
local web interface (tools/ui_server.py).

Starts sovereign_solve on a model, streams its log, and samples the solver
process while it runs: CPU time and memory (Win32 via ctypes on Windows,
/proc on Linux) every 0.2 s, and whole-device GPU utilization and memory from
nvidia-smi every 0.25 s. The solver itself writes the verified result, its
resource counters and the solution files (--json / --sol / --csv).
Python standard library only.
"""
import json, os, re, subprocess, sys, threading, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXE_NAME = "sovereign_solve" + (".exe" if os.name == "nt" else "")

ENGINES = {
    "auto": ([], "chosen by problem type (MILP -> branch-and-cut, QP -> interior point, LP -> concurrent portfolio)"),
    "concurrent": (["--concurrent"], "LP: dual simplex + primal simplex + IPM/crossover + GPU PDLP, first verified answer wins"),
    "dual": (["--simplex"], "LP: dual revised simplex"),
    "ipm": (["--ipm"], "LP / convex QP: primal-dual interior point"),
    "crossover": (["--crossover"], "LP: PDLP -> crossover -> simplex"),
    "pdlp": (["--pdlp"], "LP: first-order PDLP (GPU and CPU race)"),
    "mip": (["--mip"], "MILP: parallel branch-and-cut"),
}


def find_exe(explicit=None):
    cands = [explicit] if explicit else []
    for d in ("build/Release", "build", "build_cpu/Release"):
        cands.append(os.path.join(ROOT, d, EXE_NAME))
    for c in cands:
        if c and os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError("sovereign_solve not found; build it (cmake --build build --config Release) or pass --exe")


def gpu_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5).stdout.strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def model_stats(exe, path):
    out = subprocess.run([exe, "--stats", path], capture_output=True, text=True, timeout=600)
    lines = (out.stdout or "").strip().splitlines()
    if out.returncode != 0 or not lines:
        raise ValueError((out.stderr or out.stdout or "could not read the model").strip())
    s = {"summary": lines[0]}
    for k in ("rows", "cols", "nnz"):
        m = re.search(r"\b%s (\d+)" % k, lines[0])
        if m:
            s[k] = int(m.group(1))
    m = re.match(r"^(\S+):", lines[0])
    s["name"] = m.group(1) if m else ""
    return s


def build_args(exe, model_path, out_dir, o):
    """Command line for one solve; outputs go to out_dir."""
    engine = o.get("engine", "auto")
    if engine not in ENGINES:
        raise ValueError("unknown engine " + engine)
    args = [exe, "--live", "-v"] + ENGINES[engine][0]
    if not o.get("presolve", True):
        args.append("--no-presolve")
    if engine in ("auto", "mip"):
        args.append("--threads=%d" % max(1, min(64, int(o.get("threads") or 4))))
        args.append("--gap=%g" % max(0.0, float(o.get("gap") if o.get("gap") is not None else 1e-4)))
    args += ["--json=" + os.path.join(out_dir, "result.json"), "--sol=" + os.path.join(out_dir, "solution.sol"),
             "--csv=" + os.path.join(out_dir, "solution")]
    args.append(model_path)
    if engine == "pdlp":
        args.append(str(float(o.get("pdlp_tol") or 1e-6)))
    else:
        args.append("%g" % float(o.get("time_limit") or 60))
    return args


# ---------------------------------------------------------------- sampling
if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class _PMC(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t)]

    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.OpenProcess.restype = wintypes.HANDLE

    def _open(pid):
        return _k32.OpenProcess(0x1000, False, pid)          # PROCESS_QUERY_LIMITED_INFORMATION

    def _sample(h):
        c, e, k, u = (wintypes.FILETIME() for _ in range(4))
        if not _k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u)):
            return None
        ft = lambda f: ((f.dwHighDateTime << 32) | f.dwLowDateTime) * 1e-7
        pmc = _PMC(); pmc.cb = ctypes.sizeof(_PMC)
        if not _k32.K32GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb):
            return None
        return ft(k) + ft(u), pmc.WorkingSetSize, pmc.PrivateUsage

    def _close(h):
        _k32.CloseHandle(h)
else:
    _TICK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
    _PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096

    def _open(pid):
        return pid

    def _sample(pid):
        try:
            with open("/proc/%d/stat" % pid) as f:
                fields = f.read().rsplit(")", 1)[1].split()
            cpu = (int(fields[11]) + int(fields[12])) / _TICK
            with open("/proc/%d/statm" % pid) as f:
                size, rss = (int(v) for v in f.read().split()[:2])
            return cpu, rss * _PAGE, size * _PAGE
        except Exception:
            return None

    def _close(h):
        pass


class Run:
    """One solver process: log lines, resource samples, and the result."""

    def __init__(self, args, cwd, on_line=None, sample_gpu=True):
        self.args, self.cwd, self.on_line, self.sample_gpu = args, cwd, on_line, sample_gpu
        self.log, self.samples, self.gpu = [], [], []
        self.state, self.result, self.error = "created", None, None
        self.proc, self.returncode = None, None
        self.started = self.finished = None
        self._thread = None

    @property
    def elapsed(self):
        if self.started is None:
            return 0.0
        return (self.finished or time.time()) - self.started

    def start(self):
        self.state, self.started = "running", time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def wait(self):
        if self._thread:
            self._thread.join()
        return self

    def cancel(self):
        if self.state == "running" and self.proc:
            self.state = "cancelled"
            self.proc.kill()

    def _run(self):
        try:
            p = subprocess.Popen(self.args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                 cwd=self.cwd, encoding="utf-8", errors="replace")
        except Exception as e:
            self.state, self.error, self.finished = "error", str(e), time.time()
            return
        self.proc = p
        threading.Thread(target=self._sample_process, daemon=True).start()
        if self.sample_gpu:
            threading.Thread(target=self._sample_gpu, daemon=True).start()
        for line in p.stdout:
            line = line.rstrip("\n")
            self.log.append(line)
            if self.on_line:
                self.on_line(line)
        p.wait()
        self.returncode = p.returncode
        res = os.path.join(self.cwd, "result.json")
        if self.state == "cancelled":
            pass
        elif os.path.isfile(res):
            try:
                with open(res, encoding="utf-8") as f:
                    self.result = json.load(f)
                self.state = "done"
            except Exception as e:
                self.state, self.error = "error", "could not read result: %s" % e
        else:
            tail = "\n".join(self.log[-3:])
            self.state, self.error = "error", "the solver stopped without a result (exit code %s)%s" % (
                p.returncode, (": " + tail) if tail else "")
        self.finished = time.time()

    def _sample_process(self):
        h = _open(self.proc.pid)
        if not h:
            return
        last = None
        try:
            while self.proc.poll() is None:
                s = _sample(h)
                now = time.time()
                if s:
                    cpu, ws, priv = s
                    cores = max(0.0, (cpu - last[1]) / max(1e-6, now - last[0])) if last else None
                    last = (now, cpu)
                    self.samples.append({"t": round(now - self.started, 3), "cores": cores, "ws": ws, "priv": priv, "cpu": cpu})
                time.sleep(0.2)
        finally:
            _close(h)

    def _sample_gpu(self):
        try:
            g = subprocess.Popen(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits",
                                  "-lms", "250"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        except Exception:
            return
        try:
            for line in g.stdout:
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 2 and parts[0].isdigit():
                    self.gpu.append({"t": round(time.time() - self.started, 3), "util": int(parts[0]), "mem_mb": float(parts[1])})
                if self.proc.poll() is not None:
                    break
        finally:
            g.kill()


def metrics(run):
    """Summary numbers for one finished run (same definitions as the web UI)."""
    R = run.result or {}
    res = R.get("resources") or {}
    smp, gpu = run.samples, run.gpu
    wall = R.get("times", {}).get("total", run.elapsed)
    cpu = (res["cpu_user_seconds"] + res["cpu_kernel_seconds"]) if "cpu_user_seconds" in res else (smp[-1]["cpu"] if smp else None)
    base = gpu[0]["mem_mb"] if gpu else None
    return {
        "wall": wall, "solve": R.get("times", {}).get("solve"), "cpu": cpu,
        "avg_cores": (cpu / wall) if (cpu is not None and wall) else None,
        "peak_cores": max(s["cores"] for s in smp if s["cores"] is not None) if sum(s["cores"] is not None for s in smp) >= 2 else None,
        "peak_ws": res.get("peak_working_set_bytes") or (max(s["ws"] for s in smp) if smp else None),
        "peak_priv": res.get("peak_private_bytes") or (max(s["priv"] for s in smp) if smp else None),
        "heap_peak": res.get("heap_peak_live_bytes"), "allocs": res.get("heap_allocations"),
        "alloc_bytes": res.get("heap_bytes_allocated"),
        "gpu_util": max(g["util"] for g in gpu) if gpu else None,
        "gpu_mem": (max(g["mem_mb"] for g in gpu) - base) if gpu else None,
        "gpu_engine": bool(re.search("gpu", R.get("engine", ""), re.I)),
    }


def phases(log):
    """Time per engine phase, from the solver's verbose log."""
    agg = {}
    def add(k, v):
        agg[k] = agg.get(k, 0.0) + v
    for L in log:
        m = re.search(r"presolve: .* in \d+ passes, \d+ steps, ([\d.]+)s", L)
        if m:
            add("presolve", float(m.group(1)))
        m = re.search(r"simplex time: (.*)\(LU fill", L)
        if m:
            for k, v in re.findall(r"([a-z]+) ([\d.]+)", m.group(1)):
                add("simplex: " + k, float(v))
        m = re.search(r"heuristic (\S+)\s+([\d.]+)s", L)
        if m:
            add("MILP heuristic: " + m.group(1), float(m.group(2)))
        m = re.search(r"ipm: .*analyze ([\d.]+)s", L)
        if m:
            add("IPM: ordering + analysis", float(m.group(1)))
    return sorted(((k, v) for k, v in agg.items() if v > 0), key=lambda kv: -kv[1])


def structure(log):
    s = {}
    for L in log:
        m = re.search(r"presolve: (\d+) x (\d+) -> (\d+) x (\d+)", L)
        if m:
            s["presolve"] = "%s x %s -> %s x %s" % tuple(format(int(v), ",") for v in m.groups())
        m = re.search(r"LU fill (\d+)", L)
        if m:
            s["lu_fill"] = max(s.get("lu_fill", 0), int(m.group(1)))
        m = re.search(r"nnz\(L\) (\d+)", L)
        if m:
            s["ldl_nnz"] = max(s.get("ldl_nnz", 0), int(m.group(1)))
        m = re.search(r"probing: (\d+) binaries probed, (\d+) fixed.*?(\d+) implications, (\d+) cliques", L)
        if m:
            s["probing"] = "%s probed, %s fixed, %s implications, %s cliques" % m.groups()
    return s


def generate(exe, family, size, out_dir):
    """Model families for scaling studies; returns the path of the new model."""
    os.makedirs(out_dir, exist_ok=True)
    if family == "transport":
        out = os.path.join(out_dir, "transport_%dx%d.mps" % (size, size))
        gen = os.path.join(os.path.dirname(exe), "gen_lp" + (".exe" if os.name == "nt" else ""))
        if not os.path.isfile(gen):
            raise ValueError("gen_lp not built next to the solver")
        cmd = [gen, str(size), str(size), out, "1"]
    elif family in ("refinery_milp", "refinery_lp", "refinery_qp"):
        out = os.path.join(out_dir, "%s_%dx%d.mps" % (family, size, size))
        cmd = [sys.executable, os.path.join(ROOT, "tools", "gen_refinery.py"), "--crudes", str(size), "--periods", str(size), "--out", out]
        if family == "refinery_lp":
            cmd.append("--lp")
        if family == "refinery_qp":
            cmd.append("--qp")
    else:
        raise ValueError("unknown family " + family)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if r.returncode != 0 or not os.path.isfile(out):
        raise ValueError((r.stderr or r.stdout or "generator failed").strip())
    return out


FAMILIES = {
    "transport": ("Transportation LP: S suppliers x S customers (S^2 variables, 2 S^2 nonzeros)", "concurrent"),
    "refinery_lp": ("Refinery planning LP: k crudes x k periods", "concurrent"),
    "refinery_milp": ("Refinery planning MILP: k crudes x k periods, cargo and unit binaries", "mip"),
    "refinery_qp": ("Refinery planning convex QP: k crudes x k periods", "ipm"),
}


def fit_power(points, key):
    """Least-squares fit of log(key) = c + k log(nnz); returns (k, c, r2) or None."""
    import math
    p = [q for q in points if q.get("nnz", 0) > 0 and (q.get(key) or 0) > 0 and q.get("status") in ("optimal", "near_optimal")]
    if len(p) < 2:
        return None
    xs = [math.log10(q["nnz"]) for q in p]
    ys = [math.log10(q[key]) for q in p]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    if sxx == 0:
        return None
    k = sxy / sxx
    return k, my - k * mx, (sxy * sxy / (sxx * syy)) if syy > 0 else 1.0


def check(exe, model_path, sol_path):
    """Run the independent verifier on a solution file."""
    out = subprocess.run([exe, "--check=" + sol_path, model_path], capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise ValueError((out.stderr or out.stdout).strip() or "check failed")
    g = lambda k: (re.search(k + r":\s+(\S+)", out.stdout) or [None, None])[1]
    num = lambda v: None if v is None or "nan" in v.lower() else float(v)
    return {"objective": num(g("Objective")), "eps_P": num(g("eps_P")), "eps_D": num(g("eps_D")),
            "eps_G": num(g("eps_G")), "int_violation": num(g("max integrality violation")), "raw": out.stdout}
