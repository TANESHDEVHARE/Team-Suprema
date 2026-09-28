"""Local web interface for the sovereign solver.

    python tools/ui_server.py [--exe PATH] [--port 8765]

then open http://127.0.0.1:8765 . Python standard library only; the page
(ui/index.html) loads nothing from the internet. The server listens on
127.0.0.1 only, keeps every upload and run under ui_runs/, and never touches a
path it did not create itself. Solving, sampling and output parsing are shared
with the command-line showcase through tools/solver_runner.py.

API (JSON unless noted)
  GET  /api/info                      solver binary, CPU threads, GPU name
  GET  /api/samples                   bundled sample models
  POST /api/upload?name=<file>        body = model bytes (.mps/.qps, optionally .gz)
  POST /api/sample                    {"name": ...}  copy a bundled sample
  POST /api/generate                  {"family", "size"}  model for a scaling study
  POST /api/solve                     {"model_id", "engine", "time_limit", ...}
  GET  /api/job/<id>?since=<n>        state, new log lines, samples, result when done
  POST /api/cancel/<id>
  GET  /api/download/<id>/<kind>      kind: sol | json | variables.csv | constraints.csv | log
  POST /api/check/<model_id>          body = solution text; runs the verifier on it
"""
import argparse, gzip, json, os, re, subprocess, sys, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import solver_runner as sr

ROOT = sr.ROOT
RUNS = os.path.join(ROOT, "ui_runs")
UI = os.path.join(ROOT, "ui", "index.html")
SAMPLES = os.path.join(ROOT, "sample_problems")
MAX_UPLOAD = 1 << 30          # 1 GiB
ID_RE = re.compile(r"^[0-9a-f]{12}$")

GPU = None                    # GPU name from nvidia-smi, or None
MODELS = {}                   # model_id -> {"path", "name", "size", "stats"}
JOBS = {}                     # job_id -> {"run": sr.Run, "dir", "model_name"}


class Handler(BaseHTTPRequestHandler):
    exe = None
    server_version = "SovereignUI/1.1"

    def log_message(self, fmt, *a):          # quiet
        pass

    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def fail(self, msg, code=400):
        self.send_json({"error": msg}, code)

    def body(self, limit=MAX_UPLOAD):
        n = int(self.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("upload larger than %d MB" % (limit >> 20))
        return self.rfile.read(n) if n else b""

    # ------------------------------------------------------------------ GET
    def do_GET(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        try:
            if u.path in ("/", "/index.html"):
                with open(UI, "rb") as f:
                    data = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            elif u.path == "/api/info":
                self.send_json({"exe": self.exe, "threads": os.cpu_count(), "gpu": GPU})
            elif u.path == "/api/samples":
                items = []
                for fn in sorted(os.listdir(SAMPLES)):
                    if fn.lower().endswith((".mps", ".qps")):
                        items.append({"name": fn, "size": os.path.getsize(os.path.join(SAMPLES, fn))})
                self.send_json({"samples": items})
            elif len(parts) == 3 and parts[:2] == ["api", "job"]:
                job = JOBS.get(parts[2])
                if not job:
                    return self.fail("unknown job", 404)
                run = job["run"]
                since = int(parse_qs(u.query).get("since", ["0"])[0])
                log = run.log
                self.send_json({"state": run.state, "log": log[since:], "next": len(log), "elapsed": run.elapsed,
                                "error": run.error, "result": run.result if run.state == "done" else None,
                                "samples": run.samples, "gpu": run.gpu,
                                "command": " ".join(os.path.basename(a) if i == 0 else a for i, a in enumerate(run.args))})
            elif len(parts) == 4 and parts[:2] == ["api", "download"]:
                job = JOBS.get(parts[2])
                files = {"sol": "solution.sol", "json": "result.json", "variables.csv": "solution_variables.csv",
                         "constraints.csv": "solution_constraints.csv"}
                if not job or (parts[3] not in files and parts[3] != "log"):
                    return self.fail("not found", 404)
                if parts[3] == "log":
                    data = ("\n".join(job["run"].log) + "\n").encode()
                else:
                    fp = os.path.join(job["dir"], files[parts[3]])
                    if not os.path.isfile(fp):
                        return self.fail("not available for this run", 404)
                    with open(fp, "rb") as f:
                        data = f.read()
                stem = os.path.splitext(job["model_name"])[0]
                fname = "%s_%s" % (stem, "solution.log" if parts[3] == "log" else files[parts[3]].replace("solution_", ""))
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Disposition", 'attachment; filename="%s"' % fname)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self.fail("not found", 404)
        except Exception as e:
            self.fail(str(e), 500)

    # ----------------------------------------------------------------- POST
    def do_POST(self):
        u = urlparse(self.path)
        parts = [p for p in u.path.split("/") if p]
        try:
            if u.path == "/api/upload":
                name = os.path.basename(parse_qs(u.query).get("name", ["model.mps"])[0]) or "model.mps"
                data = self.body()
                if name.lower().endswith(".gz"):
                    data = gzip.decompress(data)
                    name = name[:-3]
                if not name.lower().endswith((".mps", ".qps")):
                    name += ".mps"
                self.send_json(self.register(name, data))
            elif u.path == "/api/sample":
                req = json.loads(self.body(1 << 16) or b"{}")
                fn = os.path.basename(req.get("name", ""))
                src = os.path.join(SAMPLES, fn)
                if not fn or not os.path.isfile(src):
                    return self.fail("unknown sample")
                with open(src, "rb") as f:
                    self.send_json(self.register(fn, f.read()))
            elif u.path == "/api/generate":
                req = json.loads(self.body(1 << 16) or b"{}")
                size = int(req.get("size") or 0)
                if size < 2 or size > 5000:
                    return self.fail("size must be between 2 and 5000")
                out = sr.generate(self.exe, req.get("family", ""), size, os.path.join(RUNS, "generated"))
                with open(out, "rb") as f:
                    self.send_json(self.register(os.path.basename(out), f.read()))
            elif u.path == "/api/solve":
                req = json.loads(self.body(1 << 16) or b"{}")
                mid = req.get("model_id", "")
                if not ID_RE.match(mid) or mid not in MODELS:
                    return self.fail("upload a model first")
                jid = uuid.uuid4().hex[:12]
                jdir = os.path.join(RUNS, "jobs", jid)
                os.makedirs(jdir)
                args = sr.build_args(self.exe, MODELS[mid]["path"], jdir, req)
                run = sr.Run(args, jdir, sample_gpu=GPU is not None).start()
                JOBS[jid] = {"run": run, "dir": jdir, "model_name": MODELS[mid]["name"]}
                self.send_json({"job_id": jid})
            elif len(parts) == 3 and parts[:2] == ["api", "cancel"]:
                job = JOBS.get(parts[2])
                if not job:
                    return self.fail("unknown job", 404)
                job["run"].cancel()
                self.send_json({"state": job["run"].state})
            elif len(parts) == 3 and parts[:2] == ["api", "check"]:
                mid = parts[2]
                if not ID_RE.match(mid) or mid not in MODELS:
                    return self.fail("unknown model", 404)
                sol = self.body(512 << 20)
                sp = os.path.join(RUNS, "models", mid, "check_%s.sol" % uuid.uuid4().hex[:8])
                with open(sp, "wb") as f:
                    f.write(sol)
                self.send_json(check(self.exe, MODELS[mid]["path"], sp))
            else:
                self.fail("not found", 404)
        except (ValueError, json.JSONDecodeError, OSError) as e:
            self.fail(str(e))
        except Exception as e:
            self.fail(str(e), 500)

    def register(self, name, data):
        mid = uuid.uuid4().hex[:12]
        d = os.path.join(RUNS, "models", mid)
        os.makedirs(d)
        path = os.path.join(d, name)
        with open(path, "wb") as f:
            f.write(data)
        stats = sr.model_stats(self.exe, path)
        MODELS[mid] = {"path": path, "name": name, "size": len(data), "stats": stats}
        return {"model_id": mid, "name": name, "size": len(data), "stats": stats}


check = sr.check


def main():
    global GPU
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    Handler.exe = sr.find_exe(a.exe)
    GPU = sr.gpu_name()
    os.makedirs(os.path.join(RUNS, "models"), exist_ok=True)
    os.makedirs(os.path.join(RUNS, "jobs"), exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print("Sovereign solver UI: http://127.0.0.1:%d   (solver: %s)" % (a.port, Handler.exe), flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
