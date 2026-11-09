"""
Dashboard - two tabs, one per stage.

Stage 1: starts/stops step 2 (step2_parallel.py), shows its progress, and every EVAL_EVERY new
scans retrains the CNN on the patches so far (dashboard/evaluate.py) and records the step 3c/3d
numbers, so you can follow them as the data grows.

Stage 2: starts/stops step 4a (step4_sequences.py) and shows its progress and alignment check
live; runs step4_check.py, step4_embed.py and step4_tracker.py (one at a time, queued) and shows
the headroom, the tracker results and their training curves. Test runs (--test) are left to the
command line on purpose: they are run once, at the very end.

Run : python dashboard/server.py        then open http://localhost:8050  (Stage 2: /#stage2)
"""
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from config import CHECK_DIR, COHORT_DIR, MAX_GB, MODEL_DIR, PATCH_DIR, ROOT, TRACK_DIR  # noqa: E402
from step4_check import read_tracks, summary  # noqa: E402
from step4_sequences import make_jobs  # noqa: E402

PORT = 8050
WINDOWS = sys.platform == "win32"
STEP2_WORKERS = 6     # scans downloaded and processed at the same time (step2_parallel.py)
STEP4_WORKERS = 6     # patients processed at the same time (step4_sequences.py)
TRACKER_PLAN = [      # CONTEXT.md 7.5, validation only: "Run the evaluation plan" queues these
    ["--scans", "2", "--overfit"], ["--scans", "2"], ["--scans", "3"],
    ["--scans", "2", "--early"], ["--scans", "1"], ["--scans", "3", "--plain"],
]
EVAL_EVERY = None     # retrain + evaluate every this many new scans; None = only once, when step 2 is done
EVAL_SPLIT = "val"    # keep "val": evaluating "test" over and over turns it into a second val set
EVAL_EPOCHS = 120    # epochs per checkpoint retrain (step3_train.py still uses config.EPOCHS)
RUNS_DIR = HERE / "runs"
HISTORY = RUNS_DIR / "history.json"


class Job:
    """One child process at a time (step 2, an evaluation, ...) with the tail of its output.
    queue() lines up more commands: each starts when the previous one exits."""

    def __init__(self, name):
        self.name, self.proc, self.started, self.stopped = name, None, None, 0.0
        self.log = deque(maxlen=300)
        self.label, self.todo, self.lock = None, deque(), threading.Lock()

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, args, label=None, clear=True):
        if clear:
            self.log.clear()
        self.started, self.label = time.time(), label
        self.proc = subprocess.Popen([sys.executable, "-u", *args], cwd=ROOT, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                                     start_new_session=not WINDOWS)   # macOS/Linux: own group, for stop()
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()

    def queue(self, commands):
        """commands: [(label, args), ...]. Starts the first one now if nothing is running."""
        with self.lock:
            first = not self.running() and not self.todo
            self.todo.extend(commands)
            if first:
                self.log.clear()
                self._next()

    def _next(self):
        if self.todo and not self.running():
            label, args = self.todo.popleft()
            self.log.append(f"{datetime.now():%H:%M:%S}  ===== {label}")
            self.start(args, label, clear=False)

    def _pump(self, proc):
        for line in proc.stdout:
            line = line.rstrip()
            if line and " - Disk size" not in line and " - Not using s5cmd" not in line \
                    and " - Successfully downloaded" not in line:   # idc-index chatter
                self.log.append(f"{datetime.now():%H:%M:%S}  {line}")
        self.log.append(f"{datetime.now():%H:%M:%S}  [exited with code {proc.wait()}]")
        with self.lock:
            self._next()

    def stop(self):
        self.todo.clear()
        if self.running():   # kill the whole tree: the venv launcher starts a second python, plus workers
            if WINDOWS:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
            else:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            self.stopped = time.time()


step2, evaluation = Job("step2"), Job("eval")
step4a, stage2 = Job("step4a"), Job("stage2")
progress = deque(maxlen=2000)     # (time, scans done) samples, for the rate and ETA
lock = threading.Lock()


def todo_total():
    """Same to-do list as step2_patches.main(): annotated scans + controls with a nodule."""
    scans = pd.read_csv(COHORT_DIR / "scans.csv")
    nodules = pd.read_csv(COHORT_DIR / "control_nodules.csv")
    return int((scans["annotated"] | scans["SeriesInstanceUID"].isin(nodules["SeriesInstanceUID"])).sum())


TOTAL = todo_total()


def scans_done():
    f = PATCH_DIR / "done.txt"
    return len(set(f.read_text().split())) if f.exists() else 0


def read_index():
    f = PATCH_DIR / "index.csv"
    if not f.exists():
        return pd.DataFrame(columns=["file", "kind", "split", "pid"])
    return pd.read_csv(f, on_bad_lines="skip")   # step 2 may be halfway through writing a line


def no_nan(x):
    """NaN is not valid JSON: the browser would reject the whole status. Send null instead."""
    if isinstance(x, float) and x != x:
        return None
    if isinstance(x, dict):
        return {k: no_nan(v) for k, v in x.items()}
    if isinstance(x, list):
        return [no_nan(v) for v in x]
    return x


def history():
    h = json.loads(HISTORY.read_text()) if HISTORY.exists() else []
    for r in h:                       # older runs did not record their epochs: count them
        if "epochs" not in r and (RUNS_DIR / r["run"] / "result.json").exists():
            r["epochs"] = len(json.loads((RUNS_DIR / r["run"] / "result.json").read_text())["curve"])
    return h


def run_curve(run):
    """Train loss / val AUC per epoch of one evaluation: live while it trains, final afterwards."""
    for name in ("curve.json", "result.json"):
        f = RUNS_DIR / run / name
        if f.exists():
            try:
                d = json.loads(f.read_text())
            except ValueError:          # caught it halfway through writing; next poll will work
                continue
            return {"run": run, "epochs": d.get("epochs", len(d["curve"])), "scans_done": d["scans_done"],
                    "n_train": d["n_train"], "curve": d["curve"]}
    return None


def curves(h):
    """The newest evaluation (running or done) and the finished one before it, for comparison."""
    finished = [r["run"] for r in h]
    runs = sorted(p.name for p in RUNS_DIR.glob("*_*") if p.is_dir())
    current = runs[-1] if runs else None
    if current and current not in finished and not evaluation.running():
        current = finished[-1] if finished else None      # an evaluation that crashed: skip it
    previous = [r for r in finished if r != current]
    return {"current": run_curve(current) if current else None,
            "previous": run_curve(previous[-1]) if previous else None}


SCRIPTS = {"step2": (r"step2_(patches|parallel)\.py", step2), "step4a": (r"step4_sequences\.py", step4a)}
_external = {name: {"t": 0.0, "pids": []} for name in SCRIPTS}


def external(name):
    """PIDs of step 2 / 4a processes this dashboard did not start (e.g. run from a terminal). Checked every 10 s."""
    pattern, job = SCRIPTS[name]
    cache = _external[name]
    if time.time() - cache["t"] > 10:
        if WINDOWS:
            ps = (f"Get-CimInstance Win32_Process | Where-Object {{ $_.CommandLine -match '{pattern}' }}"
                  " | ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId)\" }")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.split()
            pairs = list(zip(map(int, out[::2]), map(int, out[1::2])))
        else:   # macOS / Linux: every process as "pid ppid command line"
            out = subprocess.run(["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True).stdout
            rows = [line.split(None, 2) for line in out.splitlines()]
            pairs = sorted((int(r[0]), int(r[1])) for r in rows if len(r) == 3 and re.search(pattern, r[2]))
        ours = {job.proc.pid} if job.running() else set()
        for pid, parent in pairs:                     # our launcher, its python, and their workers
            if parent in ours:
                ours.add(pid)
        cache.update(t=time.time(), pids=[pid for pid, _ in pairs if pid not in ours])
    return cache["pids"]


def status():
    done, now = scans_done(), time.time()
    progress.append((now, done))
    recent = [(t, d) for t, d in progress if now - t < 1800]          # last 30 minutes
    rate = None
    if len(recent) > 1 and recent[-1][1] > recent[0][1]:
        rate = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0]) * 3600   # scans/hour

    index = read_index()
    counts = index.groupby(["split", "kind"]).size().unstack(fill_value=0) if len(index) else pd.DataFrame()
    problems = {}
    if (PATCH_DIR / "failures.csv").exists():
        p = pd.read_csv(PATCH_DIR / "failures.csv", on_bad_lines="skip")["problem"]
        problems = p.str.replace(r"\d+", "N", regex=True).value_counts().head(8).to_dict()

    ext = not step2.running() and bool(external("step2"))   # step 2 started from a terminal
    h = history()
    last = [l for l in step2.log if re.search(r"^\S+\s+\[\d+/\d+\]", l)]
    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "step2": {
            "state": "running" if step2.running() else "external" if ext
                     else "finished" if done >= TOTAL else "stopped",
            "done": done, "total": TOTAL, "rate": rate,
            "eta_h": (TOTAL - done) / rate if rate else None,
            "last": last[-1].split("  ", 1)[1] if last else None,
            "log": list(step2.log)[-40:],
        },
        "patches": {
            "total": int(len(index)),
            "patients": int(index["pid"].nunique()) if len(index) else 0,
            "by_split": {s: {k: int(v) for k, v in row.items()} for s, row in counts.iterrows()},
            "gb": sum(f.stat().st_size for f in PATCH_DIR.glob("*.npy")) / 1e9 if PATCH_DIR.exists() else 0,
            "max_gb": MAX_GB,
        },
        "problems": problems,
        "eval": {
            "running": evaluation.running(),
            "log": list(evaluation.log)[-25:],
            "every": EVAL_EVERY, "split": EVAL_SPLIT,
            "next_at": TOTAL if EVAL_EVERY is None else (h[-1]["scans_done"] if h else 0) + EVAL_EVERY,
            "history": h,
            "epochs": EVAL_EPOCHS,
            **curves(h),
        },
    }


# ---------- Stage 2 ----------------------------------------------------------------------------

progress4 = deque(maxlen=2000)    # (time, patients done) samples, for step 4a's rate and ETA
_cache = {}


def cached(key, stamp, compute):
    """compute() again only when `stamp` (a file's mtime, a time bucket) has changed."""
    if key not in _cache or _cache[key][0] != stamp:
        _cache[key] = (stamp, compute())
    return _cache[key][1]


def mtime(path):
    return path.stat().st_mtime if path.exists() else None


def read_json(path):
    try:
        return json.loads(path.read_text()) if path.exists() else None
    except ValueError:              # caught it halfway through writing; next poll will work
        return None


def disk_gb():
    """Patches + track cubes on disk. Thousands of files: recounted at most every 15 s."""
    return cached("disk", int(time.time() // 15), lambda: {
        d.name: sum(f.stat().st_size for f in d.glob("*.npy")) / 1e9 if d.exists() else 0.0
        for d in (PATCH_DIR, TRACK_DIR)})


def read_results():
    f = MODEL_DIR / "stage2_results.csv"
    if not f.exists():
        return []
    rows = pd.read_csv(f).to_dict("records")
    for r in rows:
        r["by_anchor"] = json.loads(r["by_anchor"]) if isinstance(r.get("by_anchor"), str) else {}
    return rows


def stage2_status():
    done_file = TRACK_DIR / "done.txt"
    done = len(set(done_file.read_text().split())) if done_file.exists() else 0
    total = cached("total", mtime(PATCH_DIR / "index.csv"), lambda: len(make_jobs()))
    now = time.time()
    progress4.append((now, done))
    recent = [(t, d) for t, d in progress4 if now - t < 1800]          # last 30 minutes
    rate = None
    if len(recent) > 1 and recent[-1][1] > recent[0][1]:
        rate = (recent[-1][1] - recent[0][1]) / (recent[-1][0] - recent[0][0]) * 3600   # patients/hour

    ext = not step4a.running() and bool(external("step4a"))
    tracks_csv, emb = TRACK_DIR / "tracks.csv", TRACK_DIR / "embeddings.npz"
    live = cached("summary", mtime(tracks_csv), lambda: summary(read_tracks())) if tracks_csv.exists() else None
    last = [l for l in step4a.log if re.search(r"^\S+\s+\[\d+/\d+\]", l)]
    gb = disk_gb()
    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "step4a": {
            "state": "running" if step4a.running() else "external" if ext
                     else "finished" if total and done >= total else "stopped",
            "done": done, "total": total, "rate": rate,
            "eta_h": (total - done) / rate if rate else None,
            "last": last[-1].split("  ", 1)[1] if last else None,
            "log": list(step4a.log)[-40:],
            "step2_running": step2.running() or bool(external("step2")),
        },
        "disk": {"patches": gb["patches"], "tracks": gb["tracks"], "max_gb": MAX_GB},
        "summary": live,
        "check": read_json(CHECK_DIR / "stage2_check.json"),
        "embed": {"exists": emb.exists(),
                  "stale": bool(emb.exists() and tracks_csv.exists() and mtime(emb) < mtime(tracks_csv)),
                  **(read_json(CHECK_DIR / "stage2_embed.json") or {})},
        "results": cached("results", mtime(MODEL_DIR / "stage2_results.csv"), read_results),
        "grids": {k: mtime(CHECK_DIR / f"tracks_{k}.png") for k in ("cancer", "benign")},
        "job": {"running": stage2.running(), "label": stage2.label if stage2.running() else None,
                "queue": [label for label, _ in stage2.todo], "log": list(stage2.log)[-60:]},
    }


def stage2_commands(req):
    """What the Stage 2 buttons may run: [(label, args), ...]. Never --test."""
    what = req.get("what")
    if what == "check":
        return [("step4_check", ["step4_check.py"])]
    if what == "embed":
        return [("step4_embed", ["step4_embed.py"])]
    if what == "plan":
        return [("tracker " + " ".join(a), ["step4_tracker.py", *a]) for a in TRACKER_PLAN]
    if what == "tracker":
        a = ["--scans", str(min(3, max(1, int(req.get("scans", 2)))))]
        a += [f"--{flag}" for flag in ("early", "plain", "overfit") if req.get(flag)]
        return [("tracker " + " ".join(a), ["step4_tracker.py", *a])]
    return []


def start_eval():
    with lock:
        if evaluation.running():
            return
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        evaluation.start([str(HERE / "evaluate.py"), "--run", run_id, "--split", EVAL_SPLIT,
                          "--epochs", str(EVAL_EPOCHS), "--scans-done", str(scans_done())])


def scheduler():
    """Evaluate when step 2 finishes, and (unless EVAL_EVERY is None) at the start and every EVAL_EVERY scans."""
    while True:
        time.sleep(30)
        if evaluation.running() or not (PATCH_DIR / "index.csv").exists():
            continue
        if next(PATCH_DIR.glob("*.npy"), None) is None:   # results-only copy (e.g. a git clone): nothing to train on
            continue
        done, h = scans_done(), history()
        last = h[-1]["scans_done"] if h else None
        final = done >= TOTAL and (last is None or last < done)
        periodic = EVAL_EVERY is not None and (last is None or done - last >= EVAL_EVERY)
        if final or periodic:
            start_eval()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, body, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send((HERE / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self.send(json.dumps(no_nan(status()), allow_nan=False).encode(), "application/json")
        elif self.path == "/api/stage2":
            self.send(json.dumps(no_nan(stage2_status()), allow_nan=False).encode(), "application/json")
        elif m := re.fullmatch(r"/api/stage2/curve\?setting=(k\d(?:_early)?(?:_plain)?(?:_s\d+)?)", self.path):
            f = MODEL_DIR / f"tracker_{m.group(1)}_log.csv"
            curve = pd.read_csv(f).to_dict("records") if f.exists() else []
            self.send(json.dumps(no_nan({"setting": m.group(1), "curve": curve})).encode(), "application/json")
        elif m := re.fullmatch(r"/runs/(\d{8}_\d{6})/gradcam\.png", self.path.split("?")[0]):
            f = RUNS_DIR / m.group(1) / "gradcam.png"
            self.send(f.read_bytes(), "image/png") if f.exists() else self.send(b"", "text/plain", 404)
        elif m := re.fullmatch(r"/checks/(tracks_(?:cancer|benign))\.png", self.path.split("?")[0]):
            f = CHECK_DIR / f"{m.group(1)}.png"
            self.send(f.read_bytes(), "image/png") if f.exists() else self.send(b"", "text/plain", 404)
        else:
            self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/api/step2/start" and not step2.running():
            step2.start(["step2_parallel.py", "--workers", str(STEP2_WORKERS)])
        elif self.path == "/api/step2/stop":
            step2.stop()
        elif self.path == "/api/eval/start":
            start_eval()
        elif self.path == "/api/step4a/start" and not step4a.running():
            step4a.start(["step4_sequences.py", "--workers", str(STEP4_WORKERS)])
        elif self.path == "/api/step4a/stop":
            step4a.stop()
        elif self.path == "/api/stage2/run":
            stage2.queue(stage2_commands(json.loads(body or b"{}")))
        elif self.path == "/api/stage2/stop":
            stage2.stop()
        self.send(b"{}", "application/json")


class Server(ThreadingHTTPServer):
    # On Windows, reuse lets a second dashboard share the port. On macOS/Linux it only lets a restarted
    # dashboard take the port back straight away (a second one still gets "port in use").
    allow_reuse_address = not WINDOWS

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class Server6(Server):
    address_family = socket.AF_INET6


if __name__ == "__main__":
    try:   # "localhost" can mean 127.0.0.1 or ::1 depending on the browser, so listen on both
        servers = [Server(("127.0.0.1", PORT), Handler), Server6(("::1", PORT), Handler)]
    except OSError:
        sys.exit(f"Port {PORT} is in use: is the dashboard already running? Open http://localhost:{PORT}")
    RUNS_DIR.mkdir(exist_ok=True)
    threading.Thread(target=scheduler, daemon=True).start()
    threading.Thread(target=servers[1].serve_forever, daemon=True).start()
    print(f"Dashboard on http://localhost:{PORT}  (Ctrl+C stops the dashboard AND every job it started)")
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        for job in (step2, evaluation, step4a, stage2):
            job.stop()
