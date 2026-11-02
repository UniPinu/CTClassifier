"""
Dashboard - watch step 2 run and see whether more data makes the CNN better.

Starts/stops step 2 (step2_parallel.py), shows its progress, and every EVAL_EVERY new scans
retrains the CNN on the patches so far (dashboard/evaluate.py) and records the
step 3c/3d numbers, so you can follow them as the data grows.

Run : python dashboard/server.py        then open http://localhost:8050
"""
import json
import re
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
from config import COHORT_DIR, MAX_GB, PATCH_DIR, ROOT  # noqa: E402

PORT = 8050
STEP2_WORKERS = 6     # scans downloaded and processed at the same time (step2_parallel.py)
EVAL_EVERY = None     # retrain + evaluate every this many new scans; None = only once, when step 2 is done
EVAL_SPLIT = "val"    # keep "val": evaluating "test" over and over turns it into a second val set
EVAL_EPOCHS = 120    # epochs per checkpoint retrain (step3_train.py still uses config.EPOCHS)
RUNS_DIR = HERE / "runs"
HISTORY = RUNS_DIR / "history.json"


class Job:
    """One child process (step 2 or an evaluation) with the tail of its output."""

    def __init__(self, name):
        self.name, self.proc, self.started, self.stopped = name, None, None, 0.0
        self.log = deque(maxlen=300)

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self, args):
        self.log.clear()
        self.started = time.time()
        self.proc = subprocess.Popen([sys.executable, "-u", *args], cwd=ROOT, text=True,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            line = line.rstrip()
            if line and " - Disk size" not in line and " - Not using s5cmd" not in line \
                    and " - Successfully downloaded" not in line:   # idc-index chatter
                self.log.append(f"{datetime.now():%H:%M:%S}  {line}")
        self.log.append(f"{datetime.now():%H:%M:%S}  [exited with code {self.proc.wait()}]")

    def stop(self):
        if self.running():   # kill the whole tree: the venv launcher starts a second python
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(self.proc.pid)], capture_output=True)
            self.stopped = time.time()


step2, evaluation = Job("step2"), Job("eval")
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


_external = {"t": 0.0, "pids": []}


def external_step2():
    """PIDs of step 2 processes this dashboard did not start (e.g. run from a terminal). Checked every 10 s."""
    if time.time() - _external["t"] > 10:
        ps = (r"Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'step2_(patches|parallel)\.py' }"
              " | ForEach-Object { \"$($_.ProcessId) $($_.ParentProcessId)\" }")
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.split()
        pairs = list(zip(map(int, out[::2]), map(int, out[1::2])))
        ours = {step2.proc.pid} if step2.running() else set()
        for pid, parent in pairs:                     # our launcher, its python, and their workers
            if parent in ours:
                ours.add(pid)
        _external.update(t=time.time(), pids=[pid for pid, _ in pairs if pid not in ours])
    return _external["pids"]


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

    external = not step2.running() and bool(external_step2())   # step 2 started from a terminal
    h = history()
    last = [l for l in step2.log if re.search(r"^\S+\s+\[\d+/\d+\]", l)]
    return {
        "time": datetime.now().strftime("%H:%M:%S"),
        "step2": {
            "state": "running" if step2.running() else "external" if external
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
        elif m := re.fullmatch(r"/runs/(\d{8}_\d{6})/gradcam\.png", self.path.split("?")[0]):
            f = RUNS_DIR / m.group(1) / "gradcam.png"
            self.send(f.read_bytes(), "image/png") if f.exists() else self.send(b"", "text/plain", 404)
        else:
            self.send(b"not found", "text/plain", 404)

    def do_POST(self):
        if self.path == "/api/step2/start" and not step2.running():
            step2.start(["step2_parallel.py", "--workers", str(STEP2_WORKERS)])
        elif self.path == "/api/step2/stop":
            step2.stop()
        elif self.path == "/api/eval/start":
            start_eval()
        self.send(b"{}", "application/json")


class Server(ThreadingHTTPServer):
    allow_reuse_address = False      # on Windows, reuse lets a second dashboard share the port

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
    print(f"Dashboard on http://localhost:{PORT}  (Ctrl+C stops the dashboard AND a step 2 it started)")
    try:
        servers[0].serve_forever()
    except KeyboardInterrupt:
        step2.stop()
        evaluation.stop()
