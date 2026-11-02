"""
Step 2, in parallel - the same work as step2_patches.py, but several scans at once.

Each worker process downloads, reads and crops ONE scan with step2_patches.process_scan,
exactly as before. Only the main process writes index.csv, failures.csv and done.txt,
so the files never get mixed up. It uses the same done.txt, so you can switch between
this script and step2_patches.py at any time; both resume where the other stopped.

Disk: at most WORKERS scans are in tmp_dicom/ at once (~100 MB each).

Run : python step2_parallel.py              (6 workers)
      python step2_parallel.py --workers 8
"""
import argparse
import json
import shutil
import time
import zlib
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from types import SimpleNamespace

import numpy as np
import pandas as pd

from config import ANNOTATIONS, COHORT_DIR, MAX_GB, PATCH_DIR, SEED, TMP_DIR
from step2_patches import append_csv, process_scan

client = None   # one IDCClient per worker process


def start_worker():
    global client
    import logging
    from idc_index import IDCClient
    logging.disable(logging.INFO)    # idc-index logs three lines per download
    client = IDCClient()


def run_one(scan, boxes, nodules):
    """Runs in a worker. Same as one iteration of step2_patches.main()."""
    rng = np.random.default_rng([SEED, zlib.crc32(scan.SeriesInstanceUID.encode())])  # same points every run
    try:
        rows, problems = process_scan(scan, boxes, nodules, client, rng)
    except Exception as e:
        rows, problems = [], [f"scan failed: {e}"]
    return scan, rows, problems


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    scans = pd.read_csv(COHORT_DIR / "scans.csv")
    split = pd.read_csv(COHORT_DIR / "patients.csv").set_index("pid")["split"]
    nodules = pd.read_csv(COHORT_DIR / "control_nodules.csv")
    annotations = json.load(open(ANNOTATIONS))

    todo = scans[scans["annotated"] | scans["SeriesInstanceUID"].isin(nodules["SeriesInstanceUID"])]
    todo = todo.sort_values("annotated", ascending=False)       # cancers first
    total = len(todo)

    PATCH_DIR.mkdir(exist_ok=True)
    shutil.rmtree(TMP_DIR, ignore_errors=True)       # half-downloaded scans from a killed run
    done_file = PATCH_DIR / "done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    todo = todo[~todo["SeriesInstanceUID"].isin(done)]
    used_gb = sum(f.stat().st_size for f in PATCH_DIR.glob("*.npy")) / 1e9
    n_done = total - len(todo)
    print(f"{n_done} of {total} scans already done, {len(todo)} to go, {args.workers} workers", flush=True)

    queue = iter(todo.itertuples(index=False))
    running, t0 = {}, time.time()
    with ProcessPoolExecutor(args.workers, initializer=start_worker) as pool:
        try:
            while True:
                while len(running) < 2 * args.workers:          # keep every worker busy
                    row = next(queue, None)
                    if row is None:
                        break
                    in_flight = sum(s.series_size_MB for s in running.values()) / 1000
                    if used_gb + in_flight + row.series_size_MB / 1000 > MAX_GB:
                        print(f"Stopping: the next scan would take us over {MAX_GB} GB.", flush=True)
                        queue = iter(())
                        break
                    scan = SimpleNamespace(**row._asdict())       # picklable, unlike itertuples rows
                    uid = scan.SeriesInstanceUID
                    future = pool.submit(run_one, scan, annotations.get(uid) if scan.annotated else None,
                                         nodules[nodules["SeriesInstanceUID"] == uid])
                    running[future] = scan
                if not running:
                    break

                finished, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in finished:
                    del running[future]
                    scan, rows, problems = future.result()
                    for r in rows:
                        r["split"] = split[scan.pid]
                    append_csv(PATCH_DIR / "index.csv", rows)
                    append_csv(PATCH_DIR / "failures.csv",
                               [{"series_uid": scan.SeriesInstanceUID, "pid": scan.pid, "problem": p}
                                for p in problems])
                    with open(done_file, "a") as f:
                        f.write(scan.SeriesInstanceUID + "\n")

                    n_done += 1
                    used_gb += sum((PATCH_DIR / r["file"]).stat().st_size for r in rows) / 1e9
                    print(f"[{n_done}/{total}] pid {scan.pid} T{scan.study_yr}: "
                          f"{len(rows)} patches, {len(problems)} problems, {used_gb:.1f} GB used", flush=True)
        except KeyboardInterrupt:
            print("Stopping: finishing nothing new; run again to resume.", flush=True)
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    hours = (time.time() - t0) / 3600
    print(f"Finished in {hours:.1f} h", flush=True)


if __name__ == "__main__":
    main()
