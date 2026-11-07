"""
Step 4a - Follow every known lesion back through the patient's earlier screenings.

A "track" is one lesion at several screenings, oldest first. It starts at the ANCHOR: the
newest screening where the lesion's position is known (a Sybil box, or a benign nodule that
step 2 found). From there it walks back one screening at a time:
  1. first guess : the same point, shifted by how much the lungs' centre moved between the scans
  2. refine      : look for the 48 mm cube of anatomy around the point in the earlier scan
                   (normalised cross-correlation, searching +-20 mm around the guess)
  3. snap        : if a known lesion of that year is within 15 mm, use its exact centre
Tracks never use a screening after their anchor, so no scan after the diagnosis is used.
Known lesions that no newer track reached start a track of their own.

Each patient is handled in one go: at most 2 of their scans are in memory, and the DICOM files
are deleted right after reading, as in step 2. Resumable: patients in tracks/done.txt are skipped.

Reads : cohort/scans.csv, cohort/patients.csv, patches/index.csv
Writes: tracks/*.npy         one 72^3 int16 cube per lesion per screening (cropped like step 2)
        tracks/tracks.csv    one row per cube: track_id, pid, label, anchor_yr, study_yr, source,
                             match, edge, nearest_mm, z, y, x, file, split
        tracks/failures.csv  scans we could not use
        tracks/done.txt      patients already processed

Run   : python step4_sequences.py               (6 patients at a time)
        python step4_sequences.py --workers 1   (one at a time: easier to debug)
        python step4_sequences.py --limit 10    (only the next 10 patients: a quick trial)
"""
import argparse
import shutil
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.feature import match_template

from config import (COHORT_DIR, MAX_GB, PATCH_DIR, SEARCH, SNAP_MM, STORE, TEMPLATE,
                    TMP_DIR, TRACK_DIR)
from step2_patches import append_csv, download, lung_mask, read_series, to_1mm

TMP = TMP_DIR / "step4"    # its own scratch folder, so a running step 2 is never disturbed
client = None              # one IDCClient per worker process


# ---------- 1. One scan in memory ----------------------------------------------------------

def start_worker():
    global client
    import logging
    from idc_index import IDCClient
    logging.disable(logging.INFO)          # idc-index logs three lines per download
    client = IDCClient()


def load_scan(uid):
    """Download -> read -> delete the DICOM files -> 1 mm voxels. Returns volume, lungs, info."""
    folder = TMP / uid
    for attempt in (1, 2):                 # a dropped connection loses a slice: download again once
        try:
            download(client, uid, folder)
            vol, info = read_series(folder)
            break
        except ValueError as e:
            if attempt == 2 or "missing slices" not in str(e):
                raise
        finally:
            shutil.rmtree(folder, ignore_errors=True)
    vol = to_1mm(vol, info["spacing"])
    lungs = lung_mask(vol)
    if not lungs.any():
        raise ValueError("no lungs found")
    return vol, lungs, info


def cut(vol, centre, size):
    """size^3 cube around `centre`, air (-1024) where it leaves the scan. The same crop as step 2."""
    lo = np.round(centre).astype(int) - size // 2
    out = np.full((size, size, size), -1024, dtype=vol.dtype)
    src = tuple(slice(min(max(l, 0), n), min(max(l + size, 0), n)) for l, n in zip(lo, vol.shape))
    dst = tuple(slice(s.start - l, s.stop - l) for s, l in zip(src, lo))
    out[dst] = vol[src]
    return out


# ---------- 2. Where is this point one screening earlier? -----------------------------------

def match(new_vol, old_vol, point, shift):
    """Find the anatomy around `point` (newer scan) in the older scan. Returns (point, score, edge)."""
    guess = np.round(point + shift).astype(int)                       # 1. lungs moved by `shift`
    template = np.clip(cut(new_vol, point, TEMPLATE), -1000, 400).astype(np.float32)
    region = np.clip(cut(old_vol, guess, TEMPLATE + 2 * SEARCH), -1000, 400).astype(np.float32)
    ncc = np.nan_to_num(match_template(region, template))            # 2. one score per offset
    best = np.array(np.unravel_index(np.argmax(ncc), ncc.shape))     #    (41, 41, 41) offsets
    edge = bool((best == 0).any() or (best == 2 * SEARCH).any())     #    best at the border?
    return guess + best - SEARCH, float(ncc.max()), edge


def run_patient(job):
    """All tracks of one patient, newest screening first. Returns (rows, problems)."""
    pid, tracks, problems = job["pid"], [], []
    newer = None                                       # (year, volume, lungs' centre)

    for yr, uid in reversed(job["screens"]):
        try:
            vol, lungs, _ = load_scan(uid)
            centre = np.array(ndimage.center_of_mass(lungs))
        except Exception as e:                         # skip this screening, match across it
            problems.append(f"T{yr} {uid}: {e}")
            continue
        known = job["known"].get(yr, [])               # [(z, y, x), label] of this screening
        claimed = set()

        if newer is not None:
            n_yr, n_vol, n_centre = newer
            for t in tracks:
                if n_yr not in t["points"]:            # this track ended before the newer scan
                    continue
                pos, score, edge = match(n_vol, vol, t["points"][n_yr]["pos"], centre - n_centre)
                dist = [np.linalg.norm(pos - np.array(k)) for k, _ in known]
                nearest = min(dist) if dist else np.nan
                source = "aligned"
                if dist and nearest <= SNAP_MM:        # 3. snap to the known lesion
                    i = int(np.argmin(dist))
                    pos, source = np.array(known[i][0]), "known"
                    claimed.add(i)
                t["points"][yr] = {"pos": pos, "source": source, "match": score,
                                   "edge": edge, "nearest_mm": nearest}

        for i, (k, label) in enumerate(known):         # known lesions no newer track reached
            if i not in claimed:
                tracks.append({"anchor": yr, "label": label, "points": {yr: {
                    "pos": np.array(k), "source": "known", "match": np.nan,
                    "edge": False, "nearest_mm": 0.0}}})

        for no, t in enumerate(tracks):                # crop this screening's cubes
            if yr in t["points"]:
                np.save(TRACK_DIR / f"{pid}_{no}_T{yr}.npy", cut(vol, t["points"][yr]["pos"], STORE))
        newer = (yr, vol, centre)

    rows = []
    for no, t in enumerate(tracks):
        for yr, p in sorted(t["points"].items()):     # oldest first
            z, y, x = (int(v) for v in p["pos"])
            rows.append({"track_id": f"{pid}_{no}", "pid": pid, "label": t["label"],
                         "anchor_yr": t["anchor"], "study_yr": yr, "source": p["source"],
                         "match": round(p["match"], 4), "edge": p["edge"],
                         "nearest_mm": round(p["nearest_mm"], 1), "z": z, "y": y, "x": x,
                         "file": f"{pid}_{no}_T{yr}.npy", "split": job["split"]})
    return rows, problems


def safe_run(job):
    try:
        return run_patient(job)
    except Exception as e:                             # bad patient: log it and move on
        return [], [f"patient failed: {e}"]


# ---------- 3. Which patients, and run them -------------------------------------------------

def make_jobs():
    """Patients with >= 2 screenings up to their newest known lesion. Cancers first."""
    scans = pd.read_csv(COHORT_DIR / "scans.csv")
    patients = pd.read_csv(COHORT_DIR / "patients.csv").set_index("pid")
    index = pd.read_csv(PATCH_DIR / "index.csv")
    known = index[index["kind"].isin(["cancer", "benign"])]
    # positions are only valid in the series step 2 cut them from: the one in scans.csv
    known = known.merge(scans[["pid", "study_yr", "SeriesInstanceUID"]],
                        left_on=["pid", "study_yr", "series_uid"],
                        right_on=["pid", "study_yr", "SeriesInstanceUID"])
    anchor = known.groupby("pid")["study_yr"].max()
    scans = scans[scans["pid"].isin(anchor.index)]
    scans = scans[scans["study_yr"] <= scans["pid"].map(anchor)].sort_values(["pid", "study_yr"])
    by_pid = dict(tuple(known.groupby("pid")))

    jobs = []
    for pid, s in scans.groupby("pid"):
        if len(s) < 2:
            continue
        jobs.append({
            "pid": int(pid), "split": patients.loc[pid, "split"], "label": int(patients.loc[pid, "label"]),
            "screens": list(zip(s["study_yr"].astype(int), s["SeriesInstanceUID"])),
            "size_gb": s["series_size_MB"].max() / 1000,
            "known": {int(yr): [((r.z, r.y, r.x), int(r.label)) for r in g.itertuples()]
                      for yr, g in by_pid[pid].groupby("study_yr")},
        })
    return sorted(jobs, key=lambda j: -j["label"])


def run_all(todo, workers, room_for):
    """Yields (job, rows, problems) as patients finish. room_for(n) says if n more may start."""
    if workers == 1:
        start_worker()
        for job in todo:
            if not room_for(1):
                return
            yield (job, *safe_run(job))
        return
    with ProcessPoolExecutor(workers, initializer=start_worker) as pool:
        queue, running = iter(todo), {}
        while True:
            while len(running) < 2 * workers and room_for(len(running) + 1):
                job = next(queue, None)
                if job is None:
                    break
                running[pool.submit(safe_run, job)] = job
            if not running:
                return
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                yield (running.pop(future), *future.result())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    jobs = make_jobs()
    TRACK_DIR.mkdir(exist_ok=True)
    shutil.rmtree(TMP, ignore_errors=True)            # half-downloaded scans from a killed run
    done_file = TRACK_DIR / "done.txt"
    done = set(map(int, done_file.read_text().split())) if done_file.exists() else set()
    todo = [j for j in jobs if j["pid"] not in done][:args.limit]
    n_done = len(jobs) - len([j for j in jobs if j["pid"] not in done])
    print(f"{n_done} of {len(jobs)} patients already done, {len(todo)} to go, "
          f"{sum(len(j['screens']) for j in todo)} downloads, {args.workers} workers", flush=True)

    used_gb = sum(f.stat().st_size for d in (PATCH_DIR, TRACK_DIR) for f in d.glob("*.npy")) / 1e9
    biggest = max((j["size_gb"] for j in todo), default=0)
    full = []

    def room_for(n):                                   # n scans on disk at once, at most
        if not full and used_gb + n * biggest > MAX_GB:
            full.append(True)
            print(f"Stopping: more scans would take us over {MAX_GB} GB.", flush=True)
        return not full

    for job, rows, problems in run_all(todo, args.workers, room_for):
        append_csv(TRACK_DIR / "tracks.csv", rows)
        append_csv(TRACK_DIR / "failures.csv",
                   [{"pid": job["pid"], "problem": p} for p in problems])
        with open(done_file, "a") as f:
            f.write(f"{job['pid']}\n")
        n_done += 1
        used_gb += sum((TRACK_DIR / r["file"]).stat().st_size for r in rows) / 1e9
        kind = "cancer" if job["label"] else "control"
        print(f"[{n_done}/{len(jobs)}] pid {job['pid']} ({kind}): "
              f"{len({r['track_id'] for r in rows})} tracks, {len(rows)} cubes, "
              f"{len(problems)} problems, {used_gb:.1f} GB used", flush=True)


if __name__ == "__main__":
    main()
