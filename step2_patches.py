"""
Step 2 - Cut a 72 mm cube around every lesion, ONE SCAN AT A TIME.

For each scan:  download -> read into memory -> delete DICOM -> resample to 1 mm
                -> find lesion centres -> crop cubes -> save
Only one scan is ever on disk, and the script stops before the patches pass MAX_GB.
It is resumable: run it again and it skips every series listed in patches/done.txt.

Reads : cohort/scans.csv, cohort/patients.csv, cohort/control_nodules.csv, annotations.json
Writes: patches/*.npy         one cube per patch: int16, 72x72x72, Hounsfield units
        patches/index.csv     one row per patch: file, pid, split, kind, label, centre, size
        patches/failures.csv  every nodule or scan we could not use, with the reason
        patches/done.txt      series already processed

Run   : python step2_patches.py
"""
import json
import shutil

import numpy as np
import pandas as pd
import pydicom
from idc_index import IDCClient
from scipy import ndimage

from config import (ANNOTATIONS, COHORT_DIR, MAX_GB, N_RANDOM, PATCH_DIR, SEED,
                    STORE, TMP_DIR)

# NLST lobe codes (sct_epi_loc) -> which lung. 8 = "other" -> search both.
LOBE_SIDE = {1: "right", 2: "right", 3: "right", 4: "left", 5: "left", 6: "left"}


# ---------- 1. Get one scan into memory ----------------------------------------------

def download(client, series_uid, folder):
    client.download_from_selection(seriesInstanceUID=[series_uid], downloadDir=str(folder),
                                   quiet=True, show_progress_bar=False)
    if not any(folder.rglob("*.dcm")):
        raise RuntimeError("download failed")


def read_series(folder):
    """DICOM files -> 3D array (z, y, x) of Hounsfield units, ordered feet -> head."""
    slices = [pydicom.dcmread(f) for f in folder.rglob("*.dcm")]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    orientation = [round(float(v)) for v in slices[0].ImageOrientationPatient]
    if orientation != [1, 0, 0, 0, 1, 0]:   # we assume image left = patient right
        raise ValueError(f"unusual orientation {orientation}")

    hu = [s.pixel_array * float(s.RescaleSlope) + float(s.RescaleIntercept) for s in slices]
    vol = np.clip(np.stack(hu), -1024, 3071).astype(np.int16)
    z = [float(s.ImagePositionPatient[2]) for s in slices]
    info = {
        "spacing": (float(np.median(np.diff(z))),            # mm between slices
                    float(slices[0].PixelSpacing[0]),         # mm per pixel (rows)
                    float(slices[0].PixelSpacing[1])),        # mm per pixel (columns)
        "rows": int(slices[0].Rows),
        "cols": int(slices[0].Columns),
        "sop_to_z": {s.SOPInstanceUID: i for i, s in enumerate(slices)},     # for Sybil boxes
        "number_to_z": {int(s.InstanceNumber): i for i, s in enumerate(slices)},  # for NLST slice numbers
    }
    return vol, info


def to_1mm(vol, spacing):
    """Resample so every voxel is 1 x 1 x 1 mm. Zoom factor = old spacing / 1 mm."""
    return ndimage.zoom(vol, spacing, order=1)


def lung_mask(vol):
    """Rough lung mask: air inside the body, with vessels and nodules filled in."""
    air = vol < -400
    labels, _ = ndimage.label(air)
    edges = np.concatenate([labels[:, 0].ravel(), labels[:, -1].ravel(),
                            labels[:, :, 0].ravel(), labels[:, :, -1].ravel()])
    air &= ~np.isin(labels, np.unique(edges))       # air touching the image edge = outside body
    labels, n = ndimage.label(air)
    sizes = ndimage.sum(air, labels, range(1, n + 1))
    lungs = np.isin(labels, np.flatnonzero(sizes > 200_000) + 1)   # keep parts > 200 ml
    for z in range(len(lungs)):                     # fill the holes left by vessels/nodules
        lungs[z] = ndimage.binary_fill_holes(lungs[z])
    return lungs


# ---------- 2. Where are the lesions? ----------------------------------------------------

def cancer_centres(boxes_per_slice, info):
    """Radiologist boxes (2D, per slice) -> one (centre, size) per lesion, in mm = 1 mm voxels."""
    dz, dy, dx = info["spacing"]
    boxes = []
    for sop, slice_boxes in boxes_per_slice.items():
        if sop not in info["sop_to_z"]:
            continue
        for b in slice_boxes:                       # x, y, width, height are fractions of 512
            boxes.append({
                "z": info["sop_to_z"][sop] * dz,
                "y": (b["y"] + b["height"] / 2) * info["rows"] * dy,
                "x": (b["x"] + b["width"] / 2) * info["cols"] * dx,
                "size": max(b["width"] * info["cols"] * dx, b["height"] * info["rows"] * dy),
            })
    boxes.sort(key=lambda b: b["z"])

    lesions = []    # a lesion = boxes on neighbouring slices at about the same spot
    for b in boxes:
        for lesion in lesions:
            last = lesion[-1]
            if b["z"] - last["z"] <= 3 * dz and abs(b["y"] - last["y"]) + abs(b["x"] - last["x"]) < 15:
                lesion.append(b)
                break
        else:
            lesions.append([b])

    return [((np.mean([b["z"] for b in les]), np.mean([b["y"] for b in les]),
              np.mean([b["x"] for b in les])), max(b["size"] for b in les))
            for les in lesions]


def ball(r):
    z, y, x = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return x ** 2 + y ** 2 + z ** 2 <= r ** 2


def find_nodule(vol, lungs, midline, z, side, diameter, window=15):
    """Benign nodule: the solid blob closest to `diameter` mm, within +-window mm of slice z."""
    z0, z1 = max(0, z - window), min(len(vol), z + window + 1)
    near_lungs = ndimage.binary_dilation(lungs[z0:z1], iterations=2)  # include nodules on the wall
    solid = (vol[z0:z1] > -400) & near_lungs
    if side == "right":
        solid[:, :, midline:] = False               # patient's right lung = left side of image
    if side == "left":
        solid[:, :, :midline] = False
    solid = ndimage.binary_opening(solid, structure=ball(max(1, round(diameter / 4))))  # cut vessels

    labels, _ = ndimage.label(solid)
    best, best_score = None, np.inf
    for i, box in enumerate(ndimage.find_objects(labels), start=1):
        size = max(s.stop - s.start for s in box)   # longest extent in mm
        if not 0.5 * diameter <= size <= 2 * diameter:
            continue
        cz, cy, cx = ndimage.center_of_mass(labels[box] == i)
        centre = (z0 + box[0].start + cz, box[1].start + cy, box[2].start + cx)
        score = abs(size - diameter) + abs(centre[0] - z)   # right size, close to the slice
        if score < best_score:
            best, best_score = centre, score
    return best


def random_lung_points(lungs, avoid, n, rng):
    """n random points inside the lungs, at least 30 mm away from every lesion."""
    points = []
    for _ in range(10_000):
        p = rng.integers(0, lungs.shape)
        if lungs[tuple(p)] and all(np.abs(p - np.array(a)).max() > 30 for a in avoid + points):
            points.append(p)
            if len(points) == n:
                break
    return points


# ---------- 3. One scan in, patches out ----------------------------------------------------

def process_scan(scan, boxes, nodules, client, rng):
    folder = TMP_DIR / scan.SeriesInstanceUID
    try:
        download(client, scan.SeriesInstanceUID, folder)
        vol, info = read_series(folder)
    finally:
        shutil.rmtree(folder, ignore_errors=True)  # DICOM files are deleted immediately

    vol = to_1mm(vol, info["spacing"])
    lungs = lung_mask(vol)
    if not lungs.any():
        raise ValueError("no lungs found")
    midline = int(ndimage.center_of_mass(lungs)[2])
    centres, problems = [], []                      # centres: (kind, label, (z, y, x), size_mm)

    if boxes:                                       # cancer case with radiologist boxes
        for centre, size in cancer_centres(boxes, info):
            centres.append(("cancer", 1, centre, size))

    for nod in nodules.itertuples():                # control with benign nodules
        z = info["number_to_z"].get(int(nod.sct_slice_num))
        if z is None:
            problems.append(f"slice {int(nod.sct_slice_num)} not in this series")
            continue
        side = LOBE_SIDE.get(nod.sct_epi_loc, "any")
        centre = find_nodule(vol, lungs, midline, round(z * info["spacing"][0]), side, nod.sct_long_dia)
        if centre is None:
            problems.append(f"no blob of {nod.sct_long_dia:.0f} mm near slice {int(nod.sct_slice_num)}")
            continue
        centres.append(("benign", 0, centre, nod.sct_long_dia))

    for p in random_lung_points(lungs, [c for _, _, c, _ in centres], N_RANDOM, rng):
        centres.append(("random", 0, tuple(p), 0.0))

    padded = np.pad(vol, STORE // 2, constant_values=-1024)  # so cubes near the edge still fit
    rows = []
    for k, (kind, label, centre, size) in enumerate(centres):
        z, y, x = (int(round(c)) for c in centre)
        name = f"{scan.pid}_T{scan.study_yr}_{k}_{kind}.npy"
        np.save(PATCH_DIR / name, padded[z:z + STORE, y:y + STORE, x:x + STORE])
        rows.append({"file": name, "pid": scan.pid, "study_yr": scan.study_yr,
                     "series_uid": scan.SeriesInstanceUID, "kind": kind, "label": label,
                     "z": z, "y": y, "x": x, "size_mm": round(float(size), 1)})
    return rows, problems


def append_csv(path, rows):
    if rows:
        pd.DataFrame(rows).to_csv(path, mode="a", header=not path.exists(), index=False)


def main():
    scans = pd.read_csv(COHORT_DIR / "scans.csv")
    split = pd.read_csv(COHORT_DIR / "patients.csv").set_index("pid")["split"]
    nodules = pd.read_csv(COHORT_DIR / "control_nodules.csv")
    annotations = json.load(open(ANNOTATIONS))

    # Which scans? Cases: those with boxes. Controls: those with a recorded nodule.
    todo = scans[scans["annotated"] | scans["SeriesInstanceUID"].isin(nodules["SeriesInstanceUID"])]
    todo = todo.sort_values("annotated", ascending=False)       # cancers first

    PATCH_DIR.mkdir(exist_ok=True)
    done_file = PATCH_DIR / "done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    used_gb = sum(f.stat().st_size for f in PATCH_DIR.glob("*.npy")) / 1e9
    client = IDCClient()
    rng = np.random.default_rng(SEED)

    for i, scan in enumerate(todo.itertuples(), start=1):
        uid = scan.SeriesInstanceUID
        if uid in done:
            continue
        if used_gb + scan.series_size_MB / 1000 > MAX_GB:
            print(f"Stopping: the next scan would take us over {MAX_GB} GB.")
            break
        try:
            rows, problems = process_scan(scan, annotations.get(uid) if scan.annotated else None,
                                          nodules[nodules["SeriesInstanceUID"] == uid], client, rng)
        except Exception as e:                       # bad scan: log it and move on
            rows, problems = [], [f"scan failed: {e}"]

        for r in rows:
            r["split"] = split[scan.pid]
        append_csv(PATCH_DIR / "index.csv", rows)
        append_csv(PATCH_DIR / "failures.csv",
                   [{"series_uid": uid, "pid": scan.pid, "problem": p} for p in problems])
        with open(done_file, "a") as f:
            f.write(uid + "\n")

        used_gb += sum((PATCH_DIR / r["file"]).stat().st_size for r in rows) / 1e9
        print(f"[{i}/{len(todo)}] pid {scan.pid} T{scan.study_yr}: "
              f"{len(rows)} patches, {len(problems)} problems, {used_gb:.1f} GB used")


if __name__ == "__main__":
    main()
