"""
Step 3g - Realistic evaluation: locate the cancers the way the benign nodules were located.

In steps 2-3 the cancer centres come from Sybil's boxes (drawn in hindsight, pixel-precise) and
the benign centres from the blob finder on NLST's own records (slice + lobe + size). Did the
model learn that difference? Here the VALIDATION cancers are located like benign nodules: the
same blob finder on the same kind of NLST record (nlst_ctab, abnormality 51), with no help from
the boxes. The boxes are only used afterwards, to tell whether a blob is the cancer (within
SNAP_MM of a boxed lesion). Stage 1 then scores those cubes.

If the AUC against the val benign patches holds, the way lesions were located is not what the
model learned. It also measures how often the blob finder finds a known (cancer) nodule.

Reads : nlst_ctab (idc-index), cohort/, patches/index.csv, models/stage1_final.pt
Writes: patches/realistic/*.npy, checks/stage1_realistic.csv, checks/stage1_realistic.json,
        checks/grid_realistic.png (+ .csv) for checking by eye
Run   : python step3_realistic.py              (the val cancers' annotated scans, ~150 downloads)
        python step3_realistic.py --limit 5    (quick trial)
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
from idc_index import IDCClient
from scipy import ndimage

from config import CHECK_DIR, COHORT_DIR, PATCH, PATCH_DIR, SNAP_MM, STORE
from step1_cohort import load_clinical
from step2_check import save_grid
from step2_patches import LOBE_SIDE, find_nodule
from step3_evaluate import auc_ci, load_stage1, score_patches
from step3_model import PatchDataset, pick_device, to_unit
from step4_sequences import cut, load_scan, start_worker   # download with retry, same crop

OUT = PATCH_DIR / "realistic"


def locate(job):
    """Runs in a worker: the blob finder on every NLST nodule record of one scan."""
    try:
        vol, lungs, info = load_scan(job["uid"])
    except Exception as e:
        return job, [], f"scan failed: {e}"
    midline = int(ndimage.center_of_mass(lungs)[2])
    rows = []
    for k, r in enumerate(job["records"]):
        z = info["number_to_z"].get(int(r["sct_slice_num"]))
        centre = None if z is None else find_nodule(
            vol, lungs, midline, round(z * info["spacing"][0]),
            LOBE_SIDE.get(r["sct_epi_loc"], "any"), r["sct_long_dia"])
        row = {**r, "found": centre is not None, "size_mm": r["sct_long_dia"]}
        if centre is not None:                     # which boxed lesion is closest?
            d = [np.linalg.norm(np.array(centre) - l[1:]) for l in job["lesions"]]
            row.update(dist_mm=round(min(d), 1), lesion=job["lesions"][int(np.argmin(d))][0],
                       file=f"realistic/{r['pid']}_T{r['study_yr']}_{k}.npy",
                       cube=cut(vol, centre, STORE), **dict(zip("zyx", np.round(centre).astype(int))))
        rows.append(row)
    return job, rows, None


@torch.no_grad()
def score_cubes(model, cubes, device):
    c = (STORE - PATCH) // 2                       # centre crop, as for validation
    x = torch.from_numpy(to_unit(np.stack(cubes)[:, c:c + PATCH, c:c + PATCH, c:c + PATCH]).astype(np.float32))
    return torch.cat([model(b[:, None].to(device)).cpu() for b in x.split(64)]).numpy().ravel()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    device = pick_device()

    _, ctab = load_clinical(IDCClient())
    patients = pd.read_csv(COHORT_DIR / "patients.csv")
    index = pd.read_csv(PATCH_DIR / "index.csv")
    val_cases = patients.loc[(patients["label"] == 1) & (patients["split"] == "val"), "pid"]
    boxed = index[(index["kind"] == "cancer") & index["pid"].isin(val_cases)]   # where the cancers are
    records = ctab[(ctab["sct_ab_desc"] == 51) & ctab["sct_slice_num"].between(1, 998)
                   & ctab["sct_long_dia"].notna()]
    records = records.merge(boxed[["pid", "study_yr", "series_uid"]].drop_duplicates(), on=["pid", "study_yr"])
    records[["pid", "study_yr"]] = records[["pid", "study_yr"]].astype(int)
    jobs = [{"uid": uid, "records": g.drop(columns="series_uid").to_dict("records"),
             "lesions": [(l.file, l.z, l.y, l.x) for l in boxed[boxed["series_uid"] == uid].itertuples()]}
            for uid, g in records.groupby("series_uid")][:args.limit]
    print(f"{len(boxed)} boxed val cancer lesions; {len(records)} NLST nodule records on "
          f"{len(jobs)} of their scans", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    rows, failed = [], 0
    with ProcessPoolExecutor(args.workers, initializer=start_worker) as pool:
        for i, (job, found, problem) in enumerate(pool.map(locate, jobs), start=1):
            for r in found:
                if r["found"]:
                    np.save(PATCH_DIR / r["file"], r["cube"])
            rows += found
            failed += problem is not None
            print(f"[{i}/{len(jobs)}] {len(found)} records, {sum(r['found'] for r in found)} blobs"
                  + (f", {problem}" if problem else ""), flush=True)

    df = pd.DataFrame(rows)
    if df.empty or not df["found"].any():
        print("No blobs found: nothing to score.")
        return
    blobs = df[df["found"]].copy()
    model = load_stage1(device)
    blobs["score"] = score_cubes(model, list(blobs.pop("cube")), device)
    blobs["label"] = 1
    blobs["is_cancer"] = blobs["dist_mm"] <= SNAP_MM
    df.drop(columns="cube", errors="ignore").merge(blobs[["file", "score", "is_cancer"]], on="file", how="left") \
        .to_csv(CHECK_DIR / "stage1_realistic.csv", index=False)

    # Reference: the same model on the val benign patches and on the boxed cubes of the same lesions
    val = PatchDataset("val", augment=False)
    val.df = val.df[val.df["kind"].isin(["cancer", "benign"])].reset_index(drop=True)
    val.df["score"] = score_patches(model, val, device)
    benign = val.df[val.df["kind"] == "benign"]
    hits = blobs[blobs["is_cancer"]].sort_values("dist_mm").drop_duplicates("lesion")   # 1 blob per lesion
    boxed_same = val.df[val.df["file"].isin(hits["lesion"])]

    n_rec = len(df)
    res = {"scans": len(jobs), "scan_failures": failed, "records": n_rec, "blobs": len(blobs),
           "lesions_boxed": len(boxed), "lesions_located": len(hits)}
    print(f"\nBlob finder on cancer patients' records: a blob for {len(blobs)} of {n_rec} records "
          f"({len(blobs) / max(1, n_rec):.0%}); {int(blobs['is_cancer'].sum())} blobs within {SNAP_MM} mm "
          f"of a box. Located {len(hits)} of {len(boxed)} boxed val lesions ({len(hits) / max(1, len(boxed)):.0%}).")
    if len(hits) and len(benign):
        for name, cancers in [("box-located, all val cancers", val.df[val.df["kind"] == "cancer"]),
                              ("box-located, same lesions", boxed_same),
                              ("blob-located (realistic)", hits)]:
            auc, lo, hi = auc_ci(pd.concat([cancers[["pid", "label", "score"]], benign[["pid", "label", "score"]]]))
            res[name] = [auc, lo, hi]
            print(f"  AUC vs val benign, {name:30s}: {auc:.3f} [{lo:.3f}, {hi:.3f}]  ({len(cancers)} cancers)")
        other = blobs[~blobs["is_cancer"]]
        print(f"  Other blobs in cancer patients (not the cancer): {len(other)}, median score "
              f"{np.median(1 / (1 + np.exp(-other['score']))) if len(other) else float('nan'):.2f} "
              f"(benign patches: {np.median(1 / (1 + np.exp(-benign['score']))):.2f})")
        save_grid(hits.assign(kind="cancer (blob)"), CHECK_DIR / "grid_realistic.png")
    (CHECK_DIR / "stage1_realistic.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
