"""
Step 3e - Evaluate the FROZEN Stage 1 model properly (the open items of Stage 1).

  * AUCs with a 95% bootstrap confidence interval (resampling whole patients)
  * specificity at 90% sensitivity: the share of benign nodules NOT flagged when 90% of the
    cancers are caught (the clinical framing)
  * cancer breakdown: precursor (an earlier screening of the lesion) vs newest box, and solid vs
    subsolid (centre density above / below -400 HU, the blob finder's limit for benign nodules)
  * the blob finder's find rate: the share of NLST nodule records it turned into a patch
  * best-epoch vs mean-of-the-last-10-epochs validation AUC, if the training curve was saved

The bootstrap and specificity functions are also used by step 4.

Reads : models/stage1_final.pt (+ models/stage1_final_log.csv), patches/, cohort/control_nodules.csv
Writes: models/stage1_<split>_predictions.csv, checks/stage1_eval.json
Run   : python step3_evaluate.py           validation split
        python step3_evaluate.py --test    ONCE, at the very end
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader

from config import (CHECK_DIR, COHORT_DIR, MODEL_DIR, N_BOOTSTRAP, PATCH_DIR, SEED, STAGE1_MODEL,
                    STORE)
from step3_model import PatchCNN, PatchDataset

MID = STORE // 2


# ---------- Metrics (also used by step 4) ---------------------------------------------------

def spec_at_sens(labels, scores, sens=0.9):
    """Specificity at the first threshold that catches at least `sens` of the positives."""
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(1 - fpr[np.searchsorted(tpr, sens)])


def bootstrap(df, stat, n=N_BOOTSTRAP, seed=SEED):
    """95% CI of stat(df) (a number or an array) by resampling whole PATIENTS: all rows of one
    patient go in or out together, because they are not independent."""
    rng = np.random.default_rng(seed)
    groups = list(df.groupby("pid").indices.values())     # row positions of each patient
    values = []
    for _ in range(n):
        pick = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        sample = df.iloc[pick]
        if sample["label"].nunique() == 2:                 # AUC needs both classes
            values.append(stat(sample))
    if not values:
        return np.full(2, np.nan)
    return np.percentile(np.array(values), [2.5, 97.5], axis=0)


def auc_ci(df, col="score"):
    """(AUC, lower, upper) of one score column; NaN if only one class is present."""
    if df["label"].nunique() < 2:
        return np.nan, np.nan, np.nan
    lo, hi = bootstrap(df, lambda d: roc_auc_score(d["label"], d[col]))
    return float(roc_auc_score(df["label"], df[col])), float(lo), float(hi)


# ---------- Stage 1 evaluation ----------------------------------------------------------------

def load_stage1(device="cpu"):
    """The frozen Stage 1 model, in eval() mode with every weight frozen."""
    model = PatchCNN().to(device)
    model.load_state_dict(torch.load(STAGE1_MODEL, map_location=device))
    model.eval()                          # BatchNorm uses its stored averages; dropout is off
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def score_patches(model, ds, device):
    logits = [model(x.to(device)).cpu() for x, _ in DataLoader(ds, batch_size=64, num_workers=4)]
    return torch.cat(logits).numpy().ravel()


def find_rate():
    """Benign patches / NLST nodule records on the control scans step 2 processed."""
    nodules = pd.read_csv(COHORT_DIR / "control_nodules.csv")
    done = set((PATCH_DIR / "done.txt").read_text().split())
    index = pd.read_csv(PATCH_DIR / "index.csv")
    failures = pd.read_csv(PATCH_DIR / "failures.csv")
    failed_scans = set(failures.loc[failures["problem"].str.startswith("scan failed"), "series_uid"])
    tried = nodules[nodules["SeriesInstanceUID"].isin(done - failed_scans)]
    found = int((index["kind"] == "benign").sum())
    reasons = failures.loc[~failures["problem"].str.startswith("scan failed"), "problem"]
    return {"records": len(tried), "found": found, "rate": found / max(1, len(tried)),
            "misses": reasons.str.replace(r"[\d.]+", "N", regex=True).value_counts().to_dict()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    split = "test" if args.test else "val"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ds = PatchDataset(split, augment=False)
    df = ds.df.assign(score=score_patches(load_stage1(device), ds, device))
    newest = df[df["kind"] == "cancer"].groupby("pid")["study_yr"].max()    # the newest box
    df["precursor"] = (df["kind"] == "cancer") & (df["study_yr"] < df["pid"].map(newest))
    df["centre_hu"] = [np.load(PATCH_DIR / f)[MID - 2:MID + 2, MID - 2:MID + 2, MID - 2:MID + 2].mean()
                       for f in df["file"]]
    df.to_csv(MODEL_DIR / f"stage1_{split}_predictions.csv", index=False)

    cancer, benign, rnd = (df[df["kind"] == k] for k in ("cancer", "benign", "random"))
    groups = {
        "all patches": df,
        "cancer vs benign": pd.concat([cancer, benign]),
        "cancer vs random": pd.concat([cancer, rnd]),
        "newest cancer vs benign": pd.concat([cancer[~cancer["precursor"]], benign]),
        "precursor vs benign": pd.concat([cancer[cancer["precursor"]], benign]),
        "solid cancer vs benign": pd.concat([cancer[cancer["centre_hu"] > -400], benign]),
        "subsolid cancer vs benign": pd.concat([cancer[cancer["centre_hu"] <= -400], benign]),
    }
    result = {"split": split, "groups": {}}
    print(f"Stage 1 on the {split} split ({len(df)} patches, {len(cancer)} cancer), "
          f"95% CI from {N_BOOTSTRAP} resamples of patients\n")
    print(f"{'':26s} {'cancers':>8s} {'AUC [95% CI]':>22s} {'spec @ 90% sens':>16s}")
    for name, g in groups.items():
        n_pos = int(g["label"].sum())
        if g["label"].nunique() < 2:
            print(f"{name:26s} {n_pos:8d}   (needs both classes)")
            continue
        auc, lo, hi = auc_ci(g)
        spec = spec_at_sens(g["label"], g["score"])
        result["groups"][name] = {"n": len(g), "n_cancer": n_pos, "auc": auc, "lo": lo, "hi": hi, "spec90": spec}
        print(f"{name:26s} {n_pos:8d} {auc:8.3f} [{lo:.3f}, {hi:.3f}] {spec:16.2f}")

    print(f"\nSolid share (> -400 HU at the centre): cancer {(cancer['centre_hu'] > -400).mean():.0%}, "
          f"benign {(benign['centre_hu'] > -400).mean():.0%}")
    log = MODEL_DIR / "stage1_final_log.csv"
    if log.exists():
        curve = pd.read_csv(log)
        result["best_epoch_auc"], result["last10_auc"] = curve["val_auc"].max(), curve["val_auc"].tail(10).mean()
        print(f"Training curve: best epoch val AUC {result['best_epoch_auc']:.3f}, "
              f"mean of the last 10 epochs {result['last10_auc']:.3f} (quote this one)")
    result["find_rate"] = fr = find_rate()
    print(f"Blob finder found {fr['found']} of {fr['records']} benign nodule records ({fr['rate']:.0%}). "
          f"Misses: {fr['misses']}")
    print("The hit rate (is the + on the right blob?) still needs the eye check of step2_check.py.")
    CHECK_DIR.mkdir(exist_ok=True)
    (CHECK_DIR / f"stage1_eval{'_test' if args.test else ''}.json").write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
