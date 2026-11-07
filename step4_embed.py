"""
Step 4b - Turn every track cube into 128 numbers with the FROZEN Stage 1 encoder.

The encoder comes from models/stage1_final.pt, runs in eval() mode under no_grad(), and is never
trained again. Stage 1 and the tracker therefore look through the same "eyes", so anything the
tracker gains comes from the history, not from better eyes.

For every cube in tracks.csv (same order) it stores
  centre  (n, 128)     embedding of the plain centre crop               -> validation and test
  aug     (n, 8, 128)  8 crops with Stage 1's augmentation              -> training
  logit   (n,)         Stage 1's own output; score = sigmoid(logit)     -> the Stage 1 baseline

Then two checks on the validation tracks (Stage 1 never trained on these patients):
  sanity   Stage 1 AUC on the anchor cubes should be close to its AUC on the val patches
  headroom Stage 1 AUC by years before the anchor. If it drops one screening back, that is
           where the history could help.

Reads : tracks/tracks.csv, tracks/*.npy, models/stage1_final.pt, patches/
Writes: tracks/embeddings.npz, checks/stage2_embed.json
Run   : python step4_embed.py
"""
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import CHECK_DIR, N_AUG, SEED, TRACK_DIR
from step3_evaluate import auc_ci, load_stage1
from step3_model import PatchDataset
from step3_train import DEVICE


class TrackPatches(PatchDataset):
    """Track cubes instead of step 2 patches: the same crop and augmentation code as Stage 1."""
    folder = TRACK_DIR

    def __init__(self, df, augment):
        self.df, self.augment = df.reset_index(drop=True), augment


@torch.no_grad()
def embed(model, ds):
    """Embeddings (n, 128) and Stage 1 logits (n,) of every cube in ds, in order."""
    emb, logits = [torch.zeros(0, 128)], [torch.zeros(0, 1)]
    for x, _ in DataLoader(ds, batch_size=64, num_workers=4):
        e = model.encoder(x.to(DEVICE))
        emb.append(e.cpu())
        logits.append(model.head(e).cpu())
    return torch.cat(emb).numpy(), torch.cat(logits).numpy().ravel()


def main():
    torch.manual_seed(SEED)
    tracks = pd.read_csv(TRACK_DIR / "tracks.csv")
    model = load_stage1(DEVICE)

    centre, logit = embed(model, TrackPatches(tracks, augment=False))
    aug = np.stack([embed(model, TrackPatches(tracks, augment=True))[0] for _ in range(N_AUG)], axis=1)
    np.savez(TRACK_DIR / "embeddings.npz", files=tracks["file"].to_numpy(dtype=str), centre=centre, aug=aug,
             logit=logit, score=1 / (1 + np.exp(-logit)))
    print(f"Embedded {len(tracks)} cubes: centre {centre.shape}, aug {aug.shape}")

    # Sanity: the same model on the val patches (cancer vs benign) and on the val anchor cubes
    ref = PatchDataset("val", augment=False)
    ref.df = ref.df[ref.df["kind"] != "random"].reset_index(drop=True)
    ref.df["score"] = embed(model, ref)[1]
    val = tracks.assign(score=logit, years_before=tracks["anchor_yr"] - tracks["study_yr"])
    val = val[val["split"] == "val"]
    out = {"ref": auc_ci(ref.df), "anchor": auc_ci(val[val["years_before"] == 0]), "headroom": []}
    print(f"\nStage 1 AUC, val patches, cancer vs benign : {out['ref'][0]:.3f} [{out['ref'][1]:.3f}, {out['ref'][2]:.3f}]")
    print(f"Stage 1 AUC, anchor cubes of val tracks    : {out['anchor'][0]:.3f} "
          f"[{out['anchor'][1]:.3f}, {out['anchor'][2]:.3f}]   (should be close)")

    print("\nHeadroom: Stage 1 AUC on val cubes, cancer vs benign, by screenings before the anchor")
    for k, g in val.groupby("years_before"):
        n_pos, n_neg = int(g["label"].sum()), int((g["label"] == 0).sum())
        row = {"years_before": int(k), "n_cancer": n_pos, "n_benign": n_neg}
        if n_pos and n_neg:
            row["auc"], row["lo"], row["hi"] = auc_ci(g)
            print(f"  {k} before: AUC {row['auc']:.3f} [{row['lo']:.3f}, {row['hi']:.3f}]  "
                  f"({n_pos} cancer, {n_neg} benign cubes)")
        out["headroom"].append(row)
    CHECK_DIR.mkdir(exist_ok=True)
    (CHECK_DIR / "stage2_embed.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
