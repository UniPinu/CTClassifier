"""
Step 4c - The tracker: does seeing a lesion's history beat seeing only its newest scan?

Input per track: a WINDOW of exactly k screenings, oldest first (always exactly k, so the number
of screenings can never give the label away). Each screening is the frozen Stage 1 embedding e_t
plus its change since the previous screening (0 at the first one). A GRU reads them in time order:

    x_t = [e_t, e_t - e_(t-1)]     h_t = GRU(h_(t-1), x_t)     score = sigmoid(v . h_k + c)

Dropout 0.2 on the input -> GRU(256 -> 64) -> Linear(64 -> 1): 61,889 weights.
--plain uses e_t only: GRU(128 -> 64), 37,313 weights.

Training: every epoch, a random one of the 8 augmented embeddings per screening; BCE with
pos_weight; Adam (lr 1e-3, weight decay 1e-4), batch 64, 300 epochs. The epoch with the best
VALIDATION AUC is kept, also with --test, so the test split never chooses anything.

Baselines on exactly the same windows:
  Stage 1      Stage 1's score of the newest cube in the window
  score trend  logistic regression on [newest logit, newest - oldest logit]

Run, in this order (the evaluation plan):
  python step4_tracker.py --scans 2 --overfit   does the code work? train loss must go below 0.1
  python step4_tracker.py --scans 2             does history help at the newest known screening?
  python step4_tracker.py --scans 3
  python step4_tracker.py --scans 2 --early     ... one screening BEFORE the anchor?
  python step4_tracker.py --scans 1             sanity: no history, should be about Stage 1
  python step4_tracker.py --scans 3 --plain     does the change input matter?
  python step4_tracker.py --scans 2 --test      ONCE per setting, at the very end

Reads : tracks/tracks.csv, tracks/embeddings.npz
Writes: models/tracker_<setting>.pt, models/tracker_<setting>_log.csv,
        models/stage2_pred_<setting>_<split>.csv, one row in models/stage2_results.csv
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from config import (MODEL_DIR, N_AUG, SEED, TRACK_DIR, TRACKER_BATCH, TRACKER_EPOCHS,
                    TRACKER_LR)
from step3_evaluate import bootstrap, spec_at_sens


class Tracker(nn.Module):
    def __init__(self, plain=False):
        super().__init__()
        self.plain = plain
        self.drop = nn.Dropout(0.2)
        self.gru = nn.GRU(128 if plain else 256, 64, batch_first=True)
        self.head = nn.Linear(64, 1)

    def forward(self, e):                            # e: (batch, k, 128), oldest screening first
        if not self.plain:
            change = e - torch.cat([e[:, :1], e[:, :-1]], dim=1)   # e_t - e_(t-1); 0 at the first
            e = torch.cat([e, change], dim=2)
        _, h = self.gru(self.drop(e))                # h: (1, batch, 64), state after the newest
        return self.head(h[0])                       # (batch, 1) logits


def make_windows(tracks, k, early):
    """One window per track: its last k screenings up to the anchor (--early: up to the one before).
    Returns one row per window and an (n_windows, k) array of row numbers into tracks."""
    info, rows = [], []
    for tid, t in tracks.groupby("track_id", sort=False):
        t = t.sort_values("study_yr")
        if early:
            t = t.iloc[:-1]                          # drop the anchor screening itself
        if len(t) < k:
            continue
        w = t.iloc[-k:]
        rows.append(w.index.values)
        info.append({"track_id": tid, "pid": w["pid"].iloc[0], "label": w["label"].iloc[0],
                     "split": w["split"].iloc[0], "anchor_yr": w["anchor_yr"].iloc[0],
                     "newest_yr": w["study_yr"].iloc[-1]})
    return pd.DataFrame(info), np.array(rows, dtype=int).reshape(len(rows), k)


@torch.no_grad()
def predict(model, x):
    model.eval()
    return model(x).numpy().ravel()


def train(model, x_aug, x_val, y, y_val, epochs, overfit):
    """Returns the best state (by validation AUC) and the per-epoch log."""
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=((y == 0).sum() / y.sum()).reshape(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=TRACKER_LR, weight_decay=1e-4)
    log, best_auc, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        # one of the 8 augmented embeddings per screening (overfit check: always the centre crop)
        pick = torch.zeros(x_aug.shape[:2], dtype=torch.long) if overfit else \
            torch.randint(0, N_AUG, x_aug.shape[:2])
        x = x_aug[torch.arange(len(x_aug))[:, None], torch.arange(x_aug.shape[1]), pick]  # (n, k, 128)
        total = 0.0
        for batch in torch.randperm(len(x)).split(TRACKER_BATCH):
            loss = loss_fn(model(x[batch]), y[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(batch)
        row = {"epoch": epoch, "train_loss": total / len(x)}
        if not overfit:
            row["val_auc"] = roc_auc_score(y_val.numpy(), predict(model, x_val))
            if row["val_auc"] > best_auc:
                best_auc = row["val_auc"]
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
        log.append(row)
        if epoch % 50 == 0 or epoch == epochs:
            print(f"epoch {epoch:4d}  train loss {row['train_loss']:.3f}" +
                  ("" if overfit else f"  val AUC {row['val_auc']:.3f}"), flush=True)
    return best_state, pd.DataFrame(log)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scans", type=int, default=2, help="k: screenings per window")
    parser.add_argument("--early", action="store_true", help="windows end one screening before the anchor")
    parser.add_argument("--plain", action="store_true", help="embedding only, no change input")
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    k, split = args.scans, "test" if args.test else "val"
    name = f"k{k}" + "_early" * args.early + "_plain" * args.plain + (f"_s{args.seed}" if args.seed != SEED else "")

    tracks = pd.read_csv(TRACK_DIR / "tracks.csv")
    emb = np.load(TRACK_DIR / "embeddings.npz")
    assert (emb["files"] == tracks["file"].to_numpy(dtype=str)).all(), "embeddings.npz is out of date: run step4_embed.py"
    win, rows = make_windows(tracks, k, args.early)
    win["stage1"] = emb["logit"][rows[:, -1]]                        # newest cube in the window
    win["change"] = emb["logit"][rows[:, -1]] - emb["logit"][rows[:, 0]]   # newest - oldest

    is_tr, is_val, is_ev = (win["split"] == s for s in ("train", "val", split))
    if args.overfit:                                 # 10 cancer + 10 benign windows, no augmentation
        is_tr = win.index.isin(win[is_tr].groupby("label").head(10).index)
    aug = torch.tensor(emb["aug"] if not args.overfit else emb["centre"][:, None])
    centre = torch.tensor(emb["centre"])
    y = torch.tensor(win["label"].values, dtype=torch.float32)[:, None]
    tr, val, ev = (torch.tensor(np.flatnonzero(m)) for m in (is_tr, is_val, is_ev))
    rows_t = torch.tensor(rows)
    print(f"Setting {name}: the last {k} screening(s) up to "
          f"{'one screening before ' if args.early else ''}the anchor. "
          f"Windows: train {len(tr)} ({int(y[tr].sum())} cancer), {split} {len(ev)} ({int(y[ev].sum())} cancer)")

    model = Tracker(args.plain)
    print(f"Tracker with {sum(p.numel() for p in model.parameters()):,} weights, "
          f"{'overfit check' if args.overfit else f'{TRACKER_EPOCHS} epochs'}")
    t0 = time.time()
    best_state, log = train(model, aug[rows_t[tr]], centre[rows_t[val]], y[tr], y[val],
                            TRACKER_EPOCHS, args.overfit)
    if args.overfit:
        loss = log["train_loss"].iloc[-1]
        print(f"\nOverfit check: final train loss {loss:.3f} (should be below 0.1): "
              f"{'OK' if loss < 0.1 else 'NOT OK - look for a bug'}")
        return

    model.load_state_dict(best_state)
    best = log.loc[log["val_auc"].idxmax()]
    torch.save(best_state, MODEL_DIR / f"tracker_{name}.pt")
    log.to_csv(MODEL_DIR / f"tracker_{name}_log.csv", index=False)

    # Score trend baseline: fitted on the train windows only
    trend = LogisticRegression().fit(win.loc[is_tr, ["stage1", "change"]].values, win.loc[is_tr, "label"])
    win["trend"] = trend.decision_function(win[["stage1", "change"]].values)
    win["tracker"] = predict(model, centre[rows_t])
    out = win[is_ev].reset_index(drop=True)
    out.to_csv(MODEL_DIR / f"stage2_pred_{name}_{split}.csv", index=False)

    cols = ["tracker", "stage1", "trend"]
    stat = lambda d: [roc_auc_score(d["label"], d[c]) for c in cols] + \
                     [roc_auc_score(d["label"], d["tracker"]) - roc_auc_score(d["label"], d["stage1"])]
    point, (lo, hi) = stat(out), bootstrap(out, stat)
    spec = [spec_at_sens(out["label"], out[c]) for c in cols]

    print(f"\nBest val AUC {best['val_auc']:.3f} at epoch {int(best['epoch'])} ({time.time() - t0:.0f} s). "
          f"On the {split} split, 95% CI from resampling patients:\n")
    print(f"{'':20s} {'AUC [95% CI]':>22s} {'spec @ 90% sens':>16s}")
    for i, label in enumerate(["Tracker", "Stage 1 (newest)", "Score trend"]):
        print(f"{label:20s} {point[i]:8.3f} [{lo[i]:.3f}, {hi[i]:.3f}] {spec[i]:16.2f}")
    print(f"{'Tracker - Stage 1':20s} {point[3]:+8.3f} [{lo[3]:+.3f}, {hi[3]:+.3f}]")

    by_anchor = {}
    print("\nAUC per anchor year (does the year carry the label?):")
    for yr, g in out.groupby("anchor_yr"):
        if g["label"].nunique() == 2:
            by_anchor[f"T{yr}"] = {"tracker": roc_auc_score(g["label"], g["tracker"]),
                                   "stage1": roc_auc_score(g["label"], g["stage1"]),
                                   "n": len(g), "n_cancer": int(g["label"].sum())}
            a = by_anchor[f"T{yr}"]
            print(f"  anchor T{yr}: tracker {a['tracker']:.3f}, Stage 1 {a['stage1']:.3f} "
                  f"({a['n']} windows, {a['n_cancer']} cancer)")

    result = {"time": time.strftime("%Y-%m-%d %H:%M"), "setting": name, "split": split,
              "scans": k, "early": args.early, "plain": args.plain, "seed": args.seed,
              "n_train": len(tr), "n_train_cancer": int(y[tr].sum()),
              "n_eval": len(ev), "n_eval_cancer": int(y[ev].sum()),
              "best_epoch": int(best["epoch"]), "best_val_auc": best["val_auc"]}
    for i, c in enumerate(cols + ["diff"]):
        result.update({f"auc_{c}": point[i], f"auc_{c}_lo": lo[i], f"auc_{c}_hi": hi[i]})
    result.update({f"spec90_{c}": s for c, s in zip(cols, spec)})
    result["by_anchor"] = json.dumps(by_anchor)
    results = MODEL_DIR / "stage2_results.csv"
    pd.DataFrame([result]).to_csv(results, mode="a", header=not results.exists(), index=False)
    print(f"\nSaved models/tracker_{name}.pt and a row in {results.name}")


if __name__ == "__main__":
    main()
