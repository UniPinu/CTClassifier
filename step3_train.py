"""
Step 3 - Train the 3D CNN from scratch, then test it once.

Run:
  python step3_train.py --overfit   # sanity check: train loss on 20 patches must go to ~0
  python step3_train.py             # real training; keeps the model with the best val AUC
  python step3_train.py --test      # ONCE at the very end: AUC on the test split

Options for clean experiments (step3_experiments.py runs the whole grid):
  --epochs 120        instead of EPOCHS in config.py
  --fraction 0.5      train on this share of the training PATIENTS (same cancer share)
  --seed 43           another random start
  --cosine            cosine learning-rate schedule (lr slides from 1e-3 to 0)
  --out NAME          save to models/runs/NAME/ instead of models/best.pt

Writes: models/best.pt, models/log.csv, checks/training_curves.png, models/test_predictions.csv
"""
import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CHECK_DIR, EPOCHS, LEARNING_RATE, MODEL_DIR, SEED, STAGE1_MODEL
from step3_model import PatchCNN, PatchDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict(model, loader):
    """Run the model over a whole split. Returns logits and labels."""
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(DEVICE)).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def train_one_epoch(model, loader, loss_fn, optimizer):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        loss = loss_fn(model(x), y)      # forward pass + how wrong we are
        optimizer.zero_grad()
        loss.backward()                  # backprop: which way to nudge every weight
        optimizer.step()                 # nudge them
        total += loss.item() * len(x)
    return total / len(loader.dataset)


def plot_log(log, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(log["epoch"], log["train_loss"], label="train")
    ax1.plot(log["epoch"], log["val_loss"], label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend()
    ax2.plot(log["epoch"], log["val_auc"])
    ax2.set_xlabel("epoch"); ax2.set_ylabel("val AUC")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def some_patients(df, fraction, seed):
    """All patches of a random `fraction` of the patients, stratified so the cancer share stays."""
    patient_label = df.groupby("pid")["label"].max()
    keep, _ = train_test_split(patient_label.index, train_size=fraction,
                               stratify=patient_label, random_state=seed)
    return df[df["pid"].isin(keep)].reset_index(drop=True)


def train(overfit, epochs=EPOCHS, fraction=1.0, seed=SEED, cosine=False, out=None):
    """Returns a summary dict. out=None writes models/best.pt; otherwise models/runs/<out>/."""
    t0 = time.time()
    torch.manual_seed(seed)
    train_ds = PatchDataset("train", augment=not overfit)
    val_ds = PatchDataset("val", augment=False)
    if fraction < 1:
        train_ds.df = some_patients(train_ds.df, fraction, seed)
    if overfit:   # only 10 cancers + 10 non-cancers, no augmentation
        train_ds.df = train_ds.df.groupby("label").head(10).reset_index(drop=True)
        epochs = 100

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(val_ds, batch_size=64, num_workers=4)

    # pos_weight: a cancer counts as much as (negatives / positives) non-cancers,
    # so the model cannot win by always answering "no cancer".
    n_pos = int(train_ds.df["label"].sum())
    n_neg = len(train_ds) - n_pos
    print(f"Training on {len(train_ds)} patches ({n_pos} cancer), validating on {len(val_ds)}")
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([n_neg / n_pos], device=DEVICE))

    model = PatchCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs) if cosine else None
    folder = MODEL_DIR if out is None else MODEL_DIR / "runs" / out
    folder.mkdir(parents=True, exist_ok=True)
    CHECK_DIR.mkdir(exist_ok=True)
    log, best_auc = [], 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer)
        if scheduler:
            scheduler.step()
        if overfit:
            print(f"epoch {epoch:3d}  train loss {train_loss:.3f}")
            continue
        logits, labels = predict(model, val_dl)
        val_loss = loss_fn(logits.to(DEVICE), labels.to(DEVICE)).item()
        val_auc = roc_auc_score(labels.numpy(), logits.numpy())
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_auc": val_auc})
        print(f"epoch {epoch:3d}  train loss {train_loss:.3f}  val loss {val_loss:.3f}  val AUC {val_auc:.3f}")

        if val_auc > best_auc:                   # keep the best model, not the last one
            best_auc = val_auc
            torch.save(model.state_dict(), folder / "best.pt")

    if overfit:
        verdict = "OK" if train_loss < 0.1 else "NOT OK - look for a bug before training for real"
        print(f"\nOverfit check: final train loss {train_loss:.3f} (should be below 0.1): {verdict}")
        return {"train_loss": train_loss}
    log = pd.DataFrame(log)
    log.to_csv(folder / "log.csv", index=False)
    plot_log(log, CHECK_DIR / "training_curves.png" if out is None else folder / "training_curves.png")
    last10 = log["val_auc"].tail(10).mean()
    print(f"\nBest val AUC {best_auc:.3f} (mean of the last 10 epochs {last10:.3f}), "
          f"saved to {folder / 'best.pt'}")
    return {"n_train": len(train_ds), "n_train_cancer": n_pos, "best_val_auc": best_auc,
            "best_epoch": int(log.loc[log["val_auc"].idxmax(), "epoch"]), "last10_val_auc": last10,
            "minutes": round((time.time() - t0) / 60, 1)}


def test():
    """The FROZEN model (models/stage1_final.pt), not best.pt: every training run overwrites best.pt."""
    test_ds = PatchDataset("test", augment=False)
    model = PatchCNN().to(DEVICE)
    model.load_state_dict(torch.load(STAGE1_MODEL, map_location=DEVICE))
    logits, labels = predict(model, DataLoader(test_ds, batch_size=64, num_workers=4))

    df = test_ds.df.assign(prob=torch.sigmoid(logits).numpy().ravel())
    df.to_csv(MODEL_DIR / "test_predictions.csv", index=False)
    print(f"Test AUC, all patches          : {roc_auc_score(df['label'], df['prob']):.3f}")
    for kind in ["benign", "random"]:   # hard negatives vs easy negatives
        sub = df[df["kind"].isin(["cancer", kind])]
        if sub["kind"].nunique() == 2:
            print(f"Test AUC, cancer vs {kind:8s}   : {roc_auc_score(sub['label'], sub['prob']):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--overfit", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cosine", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    if args.test:
        test()
    else:
        train(args.overfit, args.epochs, args.fraction, args.seed, args.cosine, args.out)
