"""
Step 3 - Train the 3D CNN from scratch, then test it once.

Run:
  python step3_train.py --overfit   # sanity check: train loss on 20 patches must go to ~0
  python step3_train.py             # real training; keeps the model with the best val AUC
  python step3_train.py --test      # ONCE at the very end: AUC on the test split

Writes: models/best.pt, models/log.csv, checks/training_curves.png, models/test_predictions.csv
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from config import BATCH_SIZE, CHECK_DIR, EPOCHS, LEARNING_RATE, MODEL_DIR, SEED
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


def plot_log(log):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(log["epoch"], log["train_loss"], label="train")
    ax1.plot(log["epoch"], log["val_loss"], label="val")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss"); ax1.legend()
    ax2.plot(log["epoch"], log["val_auc"])
    ax2.set_xlabel("epoch"); ax2.set_ylabel("val AUC")
    fig.tight_layout()
    fig.savefig(CHECK_DIR / "training_curves.png", dpi=80)
    plt.close(fig)


def train(overfit):
    torch.manual_seed(SEED)
    train_ds = PatchDataset("train", augment=not overfit)
    val_ds = PatchDataset("val", augment=False)
    epochs = EPOCHS
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
    MODEL_DIR.mkdir(exist_ok=True)
    CHECK_DIR.mkdir(exist_ok=True)
    log, best_auc = [], 0.0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer)
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
            torch.save(model.state_dict(), MODEL_DIR / "best.pt")

    if overfit:
        verdict = "OK" if train_loss < 0.1 else "NOT OK - look for a bug before training for real"
        print(f"\nOverfit check: final train loss {train_loss:.3f} (should be below 0.1): {verdict}")
    else:
        pd.DataFrame(log).to_csv(MODEL_DIR / "log.csv", index=False)
        plot_log(pd.DataFrame(log))
        print(f"\nBest val AUC {best_auc:.3f}, saved to {MODEL_DIR / 'best.pt'}")


def test():
    test_ds = PatchDataset("test", augment=False)
    model = PatchCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_DIR / "best.pt", map_location=DEVICE))
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
    args = parser.parse_args()
    test() if args.test else train(args.overfit)
