"""
One dashboard evaluation: retrain the CNN on the patches made so far, then compute the
step 3c AUCs and the step 3d Grad-CAM centre score on one split (val by default).

Started by dashboard/server.py. Writes dashboard/runs/<run>/{result.json, gradcam.png, best.pt}
and appends a row to dashboard/runs/history.json. Never touches models/ or checks/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
from config import BATCH_SIZE, LEARNING_RATE, PATCH, PATCH_DIR, SEED  # noqa: E402
from step3_gradcam import grad_cam  # noqa: E402
from step3_model import PatchCNN, PatchDataset  # noqa: E402
from step3_train import DEVICE, predict, train_one_epoch  # noqa: E402


def snapshot():
    """index.csv as it is right now, minus half-written lines and duplicates."""
    index = pd.read_csv(PATCH_DIR / "index.csv", on_bad_lines="skip").drop_duplicates("file", keep="last")
    return index[index["file"].map(lambda f: (PATCH_DIR / f).exists())].reset_index(drop=True)


def dataset(index, split, augment):
    ds = PatchDataset(split, augment)
    ds.df = index[index["split"] == split].reset_index(drop=True)
    return ds


def aucs(df):
    """The three step 3c numbers."""
    out = {"auc_all": roc_auc_score(df["label"], df["prob"])}
    for kind in ["benign", "random"]:
        sub = df[df["kind"].isin(["cancer", kind])]
        out[f"auc_{kind}"] = roc_auc_score(sub["label"], sub["prob"]) if sub["kind"].nunique() == 2 else None
    return out


def gradcam(model, ds, path):
    """Step 3d on this split's cancers: centre score + a 4x4 picture."""
    model = model.cpu().eval()
    cancers = ds.df.index[ds.df["kind"] == "cancer"][:64]
    x = torch.stack([ds[i][0] for i in cancers])
    cam = grad_cam(model, x)
    prob = torch.sigmoid(model(x)).detach().ravel()
    lo, hi = PATCH // 2 - 12, PATCH // 2 + 12
    total = cam.sum(dim=(1, 2, 3))
    has_heat = total > 0                              # an all-zero map would give 0 / 0
    centre = (cam[has_heat, lo:hi, lo:hi, lo:hi].sum(dim=(1, 2, 3)) / total[has_heat]).mean().item() \
        if has_heat.any() else None

    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    mid = PATCH // 2
    for ax, i in zip(axes.ravel(), range(len(x))):
        ax.imshow(x[i, 0, mid], cmap="gray", vmin=0, vmax=1)
        ax.imshow(cam[i, mid], cmap="jet", alpha=0.35, vmin=0, vmax=1)
        ax.set_title(f"p(cancer) = {prob[i]:.2f}", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return centre, len(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--scans-done", type=int, default=0)
    args = parser.parse_args()
    out = HERE / "runs" / args.run
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    torch.manual_seed(SEED)
    index = snapshot()
    train_ds = dataset(index, "train", augment=True)
    val_ds = dataset(index, "val", augment=False)
    eval_ds = dataset(index, args.split, augment=False)
    n_pos = int(train_ds.df["label"].sum())
    print(f"Training on {len(train_ds)} patches ({n_pos} cancer) from {args.scans_done} scans, "
          f"evaluating on {args.split} ({len(eval_ds)} patches)")

    loader = dict(num_workers=4, persistent_workers=True)   # don't respawn workers every epoch
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, **loader)
    val_dl = DataLoader(val_ds, batch_size=64, **loader)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(len(train_ds) - n_pos) / n_pos], device=DEVICE))
    model = PatchCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    curve, best_auc, best_state = [], -1.0, None
    for epoch in range(1, args.epochs + 1):       # same recipe as step3_train.py
        train_loss = train_one_epoch(model, train_dl, loss_fn, optimizer)
        logits, labels = predict(model, val_dl)
        val_auc = roc_auc_score(labels.numpy(), logits.numpy())
        curve.append({"epoch": epoch, "train_loss": train_loss, "val_auc": val_auc})
        (out / "curve.json").write_text(json.dumps({     # read live by the dashboard
            "run": args.run, "epochs": args.epochs, "scans_done": args.scans_done,
            "n_train": len(train_ds), "curve": curve}))
        print(f"epoch {epoch:3d}/{args.epochs}  train loss {train_loss:.3f}  val AUC {val_auc:.3f}")
        if val_auc > best_auc:
            best_auc, best_state = val_auc, {k: v.detach().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, out / "best.pt")
    logits, _ = predict(model, DataLoader(eval_ds, batch_size=64, num_workers=0))
    preds = eval_ds.df.assign(prob=torch.sigmoid(logits).numpy().ravel())
    centre, n_cam = gradcam(model, eval_ds, out / "gradcam.png")

    counts = index.groupby("kind").size().to_dict()
    result = {
        "run": args.run, "finished": time.strftime("%Y-%m-%d %H:%M"), "split": args.split,
        "scans_done": args.scans_done, "epochs": args.epochs, "minutes": round((time.time() - t0) / 60, 1),
        "n_train": len(train_ds), "n_train_cancer": n_pos, "n_eval": len(eval_ds),
        "patches": {k: int(v) for k, v in counts.items()},
        "best_val_auc": best_auc, **aucs(preds), "centre": centre, "n_cam": n_cam,
        "curve": curve,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    history_file = HERE / "runs" / "history.json"
    hist = json.loads(history_file.read_text()) if history_file.exists() else []
    hist.append({k: v for k, v in result.items() if k != "curve"})
    history_file.write_text(json.dumps(hist, indent=1))
    print(f"Done in {result['minutes']} min: AUC {result['auc_all']:.3f}, "
          f"cancer vs benign {result['auc_benign'] or float('nan'):.3f}, "
          f"cancer vs random {result['auc_random'] or float('nan'):.3f}, centre score {centre if centre is None else round(centre, 2)}")


if __name__ == "__main__":
    main()
