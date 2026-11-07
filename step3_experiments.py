"""
Step 3f - Clean Stage 1 experiments: change ONE thing, keep everything else fixed.

The dashboard history mixes data size, epochs and (early on) the split, so it cannot say what
helped. These runs can:
  learning  25 / 50 / 75 / 100% of the training patients, 120 epochs, seeds 42 and 43
  epochs    40 / 80 epochs, all patients, seed 42   (120 is the 100% run of "learning")
  cosine    120 epochs with a cosine learning-rate schedule, all patients, seed 42

Every run trains from scratch with step3_train.train() and saves to models/runs/<name>/, never
to models/best.pt or models/stage1_final.pt. Each finished run becomes a row in
models/experiments.csv; runs already there are skipped, so you can stop and resume.
Report "last10_val_auc" (mean of the last 10 epochs): the best epoch is optimistic.
About 41 min per 100% run on the school PC: roughly 6 hours for everything.

Run: python step3_experiments.py                  everything
     python step3_experiments.py --only epochs    one group
"""
import argparse

import pandas as pd

from config import MODEL_DIR, SEED
from step3_train import train

GRID = (
    [("learning", f, 120, s, False) for s in (SEED, SEED + 1) for f in (0.25, 0.5, 0.75, 1.0)]
    + [("epochs", 1.0, e, SEED, False) for e in (40, 80)]
    + [("cosine", 1.0, 120, SEED, True)]
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["learning", "epochs", "cosine"])
    args = parser.parse_args()
    results = MODEL_DIR / "experiments.csv"
    done = set(pd.read_csv(results)["name"]) if results.exists() else set()

    for group, fraction, epochs, seed, cosine in GRID:
        name = f"f{int(fraction * 100)}_e{epochs}_s{seed}" + ("_cos" if cosine else "")
        if (args.only and group != args.only) or name in done:
            continue
        print(f"\n=== {name}: {fraction:.0%} of patients, {epochs} epochs, seed {seed}"
              f"{', cosine' if cosine else ''}", flush=True)
        summary = train(False, epochs, fraction, seed, cosine, out=name)
        row = {"name": name, "group": group, "fraction": fraction, "epochs": epochs,
               "seed": seed, "cosine": cosine, **summary}
        pd.DataFrame([row]).to_csv(results, mode="a", header=not results.exists(), index=False)
        done.add(name)

    if results.exists():
        print("\n", pd.read_csv(results).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
