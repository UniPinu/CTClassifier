"""
Step 2 check - Look at the patches before you train on them.

Prints counts, the most common problems and a density check, and saves image grids
to checks/ so you can check the patches by eye.

Run: python step2_check.py
"""
import matplotlib
matplotlib.use("Agg")  # save images without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CHECK_DIR, PATCH_DIR, STORE

KINDS = ["cancer", "benign", "random"]
MID = STORE // 2


def save_grid(rows, path, n=36):
    """Middle axial slice of n patches, with a red + at the centre where the lesion should be."""
    rows = rows.head(n)
    fig, axes = plt.subplots(6, 6, figsize=(12, 12))
    for ax, (k, row) in zip(axes.ravel(), enumerate(rows.itertuples())):
        cube = np.load(PATCH_DIR / row.file)
        ax.imshow(cube[MID], cmap="gray", vmin=-1000, vmax=400)   # lung window
        ax.plot(MID, MID, "r+", markersize=12)
        ax.set_title(f"#{k}  {row.size_mm:.0f} mm", fontsize=9)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)
    rows[["file", "pid", "kind", "size_mm"]].to_csv(path.with_suffix(".csv"), index_label="tile")


def main():
    index = pd.read_csv(PATCH_DIR / "index.csv")
    CHECK_DIR.mkdir(exist_ok=True)

    print("Patches per split and kind:")
    print(index.groupby(["split", "kind"]).size().unstack(fill_value=0), "\n")
    print(f"Patients with patches: {index['pid'].nunique()}")
    print(f"Disk used by patches : {sum(f.stat().st_size for f in PATCH_DIR.glob('*.npy')) / 1e9:.1f} GB\n")

    failures = PATCH_DIR / "failures.csv"
    if failures.exists():
        problems = pd.read_csv(failures)["problem"].str.replace(r"\d+", "N", regex=True)
        print("Most common problems:")
        print(problems.value_counts().head(8), "\n")

    # Density check: a nodule is solid (> -400 HU) at the centre, open lung is not.
    print("Density at the centre (4x4x4 mm), up to 200 patches per kind:")
    for kind in KINDS:
        rows = index[index["kind"] == kind]
        if rows.empty:
            continue
        sample = rows.sample(min(200, len(rows)), random_state=0)
        centre = [np.load(PATCH_DIR / f)[MID - 2:MID + 2, MID - 2:MID + 2, MID - 2:MID + 2].mean()
                  for f in sample["file"]]
        print(f"  {kind:7s} median {np.median(centre):6.0f} HU, "
              f"solid in {np.mean(np.array(centre) > -400):.0%}")
        save_grid(sample, CHECK_DIR / f"grid_{kind}.png")

    # Shape and value check on a few files
    for f in index["file"].sample(min(20, len(index)), random_state=1):
        cube = np.load(PATCH_DIR / f)
        assert cube.shape == (STORE, STORE, STORE) and cube.dtype == np.int16, f
        assert -1100 < cube.min() and cube.max() < 3100, f
    print(f"\nShape and value check passed. Image grids saved in {CHECK_DIR}/")


if __name__ == "__main__":
    main()
