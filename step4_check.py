"""
Step 4a check - Did the tracks follow the right spot?

1. Alignment error vs the radiologists: on earlier screenings that have their own Sybil box,
   how far was the matched point (before snapping) from the box? This is a measured accuracy
   for the step that defines Stage 2. Benign nodules are shown too, but their "truth" is the
   blob finder: a large distance there usually means the blob finder picked a different blob
   that year (label noise), not that the alignment failed.
2. Match scores and the edge rate (best match at the border of the +-20 mm search).
3. How many earlier cubes were snapped to a known lesion, and how many only aligned.
4. Same crop as step 2: every anchor cube must be identical to its step 2 patch. A difference
   means one of the two downloads lost a slice (step 2 did not check for that before).
5. Track grids for checking by eye: one track per row, T0 | T1 | T2.

Run: python step4_check.py          (also while step 4a is still running)
Writes: checks/stage2_check.json, checks/tracks_cancer.png, checks/tracks_benign.png (+ .csv)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import CHECK_DIR, PATCH_DIR, STORE, TRACK_DIR

MID = STORE // 2


def read_tracks():
    df = pd.read_csv(TRACK_DIR / "tracks.csv", on_bad_lines="skip")   # step 4a may be mid-write
    df["years_before"] = df["anchor_yr"] - df["study_yr"]
    return df


def error_stats(d):
    d = d["nearest_mm"].dropna()
    if d.empty:
        return {"n": 0}
    return {"n": int(len(d)), "median_mm": float(d.median()),
            "within_10": float((d <= 10).mean()), "within_15": float((d <= 15).mean())}


def summary(df):
    """Every number of this check as a dict (also used by the dashboard)."""
    earlier = df[df["years_before"] > 0]
    tracks = df.groupby("track_id").agg(label=("label", "first"), n=("study_yr", "size"), split=("split", "first"))
    out = {
        "patients": int(df["pid"].nunique()), "tracks": int(len(tracks)), "cubes": int(len(df)),
        "by_length": {f"{lab}": {int(n): int(c) for n, c in g["n"].value_counts().sort_index().items()}
                      for lab, g in tracks.groupby("label")},
        "by_split": {s: {int(l): int(c) for l, c in g["label"].value_counts().items()}
                     for s, g in tracks.groupby("split")},
        "alignment": {},
        "source": {f"{lab}": {s: int(c) for s, c in g["source"].value_counts().items()}
                   for lab, g in earlier.groupby("label")},
    }
    for lab, name in [(1, "cancer"), (0, "benign")]:
        e = earlier[earlier["label"] == lab]
        out["alignment"][name] = {"all": error_stats(e),
                                  **{f"{k}y": error_stats(e[e["years_before"] == k]) for k in (1, 2)}}
    if len(earlier):
        m = earlier["match"].dropna()
        out["match"] = {"quantiles": {str(q): float(m.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                        "hist": np.histogram(m.clip(0, 1), bins=20, range=(0, 1))[0].tolist(),
                        "edge_rate": float((earlier["edge"].astype(str) == "True").mean())}
    return out


def same_crop(df):
    """Anchor cubes come from the same position in the same scan as a step 2 patch: identical?"""
    index = pd.read_csv(PATCH_DIR / "index.csv")
    pairs = df[df["years_before"] == 0].merge(index, on=["pid", "study_yr", "z", "y", "x"],
                                              suffixes=("", "_patch"))
    pairs = pairs[pairs["file_patch"].map(lambda f: (PATCH_DIR / f).exists())]
    same = np.array([np.array_equal(np.load(TRACK_DIR / a), np.load(PATCH_DIR / b))
                     for a, b in zip(pairs["file"], pairs["file_patch"])], dtype=bool)
    return int(same.sum()), len(same), pairs.loc[~same, "file_patch"].tolist()


def save_grid(df, label, path, n=12):
    """The longest tracks of one label: one row per track, oldest screening on the left."""
    lengths = df[df["label"] == label].groupby("track_id").size().sort_values(ascending=False, kind="stable")
    ids = lengths.index[:n]
    if not len(ids):
        return
    fig, axes = plt.subplots(len(ids), 3, figsize=(7.5, 2.6 * len(ids)), squeeze=False)
    for row, tid in zip(axes, ids):
        t = df[df["track_id"] == tid].sort_values("study_yr")
        for ax, r in zip(row[3 - len(t):], t.itertuples()):   # right-aligned: anchor in the last column
            cube = np.load(TRACK_DIR / r.file)
            ax.imshow(cube[MID], cmap="gray", vmin=-1000, vmax=400)
            ax.plot(MID, MID, "r+", markersize=12)
            score = "" if np.isnan(r.match) else f" {r.match:.2f}"
            ax.set_title(f"{tid} T{r.study_yr} {r.source}{score}", fontsize=8)
        for ax in row:
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=70)
    plt.close(fig)
    df[df["track_id"].isin(ids)][["track_id", "study_yr", "source", "match", "nearest_mm", "file"]] \
        .to_csv(path.with_suffix(".csv"), index=False)


def pct(v):
    return "–" if v is None else f"{v:.0%}"


def main():
    df = read_tracks()
    s = summary(df)
    CHECK_DIR.mkdir(exist_ok=True)
    print(f"{s['patients']} patients, {s['tracks']} tracks, {s['cubes']} cubes\n")
    print("Tracks per label (1 = cancer) and number of screenings:")
    print(pd.DataFrame(s["by_length"]).T.fillna(0).astype(int), "\n")

    print("Alignment error before snapping, on earlier screenings with a known lesion:")
    for name, truth in [("cancer", "radiologist box"), ("benign", "blob finder")]:
        for key, a in s["alignment"][name].items():
            if a["n"]:
                print(f"  {name:6s} {key:3s} vs {truth:15s}: n {a['n']:4d}, median {a['median_mm']:5.1f} mm, "
                      f"within 10 mm {pct(a['within_10'])}, within 15 mm {pct(a['within_15'])}")
    if "match" in s:
        q = s["match"]["quantiles"]
        print(f"\nMatch score (1 = identical anatomy): 10% {q['0.1']:.2f}, median {q['0.5']:.2f}, "
              f"90% {q['0.9']:.2f}. Best match at the edge of the search: {pct(s['match']['edge_rate'])}")
    print("\nEarlier cubes snapped to a known lesion vs only aligned (per label):")
    print(pd.DataFrame(s["source"]).T.fillna(0).astype(int))

    same, n, differ = same_crop(df)
    s["same_crop"] = [same, n]
    print(f"\nAnchor cube identical to its step 2 patch: {same} of {n}")
    if differ:   # one of the two downloads lost a slice; listed so step 2 can redo those scans
        pd.Series(differ, name="file").to_csv(CHECK_DIR / "stage2_crop_differs.csv", index=False)
        print(f"  {len(differ)} differ: listed in {CHECK_DIR / 'stage2_crop_differs.csv'}")

    for label, name in [(1, "cancer"), (0, "benign")]:
        save_grid(df, label, CHECK_DIR / f"tracks_{name}.png")
    (CHECK_DIR / "stage2_check.json").write_text(json.dumps(s, indent=1))
    print(f"\nTrack grids and stage2_check.json saved in {CHECK_DIR}/")


if __name__ == "__main__":
    main()
