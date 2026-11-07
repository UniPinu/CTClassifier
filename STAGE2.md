# P5 lung nodule classifier: implementation guide, Stage 2 (step 4)

Stage 1 looks at **one** cube around a lesion and answers "cancer or not?". Stage 2 asks the research question of the project:

> **Does seeing a lesion's history predict cancer better than seeing only its newest scan?**

It follows each lesion back through the patient's earlier annual screenings (T0 → T1 → T2), turns every year's cube into 128 numbers with the **frozen** Stage 1 encoder, and lets a small GRU read them in time order. The comparison with Stage 1 is always made on exactly the same lesions and screenings.

```
 patches/index.csv + cohort/ ──[4a]──▶ tracks/*.npy, tracks.csv    the same lesion at T0, T1, T2
 tracks/ + stage1_final.pt   ──[4b]──▶ tracks/embeddings.npz       128 numbers per cube (frozen encoder)
 embeddings                  ──[4c]──▶ models/stage2_results.csv   tracker vs Stage 1, with 95% CI
```

> **Status:** step 4a was tested on synthetic CTs (every position within 1.4 mm of the truth, including spots where the lesion did not exist yet) and on **11 real NLST patients** (alignment error vs the radiologists' boxes: median 3.0 mm, 11 of 11 within 10 mm). Steps 4b and 4c were tested on synthetic data. The full NLST run has not been done yet. The checks below are how you confirm that it works there.

Read `CONTEXT.md` section 7 for *why* Stage 2 is designed this way. This guide is the *how*.

---

## Contents

0. [Before you start](#0-before-you-start)
1. [Step 4a: build tracks](#step-4a-build-tracks)
2. [Step 4b: embeddings](#step-4b-embeddings)
3. [Step 4c: the tracker](#step-4c-the-tracker)
4. [The dashboard's Stage 2 tab](#the-dashboards-stage-2-tab)
5. [What to write in the report](#what-to-write-in-the-report)
6. [Reading the code](#reading-the-code)

Every step has the same parts as in `GUIDE.md`: **what it does**, **run it**, **how the code works**, **check that it worked**, **if something looks wrong**.

---

## 0. Before you start

1. **Install scikit-image** (new for step 4a): `pip install -r requirements.txt`.
2. **The frozen Stage 1 model must exist:** `models/stage1_final.pt`. It is a copy of the final 120-epoch dashboard run (`dashboard/runs/20261102_183302/best.pt`; val AUC 0.962 / 0.945 / 0.981). Its training curve is in `models/stage1_final_log.csv`. **Do not** replace it with `models/best.pt`: that file is overwritten by every `step3_train.py` run, and right now it is an early 40-epoch model (val AUC 0.75). Check it with:
   ```bash
   python step3_evaluate.py     # must print 0.962 / 0.945 / 0.981
   ```
3. Step 2 must be finished: step 4a takes the known lesion positions from `patches/index.csv`.

New settings in `config.py`:

| Setting | Value | Meaning |
|---|---|---|
| `TRACK_DIR` | `tracks/` | step 4's cubes and CSVs |
| `STAGE1_MODEL` | `models/stage1_final.pt` | the frozen Stage 1 model |
| `TEMPLATE` | 48 | mm cube of anatomy that is searched for in the earlier scan |
| `SEARCH` | 20 | mm the match may move from the first guess |
| `SNAP_MM` | 15 | a known lesion this close to the match counts as the same lesion |
| `N_AUG` | 8 | augmented embeddings per cube, for training the tracker |
| `TRACKER_EPOCHS`, `TRACKER_LR`, `TRACKER_BATCH` | 300, 1e-3, 64 | tracker training |
| `N_BOOTSTRAP` | 2000 | resamples of whole patients for every 95% CI |

---

## Step 4a: build tracks

### What it does

| Term | Meaning |
|---|---|
| **track** | one lesion at several screenings, oldest first |
| **anchor** | the newest screening where the lesion's position is *known*: a Sybil box (cancer) or a benign nodule step 2 found. A track never uses a screening after its anchor, so no post-diagnosis scans |
| **label** | 1 = the spot of a later biopsy-confirmed cancer (including its precursors); 0 = a benign nodule in a never-diagnosed control |

Who: every patient with **at least 2 screenings up to their newest known lesion**. On our data that is **1,702 patients** (310 cancer, 1,392 control) and **4,619 downloads**.

Output, `tracks/tracks.csv`, one row per cube:

| Column | Meaning |
|---|---|
| `track_id` | `<pid>_<n>` |
| `label`, `anchor_yr`, `study_yr`, `split` | as above; the split is the patient's Stage 1 split |
| `source` | `known` = the exact centre of a known lesion; `aligned` = found by template matching only |
| `match` | template-matching score of this step (1 = identical anatomy); empty at the anchor |
| `edge` | the best match lay at the border of the ±20 mm search (the true spot may be outside it) |
| `nearest_mm` | distance from the match (before snapping) to the nearest known lesion of that year; empty if there is none |
| `z, y, x` | the centre in that scan's 1 mm voxels (the same coordinates as `patches/index.csv`) |
| `file` | the 72³ int16 cube in `tracks/` |

### Run it

```bash
python step4_sequences.py --limit 10   # a quick trial on 10 patients first
python step4_sequences.py              # everything, 6 patients at a time (--workers 6)
```

On 11 real patients it took 1 min 45 s with 6 workers (28 downloads), so plan for **about 4–5 hours** for the full run, depending on the network. It needs about **5 GB** for the cubes, and at most one scan per worker is in `tmp_dicom/step4/` at a time. Stop it with Ctrl+C (or from the dashboard); running it again **resumes**, because finished patients are in `tracks/done.txt`. Each patient prints one line:

```
[37/1702] pid 100158 (cancer): 1 tracks, 3 cubes, 0 problems, 8.4 GB used
```

### How the code works

Per patient (`run_patient`), newest screening first, with at most 2 scans in memory:

1. **`load_scan`** reuses step 2's own functions (`download`, `read_series`, `to_1mm`, `lung_mask`), so every coordinate means exactly what it means in `patches/index.csv`. The DICOM files are deleted right after reading. If a slice is missing, it downloads once more (see below).
2. **`match`**, for every track that reached the newer scan:
   1. *First guess*: the point plus the shift of the lungs' centre of mass between the two scans.
   2. *Refine*: cut the 48 mm cube of anatomy around the point in the **newer** scan (HU clipped to −1000…400) and slide it over an 88 mm region around the guess in the **earlier** scan. `skimage.feature.match_template` gives a **normalised cross-correlation** score for each of the 41³ offsets; the best one is the match. Normalised means brightness and contrast differences between scanners don't matter, only the pattern.
3. **Snap**: if a known lesion of that year is within 15 mm, use its exact centre (`source = known`).
4. **Chain**: T0 is matched from T1's position, not from T2's.
5. Known lesions that no newer track reached **start their own track** (anchored at that year).
6. **`cut`** crops every cube exactly like step 2 (72³, air outside the scan).

Two more details:

* **Failed scans are skipped, not fatal.** If one screening can't be read, the tracks are matched across it (T2 → T0).
* **Disk budget.** Patches + tracks + the scans in progress stay under `MAX_GB`.

### Check that it worked

```bash
python step4_check.py        # can also run while step 4a is still going
```

1. **Alignment error vs the radiologists**: the defining number of Stage 2. On earlier screenings that have their own Sybil box, how far was the match (before snapping) from the box? On 11 real patients: **median 3.0 mm, 100% within 10 mm**. Expect a few larger misses on the full run (real lungs deform). **Write the median and the share within 10 and 15 mm into the report.** Note: if a patient has several boxed lesions and the tracked one has no box that year, the distance is to a *different* lesion, so the tail of this distribution is pessimistic.
2. **Benign rows** use the blob finder as "truth". On the 11 test patients the median was 48 mm. That is not an alignment failure (the same alignment hits the boxes within 3 mm). It means the blob finder often picks a **different blob in different years**, so it measures label noise in the benign set. It is worth a sentence in the threats to validity.
3. **Match score**: median about 0.84 on real data. **Edge rate**: should be a few percent at most.
4. **known vs aligned** per label: cancers are mostly `known` (precursor boxes); benign earlier cubes are mostly `aligned`.
5. **Same crop as step 2**: every anchor cube must be byte-identical to its step 2 patch. If some differ, they are listed in `checks/stage2_crop_differs.csv`: one of the two downloads lost a slice (see below).
6. **Eye check**: `checks/tracks_cancer.png` and `checks/tracks_benign.png`, one track per row, oldest left. The red + should sit on the same anatomy in every column. Cancers should grow from left to right; benign nodules should look the same.

### If something looks wrong

| Symptom | Likely cause and fix |
|---|---|
| Large cancer alignment errors, high edge rate | The lungs moved more than 20 mm between scans (e.g. a different breath-hold). Try `SEARCH = 30`. |
| Low match scores everywhere | Different reconstructions (soft vs sharp kernel) between years. NCC copes with brightness, not with texture. Note it as a confound. |
| `missing slices` in `tracks/failures.csv` | The download lost a file twice in a row. Delete the patient from `tracks/done.txt` and its rows from `tracks.csv`, then run again. |
| Anchor cubes differ from step 2 | See the missing-slice note below. |

> **Missing slices: a bug found during testing.** When a single DICOM file fails to download, idc-index only *logs* the error, and step 2 used to read the series anyway. Every slice above the gap then sits one slice too low. `read_series` now refuses such a series (`missing slices`), and step 4a downloads it again. Patches made by step 2 before this fix may contain a few such scans. The "same crop" check counts them for the anchor scans; `stage2_crop_differs.csv` lists them, so their scans can be re-done in step 2 (delete them from `patches/done.txt`).

---

## Step 4b: embeddings

### What it does

It runs the Stage 1 **encoder**, frozen, over every cube in `tracks.csv` and saves `tracks/embeddings.npz`:

| Array | Shape | Used for |
|---|---|---|
| `centre` | (n, 128) | the plain centre crop: validation and test |
| `aug` | (n, 8, 128) | 8 crops with Stage 1's augmentation (shift, flips, rotation): training |
| `logit`, `score` | (n,) | Stage 1's own output (score = sigmoid(logit)): the baseline |
| `files` | (n,) | the cube files, in `tracks.csv` order (step 4c checks this) |

**Why frozen:** Stage 1 and the tracker then see each cube through the same "eyes", so any gain comes from the history; ~300 cancer tracks are far too few to retrain 291 k weights; and every year is embedded by the same function, so a difference between years means the lesion changed.

### Run it

```bash
python step4_embed.py        # a few minutes on the GPU
```

### Check that it worked

It prints two checks, both on the **validation** tracks (Stage 1 never trained on these patients):

1. **Sanity**: Stage 1 AUC on the anchor cubes should be close to Stage 1's AUC on the val patches (cancer vs benign, about 0.945). Both come with a 95% CI. If the anchor AUC is much lower, the track cubes are not what step 2 cut. Run `step4_check.py` and look at "same crop".
2. **Headroom**: Stage 1 AUC by screenings before the anchor (0, 1, 2), cancer vs benign cubes at the same offset. Stage 1 already scores precursor patches lower than the newest ones (0.918 vs 0.966 on the val patches, see `step3_evaluate.py`). **If the AUC drops one screening back, that is where history can help**, and that drop is the motivation for Stage 2 in the report.

---

## Step 4c: the tracker

### What it does

Each track becomes a **window of exactly k screenings** ending at the anchor (or, with `--early`, ending one screening before it). Always exactly k, so the number of screenings can never give the label away. For each screening *t* the input is the embedding and its change since the last screening:

$$x_t=\big[e_t,\ e_t-e_{t-1}\big],\qquad h_t=\mathrm{GRU}(h_{t-1},x_t),\qquad \text{score}=\sigma(v^\top h_k+c)$$

Dropout 0.2 → `GRU(256 → 64)` → `Linear(64 → 1)`: **61,889 weights** (`--plain`: embedding only, `GRU(128 → 64)`, 37,313 weights).

| Concept | What it does |
|---|---|
| **GRU** | a recurrent layer: it reads one screening at a time and keeps a 64-number summary *h* of what it has seen. Two gates (sigmoid) decide how much of the old summary to keep and how much of the new screening to write in. |
| **change input** | the GRU could compute e_t − e_(t−1) itself, but giving it directly makes "did it grow?" easy to learn with few examples. On synthetic data it raised the 3-screening AUC from 0.64 to 0.74. |
| **augmented embeddings** | each epoch, each screening uses a random one of its 8 augmented embeddings: the tracker's version of Stage 1's augmentation. |
| **model selection** | the epoch with the best **validation** AUC is kept, also with `--test`, so the test split never chooses anything. |

**Baselines on exactly the same windows:**
1. **Stage 1**: the Stage 1 score of the newest cube in the window.
2. **Score trend**: logistic regression on [newest logit, newest − oldest logit], fitted on the train windows. If it is as good as the tracker, history helps but a two-number summary is enough.

**Note on train scores:** the train windows' cubes include the very patches Stage 1 was trained on, so their embeddings are "too easy". That is why everything is chosen and reported on validation (and, at the end, test).

### Run it, in this order

```bash
python step4_tracker.py --scans 2 --overfit   # 1. does the code work? train loss must go below 0.1
python step4_tracker.py --scans 2             # 2. does history help at the newest known screening?
python step4_tracker.py --scans 3
python step4_tracker.py --scans 2 --early     # 3. ... one screening BEFORE the anchor? (most interesting)
python step4_tracker.py --scans 1             # 4. sanity: no history, should be about Stage 1
python step4_tracker.py --scans 3 --plain     # 5. does the change input matter?
python step4_tracker.py --scans 2 --test      # 6. ONCE per setting, at the very end
```

Each run takes well under a minute (CPU). The dashboard's **Run the evaluation plan** button runs 1–5. Use `--seed 43` for a second seed: small numbers make single runs noisy.

Output per run:

```
Setting k2: the last 2 screening(s) up to the anchor. Windows: train 1180 (210 cancer), val 250 (45 cancer)
Best val AUC 0.95 at epoch 140 (22 s). On the val split, 95% CI from resampling patients:

                             AUC [95% CI]   spec @ 90% sens
Tracker              0.951 [0.920, 0.978]              0.80
Stage 1 (newest)     0.945 [0.913, 0.972]              0.75
Score trend          0.948 [0.916, 0.975]              0.78
Tracker - Stage 1   +0.006 [-0.010, +0.022]

AUC per anchor year (does the year carry the label?):
  anchor T1: tracker 0.953, Stage 1 0.947 (120 windows, 22 cancer)
```

(Illustrative numbers.) Saved: `models/tracker_<setting>.pt`, its training log, the per-window predictions (`models/stage2_pred_<setting>_<split>.csv`) and one row in `models/stage2_results.csv`.

### How to read the results

* **The 95% CI of "Tracker − Stage 1"** is computed on the *same* bootstrap samples (2,000 resamples of whole patients), so it accounts for the two scores being correlated. History helps only if the **whole interval is above 0**.
* **Specificity at 90% sensitivity**: the share of benign nodules *not* flagged while 90% of cancers are caught. "Fewer false alarms at the same sensitivity" is what history should buy.
* **Per anchor year**: if the gain only exists for one anchor year, the year (or scanner) may be carrying the label.

| Outcome | Interpretation |
|---|---|
| tracker > Stage 1, CI excludes 0, holds within each anchor year | **history helps** |
| ≈ at the newest screening, > with `--early` | history helps exactly before the lesion is obvious |
| trend ≈ tracker > Stage 1 | history helps, but a two-number summary is enough |
| ≈ everywhere | history adds nothing beyond the newest scan: still a valid answer |
| `--scans 1` ≫ Stage 1 | leakage: something other than the lesion carries the label |

On synthetic data where the change is informative, the code gives exactly the expected pattern: k=3 tracker 0.995 vs Stage 1 0.873 (Δ +0.12, CI +0.06 to +0.21); k=1 tracker 0.887 vs Stage 1 0.884 (Δ +0.003, CI −0.008 to +0.014); without the change input 0.941.

---

## The dashboard's Stage 2 tab

`python dashboard/server.py`, open http://localhost:8050 and click **Stage 2** (or open `/#stage2`).

| Part | What it shows |
|---|---|
| top bar | Start / Stop step 4a (disabled while step 2 runs), progress, speed, ETA, disk |
| **Run** | Check tracks, Compute embeddings (only when 4a is not running), Train tracker (pick k, early, plain), Overfit check, Run the evaluation plan. Jobs run one at a time in a queue; Stop clears the queue. `--test` is not on the dashboard on purpose. |
| Alignment vs radiologist boxes | live, while 4a runs: median error, within 10/15 mm, per years back |
| Template match score | histogram of match scores; edge rate |
| Tracks | tracks per label × screenings, per split, known vs aligned, the same-crop check |
| Headroom | Stage 1 AUC by screenings before the anchor, with the val-patch AUC as a dashed line |
| Does history help? | tracker − Stage 1 AUC with 95% CI, latest val run per setting |
| Tracker results | every run (`stage2_results.csv`); click a row for its training curve (dashed line = Stage 1 on the same windows) |
| Track grids | the images from `step4_check.py` |

---

## What to write in the report

| Topic | What to say |
|---|---|
| Task | Classification of a **given** lesion over time, not detection. Windows of exactly k screenings. |
| Alignment | Template matching (NCC) after a lung-centre shift; snapping to known lesions within 15 mm. **Measured accuracy** against the radiologists' boxes (median, within 10/15 mm, n). |
| Frozen encoder | Why it is frozen (fair comparison, data size, stable coordinates). Cost: the encoder learned "is it cancer now", not "will it become cancer". |
| Headroom | Stage 1 AUC by screenings before the anchor: the motivation. |
| Results | The table of settings, Δ AUC with 95% CI, specificity at 90% sensitivity, per-anchor-year AUC, the score-trend baseline. |
| Threats | Small numbers (CI width), blob-finder label noise (the benign alignment numbers), scanner/kernel changes between years, the year carrying the label, missing-slice downloads (and the fix). |

---

## Reading the code

| File | Lines | What to look at |
|---|---|---|
| `step4_sequences.py` | 255 | `run_patient` (the walk back in time), then `match` (the template matching) |
| `step4_check.py` | 145 | `summary` (every number of the check), `same_crop` |
| `step4_embed.py` | 90 | `TrackPatches` (5 lines: Stage 1's dataset on the track cubes), `embed` |
| `step4_tracker.py` | 216 | `Tracker.forward` (the change input + GRU), `make_windows`, `train`, then the baselines and the bootstrap in `main` |
| `step3_evaluate.py` | 155 | `bootstrap` (resampling whole patients), `spec_at_sens`; also used by step 4 |
| `dashboard/server.py` | | `stage2_status`, `stage2_commands` |
