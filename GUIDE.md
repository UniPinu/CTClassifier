# P5 lung nodule classifier: implementation guide (steps 1–3)

This guide takes you from the raw NLST data to a trained 3D CNN that looks at a small cube of lung around a nodule and answers "cancer or not?". You write and train the network yourselves, from scratch. Sybil is used only for its radiologist boxes, never for its model.

```
 NLST tables + Sybil boxes ──[step 1]──▶ cohort/*.csv          who is in the study, which scan to use
 cohort/*.csv              ──[step 2]──▶ patches/*.npy         a 72 mm cube around every lesion
                                         (one scan at a time, never more than 50 GB on disk)
 patches/*.npy             ──[step 3]──▶ models/best.pt        your trained CNN
                                         + test AUC, Grad-CAM heat maps
```

> **Status:** I tested the whole pipeline end to end on synthetic CT scans written as real DICOM files. Every planted lesion was found within 1 mm, resuming and the disk budget worked, and every script runs. It has **not** been run on the real NLST data yet. The checks in this guide are how you confirm that it works there too.

---

## Contents

0. [Setup](#0-setup)
1. [Step 1: Cohort](#step-1-cohort)
2. [Step 2: Patches](#step-2-patches)
3. [Step 3: Train the CNN](#step-3-train-the-cnn)
4. [What to write in the report](#what-to-write-in-the-report)
5. [Reading the code](#reading-the-code)

Every step is laid out the same way:

| Part | Tells you |
|---|---|
| **What it does** | the goal, the inputs and the outputs |
| **Run it** | the exact command |
| **How the code works** | a walk through the script, function by function |
| **Check that it worked** | concrete checks with expected results. Do not skip these. |
| **If something looks wrong** | the usual causes |

---

## 0. Setup

### Folder layout

```
p5/
├── GUIDE.md               this file
├── requirements.txt
├── config.py              ALL paths and settings, in one place
├── step1_cohort.py
├── step2_patches.py
├── step2_check.py
├── step3_model.py         dataset + network (imported by the two below)
├── step3_train.py
├── step3_gradcam.py
├── step3_evaluate.py      CIs, breakdowns, find rate of the frozen model (3e)
├── step3_experiments.py   clean learning-curve / epoch experiments (3f)
├── step3_realistic.py     cancers located like benign nodules (3g)
├── step4_*.py             Stage 2: see STAGE2.md
├── annotations.json       you download this (see below)
│
├── cohort/                created by step 1
├── patches/               created by step 2
├── tmp_dicom/             step 2's scratch space: holds ONE scan at a time
├── checks/                images for checking by eye (steps 2 and 3)
└── models/                created by step 3
```

### Install

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For the GPU, install PyTorch using the selector on pytorch.org, choosing the CUDA version. The plain `pip install torch` may give you a CPU-only build. Then check:

```bash
python -c "import torch; print(torch.cuda.is_available())"     # must print True on the school PC
```

### Get the radiologist boxes

The Sybil README (github.com/reginabarzilaygroup/Sybil), under **Annotations**, links to a JSON file on Google Drive. Save it as `annotations.json` in the project folder. Its format:

```json
{ "<SeriesInstanceUID>": {
    "<SOPInstanceUID of one slice>": [ {"x": 0.62, "y": 0.41, "width": 0.037, "height": 0.033} ],
    "<another slice>": [] } }
```

`x, y` is the box's top-left corner, and every value is a fraction of the 512×512 image.

### Settings

Everything you might want to change is in `config.py`: folders, `MAX_GB = 50`, `CONTROLS_PER_CASE = 3`, patch sizes, `EPOCHS` and so on. Don't hard-code numbers in the scripts.

### Disk budget

| What | Size |
|---|---|
| idc-index metadata + NLST clinical tables | < 1 GB |
| `tmp_dicom/`, one scan at a time | ~100 MB, deleted right after reading |
| `patches/`, 0.75 MB per patch | roughly 8,000–12,000 patches ≈ **6–9 GB** |
| models, images, CSVs | < 100 MB |

`step2_patches.py` checks the budget before every download and stops cleanly before going over `MAX_GB`.

---

## Step 1: Cohort

### What it does

It decides **who** is in the study, **which split** each person is in, and **which CT series** to use for each screening. Nothing is downloaded except small metadata tables.

| Output | One row per | Key columns |
|---|---|---|
| `cohort/patients.csv` | patient | `pid`, `label` (1 = cancer), `n_screens`, `split` |
| `cohort/scans.csv` | screening (patient × year) | `pid`, `study_yr`, `SeriesInstanceUID`, `annotated`, `series_size_MB` |
| `cohort/control_nodules.csv` | benign nodule | `pid`, `study_yr`, `sct_slice_num`, `sct_epi_loc` (lobe), `sct_long_dia` (mm) |

### Run it

```bash
python step1_cohort.py
```

### How the code works

| Function | What it does |
|---|---|
| `load_clinical` | Downloads NLST's participant table (`nlst_prsn`) and CT findings table (`nlst_ctab`) through idc-index. NLST writes missing values as codes like `.N`, and `pd.to_numeric(errors="coerce")` turns them into `NaN`. |
| `load_series` | Lists every NLST CT series, with slice thickness and image type. NLST's study dates are placeholders: 1999 = T0, 2000 = T1, 2001 = T2. It drops localizers (flat scout images), series with fewer than 50 slices, and the rare compressed series. |
| `pick_one_series` | Each screening was reconstructed 2–3 times, so it keeps the **thinnest** series, as Sybil does. |
| `load_annotated_series` | Finds the series that have at least one radiologist box. These patients are your **cancer cases**. |
| `find_controls` | A **control** must meet four conditions: (1) never diagnosed with lung cancer, (2) known cancer-free at least a year after their last scan, (3) at least one non-calcified nodule ≥ 4 mm (`sct_ab_desc == 51`) with a known slice number, (4) at least 2 screenings. It samples `CONTROLS_PER_CASE` × (number of cases) of them. |
| `build_cohort` | Glues these together. For annotated screenings it uses the **annotated** series, because the boxes are pixel positions in that exact series. |
| `split_patients` | Splits 70/15/15 **by patient**, stratified so every split gets the same share of cancers. |

**Why split by patient:** if one patient's scans were in both train and test, the model could recognise the patient instead of the cancer, and your test score would be too good.

### Check that it worked

1. **The printed table.** Expect roughly 581 cases (somewhat fewer if a few annotated series are not in the index) and about 3× as many controls. The cancer share (column `1`) should be about 25% in **every** split.
2. **The asserts passed.** The script stops with an error if a patient appears twice or a screening has two series.
3. **Spot-check one control by hand:**
   ```python
   import pandas as pd
   p = pd.read_csv("cohort/patients.csv"); n = pd.read_csv("cohort/control_nodules.csv")
   pid = p[p.label == 0].pid.iloc[0]
   print(n[n.pid == pid])            # at least one nodule: sct_ab_desc 51, a slice number, a size
   ```
4. **No patient in two splits:** `p.groupby("pid").split.nunique().max() == 1`.

### If something looks wrong

* **`get_clinical_table` returns `None` or errors:** the clinical download (through `s5cmd`) was blocked. Download the NLST clinical ZIP from the TCIA NLST page instead and read the CSVs with `pd.read_csv`. The column names are the same.
* **"0 found in NLST":** `annotations.json` is missing or is the wrong file.

---

## Step 2: Patches

### What it does

For every lesion it cuts out a **72 × 72 × 72 mm cube**, resampled so that 1 voxel = 1 mm, and saves it as a small `.npy` file. There are three kinds of patch:

| kind | label | where the centre comes from |
|---|---|---|
| `cancer` | 1 | the radiologists' boxes, on cancer cases. This includes earlier scans of a spot that later became cancer. |
| `benign` | 0 | NLST's nodule records for controls (slice number + lobe + size), via a **blob finder** |
| `random` | 0 | a random point inside the lungs, ≥ 30 mm from every lesion. These are the **easy negatives**. |

**Why 72 if the network sees 64?** The extra 8 voxels let training cut the 64³ window at a slightly different place each time, shifting it by up to ±4 mm. See augmentation in step 3.

**Which scans:** the annotated scans of the cases, plus every control screening that has a recorded nodule. Cancers go first, so if anything stops the run early, you have your positives.

### One scan at a time

```
for each scan in the to-do list:
    1. download   ──▶ tmp_dicom/<series>/          (~100 MB of DICOM files)
    2. read       ──▶ 3D array of Hounsfield units in memory
                      ...and tmp_dicom/<series>/ is deleted immediately
    3. resample   ──▶ 1 × 1 × 1 mm voxels
    4. lung mask  ──▶ where the lungs are
    5. centres    ──▶ cancer (boxes) · benign (blob finder) · random
    6. crop+save  ──▶ patches/<pid>_T<year>_<n>_<kind>.npy    (72³ int16, 0.75 MB)
    7. log        ──▶ index.csv · failures.csv · done.txt
```

Only one scan is ever on disk. Before each download the script adds that scan's size to the disk already used and **stops** if the total would pass `MAX_GB`.

### Run it

```bash
python step2_patches.py
```

There are roughly 4,000–6,000 scans. On a test machine with two slow CPU cores a scan took about 14 s, plus download time, so plan for **one or two nights**. You can stop it at any time with **Ctrl+C**. Running it again **resumes**: every finished series is in `patches/done.txt` and gets skipped.

Each scan prints one line:
```
[37/4512] pid 100158 T1: 3 patches, 0 problems, 0.4 GB used
```

### How the code works

**Getting a scan into memory**

| Function | What it does |
|---|---|
| `download` | Asks idc-index for exactly one series. |
| `read_series` | Reads all slices, sorts them by height (`ImagePositionPatient[2]`, feet → head) and converts to Hounsfield units (`pixel × RescaleSlope + RescaleIntercept`). It also returns the spacing and two look-up tables: slice UID → slice index (for Sybil's boxes) and `InstanceNumber` → slice index (for NLST's slice numbers). It refuses scans with unusual orientation, because the code assumes image-left = patient-right. |
| `to_1mm` | `scipy.ndimage.zoom` with factor = old spacing / 1 mm. For example, 2.5 mm slices are stretched ×2.5. |
| `lung_mask` | Takes air (< −400 HU) inside the body. Air that touches the image edge is outside the body and is removed. It keeps the parts larger than 200 ml and fills the holes left by vessels and nodules. |

**Finding the centres**

* **`cancer_centres`:** each box is 2D, on one slice. It converts each box to mm: fraction × 512 × pixel spacing. Then it groups boxes on neighbouring slices at about the same spot into one lesion, and the lesion's centre is the average of its boxes.
* **`find_nodule`** (benign) works in five steps:
  1. Look up the slice whose `InstanceNumber` equals NLST's `sct_slice_num`.
  2. Take a slab of ±15 mm around it, and keep solid voxels (> −400 HU) in or right next to the lungs.
  3. Keep only the correct lung. The lobe code gives the side: 1–3 = right, 4–6 = left. The patient's right is on the **left** of the image.
  4. **Morphological opening** with a ball about a quarter of the nodule's diameter. Vessels are thinner than the ball and disappear; the nodule is round and survives.
  5. Of the remaining blobs, pick the one whose size is closest to the recorded diameter (it must be within 0.5–2×) and closest to the slice.
* **`random_lung_points`:** tries random voxels until it finds ones inside the lungs and far from every lesion.

**The slice-number problem.** The radiologist's slice number refers to the series *they* read, which may be a different reconstruction from the one you downloaded. If those differ, slice 112 in theirs is not slice 112 in yours. The code copes in three ways: it searches ±15 mm instead of one slice, it requires the blob to have the right size, and it logs every miss in `failures.csv`. The eye check below tells you how often it worked.

**Saving.** The volume is padded with 36 voxels of air on every side, so a cube near the edge still fits. Each cube is saved as `int16` Hounsfield units, and a row goes into `index.csv`:

```
file,pid,study_yr,series_uid,kind,label,z,y,x,size_mm,split
100158_T1_0_cancer.npy,100158,1,1.2.840.113654...,cancer,1,150,159,94,14.7,train
```

### Check that it worked

Run the check script. You can run it while step 2 is still going, to catch problems early.

```bash
python step2_check.py
```

1. **Counts** per split and kind. Every split should have all three kinds.
2. **Most common problems.** For example, `slice N not in this series` or `no blob of N mm near slice N`. A few percent is normal; a large share means the blob finder needs work (see below).
3. **Density at the centre.** A nodule is solid; open lung is not. Expected:

   | kind | median HU at the centre | "solid" share |
   |---|---|---|
   | cancer | above −400 (often near 0) | high, ideally > 90% |
   | benign | above −400 | high, ideally > 80% |
   | random | around −850 | low (only vessels) |

   If `benign` is low, the blob finder often misses. If `cancer` is low, the box conversion has a bug.
4. **Eye check.** Open `checks/grid_benign.png`. It shows the middle slice of 36 benign patches, with a red **+** where the nodule should be. Count the tiles where the + is **not** on a round blob, then compute the hit rate = (36 − misses) / 36. Do the same for `grid_cancer.png` and `grid_random.png`. The matching `.csv` next to each image tells you which file each tile is. **Write the hit rates down: they go in the report.** To look at another 36, change `random_state=0` in `step2_check.py`.
5. **Shape and value check.** This runs automatically: every patch must be 72³ `int16` with plausible HU values.
6. **Disk.** The printed "Disk used by patches" must stay under your budget.

### If something looks wrong

| Symptom | Likely cause and fix |
|---|---|
| Many `slice N not in this series` | The radiologist read a different reconstruction. Try changing `pick_one_series` for controls so it prefers a series with at least as many slices as the highest `sct_slice_num`. |
| Benign hit rate < ~80% | Make `window` smaller (fewer wrong candidates) or loosen the size range. Look at the misses: are they vessels (make the opening ball bigger) or the chest wall (lower the dilation)? |
| `scan failed: unusual orientation` | Rare. These scans are skipped on purpose. |
| `scan failed: download failed` | A network problem. Delete that line from `done.txt` and run again. |
| It stops with "Stopping: the next scan would take us over 50 GB" | The budget is working. Check whether `patches/` holds files you don't need. |

---

## Step 3: Train the CNN

### What it does

It trains a small 3D CNN to output one number per patch: a high value means cancer. Then it measures how well that works on patients the model has never seen, and checks *where* in the patch the model is looking.

### The network

```
input patch          (batch, 1, 64, 64, 64)    one channel: HU scaled to 0..1
block 1  ──▶         (batch, 16, 32, 32, 32)   simple features: edges, bright spots
block 2  ──▶         (batch, 32, 16, 16, 16)
block 3  ──▶         (batch, 64,  8,  8,  8)
block 4  ──▶         (batch, 128, 4,  4,  4)   complex features: "round, spiky blob"
global average pool  (batch, 128)              the embedding: 128 numbers per patch
dropout + linear     (batch, 1)                one logit: > 0 means "cancer"
```

Each block is **Conv3d → BatchNorm → ReLU → MaxPool**. That's 291,345 weights in total. That is tiny, on purpose: you have only hundreds of cancers.

### The concepts, briefly

| Concept | What it does and why it's here |
|---|---|
| **Conv3d** (3×3×3) | Slides small learnable 3D filters over the cube. Early filters learn edges, later ones learn shapes. |
| **ReLU** | Sets negative values to 0. Without it, stacking layers would just be one big linear function. |
| **MaxPool** (2) | Halves each dimension. Later layers then "see" a larger part of the patch, and there is less to compute. |
| **BatchNorm** | Re-centres and re-scales each channel during training. It keeps training stable and lets you use a normal learning rate. |
| **Global average pooling** | Averages each of the 128 channels over the 4³ grid. That gives 128 inputs to the last layer instead of 8,192, so far fewer weights to overfit. |
| **Dropout** (0.3) | Randomly zeroes 30% of the embedding during training, so the model can't rely on one feature. |
| **Augmentation** | Each time a patch is used: a random shift (±4 mm), random flips and a random 90° turn. The model never sees exactly the same cube twice. With only hundreds of cancers, this is your **main defence against overfitting**. It also stops the model from learning "the blob is always exactly in the centre". Validation and test use the plain centre crop. |
| **BCE loss + `pos_weight`** | Binary cross-entropy punishes confident wrong answers. Since about 85% of patches are not cancer, `pos_weight` = (non-cancers / cancers) makes each cancer count that many times. Without it, answering "never cancer" would already look good. |
| **Adam**, lr = 1e-3 | The optimizer: how the weights are nudged after each batch. |
| **AUC** | The chance that a random cancer patch scores higher than a random non-cancer patch. 0.5 = coin flip, 1.0 = perfect. It doesn't depend on a threshold or on the class balance, unlike accuracy. |
| **Grad-CAM** | A heat map of which parts of the patch pushed the score towards "cancer". It's how you check the model looks at the nodule and not the background. |

### How the code works

* **`step3_model.py`**
  * `PatchDataset` reads `index.csv`, keeps one split and turns each `.npy` into a tensor. `to_unit` clips HU to [−1000, 400] and scales to 0..1, then the crop and augmentation are applied.
  * `PatchCNN` is the network above.
* **`step3_train.py`**
  * `train_one_epoch` is the core of all deep learning, in four lines: forward pass → loss → `backward()` → `optimizer.step()`.
  * `predict` runs the model over a split without training (`model.eval()`, `torch.no_grad()`).
  * After every epoch it computes val loss and val AUC and saves the model **only if val AUC improved**. That way you keep the best model, not the last one.

### Run it, in this order

#### 3a. Overfit check (2 minutes)

```bash
python step3_train.py --overfit
```

This trains on just 10 cancer + 10 non-cancer patches, with no augmentation. A working network **must** be able to memorise 20 examples, so the train loss should fall towards 0:

```
epoch   1  train loss 0.726
...
epoch 100  train loss 0.010
Overfit check: final train loss 0.010 (should be below 0.1): OK
```

If it says **NOT OK**, there is a bug: labels, data loading or the model. Fix it before going on. Real training on a broken pipeline wastes hours and still produces numbers that look real.

#### 3b. Train

```bash
python step3_train.py
```

```
Training on 7316 patches (790 cancer), validating on 1540
epoch   1  train loss 0.981  val loss 0.902  val AUC 0.712
epoch   2  ...
Best val AUC 0.873, saved to models/best.pt
```

These numbers are only an illustration. It writes `models/best.pt`, `models/log.csv` and `checks/training_curves.png`.

**Read the curves** in `checks/training_curves.png`:

| What you see | What it means | What to try |
|---|---|---|
| train ↓, val ↓, val AUC ↑ then flat | good | done |
| train ↓, val ↑ (gap opens) | **overfitting**: memorising the training patches | more augmentation, more dropout, fewer channels |
| both flat and high | **underfitting** or a bug | rerun 3a; try lr 3e-4; more channels |
| val AUC jumps around a lot | the val split is small | look at the trend, not single epochs |

**Change one thing per run** and log every run in a spreadsheet (setting changed → best val AUC). Choose settings on **validation only**.

#### 3c. Test, once

```bash
python step3_train.py --test
```

```
Test AUC, all patches          : 0.861
Test AUC, cancer vs benign     : 0.742
Test AUC, cancer vs random     : 0.968
```

(Example numbers.)

* **Cancer vs random** should be high: "nodule vs no nodule" is easy.
* **Cancer vs benign** is the **real** question, cancerous nodule vs harmless nodule. It will be lower, and it's the number to discuss.
* Run this **once**, at the end. If you change the model after seeing test results, the test set has become a second validation set and its AUC no longer means anything.
* Don't be surprised if every cancer gets a "probability" like 0.2 while the AUC is excellent. AUC only measures the **ranking** (do cancers score higher than non-cancers?), and `pos_weight` shifts the scores. Treat the output as a risk score, not a calibrated probability.

Predictions are saved to `models/test_predictions.csv`, which is useful for plots in the report.

#### 3d. Grad-CAM

```bash
python step3_gradcam.py
```

```
Centre score on 64 test cancers: 0.41 (0.05 = no focus on the centre)
```

* **Centre score** = the share of the heat map inside the central 24³ mm, where the lesion is. Heat spread evenly would give (24/64)³ ≈ 0.05. Clearly above that means the model focuses on the lesion.
* **`checks/gradcam.png`** shows 16 test cancers with the heat map on top. Look for heat **on the nodule**. Heat on the chest wall, the patch edges or empty lung means the model has found a shortcut, and that's worth a paragraph in the report even if the AUC looks good.

#### Freeze the final model

Every `step3_train.py` run overwrites `models/best.pt`. The model you report, and that Stage 2 builds on, is frozen as **`models/stage1_final.pt`** (`config.STAGE1_MODEL`). Ours is the final 120-epoch dashboard run, `dashboard/runs/20261102_183302/best.pt`, with its training curve in `models/stage1_final_log.csv`. `step3_train.py --test` and all scripts below use the frozen model.

#### 3e. Evaluate the frozen model properly

```bash
python step3_evaluate.py           # validation split; --test ONCE at the very end
```

It prints every AUC with a **95% bootstrap CI** (2,000 resamples of whole *patients*: a patient's patches are not independent), **specificity at 90% sensitivity**, and two breakdowns of the cancers: **precursor vs newest** box and **solid vs subsolid** (centre above / below −400 HU, the blob finder's limit). It also gives the blob finder's **find rate** and the **mean of the last 10 epochs** (quote that, not the best epoch). Validation results of the frozen model:

| | cancers | AUC [95% CI] | spec @ 90% sens |
|---|---|---|---|
| all patches | 180 | 0.962 [0.943, 0.979] | 0.91 |
| cancer vs benign | 180 | 0.945 [0.920, 0.967] | 0.85 |
| cancer vs random | 180 | 0.981 [0.967, 0.993] | 0.97 |
| newest cancer vs benign | 102 | 0.966 [0.945, 0.983] | 0.90 |
| precursor vs benign | 78 | 0.918 [0.867, 0.958] | 0.83 |
| solid cancer vs benign | 99 | 0.967 [0.948, 0.982] | 0.89 |
| subsolid cancer vs benign | 81 | 0.918 [0.867, 0.961] | 0.83 |

Mean of the last 10 epochs: 0.952 (best epoch 0.962). Blob finder: a patch for 5,196 of 6,074 benign records (86%). Subsolid cancers are *harder*, not easier, so "subsolid = cancer" is not what drives the AUC. Precursors are harder too: that is the headroom for Stage 2. The hit rate still needs the eye check of step 2.

#### 3f. Clean experiments

```bash
python step3_experiments.py        # ~6 h; resumable; --only learning | epochs | cosine
```

The dashboard history changed data size, epochs and the split together. These runs change **one** thing: a learning curve (25/50/75/100% of the training patients, 120 epochs, 2 seeds), epochs (40/80/120) and a cosine learning-rate schedule. Each run is saved in `models/runs/<name>/` (never `best.pt`) and becomes a row in `models/experiments.csv`. The same options exist on `step3_train.py`: `--epochs`, `--fraction`, `--seed`, `--cosine`, `--out`.

#### 3g. Realistic evaluation

```bash
python step3_realistic.py          # ~110 downloads, ~10 min with 6 workers
```

Cancers come from hindsight boxes, benign nodules from the blob finder: did the model learn the difference? Here the val cancers are located **like the benign nodules**: the blob finder on NLST's own nodule records, with the boxes used only afterwards, to see which blob is the cancer. Result on our data: the blob finder located 46 of 180 boxed val lesions (it finds solid nodules, and many precursors have no record). On those 46, Stage 1 scores **0.987 [0.975, 0.994]** with blob-located cubes vs **0.985** with box-located cubes of the same lesions (AUC vs val benign). The location method is not what the model learned. Look at `checks/grid_realistic.png` to see that the blobs are nodules.

---

## What to write in the report

| Topic | What to say |
|---|---|
| Task | Classification of a **given** location, not detection. The model is told where the nodule is. |
| Labels | Cancer = radiologist boxes on biopsy-confirmed cancers (and their earlier precursors). Benign = NLST nodule records in people who never got cancer. |
| Label noise | Benign centres come from a heuristic blob finder. Report its eye-check hit rate. |
| Shortcut risk | Positives and negatives were located differently. Mitigations: random shifts, Grad-CAM centre score, cancer-vs-benign AUC reported separately. |
| Leakage | Split by patient; test set used once. |
| Results | Overfit check, training curves, val AUC per experiment (your spreadsheet), test AUCs, Grad-CAM examples. |

---

## Reading the code

For code review, read in this order. Each file is short and does one thing.

| File | Lines | What to look at |
|---|---|---|
| `config.py` | 43 | every setting |
| `step1_cohort.py` | 147 | `find_controls` (who counts as a control), `split_patients` |
| `step2_patches.py` | 262 | `process_scan` (the per-scan recipe), then `cancer_centres` and `find_nodule` |
| `step2_check.py` | 75 | the density check |
| `step3_model.py` | 76 | `__getitem__` (augmentation) and `PatchCNN` |
| `step3_train.py` | 172 | `train_one_epoch` (4 lines), the best-model logic, `test` |
| `step3_gradcam.py` | 73 | `grad_cam` |
| `step3_evaluate.py` | 155 | `bootstrap` (whole patients), `spec_at_sens`, the breakdowns in `main` |
| `step3_experiments.py` | 57 | `GRID`: one change per run |
| `step3_realistic.py` | 147 | `locate` (the blob finder on a cancer patient's records) |

### Next: stage 2 (the tracker)

The trained `PatchCNN.encoder` turns any patch into 128 numbers. Stage 2 reuses it as a frozen encoder and runs a small GRU over the embeddings from T0 → T1 → T2 at the same spot. For that you need cubes from the *other* screenings of each lesion, which means aligning scans across years. See **`STAGE2.md`**.
