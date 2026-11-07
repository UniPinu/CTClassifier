# CONTEXT.md: P5 longitudinal lung nodule project

This is the single reference for **what we are building, why it is built this way, how each part works, and how it is evaluated**. It covers Stage 1 (done) and Stage 2 (implemented, tested on synthetic data and on 11 real NLST patients, not yet run on the full cohort). For step-by-step instructions, see `GUIDE.md` (steps 1–3) and `STAGE2.md` (step 4). The design rationale is also in the paper draft.

---

## 0. Summary

We train our **own** neural network, from scratch, on the National Lung Screening Trial (NLST). It works in two stages:

* **Stage 1.** A small 3D CNN (291,345 weights) looks at a 64×64×64 mm cube centred on a lung lesion and outputs one score: *cancer or not*. It is trained on radiologist-boxed cancers, benign nodules from the trial records, and random lung tissue. Done; validation AUC 0.962 overall, **0.945 vs benign nodules**.
* **Stage 2.** The same lesion is followed back through the patient's earlier annual screenings (T0 → T1 → T2). The frozen Stage 1 encoder turns each year's cube into 128 numbers, and a GRU reads them in time order. The research question:

> **Does seeing a lesion's history predict cancer better than seeing only its newest scan?**

The model is always **told where the lesion is**. It classifies lesions; it does not find them.

---

## 1. Framing and constraints

| | |
|---|---|
| Project | P5 semester project, Computer Science, Aalborg University, Fall 2026. Group with no prior ML experience; **the learning process is the goal**, not beating published models |
| Supervisor's requirement | build and train our **own** model. Re-using Sybil's risk outputs (or Sybil as an encoder) is not enough |
| Hardware | one school PC: RTX 4080 (16 GB VRAM), 128 GB RAM |
| Disk budget | **50 GB maximum**, so data is streamed one scan at a time and only small cubes are kept |
| Code style | barebones, short scripts, reviewable by bachelor students; all settings in `config.py` |
| Related work | **Sybil** (Mikhael et al., JCO 2023): 3D CNN (R3D-18 backbone, 33.4 M weights) on a *single* whole LDCT, 1–6 year risk, AUC 0.92 at 1 year on NLST. **Ardila et al.** (Nat Med 2019): end-to-end model using current + prior CT |

---

## 2. Data

### 2.1 NLST at a glance (CT arm, from the IDC index)

| Item | Value |
|---|---|
| CT patients | 26,254 |
| CT screenings (studies) | 73,113: T0, T1, T2, one year apart |
| CT series | 203,111 (1–4 reconstructions per screening, plus localizers) |
| Total size | 11.1 TB |
| Patients with 3 / 2 / 1 screenings | 22,264 (84.8%) / 2,076 / 1,829 (and 85 with 4+) |
| Thinnest series per screening | ≤ 2.5 mm in 94% (≤1.3 mm 9%, 2.0 mm 38%, 2.5 mm 47%, 3.0–3.2 mm 6%) |
| One series per screening | ~91 MB, ~254 MB per patient |

Access: `idc-index` (Python). `client.index` lists series; `client.ct_index` adds slice thickness, kernel and image type. Clinical tables come via `client.get_clinical_table(...)`, or as CSVs from the TCIA NLST page. **DICOM study dates are placeholders: 1999 = T0, 2000 = T1, 2001 = T2.**

### 2.2 Clinical tables

| Table | One row per | Key columns |
|---|---|---|
| `nlst_prsn` | participant | `pid`, `can_scr` (0 = no cancer), `candx_days` (days randomisation → diagnosis), `canc_free_days` (last known cancer-free), `cancyr`, `scr_days0/1/2` (day of each screening), `loc*` (cancer lobe flags) |
| `nlst_ctab` | abnormality per screening | `study_yr`, `sct_ab_desc` (**51 = non-calcified nodule ≥ 4 mm**), `sct_slice_num` (slice of greatest diameter; 999 = missing), `sct_epi_loc` (lobe: 1 RUL, 2 RML, 3 RLL, 4 LUL, 5 lingula, 6 LLL, 8 other), `sct_long_dia`, `sct_perp_dia`, `sct_margins`, `sct_pre_att` (soft tissue / ground glass / mixed) |
| `nlst_ctabc` | abnormality at the **comparison read** | `sct_ab_preexist` (seen before?), `sct_ab_gwth` (**interval growth**), `sct_ab_attn` (density change), `visible_days` (earliest date visible) |
| `nlst_canc` | lung cancer | `study_yr` / `candx_days` of diagnosis, `lc_topog` (lobe), `de_type` (histology), `de_stag` (stage), `lesionsize` |

Missing values use codes like `.N` / `.M`; convert with `pd.to_numeric(errors="coerce")`. The public tables have no `fup_days`, so we use `canc_free_days` instead. **No column links the same nodule across years.**

### 2.3 Two kinds of lesion location

| | **NLST records** (`nlst_ctab`) | **Sybil boxes** |
|---|---|---|
| Drawn by / when | trial radiologists, while reading each screening (2002–04) | two thoracic radiologists, retrospectively (~2020), using MD.ai |
| Knew the outcome? | **no** | **yes**: guided by where the biopsy-confirmed cancer was |
| Covers | everyone, every screening, every nodule ≥ 4 mm (mostly benign) | 581 patients diagnosed within 1 year of a screening; only the cancer, plus **precursors at the same spot** on earlier scans |
| Location | slice number + lobe (no x/y; slice may refer to a different series) | 2D box on every slice of the lesion, pixel-precise, on a specific series |
| Size | ~177,500 records (all types) | 9,280 boxes on 970 series |

Sybil JSON format: `{SeriesInstanceUID: {SOPInstanceUID: [{"x", "y", "width", "height"}, ...]}}`. `(x, y)` is the top-left corner, and all values are fractions of the 512×512 image. Match slices on the `SOPInstanceUID` **tag**, never the file name. In IDC the same boxes exist as DICOM SR (`analysis_result_id = nlst_sybil`).

**Implication:** our positives are hindsight-guided and precise, while our negatives are prospective records located by a heuristic. That asymmetry is the main validity risk (section 9).

### 2.4 Longitudinal facts (computed from the IDC index)

* Sybil-annotated screenings per patient: 1 / 2 / 3 = **309 / 161 / 111**. Patterns: T0 only 262, T0+T1 130, T0+T1+T2 111, T1+T2 26, T2 24, T1 23, T0+T2 5.
* Screenings up to and including the newest box: 1 / 2 / 3 = **266 / 160 / 155**.
  * **315 cancer patients can be tracked** (≥ 2 screenings); 155 of them over all three. Newest box at T1: 150, at T2: 165.
  * Of their **470 earlier screenings, 383 (81%) have their own box**: ground truth for following a lesion back in time. 87 have no box.
  * Only 4 patients have a screening after their newest box.
* The same patient's scans across 3 years (22,131 patients):

  | Same in all 3 years | |
  |---|---|
  | slice thickness | 88.8% |
  | manufacturer | 93.1% |
  | scanner model | 57.1% |
  | kernel | 53.8% |
  | pixel spacing within 0.05 mm | 64.1% |

  The slice-count ratio between years has a median of 1.10 (1.37 at the 90th percentile). Consequences: always resample to 1 mm, never compare slice numbers across years, and treat scanner or texture change as a possible confound.

---

## 3. Pipeline overview

```
NLST tables + Sybil boxes ─[step 1]→ cohort/          who, which split, which series
cohort/                   ─[step 2]→ patches/         72³ cube per lesion (one scan at a time)
patches/                  ─[step 3]→ models/          Stage 1 PatchCNN + evaluation
patches/ + cohort/        ─[step 4a]→ tracks/         same lesion at T0, T1, T2
tracks/ + stage1_final.pt ─[step 4b]→ embeddings.npz  128 numbers per cube (frozen encoder)
embeddings                ─[step 4c]→ Stage 2 results tracker vs Stage 1, with 95% CI
```

| File | Lines | Role |
|---|---|---|
| `config.py` | 43 | every path and setting |
| `step1_cohort.py` | 147 | cohort, series choice, patient split |
| `step2_patches.py` | 262 | download → read → resample → find centres → crop (refuses series with a missing slice) |
| `step2_parallel.py` | 121 | step 2 with several scans at once (used by the dashboard) |
| `step2_check.py` | 75 | counts, density check, image grids |
| `step3_model.py` | 76 | `PatchDataset` (augmentation) + `PatchCNN` |
| `step3_train.py` | 172 | training, overfit check, single test run; `--epochs/--fraction/--seed/--cosine/--out` |
| `step3_gradcam.py` | 73 | Grad-CAM + centre score |
| `step3_evaluate.py` | 155 | frozen model: bootstrap CIs, spec @ 90% sens, precursor/subsolid breakdown, find rate |
| `step3_experiments.py` | 57 | clean experiments: learning curve, epochs, cosine (6.5 item 4) |
| `step3_realistic.py` | 147 | cancers located with the blob finder from `nlst_ctab` (6.5 item 5) |
| `step4_sequences.py` | 255 | build tracks across screenings (6 patients at a time, resumable) |
| `step4_check.py` | 145 | alignment error vs radiologist boxes, match scores, same-crop check, track grids |
| `step4_embed.py` | 90 | frozen-encoder embeddings, sanity check, headroom |
| `step4_tracker.py` | 216 | GRU tracker, baselines, bootstrap CI |
| `dashboard/server.py`, `index.html` | 441, 829 | two tabs: Stage 1 (step 2 + retrains) and Stage 2 (step 4a progress, checks, tracker runs) |

Key settings: `MAX_GB = 50`, `CONTROLS_PER_CASE = 3`, `SEED = 42`, `STORE = 72`, `PATCH = 64`, `N_RANDOM = 1`, `BATCH_SIZE = 32`, `LEARNING_RATE = 1e-3`, `EPOCHS` (the final Stage 1 run used **120**), `STAGE1_MODEL = models/stage1_final.pt`, `TEMPLATE = 48`, `SEARCH = 20`, `SNAP_MM = 15`, `N_AUG = 8`, `TRACKER_EPOCHS = 300`, `N_BOOTSTRAP = 2000`.

---

## 4. Step 1: Cohort

* **Cases:** every patient with at least one Sybil box (≈ 581).
* **Controls:** all four must hold:
  1. `can_scr == 0` and no `candx_days`;
  2. `canc_free_days ≥ last screening day + 365`;
  3. at least one `sct_ab_desc == 51` record with a valid slice number and diameter;
  4. ≥ 2 usable screenings.

  Sampled at 3 per case (≈ 1,743).
* **Series per screening:** drop localizers, series with fewer than 50 slices, and compressed (JPEG) series. Keep the **thinnest**, ties broken by more slices. For annotated screenings, use the **annotated** series, because the boxes are pixel positions in that series.
* **Split:** 70/15/15 **by patient**, stratified by label, seed 42. Asserts: no patient twice, no screening with two series.
* **Outputs:** `patients.csv` (pid, label, n_screens, split), `scans.csv` (one row per screening, with `annotated` and `series_size_MB`), `control_nodules.csv` (slice, lobe, diameter, series).

---

## 5. Step 2: Patches (one scan at a time)

**Loop per scan:** budget check (patches so far + next scan ≤ 50 GB) → download one series → read into memory → **delete the DICOM files** → resample → find centres → crop → append to `index.csv`, `failures.csv` and `done.txt` (resumable).

| Operation | How |
|---|---|
| Read | sort slices by `ImagePositionPatient[2]`, HU = pixel × slope + intercept, clip −1024…3071; refuse non-standard orientation (image left = patient right) |
| Resample | `scipy.ndimage.zoom` with factor = spacing / 1 mm, so 1 voxel = 1 mm |
| Lung mask | air < −400 HU, remove air touching the in-plane image edge, keep components > 200 ml, fill holes slice by slice |
| **Cancer centre** | box centre in mm: $\big(k\Delta_z,\ (y+\tfrac h2)N_\text{rows}\Delta_y,\ (x+\tfrac w2)N_\text{cols}\Delta_x\big)$. A box joins a lesion if it lies ≤ 3 slices after the lesion's last box and its in-plane offset (\|Δx\| + \|Δy\|) is < 15 mm. Lesion centre = mean of its boxes |
| **Benign centre** (blob finder) | slice = `InstanceNumber == sct_slice_num`; slab ±15 mm; solid (> −400 HU) within 2 voxels of the lung; keep the lung given by the lobe code; morphological opening with a ball of radius max(1, round(d/4)) to remove vessels; among blobs 0.5–2× the recorded diameter, minimise \|size − d\| + \|z − slice\| |
| **Random centre** | one random lung voxel per scan, ≥ 30 mm (Chebyshev) from every lesion: an easy negative |
| Crop | 72³ cube (pad with −1024), saved as int16 HU, 0.75 MB |

**Real run (snapshot at 84%, 3,877 of 4,605 scans):** 8,971 patches: 1,025 cancer, 4,132 benign, 3,814 random, from 1,969 patients, 6.7 GB. About 990 scans per hour. Final total ≈ 8 GB. Exact final counts are in `patches/index.csv` and `step2_check.py`.

**Checks:** density at the centre (cancer and benign solid, random ≈ −850 HU); eye check of 36-patch grids with a **hit rate for the blob finder**, which goes in the report; failure reasons.

---

## 6. Stage 1: the PatchCNN

### 6.1 Architecture

$$e_\theta(x) = \mathrm{GAP}\Big(B_4\big(B_3(B_2(B_1(x)))\big)\Big)\in\mathbb{R}^{128},\qquad B_\ell(h)=\mathrm{MaxPool}_2\Big(\mathrm{ReLU}\big(\mathrm{BN}(\mathrm{Conv}_{3^3}(h))\big)\Big)$$

$$f_\theta(x) = \sigma\big(w^\top e_\theta(x) + b\big)$$

| Layer | Output | Receptive field | Weights |
|---|---|---|---|
| input (HU clipped −1000…400, scaled 0–1) | 1 × 64³ | 1 mm | – |
| block 1 | 16 × 32³ | 4 mm | 464 |
| block 2 | 32 × 16³ | 10 mm | 13,888 |
| block 3 | 64 × 8³ | 22 mm | 55,424 |
| block 4 | 128 × 4³ | 46 mm | 221,440 |
| global average pool | 128 (**the embedding**) | whole patch | – |
| dropout 0.3 + linear | 1 logit | | 129 |
| **total** | | | **291,345** |

In code: `model(x)` → logit; `model.encoder(x)` → 128 numbers; `model.head(e)` → logit.

**What each part does:**
* **Conv3d:** 3×3×3 learned pattern detectors, applied everywhere.
* **BatchNorm:** re-centres each channel to keep training stable; uses stored averages in `eval()`.
* **ReLU:** keeps positive evidence and adds non-linearity.
* **MaxPool:** halves the size, so later layers see more context.
* **Global average pooling:** each of the 128 channels becomes "how much of pattern j is in this cube".

**Why this design** (constraints from section 1):
* **Patches:** a 10 mm nodule is 0.001% of a chest CT but 0.2% of a 64³ cube.
* **Small network:** about 115× fewer weights than Sybil's backbone, because there are only hundreds of positives and no pre-training.
* **3D, not 2D:** malignancy shows in 3D shape.
* **Standard choices:** VGG-style 3×3 stacks, channels doubling as resolution halves, BN, GAP instead of flatten, dropout.

### 6.2 Training

* **Augmentation** (training only): random shift of the 64³ window inside the 72³ cube (±4 mm), flip per axis (p = 0.5), random 90° axial rotation. The shift also stops the model learning "perfectly centred = cancer". Validation and test use the centre crop.
* **Loss:** `BCEWithLogitsLoss(pos_weight = N−/N+)`:
  $$\mathcal L=-\tfrac1N\sum_i\big[w\,y_i\log f(x_i)+(1-y_i)\log(1-f(x_i))\big]$$
  The know-nothing baseline is $\ln 2\cdot 2N_-/N$ (≈ 1.15–1.26 for our class mixes). Raw losses are not comparable across class mixes.
* **Optimizer:** Adam, lr 1e-3, batch 32, `num_workers = 4`. The final run used 120 epochs (about 41 min on 7,562 training patches).
* **Model selection:** keep the epoch with the best validation AUC (`models/best.pt`). **Freeze it as `models/stage1_final.pt`**, because every training run overwrites `best.pt`. Done: the frozen model is the final dashboard run (`dashboard/runs/20261102_183302/best.pt`, curve in `models/stage1_final_log.csv`). `models/best.pt` is an older 40-epoch run (val AUC 0.75) and must not be used.
* **Overfit check** (`--overfit`): 10 + 10 patches, no augmentation; the train loss must go below 0.1. Judge the train loss, not eval-mode metrics (BatchNorm statistics lag with one batch).

### 6.3 Evaluation

| Metric | Meaning |
|---|---|
| **AUC, all patches** | P(cancer patch scores higher than a non-cancer patch); no threshold, insensitive to class balance |
| **AUC cancer vs benign** | the clinically relevant question |
| **AUC cancer vs random** | the easy one: lesion vs open lung |
| **Grad-CAM centre score** | share of heat in the central 24³ mm (uniform = 0.05): does the model look at the lesion? |
| specificity at 90% sensitivity | share of benign nodules *not* flagged when catching 90% of cancers (clinical framing; recommended) |

The test split is evaluated **once**, after all decisions (`--test`). Report validation numbers until then.

### 6.4 Results so far (validation split unless noted)

| Run | Scans | Train patches (cancer) | Epochs | AUC | vs benign | vs random | Centre |
|---|---|---|---|---|---|---|---|
| final | 4,605 | 7,562 (704) | 120 | **0.962** | **0.945** | **0.981** | **0.77** |
| | 3,141 | 5,040 (704) | 120 | 0.958 | 0.939 | 0.977 | 0.76 |
| | 2,641 | 4,157 (704) | 80 | 0.949 | 0.922 | 0.973 | 0.64 |
| | 2,130 | 3,237 (704) | 40 | 0.894 | 0.873 | 0.910 | 0.35 |
| | 850 | 1,272 (552) | 40 | 0.816 | 0.792 | 0.818 | 0.28 |
| | 200 | 345 (77) | 40 | 0.703 | 0.750 | 0.678 | 0.04 |

The early rows may have been scored on the test split; the dashboard switched to validation at the 2,641-scan run.

**Frozen final model, validation split, 95% CI from 2,000 resamples of patients** (`step3_evaluate.py`):

| | cancers | AUC [95% CI] | spec @ 90% sens |
|---|---|---|---|
| all patches | 180 | 0.962 [0.943, 0.979] | 0.91 |
| cancer vs benign | 180 | 0.945 [0.920, 0.967] | 0.85 |
| cancer vs random | 180 | 0.981 [0.967, 0.993] | 0.97 |
| newest cancer vs benign | 102 | 0.966 [0.945, 0.983] | 0.90 |
| precursor vs benign | 78 | 0.918 [0.867, 0.958] | 0.83 |
| solid cancer vs benign (centre > −400 HU) | 99 | 0.967 [0.948, 0.982] | 0.89 |
| subsolid cancer vs benign | 81 | 0.918 [0.867, 0.961] | 0.83 |

Mean of the last 10 epochs: **0.952** (best epoch 0.962). Subsolid cancers are harder, not easier, so "subsolid = cancer" does not drive the AUC. Precursors are harder: the headroom for Stage 2.

**Realistic evaluation** (`step3_realistic.py`): val cancers located like benign nodules, with the blob finder on NLST's own records (`nlst_ctab`, 222 records on 109 scans; boxes only used afterwards to see which blob is the cancer). The finder located **46 of 180** boxed val lesions (it finds solid nodules; many precursors have no record). On those 46: AUC vs val benign **0.987 [0.975, 0.994]** blob-located vs **0.985** box-located (same lesions). The location method is not what the model learned, at least for the lesions the finder can locate.

**How to read them:**
* "Best epoch" values are optimistic. Quote the **mean of the last 10 epochs** and a bootstrap 95% CI.
* The history **confounds data size, epochs and (early) split**. The jump at 2,641 scans coincides with 40 → 80 → 120 epochs, not with more cancers (704 since 1,355 scans).
* **Stage 1 has levelled off.** Adding negatives no longer helps, so the number of cancers is the limit.
* **Grad-CAM:** early models put heat on the **chest wall and lung border** (a shortcut); the final model puts it on the lesion. Low-scoring cancers are mostly cubes with no visible lesion at the centre (likely precursors). Juxtapleural nodules score lower.
* **Scores are not probabilities.** `pos_weight` multiplies the odds by about w = N−/N+ (≈ 6 at 3,141 scans, ≈ 10 in the final run). Corrected odds ≈ shown odds / w (for w ≈ 6: shown 0.80 ≈ 0.39, 0.99 ≈ 0.94), and they are relative to the training mix (9–14% cancer), not real-world prevalence. Call it a "cancer score".
* **Why 0.94 is not the whole story:** the model is told where to look, most cancer cubes come from the scan just before diagnosis, the negatives include easy random lung, and the benign set is solid-only.

### 6.5 Open items for Stage 1

1. Eye-check hit rate and find rate of the blob finder. Find rate ✅ **86%** (5,196 of 6,074 benign records; misses: 798 "no blob", 80 "slice not in series"). Hit rate ⏳ needs the eye check (`step2_check.py` grids).
2. ✅ Cancer breakdown: precursor vs newest, solid vs subsolid (table in 6.4).
3. ✅ Bootstrap CI on the final validation AUC (table in 6.4).
4. ⏳ Clean experiments with everything else fixed: a learning curve (25/50/75/100% of patients, 120 epochs, 2 seeds) and epochs (40/80/120). Implemented (`step3_experiments.py`, ~6 h, resumable), not run.
5. ✅ **Realistic evaluation:** locate cancers with the blob finder from `nlst_ctab` too (no Sybil boxes), and check that the AUC holds (6.4: it holds, 0.987 vs 0.985 on 46 lesions).
6. ⏳ Optional: cosine learning-rate schedule. Implemented (`--cosine`, part of `step3_experiments.py`), not run.
7. ⏳ One `--test` run when Stage 1 is final (`step3_evaluate.py --test`, uses `stage1_final.pt`).

---

## 7. Stage 2: following a lesion across screenings

### 7.1 Definitions

| Term | Meaning |
|---|---|
| **track** | one lesion at several screenings, oldest first |
| **anchor** | the newest screening where the lesion's position is *known* (Sybil box, or benign nodule found in step 2). Tracks never use screenings after the anchor, so no post-diagnosis scans |
| **track label** | 1 = location of a later biopsy-confirmed cancer (including precursors); 0 = benign nodule in a never-diagnosed control |
| **window of k** | the last *k* screenings up to the anchor; **always exactly k** |
| **early window** | the same, ending one screening **before** the anchor (`--early`) |

### 7.2 Step 4a: build tracks (`step4_sequences.py`)

Per patient (≥ 2 screenings up to the newest known lesion): load every screening up to the anchor with step 2's own functions (same coordinates as step 2), at most 3 scans in memory, DICOM deleted immediately. Then for each known lesion, newest first, walk back one screening at a time:

1. **First guess:** point + (lung centre of mass in the earlier scan − in the newer scan).
2. **Refine with template matching:** take the 48 mm cube of anatomy around the point in the newer scan (HU clipped to −1000…400) and search a (48 + 2·20) mm region around the guess in the earlier scan with **normalised cross-correlation** (`skimage.feature.match_template`). The best offset is the match; store the `match` score and an `edge` flag (best offset at the border of the search).
3. **Snap:** if a known lesion of that year is within 15 mm, use its exact centre (`source = known`); otherwise `source = aligned`. The pre-snap distance is stored as `nearest_mm`.
4. **Chain:** T0 is matched from T1, not from T2.

Known lesions that no newer track claimed start their own track. Each cube is cropped exactly like step 2 (72³ int16). A screening that cannot be read is skipped and the track is matched across it. 6 patients run in parallel (`--workers`), resumable via `tracks/done.txt`.

**Output** `tracks/tracks.csv`: `track_id, pid, label, anchor_yr, study_yr, source, match, edge, nearest_mm, z, y, x, file, split`.

**Check** (`step4_check.py`):
* **Alignment error vs radiologist boxes** on earlier screenings that have their own box (the 383 above): median and share within 10 / 15 mm. This is a measured accuracy for the defining step of Stage 2.
* Match-score distribution and edge rate.
* `known` vs `aligned` per label.
* Track grids for checking by eye (T0 | T1 | T2 per row).

**Expected scale:** **1,702 patients** (310 cancer, 1,392 control) with ≥ 2 screenings up to their newest known lesion, **4,619 downloads**, about 4–5 h with 6 workers (measured: 11 patients in 1 min 45 s), ~6,000 cubes ≈ 5 GB.

### 7.3 Step 4b: embeddings (`step4_embed.py`)

The Stage 1 encoder is **frozen**: loaded from `stage1_final.pt`, run in `eval()` under `torch.no_grad()`, never trained again. For every cube it computes:

* `centre` (n, 128): plain centre crop (used for validation and test);
* `aug` (n, 8, 128): 8 versions with Stage 1's augmentation (used for training);
* `score` (n,): Stage 1 cancer score, the baseline.

It reuses Stage 1's data code (`TrackPatches` subclasses `PatchDataset`).
**Sanity check:** Stage 1 AUC on the anchor cubes of validation tracks should be close to the Stage 1 validation AUC.

**Why freeze:**
1. **Fair comparison:** Stage 1 and the tracker use the same "eyes", so any gain comes from history.
2. **Data:** ~315 cancer tracks cannot retrain 291 k weights without overfitting or forgetting.
3. **Stable coordinates:** the same function embeds every year, so differences mean lesion change.
4. **Speed:** vectors train in seconds.

**Cost:** the encoder learned "is it cancer *now*", not "will it become cancer". A later option is to unfreeze only block 4 with lr 1e-4, as a separate experiment.

### 7.4 Step 4c: the tracker (`step4_tracker.py`)

$$x_t=\big[e_\theta(x^t),\ e_\theta(x^t)-e_\theta(x^{t-1})\big],\qquad h_t=\mathrm{GRU}(h_{t-1},x_t),\qquad \hat r=\sigma(v^\top h_T+c)$$

* Dropout 0.2 on the input → `nn.GRU(256 → 64)` → `Linear(64 → 1)`. **61,889 weights.**
* `--plain` gives the paper's original design (embedding only, `GRU(128 → 64)`, **37,313 weights**), used as an ablation. The change input was added because, on synthetic data, it raised 3-screening AUC from 0.64 to 0.74.
* **GRU internals:** sigmoid gates z_t (update) and r_t (reset), tanh candidate memory. h_t is a running summary *within* one track. Learning only changes the weights.
* **Training:** a random one of the 8 augmented embeddings per screening each epoch; BCE with `pos_weight`; Adam lr 1e-3, weight decay 1e-4, batch 64, 300 epochs. Keep the best validation AUC; model selection is always on validation, even with `--test`.
* **Baselines on exactly the same windows:**
  1. **Stage 1:** score of the newest cube only.
  2. **Score trend:** logistic regression on [newest score, newest − oldest score].
* **Outputs:** printed AUCs; tracker − Stage 1 with a **95% bootstrap CI** (2,000 resamples of whole patients); AUC per anchor year; `models/tracker_k*.pt`; a row in `models/stage2_results.csv`.

### 7.5 Evaluation plan, in order

| Run | Question |
|---|---|
| `--scans 2 --overfit` | does the code work? (train loss < 0.1) |
| `--scans 2`, `--scans 3` | does history help at the newest known screening? |
| `--scans 2 --early` | does history help **a year earlier**, when the lesion is faint? (most clinically interesting) |
| `--scans 1` | sanity: without history the tracker should ≈ Stage 1 |
| `--scans 3 --plain` | does the explicit change input matter? |
| `--test` (once per setting, at the end) | final numbers |

**Before training,** measure the **headroom**: Stage 1 AUC by `years_before = anchor_yr − study_yr`, cancer vs benign cubes at the same offset. If Stage 1 drops sharply one screening back, that is the motivation for Stage 2.

**Secondary metric:** specificity at 90% sensitivity. Near AUC ≈ 0.94, going to 0.96 removes a third of the ranking errors, and "fewer false alarms at the same sensitivity" is what history should buy (stable benign nodules pulled down, growing cancers pushed up).

**How to read the results:**

| Outcome | Interpretation |
|---|---|
| tracker > Stage 1, CI excludes 0, holds within each anchor year | **history helps** |
| ≈ at the newest screening, > in `--early` | history helps exactly before the lesion is obvious |
| trend ≈ tracker > Stage 1 | history helps, but a 2-number summary is enough |
| ≈ everywhere | history adds nothing beyond the newest scan; still a valid answer |
| `--scans 1` ≫ Stage 1 | leakage: something other than the lesion carries the label |

### 7.6 Verified so far

**Synthetic** multi-year DICOM patients (anatomy shifted up to 12 mm and scaled ±2% between years; cancers grow, benign nodules stable; run through the real step 2 and step 4a code with only the download replaced):
* every tracked position within **1.4 mm** of the truth (median 0.9 mm), including spots where the lesion did not exist yet and matching across a missing screening;
* box-based error median 1.0 mm; screenings after the anchor are never used;
* anchor cubes byte-identical to the step 2 patches;
* embeddings: shapes, order and augmentation correct;
* tracker overfit loss 0.002; on synthetic embeddings where the change carries the signal: k=3 tracker 0.995 vs Stage 1 0.873 (Δ CI +0.06 to +0.21), k=1 tracker ≈ Stage 1 (Δ +0.003, CI −0.008 to +0.014), `--plain` 0.941.

**Real NLST, 11 patients** (8 cases with boxes in ≥ 2 years, 3+1 controls; 28 downloads):
* alignment error vs radiologist boxes: **median 3.0 mm, 11 of 11 within 10 mm** (1 year back: 3.1 mm, n 8; 2 years back: 2.4 mm, n 3); match score median 0.84; no edge hits;
* benign vs blob finder: median 48 mm, 1 of 6 within 15 mm. The alignment is fine (see the boxes): the blob finder picks **different blobs in different years**, which is label noise in the benign set;
* 16 of 17 anchor cubes identical to step 2. The odd one came from a download that silently lost a slice (below).

**Missing-slice downloads (found and fixed).** When one DICOM file fails, idc-index only logs the error and `download` saw "some files" and carried on, so the series was read with a gap and every slice above it sat one slice too low. `read_series` now refuses a series whose slice gaps exceed 1.5× the median (`missing slices`), and step 4a downloads it once more. Step 2 ran before this fix, so a few Stage 1 patches may come from such scans. `step4_check.py` compares every anchor cube with its step 2 patch and lists the differences (`checks/stage2_crop_differs.csv`).

### 7.7 Possible extension: time to diagnosis

Instead of one logit, predict *when* (survival analysis). This needs **censoring** for controls, using $P(T>c)$ with $c$ from `canc_free_days`, and either a log-normal (Gaussian on log time) or discrete yearly hazard bins, as Sybil uses. Evaluate with the C-index. Caveats: cancer times span only about 0–2 years in our cohort, and for screen-detected cancers the time partly reflects the clinical work-up, not tumour growth. Keep the binary tracker as the main result.

---

## 8. Concepts cheat sheet

| Concept | One line |
|---|---|
| embedding | the 128 numbers before the head; direction w = "cancer-like"; logit = w·e + b |
| freeze vs fine-tune | freeze = never update the encoder in Stage 2; fine-tune = keep training it with the GRU |
| inference vs training | running a model never changes it; only `loss.backward()` + `optimizer.step()` do (and BatchNorm stats in `train()` mode) |
| sigmoid vs ReLU | ReLU in hidden conv layers; sigmoid for the output score and GRU gates; tanh for GRU memory |
| `BCEWithLogitsLoss` | applies the sigmoid internally, so the model outputs raw logits |
| AUC | ranking quality, threshold-free; ±0.03 noise on ~150 validation cancers, ±0.05+ on ~50 tracks |
| bootstrap CI | resample whole patients, recompute the metric 2,000 times, take the 2.5–97.5% range |
| overfitting | train ↓ while validation ↑; the defences are augmentation, dropout, small models and early model selection |
| shortcut learning | a high score for the wrong reason (centring, chest wall, density, year) |

---

## 9. Threats to validity (all stages)

| Risk | Mitigation |
|---|---|
| model is told the location (classification, not detection) | stated as scope; realistic test with blob-finder-located cancers; sliding window is future work |
| positives (hindsight boxes) vs negatives (prospective records + heuristic) located differently | random ±4 mm shifts, Grad-CAM centre score, cancer vs benign AUC separately, realistic evaluation |
| benign finder is solid-only (> −400 HU), so "subsolid = cancer" is possible | solid/subsolid breakdown of cancers |
| blob-finder label noise | eye-check hit rate, density check, failure log; across years the finder often picks different blobs (benign alignment check, 7.6) |
| silently incomplete downloads (a missing slice shifts the scan) | `read_series` refuses gaps; step 4a re-downloads; same-crop check counts affected step 2 patches |
| patient leakage | split by patient, asserted |
| test reuse / best-epoch optimism | single `--test`; report last-10-epoch means and CIs |
| sequence length carries the label (Stage 2) | windows of exactly k |
| year or scanner carries the label (Stage 2) | per-anchor-year AUC |
| matching errors (Stage 2) | measured against 383 radiologist boxes; `match` / `edge` per cube |
| small numbers (Stage 2) | bootstrap CIs; consider seeds or cross-validation |
| population | NLST: heavy smokers aged 55–74, mostly white; claims limited to that population |

---

## 10. Status

| Part | Status |
|---|---|
| Step 1 cohort | ✅ run on NLST |
| Step 2 patches | ✅ 4,605 scans processed |
| Stage 1 model | ✅ trained (val 0.962 [0.943, 0.979] / 0.945 [0.920, 0.967] / 0.981, centre 0.77); ✅ frozen as `models/stage1_final.pt`; **test sealed** |
| Stage 1 open items | ✅ 2, 3, 5 and the find rate; ⏳ hit-rate eye check, clean experiments (implemented), test run; see 6.5 |
| Step 4a–c (Stage 2) | ✅ implemented; verified on synthetic data and step 4a on 11 real patients; ⏳ full run on NLST |
| Dashboard | ✅ Stage 1 and Stage 2 tabs |
| Paper draft | ✅ v1; ⏳ update the Stage 2 section (change input, 61,889 weights) and add real results |

**Next actions, in order:**
1. Eye-check hit rate of the blob finder (`step2_check.py` grids).
2. `step4_sequences.py` (dashboard: Stage 2 → Start step 4a, ~4–5 h), then `step4_check.py`. Look at the alignment error and the same-crop count.
3. `step4_embed.py` and the headroom check.
4. Tracker runs (7.5; dashboard: Run the evaluation plan), then a second seed.
5. Clean Stage 1 experiments (`step3_experiments.py`, ~6 h).
6. Freeze everything, then run `--test` once per setting (`step3_evaluate.py --test`, `step4_tracker.py --scans k --test`).
