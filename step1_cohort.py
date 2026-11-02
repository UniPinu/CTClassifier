"""
Step 1 - Build the cohort: who is in the study, and which CT series to use per screen.

Reads : NLST clinical tables + CT series list (both via idc-index), annotations.json
Writes: cohort/patients.csv          pid, label (1 = cancer case), n_screens, split
        cohort/scans.csv             one row per screen: pid, study_yr, SeriesInstanceUID, annotated, ...
        cohort/control_nodules.csv   benign nodules of controls: slice number, lobe, size

Run   : python step1_cohort.py
"""
import json

import pandas as pd
from idc_index import IDCClient
from sklearn.model_selection import train_test_split

from config import ANNOTATIONS, COHORT_DIR, CONTROLS_PER_CASE, SEED

KEY = ["pid", "study_yr"]  # one screen = one patient in one study year


def load_clinical(client):
    """Participant table (one row per person) and CT abnormality table (one row per finding)."""
    client.fetch_index("clinical_index")  # also downloads the NLST clinical tables
    prsn = client.get_clinical_table("nlst_prsn")
    ctab = client.get_clinical_table("nlst_ctab")
    prsn_cols = ["pid", "can_scr", "candx_days", "canc_free_days",
                 "scr_days0", "scr_days1", "scr_days2"]
    ctab_cols = ["pid", "study_yr", "sct_ab_desc", "sct_slice_num",
                 "sct_epi_loc", "sct_long_dia"]
    # NLST marks missing values with codes like ".N" / ".M": anything non-numeric -> NaN
    prsn = prsn[prsn_cols].apply(pd.to_numeric, errors="coerce")
    ctab = ctab[ctab_cols].apply(pd.to_numeric, errors="coerce")
    return prsn, ctab


def load_series(client):
    """Every usable axial CT series in NLST, with patient id and study year."""
    client.fetch_index("ct_index")  # adds slice thickness and image type per series
    s = client.index.query("collection_id == 'nlst' and Modality == 'CT'")
    s = s.merge(client.ct_index[["SeriesInstanceUID", "SliceThickness", "ImageType"]],
                on="SeriesInstanceUID", how="left")
    s["pid"] = s["PatientID"].astype(int)
    s["study_yr"] = pd.to_datetime(s["StudyDate"]).dt.year - 1999  # 1999=T0, 2000=T1, 2001=T2

    usable = (
        ~s["ImageType"].astype(str).str.contains("LOCALIZER")      # flat scout images
        & (s["instanceCount"] >= 50)                                 # too few slices
        & ~s["transfer_syntax_name"].astype(str).str.contains("JPEG")  # compressed (rare)
    )
    cols = KEY + ["SeriesInstanceUID", "SliceThickness", "instanceCount", "series_size_MB"]
    return s.loc[usable, cols]


def pick_one_series(series):
    """Each screen has several reconstructions: keep the thinnest slices (like Sybil)."""
    ranked = series.sort_values(["SliceThickness", "instanceCount"], ascending=[True, False])
    return ranked.groupby(KEY).head(1)


def load_annotated_series(series):
    """Series that have at least one radiologist box in Sybil's annotations."""
    ann = json.load(open(ANNOTATIONS))
    uids = [uid for uid, slices in ann.items() if any(len(b) > 0 for b in slices.values())]
    annotated = series[series["SeriesInstanceUID"].isin(uids)]
    print(f"Annotated series: {len(uids)} in JSON, {len(annotated)} found in NLST")
    return pick_one_series(annotated)  # if a screen has 2 annotated series, keep one


def find_controls(prsn, ctab, picked):
    """Never diagnosed, known cancer-free 1 year after last screen, has a nodule, >= 2 screens."""
    last_screen = prsn[["scr_days0", "scr_days1", "scr_days2"]].max(axis=1)
    cancer_free = ((prsn["can_scr"] == 0)
                   & prsn["candx_days"].isna()
                   & (prsn["canc_free_days"] >= last_screen + 365))
    nodules = ctab[(ctab["sct_ab_desc"] == 51)               # non-calcified nodule >= 4 mm
                   & ctab["sct_slice_num"].between(1, 998)   # 999 = missing
                   & ctab["sct_long_dia"].notna()]
    n_screens = picked.groupby("pid").size()
    ok = (cancer_free
          & prsn["pid"].isin(nodules["pid"])
          & prsn["pid"].map(n_screens).fillna(0).ge(2))
    return prsn.loc[ok, "pid"], nodules


def split_patients(patients):
    """70/15/15 by patient, stratified: every split gets the same share of cancers."""
    train, rest = train_test_split(patients, test_size=0.30,
                                   stratify=patients["label"], random_state=SEED)
    val, test = train_test_split(rest, test_size=0.50,
                                 stratify=rest["label"], random_state=SEED)
    return pd.concat([train.assign(split="train"),
                      val.assign(split="val"),
                      test.assign(split="test")])


def build_cohort(series, prsn, ctab):
    picked = pick_one_series(series)
    annotated = load_annotated_series(series)

    # Cases: everyone with a box. Controls: random sample from the eligible pool.
    case_pids = pd.Series(annotated["pid"].unique())
    pool, nodules = find_controls(prsn, ctab, picked)
    pool = pool[~pool.isin(case_pids)]
    control_pids = pool.sample(min(len(pool), CONTROLS_PER_CASE * len(case_pids)),
                               random_state=SEED)
    patients = pd.concat([pd.DataFrame({"pid": case_pids, "label": 1}),
                          pd.DataFrame({"pid": control_pids.values, "label": 0})])

    # Scans: thinnest series per screen, but the annotated series where boxes exist
    # (boxes are pixel positions in that exact series).
    is_annotated = picked.set_index(KEY).index.isin(annotated.set_index(KEY).index)
    scans = pd.concat([annotated.assign(annotated=True),
                       picked[~is_annotated].assign(annotated=False)])
    scans = scans[scans["pid"].isin(patients["pid"])].sort_values(KEY)

    patients["n_screens"] = patients["pid"].map(scans.groupby("pid").size())
    patients = split_patients(patients)

    # Benign nodules of the controls, on the series we chose for that screen
    control_nodules = nodules[nodules["pid"].isin(control_pids)]
    control_nodules = control_nodules.merge(scans[KEY + ["SeriesInstanceUID"]], on=KEY)

    assert patients["pid"].is_unique, "a patient appears twice"
    assert not scans.duplicated(KEY).any(), "two series for one screen"
    return patients, scans, control_nodules


def main():
    client = IDCClient()
    prsn, ctab = load_clinical(client)
    series = load_series(client)
    patients, scans, control_nodules = build_cohort(series, prsn, ctab)

    COHORT_DIR.mkdir(exist_ok=True)
    patients.to_csv(COHORT_DIR / "patients.csv", index=False)
    scans.to_csv(COHORT_DIR / "scans.csv", index=False)
    control_nodules.to_csv(COHORT_DIR / "control_nodules.csv", index=False)

    print("\nPatients per split (0 = control, 1 = cancer):")
    print(patients.groupby(["split", "label"]).size().unstack())
    print(f"\n{len(scans)} screens in total, {scans['annotated'].sum()} with boxes")
    print(f"{len(control_nodules)} benign nodules recorded for controls")


if __name__ == "__main__":
    main()
