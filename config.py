"""All paths and settings in one place. Every script imports from here."""
from pathlib import Path

ROOT = Path(__file__).parent

# Inputs
ANNOTATIONS = ROOT / "annotations.json"   # Sybil's radiologist boxes (download link in Sybil README)

# Outputs, one folder per step
COHORT_DIR = ROOT / "cohort"              # step 1: patients.csv, scans.csv, control_nodules.csv
PATCH_DIR = ROOT / "patches"              # step 2: one .npy file per patch + index.csv
TMP_DIR = ROOT / "tmp_dicom"              # step 2: the ONE scan currently being processed
CHECK_DIR = ROOT / "checks"               # step 2 + 3: images for checking by eye
MODEL_DIR = ROOT / "models"               # step 3: trained weights + training log

# Budget
MAX_GB = 50                               # never use more disk than this

# Cohort
CONTROLS_PER_CASE = 3
SEED = 42

# Patches
STORE = 72        # stored cube size in voxels (1 voxel = 1 mm) ...
PATCH = 64        # ... the network sees 64^3, so training can shift the crop by up to +-4 mm
N_RANDOM = 1      # random lung patches ("easy negatives") per scan

# Training
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
