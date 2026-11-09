"""
Step 3 - The dataset (patches -> tensors, with augmentation) and the 3D CNN.
Imported by step3_train.py and step3_gradcam.py.
"""
import functools
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from config import PATCH, PATCH_DIR, STORE


def to_unit(hu):
    """Hounsfield units -> 0..1, using a lung window (-1000 = air ... 400 = bone-ish)."""
    return (np.clip(hu, -1000, 400) + 1000) / 1400


class PatchDataset(Dataset):
    folder = PATCH_DIR                    # where the .npy files are (step 4 reads tracks/ instead)

    def __init__(self, split, augment):
        index = pd.read_csv(PATCH_DIR / "index.csv")
        self.df = index[index["split"] == split].reset_index(drop=True)
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        cube = torch.from_numpy(to_unit(np.load(self.folder / row.file)).astype(np.float32))
        slack = STORE - PATCH                                    # 8 voxels -> shifts of up to +-4 mm

        if self.augment:
            z, y, x = torch.randint(0, slack + 1, (3,)).tolist()  # small random shift
            cube = cube[z:z + PATCH, y:y + PATCH, x:x + PATCH]
            for axis in range(3):                                 # random flips
                if torch.rand(1) < 0.5:
                    cube = cube.flip(axis)
            k = int(torch.randint(0, 4, (1,)))                    # random 90 degree turn
            cube = torch.rot90(cube, k, dims=(1, 2))
        else:
            c = slack // 2                                        # centre crop
            cube = cube[c:c + PATCH, c:c + PATCH, c:c + PATCH]

        x = cube.contiguous().unsqueeze(0)                        # (1, 64, 64, 64)
        y = torch.tensor([float(row.label)])
        return x, y


def block(c_in, c_out):
    """conv3d -> batch norm -> ReLU -> max pool. Halves the size of the cube."""
    return nn.Sequential(
        nn.Conv3d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(c_out),
        nn.ReLU(),
        nn.MaxPool3d(2),
    )


class PatchCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            block(1, 16),     # 64^3 -> 32^3
            block(16, 32),    # 32^3 -> 16^3
            block(32, 64),    # 16^3 ->  8^3
            block(64, 128),   #  8^3 ->  4^3
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),     # -> 128 numbers per patch (the embedding)
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):                 # x: (batch, 1, 64, 64, 64)
        return self.head(self.encoder(x))  # (batch, 1) logits: > 0 means "cancer"


@functools.cache
def pick_device():
    """NVIDIA GPU (cuda) > Apple GPU (mps, MacBooks with M chips) > cpu.
    Force one with the environment variable DEVICE, e.g.  DEVICE=cpu python step3_train.py"""
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        try:   # older PyTorch versions lack some 3D layers on mps: try one tiny training step
            model = PatchCNN().to("mps")
            model(torch.zeros(2, 1, PATCH, PATCH, PATCH, device="mps")).sum().backward()
            return "mps"
        except (RuntimeError, NotImplementedError) as e:
            print(f"Apple GPU (mps) can't run the 3D CNN with this PyTorch ({type(e).__name__}), using the CPU")
    return "cpu"
