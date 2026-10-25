from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

# S2 L1C has 13 bands: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
# Prithvi's 6 bands: Blue B2, Green B3, Red B4, NIR-narrow B8A, SWIR1 B11, SWIR2 B12
BAND_NAMES = ["B2", "B3", "B4", "B8A", "B11", "B12"]
BAND_IDX_1BASED = [2, 3, 4, 9, 12, 13]  # rasterio is 1-based
IGNORE_INDEX = -1  # no-data label value (verify on Day 1)


def read_split(csv_path):
    """Split CSVs contain S1 filenames; convert them to S2 filenames."""
    df = pd.read_csv(csv_path, header=None, names=["s1", "label"])
    df["s2"] = df["s1"].str.replace("_S1Hand", "_S2Hand", regex=False)
    return df



class Sen1Floods11(Dataset):
    def __init__(self, root, csv_path, mean=None, std=None, augment=False):
        root = Path(root)
        df = read_split(csv_path)
        self.s2_paths = [root / "S2Hand" / f for f in df.s2]
        self.lbl_paths = [root / "LabelHand" / f for f in df.label]
        self.events = [f.split("_")[0] for f in df.s2]
        self.mean = None if mean is None else np.asarray(mean, np.float32)[:, None, None]
        self.std = None if std is None else np.asarray(std, np.float32)[:, None, None]
        self.augment = augment

    def __len__(self):
        return len(self.s2_paths)

    def read_raw(self, i):
        with rasterio.open(self.s2_paths[i]) as src:
            img = src.read(BAND_IDX_1BASED).astype(np.float32)  # (6,H,W)
        with rasterio.open(self.lbl_paths[i]) as src:
            lbl = src.read(1).astype(np.int64)  # (H,W)
        return np.nan_to_num(img, nan=0.0), lbl

    def __getitem__(self, i):
        img, lbl = self.read_raw(i)
        if self.mean is not None:
            img = (img - self.mean) / self.std
        if self.augment:
            if np.random.rand() < 0.5:
                img, lbl = img[:, :, ::-1], lbl[:, ::-1]
            if np.random.rand() < 0.5:
                img, lbl = img[:, ::-1, :], lbl[::-1, :]
            k = np.random.randint(4)
            img, lbl = np.rot90(img, k, (1, 2)), np.rot90(lbl, k, (0, 1))
        img = torch.from_numpy(np.ascontiguousarray(img))
        lbl = torch.from_numpy(np.ascontiguousarray(lbl))
        return {
            "image": img,
            "label": lbl,
            "valid": lbl != IGNORE_INDEX,
            "event": self.events[i],
            "name": self.s2_paths[i].name,
        }


def compute_stats(ds, n=100):
    """Per-band mean/std over up to n chips, ignoring no-data pixels."""
    idx = np.linspace(0, len(ds) - 1, min(n, len(ds))).astype(int)
    s, s2, cnt = np.zeros(6), np.zeros(6), 0
    for i in idx:
        img, lbl = ds.read_raw(i)
        x = img[:, lbl != IGNORE_INDEX].astype(np.float64)
        s += x.sum(1)
        s2 += (x ** 2).sum(1)
        cnt += x.shape[1]
    mean = s / cnt
    return mean, np.sqrt(s2 / cnt - mean ** 2)