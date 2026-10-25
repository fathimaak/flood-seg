from pathlib import Path

import torch

from src.dataset import Sen1Floods11

ROOT = Path(__file__).resolve().parents[1] / "data" / "sen1floods11"


def test_item_shapes_and_labels():
    ds = Sen1Floods11(ROOT, ROOT / "splits" / "flood_train_data.csv", augment=True)
    b = ds[0]
    assert b["image"].shape[0] == 6
    assert b["image"].shape[1:] == b["label"].shape
    assert set(torch.unique(b["label"]).tolist()) <= {-1, 0, 1}

def test_all_split_files_exist():
    for csv in (ROOT / "splits").glob("*.csv"):
        ds = Sen1Floods11(ROOT, csv)
        missing = [p.name for p in ds.s2_paths + ds.lbl_paths if not p.exists()]
        assert not missing, (
            f"{csv.name}: {len(missing)} missing, e.g. {missing[:3]}"
        )