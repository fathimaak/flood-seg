"""Diagnose the no-data region and check that the demo's inference path reproduces eval.json.
Run from the repo root: python scripts/verify_app_pipeline.py"""
import sys
from pathlib import Path

import numpy as np
import rasterio

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.dataset import Sen1Floods11, read_label, read_s2_raw, valid_input_mask
from src.inference import discover, predict_classes
from src.metrics import Metrics

DATA = REPO / "data" / "sen1floods11"
CHIP = "Bolivia_103757_S2Hand.tif"


def share(a):
    return float(a.mean()) if a.size else float("nan")


print("== 1. Raw chip diagnostics:", CHIP)
path = DATA / "S2Hand" / CHIP
with rasterio.open(path) as src:
    print(f"bands={src.count} size={src.width}x{src.height} dtype={set(src.dtypes)} "
          f"nodata={src.nodata} crs={src.crs}")
    print("band descriptions:", src.descriptions)
raw = read_s2_raw(path)
lbl = read_label(DATA / "LabelHand" / CHIP.replace("_S2Hand", "_LabelHand"))
valid = valid_input_mask(raw)
unl = lbl == -1
allzero = (raw == 0).all(0)
print(f"NaN: {np.isnan(raw).any(0).mean():.2%} | all six bands == 0: {allzero.mean():.2%} | "
      f"some (not all) bands == 0: {((raw == 0).any(0) & ~allzero).mean():.2%}")
print(f"invalid input (app rule): {(~valid).mean():.2%} | label == -1: {unl.mean():.2%}")
print(f"share of invalid-input pixels that are labelled -1: {share(unl[~valid]):.2%}")
print(f"labelled pixels with invalid input: {int(((~unl) & (~valid)).sum())}")

bundles, problems, warn = discover()
print("\n== 2. Checkpoints")
for k, b in bundles.items():
    print(f"OK   {b.label}: {b.ckpt_path.relative_to(REPO)} (epoch {b.epoch})")
for k, why in problems.items():
    print(f"FAIL {k}: {why}")
if warn:
    print("WARNING:", warn)

print("\n== 3. What the model outputs inside the invalid area of this chip")
for k, b in bundles.items():
    p = predict_classes(b, raw)
    if (~valid).any():
        print(f"{b.label}: {share(p[~valid] == 1):.2%} of invalid-input pixels are predicted 'flood' "
              "(meaningless; the demo now masks them)")

print("\n== 4. Demo inference path vs recorded eval.json (flood IoU, label != -1 protocol)")
for split in ("bolivia", "test"):
    ds = Sen1Floods11(DATA, DATA / "splits" / f"flood_{split}_data.csv")
    for k, b in bundles.items():
        m, excl = Metrics(), 0
        for s2, lb, ev in zip(ds.s2_paths, ds.lbl_paths, ds.events):
            img, l = read_s2_raw(s2), read_label(lb)
            m.update(predict_classes(b, img)[None], l[None], [ev])
            excl += int(((l != -1) & ~valid_input_mask(img)).sum())
        o, _ = m.compute()
        rec = (b.recorded or {}).get(split)
        verdict = "no recorded value" if rec is None else (
            "MATCH" if abs(rec - o["flood_iou"]) < 0.002 else "MISMATCH - investigate")
        print(f"{split:8s} {b.label:24s} app-path IoU {o['flood_iou']:.4f} | recorded "
              f"{'n/a' if rec is None else f'{rec:.4f}'} | {verdict} | labelled pixels with invalid input: {excl}")