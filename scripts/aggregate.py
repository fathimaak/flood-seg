"""Collect checkpoints/*/eval.json into mean±std tables."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

rows = []
for p in Path("checkpoints").glob("*/eval.json"):
    m = re.match(r"(.+)_seed(\d+)_frac([\d.]+)", p.parent.name)
    if not m:
        continue
    model, seed, frac = m.group(1), int(m.group(2)), float(m.group(3))
    for split, r in json.loads(p.read_text()).items():
        o = r["overall"]
        rows.append(dict(model=model, seed=seed, frac=frac, split=split,
                         flood_iou=o["flood_iou"], f1=o["f1"],
                         precision=o["precision"], recall=o["recall"]))

df = pd.DataFrame(rows)
Path("results").mkdir(exist_ok=True)
df.to_csv("results/runs.csv", index=False)

g = df.groupby(["model", "frac", "split"]).agg(
    n=("seed", "count"),
    iou_mean=("flood_iou", "mean"), iou_std=("flood_iou", "std"),
    f1_mean=("f1", "mean"), f1_std=("f1", "std")).reset_index()
g.to_csv("results/summary.csv", index=False)
print(g.round(4).to_string(index=False))