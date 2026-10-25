import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dataset import BAND_NAMES, IGNORE_INDEX, Sen1Floods11, read_split
from src.utils import show_chip

ROOT = Path(__file__).resolve().parents[1] / "data" / "sen1floods11"
FIG = Path(__file__).resolve().parents[1] / "results" / "figures"

# 1. Splits and events
for csv in sorted((ROOT / "splits").glob("*.csv")):
    df = read_split(csv)
    ev = Counter(df.s2.str.split("_").str[0])
    print(f"{csv.name}: {len(df)} chips | events: {dict(ev)}")

# 2. File counts
print("S2Hand:", len(list((ROOT / "S2Hand").glob("*.tif"))),
      "| LabelHand:", len(list((ROOT / "LabelHand").glob("*.tif"))))

# 3. Band count in a raw chip (expect 13)
ds = Sen1Floods11(ROOT, ROOT / "splits" / "flood_train_data.csv")
with rasterio.open(ds.s2_paths[0]) as src:
    print("bands in raw S2 chip:", src.count)

# 4. Value ranges and labels
img, lbl = ds.read_raw(0)
print("image", img.shape, img.dtype)
for n, b in zip(BAND_NAMES, img):
    print(f"  {n}: {b.min():.0f}..{b.max():.0f} mean {b.mean():.0f}")
print("labels:", dict(zip(*[x.tolist() for x in np.unique(lbl, return_counts=True)])))

# 5. Flood fraction over train split
w = n = 0
for i in range(len(ds)):
    _, l = ds.read_raw(i)
    w += (l == 1).sum()
    n += (l != IGNORE_INDEX).sum()
print(f"flood fraction among valid train pixels: {w / n:.3%}")

# 6. Figures
for k in [0, len(ds) // 2, len(ds) - 1]:
    show_chip(ds, k, save_to=FIG / f"chip_{k}.png")
print("saved sample figures to", FIG)