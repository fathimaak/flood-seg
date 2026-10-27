import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
from src.dataset import Sen1Floods11
from src.models import build_model
from src.utils import CMAP, rgb_from_chip

DATA = ROOT_DIR / "data" / "sen1floods11"
FIG = ROOT_DIR / "results" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(ROOT_DIR / "results" / "runs.csv")
models = list(df.model.unique())

# 1. Label-fraction curve
fig, ax = plt.subplots(figsize=(6, 4))
styles = {"valid": "tab:green", "test": "tab:blue", "bolivia": "tab:red"}
for m, ls in zip(models, ["-", "--", ":"]):
    for s, c in styles.items():
        g = df[(df.model == m) & (df.split == s) & (df.frac.isin([0.1, 0.25, 0.5, 1.0]))] \
            .groupby("frac").flood_iou.agg(["mean", "std"])
        ax.errorbar(g.index * 100, g["mean"], yerr=g["std"], color=c, ls=ls,
                    marker="o", capsize=3, label=f"{m} / {s}")
ax.set_xlabel("% of training chips"); ax.set_ylabel("Flood IoU"); ax.grid(alpha=.3)
ax.legend(fontsize=7); plt.tight_layout()
plt.savefig(FIG / "label_fraction.png", dpi=150); plt.close()

# 2. Per-event IoU at 100% labels (mean +- std over seeds)
def per_event(m):
    acc = {}
    for p in (ROOT_DIR / "checkpoints").glob(f"{m}_seed*_frac1.0/eval.json"):
        r = json.loads(p.read_text())
        for split in ("test", "bolivia"):
            for e, v in r[split]["per_event"].items():
                acc.setdefault(e, []).append(v["flood_iou"])
    return {e: (np.mean(v), np.std(v)) for e, v in acc.items()}

pe = {m: per_event(m) for m in models}
events = sorted(pe[models[0]], key=lambda e: pe[models[0]][e][0])
x, w = np.arange(len(events)), 0.8 / len(models)
fig, ax = plt.subplots(figsize=(9, 4))
for k, m in enumerate(models):
    ax.bar(x + k * w, [pe[m][e][0] for e in events], w,
           yerr=[pe[m][e][1] for e in events], capsize=2, label=m)
ax.set_xticks(x + w * (len(models) - 1) / 2)
ax.set_xticklabels([e + (" (unseen)" if e == "Bolivia" else "") for e in events],
                   rotation=40, ha="right")
ax.set_ylabel("Flood IoU"); ax.legend(); plt.tight_layout()
plt.savefig(FIG / "per_event_iou.png", dpi=150); plt.close()

# 3. Qualitative grid (seed 0, 100% labels)
device = "cuda" if torch.cuda.is_available() else "cpu"
nets = {}
for m in models:
    ck = torch.load(ROOT_DIR / "checkpoints" / f"{m}_seed0_frac1.0" / "best.pt",
                    map_location=device, weights_only=False)
    net = build_model(ck["cfg"].get("arch", "unet"), ck["cfg"]["encoder"], None).to(device)
    net.load_state_dict(ck["model"]); net.eval()
    nets[m] = net
    mean, std = ck["mean"], ck["std"]

picks = [("bolivia", 0), ("bolivia", 7), ("test", 5), ("test", 40), ("test", 70)]
fig, axes = plt.subplots(len(picks), 2 + len(models),
                         figsize=(3.2 * (2 + len(models)), 3.2 * len(picks)))
for r, (split, i) in enumerate(picks):
    ds = Sen1Floods11(DATA, DATA / "splits" / f"flood_{split}_data.csv", mean, std)
    raw, lbl = ds.read_raw(i)
    x_in = ds[i]["image"][None].to(device)
    axes[r, 0].imshow(rgb_from_chip(raw))
    axes[r, 0].set_ylabel(f"{split}\n{ds.events[i]}")
    axes[r, 1].imshow(lbl + 1, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    for c, (m, net) in enumerate(nets.items()):
        with torch.no_grad():
            pred = net(x_in).argmax(1)[0].cpu().numpy()
            pred = np.where(lbl == -1, -1, pred)   # don't draw predictions on ignored/no-data pixels
        axes[r, 2 + c].imshow(pred + 1, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
        if r == 0:
            axes[r, 2 + c].set_title(m)
    if r == 0:
        axes[r, 0].set_title("RGB"); axes[r, 1].set_title("Ground truth")
for a in axes.ravel():
    a.set_xticks([]); a.set_yticks([])
plt.tight_layout(); plt.savefig(FIG / "qualitative.png", dpi=120); plt.close()
print("saved figures to", FIG)