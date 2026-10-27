import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from src.dataset import Sen1Floods11, compute_stats, valid_input_mask

# Single source of truth for colours (report figures, demo panels and demo legend).
COLORS = {"land": "#d9d9d9", "flood": "#1f77b4", "ignored": "#888888", "invalid": "#2b2b2b"}
CMAP = ListedColormap([COLORS["ignored"], COLORS["land"], COLORS["flood"]])  # index = class + 1


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rgb_from_chip(img, valid=None):
    """(6,H,W) raw reflectance -> (H,W,3) float in [0,1]. The contrast stretch uses valid
    pixels only; invalid pixels are left black (not filled with invented imagery)."""
    rgb = np.stack([img[2], img[1], img[0]], -1).astype(np.float32)
    if valid is None:
        valid = valid_input_mask(img)
    out = np.zeros_like(rgb)
    if valid.any():
        lo, hi = np.percentile(rgb[valid], (2, 98))
        out = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    out[~valid] = 0
    return out


def rgb_uint8(img, valid):
    out = (rgb_from_chip(img, valid) * 255).astype(np.uint8)
    out[~valid] = hex_to_rgb(COLORS["invalid"])
    return out


def class_map_rgb(cls, invalid=None):
    """cls: (H,W) with -1 ignored, 0 land, 1 flood -> uint8 RGB. invalid: optional bool mask."""
    out = np.zeros(cls.shape + (3,), np.uint8)
    out[:] = hex_to_rgb(COLORS["ignored"])
    out[cls == 0] = hex_to_rgb(COLORS["land"])
    out[cls == 1] = hex_to_rgb(COLORS["flood"])
    if invalid is not None:
        out[invalid] = hex_to_rgb(COLORS["invalid"])
    return out


def show_chip(ds, i, pred=None, save_to=None):
    img, lbl = ds.read_raw(i)
    valid = valid_input_mask(img)
    n = 3 if pred is not None else 2
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 5))
    ax[0].imshow(rgb_from_chip(img, valid))
    ax[0].set_title("RGB (B4,B3,B2)")
    ax[1].imshow(lbl + 1, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax[1].set_title(f"Label {np.unique(lbl).tolist()}")
    if pred is not None:
        pred = np.where(lbl == -1, -1, pred)   # do not show predictions on ignored pixels
        ax[2].imshow(pred + 1, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
        ax[2].set_title("Prediction")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{ds.s2_paths[i].name} | event: {ds.events[i]}")
    plt.tight_layout()
    if save_to:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=120)
        plt.close(fig)
    else:
        plt.show()


def get_stats(root, cache="results/stats.json"):
    """Per-band mean/std of the training split, cached to disk."""
    cache = Path(cache)
    if cache.exists():
        d = json.loads(cache.read_text())
        return d["mean"], d["std"]
    ds = Sen1Floods11(root, Path(root) / "splits" / "flood_train_data.csv")
    mean, std = compute_stats(ds, n=len(ds))
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({"mean": mean.tolist(), "std": std.tolist()}))
    return mean.tolist(), std.tolist()