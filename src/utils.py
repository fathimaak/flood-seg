from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

CMAP = ListedColormap(["#888888", "#d9d9d9", "#1f77b4"])  # no-data, land, water


def rgb_from_chip(img):
    """img: (6,H,W) raw reflectance in order B2,B3,B4,... -> stretched RGB."""
    rgb = np.stack([img[2], img[1], img[0]], -1)
    lo, hi = np.percentile(rgb, (2, 98))
    return np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)


def show_chip(ds, i, pred=None, save_to=None):
    img, lbl = ds.read_raw(i)
    n = 3 if pred is not None else 2
    fig, ax = plt.subplots(1, n, figsize=(5 * n, 5))
    ax[0].imshow(rgb_from_chip(img))
    ax[0].set_title("RGB (B4,B3,B2)")
    ax[1].imshow(lbl + 1, cmap=CMAP, vmin=0, vmax=2, interpolation="nearest")
    ax[1].set_title(f"Label {np.unique(lbl).tolist()}")
    if pred is not None:
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