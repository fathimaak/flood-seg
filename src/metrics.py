from collections import defaultdict

import numpy as np

from src.dataset import IGNORE_INDEX


def _summarize(c):
    tp, fp, fn, tn = [float(x) for x in c]
    eps = 1e-9
    flood_iou = tp / (tp + fp + fn + eps)
    land_iou = tn / (tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec = tp / (tp + fn + eps)
    return {"flood_iou": flood_iou, "mean_iou": (flood_iou + land_iou) / 2,
            "f1": 2 * prec * rec / (prec + rec + eps),
            "precision": prec, "recall": rec, "n_pixels": int(tp + fp + fn + tn)}


class Metrics:
    """Pixel-pooled confusion counts (flood = positive), overall and per event."""

    def __init__(self):
        self.c = defaultdict(lambda: np.zeros(4, dtype=np.int64))

    def update(self, pred, label, events):
        for p, l, e in zip(pred, label, events):
            v = l != IGNORE_INDEX
            p, l = p[v], l[v]
            self.c[e] += np.array([((p == 1) & (l == 1)).sum().item(),
                                   ((p == 1) & (l == 0)).sum().item(),
                                   ((p == 0) & (l == 1)).sum().item(),
                                   ((p == 0) & (l == 0)).sum().item()])

    def compute(self):
        total = sum(self.c.values())
        return _summarize(total), {e: _summarize(c) for e, c in sorted(self.c.items())}