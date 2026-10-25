import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.dataset import Sen1Floods11
from src.metrics import Metrics
from src.models import build_unet

ROOT = Path(__file__).resolve().parents[1] / "data" / "sen1floods11"


@torch.no_grad()
def evaluate(model, ds, device, batch_size=4):
    model.eval()
    m = Metrics()
    for b in DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0):
        pred = model(b["image"].to(device)).argmax(1).cpu()
        m.update(pred, b["label"], b["event"])
    return m.compute()


def evaluate_splits(model, splits, mean, std, device):
    out = {}
    for s in splits:
        ds = Sen1Floods11(ROOT, ROOT / "splits" / f"flood_{s}_data.csv", mean, std)
        overall, per_event = evaluate(model, ds, device)
        out[s] = {"overall": overall, "per_event": per_event}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--splits", nargs="+", default=["valid", "test", "bolivia"])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_unet(ck["cfg"]["encoder"], None).to(device)
    model.load_state_dict(ck["model"])
    res = evaluate_splits(model, args.splits, ck["mean"], ck["std"], device)
    for s, r in res.items():
        o = r["overall"]
        print(f"{s:8s} IoU {o['flood_iou']:.4f} | F1 {o['f1']:.4f} | "
              f"P {o['precision']:.4f} | R {o['recall']:.4f} | mIoU {o['mean_iou']:.4f}")
        for e, v in r["per_event"].items():
            print(f"    {e:10s} IoU {v['flood_iou']:.4f}")
    out = Path(args.ckpt).parent / "eval.json"
    out.write_text(json.dumps(res, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()