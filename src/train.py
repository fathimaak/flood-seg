import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset

from src.dataset import IGNORE_INDEX, Sen1Floods11
from src.eval import ROOT, evaluate, evaluate_splits
from src.models import build_unet
from src.utils import get_stats


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def random_crop(x, y, size):
    H, W = y.shape[-2:]
    i, j = np.random.randint(0, H - size + 1), np.random.randint(0, W - size + 1)
    return x[..., i:i + size, j:j + size], y[..., i:i + size, j:j + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/unet_baseline.yaml")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--label_fraction", type=float)
    ap.add_argument("--run_name")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    for k in ["epochs", "seed", "label_fraction", "run_name"]:
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    cfg["run_name"] = cfg["run_name"] or f"unet_{cfg['encoder']}_seed{cfg['seed']}_frac{cfg['label_fraction']}"

    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print("device:", device, "| run:", cfg["run_name"])

    mean, std = get_stats(ROOT)
    train_full = Sen1Floods11(ROOT, ROOT / "splits" / "flood_train_data.csv", mean, std, augment=True)
    k = max(1, round(len(train_full) * cfg["label_fraction"]))
    idx = np.random.RandomState(cfg["seed"]).permutation(len(train_full))[:k].tolist()
    train_ds = Subset(train_full, idx)
    valid_ds = Sen1Floods11(ROOT, ROOT / "splits" / "flood_valid_data.csv", mean, std)
    print(f"train chips: {len(train_ds)} | valid chips: {len(valid_ds)}")

    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                        num_workers=cfg["num_workers"], drop_last=True)
    model = build_unet(cfg["encoder"], cfg["encoder_weights"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(cfg["class_weights"], device=device),
                                  ignore_index=IGNORE_INDEX)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path("checkpoints") / cfg["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    log = csv.writer(open(out_dir / "log.csv", "w", newline=""))
    log.writerow(["epoch", "train_loss", "valid_flood_iou", "valid_f1", "seconds"])

    best = -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        t0, losses = time.time(), []
        for b in loader:
            x, y = random_crop(b["image"].to(device), b["label"].to(device), cfg["crop"])
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                out = model(x)
            loss = loss_fn(out.float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()
        overall, _ = evaluate(model, valid_ds, device)
        sec = time.time() - t0
        print(f"epoch {epoch:3d} | loss {np.mean(losses):.4f} | valid IoU {overall['flood_iou']:.4f} "
              f"| F1 {overall['f1']:.4f} | {sec:.0f}s")
        log.writerow([epoch, np.mean(losses), overall["flood_iou"], overall["f1"], sec])
        state = {"model": model.state_dict(), "cfg": cfg, "mean": mean, "std": std, "epoch": epoch}
        torch.save(state, out_dir / "last.pt")
        if overall["flood_iou"] > best:
            best = overall["flood_iou"]
            torch.save(state, out_dir / "best.pt")

    ck = torch.load(out_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    res = evaluate_splits(model, ["valid", "test", "bolivia"], mean, std, device)
    (out_dir / "eval.json").write_text(json.dumps(res, indent=2))
    print(f"\nbest epoch {ck['epoch']} (valid IoU {best:.4f})")
    for s, r in res.items():
        print(f"{s:8s} IoU {r['overall']['flood_iou']:.4f} | F1 {r['overall']['f1']:.4f}")


if __name__ == "__main__":
    main()