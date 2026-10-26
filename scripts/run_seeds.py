"""Run the U-Net baseline over seeds and label fractions. Skips runs that already finished."""
import argparse
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
ap.add_argument("--fractions", type=float, nargs="+", default=[1.0])
ap.add_argument("--config", default="configs/unet_baseline.yaml")
args = ap.parse_args()

for frac in args.fractions:
    for seed in args.seeds:
        name = f"unet_resnet34_seed{seed}_frac{frac}"
        if (Path("checkpoints") / name / "eval.json").exists():
            print("skip", name)
            continue
        subprocess.run([sys.executable, "-m", "src.train", "--config", args.config,
                        "--seed", str(seed), "--label_fraction", str(frac),
                        "--run_name", name], check=True)