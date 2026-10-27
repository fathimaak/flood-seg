"""Shared inference code for the demo and the verification script.
Reuses src.models and the same preprocessing as src.dataset / src.train."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from src.dataset import BAND_IDX_1BASED
from src.models import ARCHS, build_model

REPO = Path(__file__).resolve().parents[1]
CKPT_DIR = REPO / "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCH_LABELS = {"unet": "U-Net (ResNet-34)", "deeplabv3plus": "DeepLabV3+ (ResNet-34)"}
MAX_SIDE = 2048


class ModelUnavailable(Exception):
    pass


class InputError(Exception):
    pass


@dataclass
class ModelBundle:
    arch: str
    label: str
    ckpt_path: Path
    model: torch.nn.Module
    mean: np.ndarray        # (6,1,1) float32, from the checkpoint (training statistics)
    std: np.ndarray
    epoch: int
    recorded: dict          # flood IoU per split from this run's eval.json (or None)


def load_bundle(arch, encoder="resnet34", seed=0, frac=1.0):
    run = f"{arch}_{encoder}_seed{seed}_frac{frac}"
    path = CKPT_DIR / run / "best.pt"
    if arch not in ARCHS:
        raise ModelUnavailable(f"unknown architecture '{arch}'")
    if not path.exists():
        raise ModelUnavailable(f"checkpoint not found: {path}")
    try:
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
    except Exception as e:
        raise ModelUnavailable(f"could not read {path.name}: {e}")
    cfg = ck.get("cfg", {})
    ck_arch = cfg.get("arch", "unet")   # runs trained before the 'arch' key existed were U-Nets
    if ck_arch != arch or cfg.get("encoder", encoder) != encoder:
        raise ModelUnavailable(f"{path} was trained as {ck_arch}/{cfg.get('encoder')}, "
                               f"expected {arch}/{encoder}")
    model = build_model(arch, encoder, None).to(DEVICE)
    try:
        model.load_state_dict(ck["model"])          # strict: any mismatch raises
    except RuntimeError as e:
        raise ModelUnavailable(f"weights in {path.name} do not fit {arch}: {str(e)[:200]}")
    model.eval()
    recorded = None
    ev = CKPT_DIR / run / "eval.json"
    if ev.exists():
        try:
            r = json.loads(ev.read_text())
            recorded = {s: r[s]["overall"]["flood_iou"] for s in ("valid", "test", "bolivia") if s in r}
        except Exception:
            recorded = None
    return ModelBundle(arch, ARCH_LABELS.get(arch, arch), path, model,
                       np.asarray(ck["mean"], np.float32)[:, None, None],
                       np.asarray(ck["std"], np.float32)[:, None, None],
                       ck.get("epoch", -1), recorded)


def discover(encoder="resnet34", seed=0, frac=1.0):
    """Returns (available bundles, {arch: reason it is unavailable}, optional warning)."""
    bundles, problems = {}, {}
    for arch in ARCHS:
        try:
            bundles[arch] = load_bundle(arch, encoder, seed, frac)
        except ModelUnavailable as e:
            problems[arch] = str(e)
    warning = None
    if len(bundles) > 1:
        ref = next(iter(bundles.values()))
        if not all(np.allclose(b.mean, ref.mean) and np.allclose(b.std, ref.std)
                   for b in bundles.values()):
            warning = "The two checkpoints store different normalization statistics."
    return bundles, problems, warning


@torch.no_grad()
def predict_classes(bundle, img_raw):
    """img_raw: (6,H,W) float32 in the order B2,B3,B4,B8A,B11,B12; may contain NaN.
    Preprocessing is identical to training: NaN->0, then (x-mean)/std. Returns (H,W) int64
    classes (0 land, 1 flood) for EVERY pixel; callers must mask invalid input pixels."""
    x = np.nan_to_num(img_raw, nan=0.0)
    x = ((x - bundle.mean) / bundle.std).astype(np.float32)
    x = torch.from_numpy(x)[None].to(DEVICE)
    H, W = x.shape[-2:]
    ph, pw = (-H) % 32, (-W) % 32
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="reflect")
    bundle.model.eval()
    return bundle.model(x).argmax(1)[0, :H, :W].cpu().numpy().astype(np.int64)


def load_upload(path):
    """Validate and read an uploaded GeoTIFF -> (6,H,W) float32 (NaN kept), band count."""
    p = Path(path)
    if p.suffix.lower() not in (".tif", ".tiff"):
        raise InputError("Please upload a GeoTIFF (.tif / .tiff). Ordinary RGB images cannot be "
                         "used: the models need six Sentinel-2 bands.")
    try:
        src = rasterio.open(p)
    except Exception as e:
        raise InputError(f"This file could not be read as a GeoTIFF ({e}).")
    with src:
        n = src.count
        if n == 13:
            idx = BAND_IDX_1BASED          # full Sen1Floods11 Sentinel-2 L1C chip
        elif n == 6:
            idx = list(range(1, 7))        # assumed to be B2,B3,B4,B8A,B11,B12 in that order
        else:
            raise InputError(f"This file has {n} band(s). Expected either the 13-band Sen1Floods11 "
                             "Sentinel-2 L1C chip, or exactly 6 bands ordered B2, B3, B4, B8A, B11, B12.")
        if min(src.height, src.width) < 64:
            raise InputError("Image is too small (minimum 64 × 64 pixels).")
        if max(src.height, src.width) > MAX_SIDE:
            raise InputError(f"Image is larger than {MAX_SIDE} px on a side; please crop it first.")
        img = src.read(idx).astype(np.float32)
        nodata = src.nodata
    if nodata is not None and np.isfinite(nodata):
        img[img == nodata] = np.nan        # flag, don't fill
    finite = img[np.isfinite(img)]
    if finite.size == 0:
        raise InputError("The file contains no valid pixel values.")
    if finite.max() < 20:
        raise InputError("Values look like 0–1 reflectance. The models expect the scaled values "
                         "used in Sen1Floods11 (roughly 0–10000), so this file is not compatible.")
    return img, n