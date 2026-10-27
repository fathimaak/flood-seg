import sys
from pathlib import Path

import gradio as gr
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.dataset import Sen1Floods11, read_label, read_s2_raw, valid_input_mask
from src.inference import DEVICE, InputError, discover, load_upload, predict_classes
from src.metrics import Metrics
from src.utils import COLORS, class_map_rgb, rgb_uint8

DATA = REPO / "data" / "sen1floods11"
UNSEEN_EVENTS = {"Bolivia"}
DEFAULT_CHIP = "Bolivia_103757_S2Hand.tif"
SRC_SAMPLE, SRC_UPLOAD = "Built-in sample chip", "Upload GeoTIFF"

BUNDLES, PROBLEMS, STATS_WARNING = discover()

# ---------------------------------------------------------------- sample chips
SAMPLES = {}      # filename -> (s2 path, label path, event, split)
COVERAGE = {}     # filename -> fraction of pixels that are labelled
for _split in ("bolivia", "test"):
    _csv = DATA / "splits" / f"flood_{_split}_data.csv"
    if _csv.exists():
        _ds = Sen1Floods11(DATA, _csv)
        for _s2, _lb, _ev in zip(_ds.s2_paths, _ds.lbl_paths, _ds.events):
            SAMPLES[_s2.name] = (_s2, _lb, _ev, _split)
            COVERAGE[_s2.name] = float((read_label(_lb) != -1).mean())

EVENTS = sorted({v[2] for v in SAMPLES.values()}, key=lambda e: (e not in UNSEEN_EVENTS, e))


def event_choices():
    return [((f"{e} - unseen event" if e in UNSEEN_EVENTS else f"{e} - seen in training"), e) for e in EVENTS]


def chips_for(ev):
    return [(f"{k.replace('_S2Hand.tif', '')}  ({100 * COVERAGE[k]:.0f}% labelled)", k)
            for k, v in SAMPLES.items() if v[2] == ev]


def default_chip(ev):
    keys = [k for _, k in chips_for(ev)]
    return max(keys, key=lambda k: COVERAGE[k]) if keys else None   # best-covered chip


def chip_info(key):
    if not key or key not in SAMPLES:
        return "No sample chip selected."
    _, _, ev, _ = SAMPLES[key]
    if ev in UNSEEN_EVENTS:
        role = ("**Unseen evaluation event.** No chip from this event was used for training or "
                "validation, so this tests generalization.")
    else:
        role = ("Event seen during training. This chip comes from the held-out test split, so the "
                "model did not train on this chip itself.")
    return (f"**Event:** {ev}  \n{role}  \n**Labelled coverage:** {100 * COVERAGE[key]:.0f}% of the chip "
            "(the rest has no satellite data).  \nA sample chip is a small tile (512 × 512 pixels, roughly "
            "5 km × 5 km) cut from a much larger Sentinel-2 satellite scene.")


def legend_html():
    items = [("Land", COLORS["land"]), ("Flood", COLORS["flood"]),
             ("Ignored / unlabelled (ground truth)", COLORS["ignored"]),
             ("Invalid input (no data, not predicted)", COLORS["invalid"])]
    return '<div class="legend">' + "".join(
        f'<span class="lg"><i style="background:{c}"></i>{n}</span>' for n, c in items) + "</div>"


def model_info(key):
    if key not in BUNDLES:
        return "No model selected."
    b = BUNDLES[key]
    txt = (f"**{b.label}** - checkpoint `{b.ckpt_path.relative_to(REPO)}` "
           f"(best validation epoch {b.epoch}), running on **{DEVICE}**.")
    if b.recorded:
        txt += ("\n\nRecorded flood IoU of this training run on its full splits (not this chip): "
                + " · ".join(f"{k} {v:.3f}" for k, v in b.recorded.items()))
    return txt


def models_unavailable_md():
    if not PROBLEMS and not STATS_WARNING:
        return ""
    lines = [f"- **{k}** unavailable: {v}" for k, v in PROBLEMS.items()]
    if STATS_WARNING:
        lines.append(f"- Warning: {STATS_WARNING}")
    return "\n".join(lines)


# ------------------------------------------------------------------- inference
def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def _run(source, chip_key, upload_path, model_key, compare):
    if not BUNDLES:
        raise InputError("No trained checkpoint could be loaded. See 'Model status' below.")
    if source == SRC_SAMPLE:
        if not chip_key or chip_key not in SAMPLES:
            raise InputError("Select a sample chip first.")
        s2, lb, ev, split = SAMPLES[chip_key]
        img, lbl = read_s2_raw(s2), read_label(lb)
        name = s2.name
        origin = f"Sen1Floods11 sample - {ev} ({'unseen event' if ev in UNSEEN_EVENTS else 'seen event, held-out test chip'})"
    else:
        if not upload_path:
            raise InputError("Upload a GeoTIFF first (or switch to a built-in sample).")
        img, nb = load_upload(upload_path)
        lbl, name = None, Path(upload_path).name
        origin = f"Uploaded file ({nb} bands), no ground truth"
    if model_key not in BUNDLES:
        raise InputError("Select an available model.")

    H, W = img.shape[1:]
    valid = valid_input_mask(img)
    n_inv, total = int((~valid).sum()), valid.size
    if valid.sum() == 0:
        raise InputError("Every pixel is invalid/no-data, so nothing can be predicted.")
    if n_inv / total > 0.5:
        gr.Warning(f"{100 * n_inv / total:.0f}% of this image is invalid/no-data. "
                   "Predictions only cover the remaining pixels.")

    keys = [model_key] + ([k for k in BUNDLES if k != model_key] if compare and len(BUNDLES) > 1 else [])
    preds = {k: predict_classes(BUNDLES[k], img) for k in keys}     # identical input for every model

    rgb = rgb_uint8(img, valid)
    gt_img = class_map_rgb(lbl) if lbl is not None else None
    pred_imgs = {k: class_map_rgb(p, invalid=~valid) for k, p in preds.items()}

    rows = ["| Item | Value |", "|---|---|", f"| Input | `{name}` |", f"| Source | {origin} |",
            f"| Size | {W} × {H} px |",
            f"| Invalid input pixels | {n_inv:,} ({100 * n_inv / total:.1f}%), drawn dark and excluded from all statistics |"]
    if lbl is not None:
        lab = lbl != -1
        rows.append(f"| Labelled pixels | {int(lab.sum()):,} ({100 * lab.sum() / total:.1f}%) |")
        gt_flood = 100 * (lbl == 1).sum() / max(lab.sum(), 1)
        rows.append(f"| **Ground-truth** flood fraction | {gt_flood:.1f}% of labelled pixels |")
    rows += ["", "| Model | **Predicted** flood fraction (of valid input pixels) |", "|---|---|"]
    for k, p in preds.items():
        rows.append(f"| {BUNDLES[k].label} | {100 * ((p == 1) & valid).sum() / valid.sum():.1f}% |")
    if len(preds) > 1:
        a, b = list(preds.values())[:2]
        rows.append(f"\nThe two models give the same class on {100 * (a == b)[valid].mean():.1f}% of valid pixels.")
    summary = "\n".join(rows)

    metrics = ""
    if lbl is not None:
        lbl_eval = np.where(valid, lbl, -1)      # ignored OR invalid-input pixels are excluded
        n_eval = int((lbl_eval != -1).sum())
        excluded = int(((lbl != -1) & ~valid).sum())
        mrows = ["| Model | Flood IoU | F1 | Precision | Recall |", "|---|---|---|---|---|"]
        for k, p in preds.items():
            gt_f, pr_f = int((lbl_eval == 1).sum()), int(((p == 1) & (lbl_eval != -1)).sum())
            if n_eval == 0 or gt_f + pr_f == 0:
                mrows.append(f"| {BUNDLES[k].label} | n/a | n/a | n/a | n/a |")
                continue
            m = Metrics()
            m.update(p[None], lbl_eval[None], ["chip"])
            o, _ = m.compute()
            mrows.append(f"| {BUNDLES[k].label} | {_fmt(o['flood_iou'])} | {_fmt(o['f1'])} | "
                         f"{_fmt(o['precision'])} | {_fmt(o['recall'])} |")
        metrics = ("\n".join(mrows) +
                   f"\n\nComputed on **this chip only**, over {n_eval:,} labelled pixels with valid input "
                   f"({excluded:,} labelled pixels excluded for invalid input). Single-chip scores are noisy "
                   "and are not comparable with the full test-set results reported in the project.")
    status = f"✅ Prediction complete for `{name}`."
    if lbl is not None and source == SRC_SAMPLE and SAMPLES[chip_key][2] in UNSEEN_EVENTS:
        status += " This chip is from the unseen Bolivia event."

    outs = [status, rgb]
    outs.append(gr.update(value=gt_img, visible=True) if gt_img is not None else gr.update(value=None, visible=False))
    k0 = keys[0]
    outs.append(gr.update(value=pred_imgs[k0], label=f"Prediction - {BUNDLES[k0].label}", visible=True))
    if len(keys) > 1:
        k1 = keys[1]
        outs.append(gr.update(value=pred_imgs[k1], label=f"Prediction - {BUNDLES[k1].label}", visible=True))
    else:
        outs.append(gr.update(value=None, visible=False))
    outs += [summary, metrics]
    return tuple(outs)


def run(source, chip_key, upload_path, model_key, compare):
    try:
        return _run(source, chip_key, upload_path, model_key, compare)
    except InputError as e:
        raise gr.Error(str(e))
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Prediction failed ({type(e).__name__}): {e}")


# ------------------------------------------------------------------------ UI
CSS = """
<style>
.gradio-container {max-width: 1400px !important;}
.hero {background: linear-gradient(135deg,#0f2a43,#0e5a73); color:#fff; padding:22px 28px;
       border-radius:12px; margin-bottom:6px;}
.hero, .hero h1, .hero p {color:#ffffff !important;}
.hero h1 {margin:0 0 6px 0; font-size:1.8rem;}
.hero p {margin:0; opacity:.92; max-width:950px;}
.card code {white-space: pre-wrap; word-break: break-all;}
.card {border:1px solid #d7dee6; border-radius:12px; padding:14px;}
.legend {display:flex; flex-wrap:wrap; gap:18px; padding:6px 2px; font-size:.9rem;}
.legend .lg {display:inline-flex; align-items:center; gap:6px;}
.legend i {width:16px; height:16px; border-radius:3px; border:1px solid rgba(0,0,0,.4); display:inline-block;}
</style>
"""
HERO = """
<div class="hero"><h1>Flood Extent Segmentation</h1>
<p>Deep-learning flood mapping from six-band Sentinel-2 imagery, trained on Sen1Floods11. Compare U-Net and
DeepLabV3+ on built-in sample chips (including the unseen Bolivia event) or on your own compatible GeoTIFF.</p></div>
"""
HOWTO = """
1. Choose **Built-in sample chip** (default: Bolivia, an unseen event) or **Upload GeoTIFF**.
2. Pick a model, or tick **Compare** to run both on exactly the same input.
3. Press **Predict**. Pixels with no satellite data are drawn dark and are never counted.

**Upload format:** a GeoTIFF with either the 13-band Sen1Floods11 Sentinel-2 L1C chip, or exactly 6 bands ordered
B2, B3, B4, B8A, B11, B12, with values on the same scale as Sen1Floods11 (about 0–10000). Ordinary RGB photos or
screenshots will be rejected.
"""

with gr.Blocks() as demo:
    gr.HTML(CSS + HERO)
    with gr.Accordion("How to use", open=False):
        gr.Markdown(HOWTO)
    with gr.Row():
        with gr.Column(scale=1, min_width=340, elem_classes=["card"]):
            source = gr.Radio([SRC_SAMPLE, SRC_UPLOAD], value=SRC_SAMPLE, label="Input source")
            with gr.Group(visible=True) as sample_group:
                ev0 = "Bolivia" if "Bolivia" in EVENTS else (EVENTS[0] if EVENTS else None)
                event_dd = gr.Dropdown(event_choices(), value=ev0, label="Event / location",
                                       info="Bolivia was never used in training or validation.")
                chip_dd = gr.Dropdown(chips_for(ev0) if ev0 else [], value=default_chip(ev0) if ev0 else None,
                                      label="Sample chip")
                chip_md = gr.Markdown(chip_info(default_chip(ev0) if ev0 else None))
            with gr.Group(visible=False) as upload_group:
                upload = gr.File(label="Drop a six-band Sentinel-2 GeoTIFF here", file_types=[".tif", ".tiff"],
                                 type="filepath")
                gr.Markdown("Needs 13 bands (Sen1Floods11 chip) or 6 bands ordered B2, B3, B4, B8A, B11, B12.")
            model_dd = gr.Dropdown([(b.label, k) for k, b in BUNDLES.items()],
                                   value=next(iter(BUNDLES), None), label="Model",
                                   info="Only models whose trained checkpoint loaded successfully are listed.")
            compare = gr.Checkbox(False, label="Compare both models on the same input",
                                  interactive=len(BUNDLES) > 1)
            model_md = gr.Markdown(model_info(next(iter(BUNDLES), None)))
            btn = gr.Button("Predict", variant="primary", interactive=bool(BUNDLES))
            status = gr.Markdown("Ready. Choose an input and press **Predict**." if BUNDLES else
                                 "⚠️ No trained checkpoint could be loaded. See *Model status* below.")
        with gr.Column(scale=3):
            gr.HTML(legend_html())
            with gr.Row():
                rgb_img = gr.Image(label="Sentinel-2 RGB (B4, B3, B2; contrast-stretched, for viewing only)",
                                   interactive=False, height=430)
                gt_img = gr.Image(label="Ground truth", interactive=False, height=430)
            with gr.Row():
                pred_a = gr.Image(label="Prediction", interactive=False, height=430)
                pred_b = gr.Image(label="Prediction (second model)", interactive=False, height=430, visible=False)
            summary_md = gr.Markdown()
            metrics_md = gr.Markdown()
    with gr.Accordion("Model status", open=bool(PROBLEMS)):
        gr.Markdown(models_unavailable_md() or "All configured checkpoints loaded successfully.")

    def toggle_source(s):
        return gr.update(visible=s == SRC_SAMPLE), gr.update(visible=s == SRC_UPLOAD)

    def on_event(ev):
        d = default_chip(ev)
        return gr.update(choices=chips_for(ev), value=d), chip_info(d)

    source.change(toggle_source, source, [sample_group, upload_group])
    event_dd.change(on_event, event_dd, [chip_dd, chip_md])
    chip_dd.change(chip_info, chip_dd, chip_md)
    model_dd.change(model_info, model_dd, model_md)

    btn.click(lambda: gr.update(interactive=False, value="Running…"), None, btn, queue=False).then(
        run, [source, chip_dd, upload, model_dd, compare],
        [status, rgb_img, gt_img, pred_a, pred_b, summary_md, metrics_md],
        show_progress="full", concurrency_limit=1,
    ).then(lambda: gr.update(interactive=bool(BUNDLES), value="Predict"), None, btn, queue=False)

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="127.0.0.1")   # local only; no public share link