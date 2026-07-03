"""Generate the step-by-step teaching notebook (07-Object-Detection.ipynb).

The notebook mirrors ``object_detection.py`` one step at a time so a learner can open it, run every cell live,
and *see* object detection built and measured on a real photograph: box formats, IoU from scratch (verified
against torchvision), NMS from scratch (verified against torchvision on the real overlapping boxes a real
detector emits), anchor box-regression, a real pretrained Faster R-CNN run on a real image, the raw->kept NMS
dedup, Average Precision from scratch (pinned to a hand-verified 11/12 worked example, then applied to the
real detections before vs after NMS), per-class AP / mAP / the COCO mAP@[.5:.95] sweep, and IoU on a real box.
Each numbered step has a short markdown lead-in (the intuition) followed by a focused code cell with real output.

    python build_notebook_07.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../07-Object-Detection/code/07-Object-Detection.ipynb"

This generator lives in the domain-level ``07. Computer Vision/tools/`` folder; the notebook it writes (and the
module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited .ipynb) so the
notebook and the module stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "07-Object-Detection" / "code"
NB_PATH = _CHAPTER_CODE / "07-Object-Detection.ipynb"

_CELL_ID = 0


def _next_id() -> str:
    global _CELL_ID
    _CELL_ID += 1
    return f"cell-{_CELL_ID:02d}"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []


def add_md(source: str) -> None:
    CELLS.append(md(source))


def add_code(source: str) -> None:
    CELLS.append(code(source))


# ============================ Title ============================================================
add_md(
    "# Object detection — a runnable, measured build\n"
    "\n"
    "Image *classification* answers **what** is in a picture; **object detection** answers **what, where, and "
    "how many** — a class *and* a bounding box for every object in the scene. This notebook builds detection's "
    "three load-bearing ideas **from scratch** and checks each against a trusted reference, then runs a **real** "
    "pretrained detector on a **real** photograph and measures everything:\n"
    "\n"
    "- **IoU** (Intersection over Union) — the box-overlap metric — checked against `torchvision.ops.box_iou`.\n"
    "- **NMS** (Non-Maximum Suppression) — the duplicate-removal step — checked against `torchvision.ops.nms` on "
    "the *real* overlapping boxes the detector emits.\n"
    "- **AP** (Average Precision) — the detection score — pinned to a hand-verified worked example (AP = 11/12), "
    "then applied to the real detections.\n"
    "\n"
    "It imports the **exact same functions** as the companion page and its figures (from `object_detection.py`), "
    "so the numbers here are the numbers there. Everything is **CPU-pinned and seeded** for a reproducible trace. "
    "If the COCO image can't be downloaded, the module falls back to a real bundled photo (Grace Hopper) — the "
    "detector still runs on a real image and every check still executes.\n"
    "\n"
    "> Companion page: **Object Detection**. Run top-to-bottom (Kernel → Restart & Run All). The detector weights "
    "download once (~160 MB) to a temp cache, never into the repo."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup: import the real module and print versions\n"
    "\n"
    "We import the pipeline from the chapter module so this notebook runs the *same code* the page and figures "
    "use, and print the library versions + the accelerator we detected (we still run on CPU on purpose — GPU "
    "kernels are nondeterministic and every number here must be bit-reproducible)."
)
add_code(
    "import numpy as np\n"
    "import torch\n"
    "import torchvision\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.patches as mpatches\n"
    "\n"
    "import object_detection as od\n"
    "\n"
    "print(f'torch {torch.__version__} | torchvision {torchvision.__version__} | numpy {np.__version__}')\n"
    "print(f'accelerator available: {od.detect_accelerator()}  (reported metrics are CPU-pinned, seed={od.SEED})')"
)

# ---- Step 1: box formats ----
add_md(
    "## Step 1 — A box is four numbers — but *which* four? (formats)\n"
    "\n"
    "Every box has four numbers, but three conventions are in daily use and silently mixing them is the #1 "
    "detection bug:\n"
    "\n"
    "- **xyxy** = $(x_{min}, y_{min}, x_{max}, y_{max})$ — corners; what IoU, NMS, and drawing want.\n"
    "- **xywh** = $(x_{min}, y_{min}, w, h)$ — top-left + size; the COCO annotation format.\n"
    "- **cxcywh** = $(x_{center}, y_{center}, w, h)$ — center + size; what anchor regression predicts.\n"
    "\n"
    "We implement the conversions and check them against `torchvision.ops.box_convert` (and that a round-trip is "
    "lossless)."
)
add_code(
    "fc = od.verify_box_formats()\n"
    "print(f'max|xywh   - torchvision| = {fc.max_err_xywh:.1e}')\n"
    "print(f'max|cxcywh - torchvision| = {fc.max_err_cxcywh:.1e}')\n"
    "print(f'xyxy -> cxcywh -> xyxy round-trip error = {fc.max_err_roundtrip:.1e}')\n"
    "box = np.array([[10., 20., 110., 220.]])\n"
    "print(f'\\nexample: xyxy {box[0].tolist()}  ->  xywh {od.xyxy_to_xywh(box)[0].tolist()}  '\n"
    "      f'->  cxcywh {od.xyxy_to_cxcywh(box)[0].tolist()}')\n"
    "assert fc.max_err_cxcywh < 1e-9\n"
    "print('OK: our conversions match torchvision')"
)

# ---- Step 2: IoU ----
add_md(
    "## Step 2 — IoU: how much do two boxes overlap?\n"
    "\n"
    "**Intersection over Union** is the overlap metric everything downstream uses:\n"
    "\n"
    "$$\\mathrm{IoU}(A, B) = \\frac{\\mathrm{area}(A \\cap B)}{\\mathrm{area}(A \\cup B)}.$$\n"
    "\n"
    "The intersection rectangle is `[max of the mins, min of the maxes]`, clamped so no overlap gives 0; the "
    "union is `area(A) + area(B) − intersection`. IoU is 0 for disjoint boxes and 1 for identical boxes. We "
    "implement a vectorized IoU *matrix* (every A against every B) and check it against `torchvision.ops.box_iou`."
)
add_code(
    "iou = od.verify_iou()\n"
    "print(f'max|IoU - torchvision.ops.box_iou| = {iou.max_err_vs_torchvision:.2e}')\n"
    "# a hand-checkable pair: two 10x10 boxes shifted by 5 in x -> intersection 5x10=50, union 150 -> 1/3\n"
    "a = np.array([[0., 0., 10., 10.]])\n"
    "b = np.array([[5., 0., 15., 10.]])\n"
    "print(f'IoU(two 10x10 boxes shifted 5px) = {od.box_iou_from_scratch(a, b)[0,0]:.4f}  (= 50/150 = 0.3333)')\n"
    "assert iou.max_err_vs_torchvision < 1e-6\n"
    "print('OK: our IoU IS torchvision IoU')"
)

# ---- Step 3: NMS ----
add_md(
    "## Step 3 — NMS: keep one box per object\n"
    "\n"
    "A detector fires many overlapping boxes on the same object. **Non-Maximum Suppression** removes the "
    "duplicates greedily: sort boxes by score; take the top one, keep it, and discard every remaining box whose "
    "IoU with it exceeds a threshold (they're duplicates of the same object); repeat. We implement it and check "
    "it against `torchvision.ops.nms` on a small set of overlapping boxes (identical kept indices)."
)
add_code(
    "boxes = np.array([[10.,10,110,110],[14,12,112,108],[200,200,300,300],[8,14,105,112],[205,198,298,305]])\n"
    "scores = np.array([0.9, 0.82, 0.75, 0.60, 0.55])\n"
    "kept = od.nms_from_scratch(boxes, scores, iou_thresh=0.5)\n"
    "print(f'{len(boxes)} boxes -> NMS keeps indices {kept.tolist()} (the two distinct objects, best score each)')\n"
    "chk = od.verify_nms(boxes, scores, 0.5)\n"
    "print(f'kept indices == torchvision.ops.nms: {chk.matches_torchvision}')\n"
    "assert chk.matches_torchvision\n"
    "print('OK: our NMS IS torchvision NMS')"
)

# ---- Step 4: anchors / box regression ----
add_md(
    "## Step 4 — Anchors + box regression: predicting a box as a *correction*\n"
    "\n"
    "Detectors don't predict box coordinates directly. They start from **anchors** (reference boxes tiled over "
    "the image) and predict a *correction* $(t_x, t_y, t_w, t_h)$ that nudges an anchor onto the object "
    "(Faster R-CNN):\n"
    "\n"
    "$$t_x = \\frac{g_x - a_x}{a_w},\\ t_y = \\frac{g_y - a_y}{a_h},\\ t_w = \\log\\frac{g_w}{a_w},\\ "
    "t_h = \\log\\frac{g_h}{a_h}.$$\n"
    "\n"
    "Centers shift in units of the anchor's size (scale-invariant); sizes are predicted in log-space (so a box "
    "can never go negative). If our encode/decode is a true inverse, `decode(encode(gt, anchor), anchor) == gt`."
)
add_code(
    "rc = od.verify_box_regression()\n"
    "print(f'max|decode(encode(gt)) - gt| = {rc.max_roundtrip_err:.1e}')\n"
    "anchor = np.array([[50., 50., 150., 150.]])\n"
    "gt = np.array([[60., 70., 170., 180.]])\n"
    "delta = od.encode_boxes(gt, anchor)\n"
    "print(f'encode -> (tx,ty,tw,th) = {tuple(round(float(v),3) for v in delta[0])}')\n"
    "print(f'decode -> {od.decode_boxes(delta, anchor)[0].round(2).tolist()}  (recovers gt {gt[0].tolist()})')\n"
    "assert rc.max_roundtrip_err < 1e-9\n"
    "print('OK: the (tx,ty,tw,th) parametrization is an exact inverse')"
)

# ---- Step 5: real image ----
add_md(
    "## Step 5 — Load a real photograph\n"
    "\n"
    "Detection only means something on a real, multi-object scene. We load a real COCO photo (two cats on a couch "
    "with two remotes). If the download is unavailable, the module falls back to a real bundled photo so the "
    "notebook still runs on a real image."
)
add_code(
    "image = od.load_real_image()\n"
    "print(f'image  : {image.name}   [{image.source}]')\n"
    "print(f'shape  : {image.array.shape}  (H, W, 3 uint8)')\n"
    "print(f'ground-truth classes (hand-specified for scoring): {list(image.ground_truth)}')\n"
    "plt.figure(figsize=(6.5, 5))\n"
    "plt.imshow(image.array)\n"
    "plt.axis('off')\n"
    "plt.title(f'Real image: {image.name}')\n"
    "plt.show()"
)

# ---- Step 6: run the real detector ----
add_md(
    "## Step 6 — Run a real pretrained detector\n"
    "\n"
    "We load **Faster R-CNN (ResNet-50 FPN)** with COCO weights — a real two-stage detector — and run it on the "
    "image. It returns, for each detection, a **box**, a **class label** (one of 80 COCO categories), and a "
    "**score**. We keep detections scoring ≥ 0.5 and draw them. (Weights download once to a temp cache.)"
)
add_code(
    "model, categories = od.load_detector(score_thresh=0.5, nms_thresh=0.5)\n"
    "det = od.run_detector(model, categories, image.array)\n"
    "for name, score in sorted(zip(det.names, det.scores), key=lambda t: -t[1]):\n"
    "    print(f'  {name:10s} {score:.3f}')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(7, 5.2))\n"
    "ax.imshow(image.array)\n"
    "ax.axis('off')\n"
    "for bx, nm, sc in zip(det.boxes, det.names, det.scores):\n"
    "    c = '#2E7A5A' if sc >= 0.6 else '#7A6528'\n"
    "    ax.add_patch(mpatches.Rectangle((bx[0],bx[1]), bx[2]-bx[0], bx[3]-bx[1], fill=False, edgecolor=c, lw=2))\n"
    "    ax.text(bx[0], bx[1]-3, f'{nm} {sc:.2f}', color='white', fontsize=8,\n"
    "            bbox=dict(boxstyle='square,pad=0.1', fc=c, ec='none'))\n"
    "ax.set_title('Real Faster R-CNN detections (green >=0.6, amber borderline)')\n"
    "plt.show()"
)

# ---- Step 7: raw boxes before NMS ----
add_md(
    "## Step 7 — Peek *before* NMS: the raw overlapping flood\n"
    "\n"
    "The clean result above already had NMS applied inside the detector. To *see* what NMS does, we re-run the "
    "detector with its own NMS effectively disabled (`nms_thresh = 0.95`) and a low score threshold, then take "
    "all boxes of one class. A single object attracts **many** overlapping boxes — the raw material NMS cleans up."
)
add_code(
    "raw_model, _ = od.load_detector(score_thresh=0.05, nms_thresh=0.95)\n"
    "det_raw = od.run_detector(raw_model, categories, image.array)\n"
    "cls = od._pick_nms_class(det_raw, image.ground_truth)\n"
    "mask = np.array([n == cls for n in det_raw.names]) & (det_raw.scores >= 0.30)\n"
    "raw_boxes, raw_scores = det_raw.boxes[mask], det_raw.scores[mask]\n"
    "print(f\"class '{cls}': {len(raw_boxes)} raw overlapping boxes (score >= 0.30) before NMS\")"
)

# ---- Step 8: NMS on the real boxes ----
add_md(
    "## Step 8 — Apply NMS to the real boxes (and verify against torchvision)\n"
    "\n"
    "Now run our from-scratch NMS on those real overlapping boxes: the raw flood collapses to one box per real "
    "object. We confirm our kept indices are *identical* to `torchvision.ops.nms`, then draw before/after."
)
add_code(
    "chk = od.verify_nms(raw_boxes, raw_scores, 0.5)\n"
    "kept = od.nms_from_scratch(raw_boxes, raw_scores, 0.5)\n"
    "print(f'{len(raw_boxes)} raw boxes -> {len(kept)} kept   (kept == torchvision.ops.nms: {chk.matches_torchvision})')\n"
    "assert chk.matches_torchvision and len(kept) < len(raw_boxes)\n"
    "\n"
    "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))\n"
    "for ax in (a1, a2):\n"
    "    ax.imshow(image.array)\n"
    "    ax.axis('off')\n"
    "for bx in raw_boxes:\n"
    "    a1.add_patch(mpatches.Rectangle((bx[0],bx[1]), bx[2]-bx[0], bx[3]-bx[1], fill=False, edgecolor='#8B3B4A', lw=1))\n"
    "a1.set_title(f'before NMS: {len(raw_boxes)} raw {cls} boxes')\n"
    "for i in kept:\n"
    "    bx = raw_boxes[i]\n"
    "    a2.add_patch(mpatches.Rectangle((bx[0],bx[1]), bx[2]-bx[0], bx[3]-bx[1], fill=False, edgecolor='#2E7A5A', lw=2.2))\n"
    "a2.set_title(f'after NMS: {len(kept)} kept')\n"
    "plt.show()"
)

# ---- Step 9: AP worked example ----
add_md(
    "## Step 9 — Average Precision, from a worked example (AP = 11/12)\n"
    "\n"
    "A detector needs a single score. **Average Precision** is the area under the **precision-recall curve**. "
    "Walking the detections from highest score to lowest, each is a **true positive** (matches an unclaimed "
    "ground-truth box at IoU ≥ 0.5) or a **false positive**; precision = TP/(TP+FP) and recall = TP/#GT are "
    "accumulated down the list, and AP integrates the (interpolated) curve.\n"
    "\n"
    "To pin the implementation, here is a hand-verifiable case: 3 ground-truth boxes and 5 detections whose "
    "match sequence is **TP, TP, FP, TP, FP**. By hand, AP = 1/3 + 1/3 + 1/4 = **11/12 ≈ 0.9167**. Our code must "
    "reproduce that exactly."
)
add_code(
    "kc = od.verify_ap_known_case()\n"
    "print(f'AP (from scratch) = {kc.ap_measured:.6f}   expected 11/12 = {kc.ap_expected:.6f}')\n"
    "assert abs(kc.ap_measured - 11/12) < 1e-9\n"
    "c = kc.curve\n"
    "plt.figure(figsize=(5, 4))\n"
    "plt.step(np.concatenate(([0], c.recall)), np.concatenate(([1], c.precision)), where='post', color='#4A5B6E')\n"
    "plt.scatter(c.recall, c.precision, color='#4A5B6E', zorder=3)\n"
    "plt.xlabel('recall')\n"
    "plt.ylabel('precision')\n"
    "plt.ylim(0, 1.05)\n"
    "plt.xlim(0, 1.02)\n"
    "plt.title(f'PR curve — AP = area = {kc.ap_measured:.3f} = 11/12')\n"
    "plt.show()\n"
    "print('OK: AP reproduces the hand-verified worked example')"
)

# ---- Step 10: AP on real detections before/after NMS ----
add_md(
    "## Step 10 — AP on the real detections: why NMS matters for the *score*\n"
    "\n"
    "NMS isn't cosmetic — it changes the score. Take the *same* real detections for one class and compute AP two "
    "ways: **before NMS** (the duplicate boxes each become a false positive, crushing precision) and **after "
    "NMS** (one clean box per object). NMS *raises* AP by removing duplicate false positives."
)
add_code(
    "eval_model, _ = od.load_detector(score_thresh=0.05, nms_thresh=0.5)   # standard eval: post-NMS, low score\n"
    "det_eval = od.run_detector(eval_model, categories, image.array)\n"
    "gt = image.ground_truth[cls]\n"
    "pre_mask = np.array([n == cls for n in det_raw.names])\n"
    "post_mask = np.array([n == cls for n in det_eval.names])\n"
    "pre = od.average_precision(det_raw.boxes[pre_mask], det_raw.scores[pre_mask], gt, 0.5)\n"
    "post = od.average_precision(det_eval.boxes[post_mask], det_eval.scores[post_mask], gt, 0.5)\n"
    "print(f'pre-NMS : {pre.n_det:3d} detections -> AP@0.5 = {pre.ap:.4f}')\n"
    "print(f'post-NMS: {post.n_det:3d} detections -> AP@0.5 = {post.ap:.4f}   (NMS lifts AP by {post.ap-pre.ap:+.4f})')\n"
    "for prc, col, lab in [(pre,'#8B3B4A','pre-NMS'),(post,'#2E7A5A','post-NMS')]:\n"
    "    plt.step(np.concatenate(([0],prc.recall)), np.concatenate(([1],prc.precision)), where='post',\n"
    "             color=col, label=f'{lab} AP={prc.ap:.2f}')\n"
    "plt.xlabel('recall')\n"
    "plt.ylabel('precision')\n"
    "plt.legend()\n"
    "plt.ylim(0, 1.05)\n"
    "plt.title(f\"'{cls}' PR curve\")\n"
    "plt.show()\n"
    "assert post.ap >= pre.ap"
)

# ---- Step 11: mAP + COCO sweep ----
add_md(
    "## Step 11 — mAP and the COCO mAP@[.5:.95] sweep\n"
    "\n"
    "**mAP** (mean Average Precision) averages AP over classes. The looser **AP@0.5** only asks for IoU ≥ 0.5; "
    "the headline **COCO mAP@[.5:.95]** averages AP over ten IoU thresholds from 0.50 to 0.95, so it rewards "
    "*tight* localization, not just detection — which is why it is always lower. On this single easy image the "
    "detector is excellent, so the numbers are high; a single image is illustrative, **not a benchmark**."
)
add_code(
    "sweep = od.COCO_IOU_SWEEP\n"
    "for name, gt_boxes in image.ground_truth.items():\n"
    "    m = np.array([n == name for n in det_eval.names])\n"
    "    ap50 = od.average_precision(det_eval.boxes[m], det_eval.scores[m], gt_boxes, 0.5).ap\n"
    "    apco = od.mean_ap_over_iou(det_eval.boxes[m], det_eval.scores[m], gt_boxes, sweep)\n"
    "    print(f'  {name:8s}: AP@0.5 = {ap50:.3f}   AP@[.5:.95] = {apco:.3f}')\n"
    "exp = od.run_experiment()\n"
    "print(f'\\n  mAP@0.5        = {exp.map_50:.4f}')\n"
    "print(f'  COCO mAP@[.5:.95] = {exp.coco_map:.4f}  (stricter IoU bar -> lower, by design)')"
)

# ---- Step 12: IoU on a real detection ----
add_md(
    "## Step 12 — IoU on a real detection vs its ground truth\n"
    "\n"
    "Finally, close the loop: the top detection's box against its ground-truth box. A good detection has a high "
    "IoU — that's exactly the quantity AP thresholds on."
)
add_code(
    "pair, val, name = od._iou_demo_pair(det, image.ground_truth)\n"
    "print(f\"IoU(top '{name}' detection, its ground truth) = {val:.4f}\")\n"
    "fig, ax = plt.subplots(figsize=(6.5, 5))\n"
    "ax.imshow(image.array)\n"
    "ax.axis('off')\n"
    "for bx, c, lab in [(pair[1],'#3A6B96','ground truth'), (pair[0],'#2E7A5A','detection')]:\n"
    "    ax.add_patch(mpatches.Rectangle((bx[0],bx[1]), bx[2]-bx[0], bx[3]-bx[1], fill=False, edgecolor=c, lw=2.4))\n"
    "    ax.text(bx[0], bx[1]-3, lab, color='white', fontsize=8, bbox=dict(boxstyle='square,pad=0.1', fc=c, ec='none'))\n"
    "ax.set_title(f'IoU = {val:.3f} (a tight, correct box)')\n"
    "plt.show()"
)

# ---- close ----
add_md(
    "## Recap\n"
    "\n"
    "Detection = **classification + localization**, and it rests on three ideas you just built and verified on a "
    "real image: **IoU** measures box overlap (checked against torchvision), **NMS** removes duplicate boxes "
    "(checked against torchvision, and it *raises* AP by deleting duplicate false positives), and **AP** is the "
    "area under the precision-recall curve (pinned to the hand-verified 11/12 worked example). A real Faster "
    "R-CNN found the cats and remotes; its raw output was a flood of overlapping boxes that NMS collapsed to one "
    "per object; and mAP@0.5 vs the stricter COCO mAP@[.5:.95] showed how the score tightens with the IoU bar.\n"
    "\n"
    "See the companion page for the two-stage vs one-stage architecture story (R-CNN → Fast → Faster → YOLO → "
    "RetinaNet → DETR), anchors and the class-imbalance problem focal loss solves, the pitfalls, and references — "
    "and the sibling chapters on image classification (04, a detector is a classifier over regions), the CNN "
    "backbone (05 DL 13), and detection metrics (10)."
)


def build() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {NB_PATH}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
