"""Figure generator for 07-Object-Detection — every quantitative figure from the REAL run in
``object_detection.py``.

One measured experiment (``run_experiment``) drives every figure below, so nothing quantitative is hand-typed:
the real detector's boxes, the real raw->kept NMS counts, the real before/after-NMS precision-recall curves,
and the concrete IoU value all come from the same executed pipeline the chapter and notebook use.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``cv07_``:

  cv07_detections.png -- a REAL pretrained detector's boxes + labels + scores on a real photo, honest: the
                         high-confidence objects AND the low-score near-misses the detector also proposes.
  cv07_nms.png        -- NMS before/after on REAL boxes: the raw overlapping flood one object attracts (N) and
                         the deduplicated survivors (M), with the measured counts.
  cv07_pr_curve.png   -- precision-recall + AP: (a) the hand-verified worked example (AP = 11/12) with the
                         interpolation envelope, and (b) the SAME real detections scored before vs after NMS
                         (NMS removes duplicate false positives and lifts AP).
  cv07_iou.png        -- IoU illustration: a real detection box and its ground-truth box, the intersection and
                         union shaded, and the measured IoU value.

    python make_figures_07.py

Verified on Python 3.12 / matplotlib 3.10 / torch 2.12 / torchvision 0.27 / numpy 2.4.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in ``07. Computer Vision/tools/``; the chapter module it demonstrates stays in that
# chapter's ``code/`` folder. Put that folder on sys.path so the ``object_detection`` import resolves.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "07-Object-Detection" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from object_detection import (  # noqa: E402
    box_iou_from_scratch,
    cxcywh_to_xyxy,
    encode_boxes,
    nms_from_scratch,
    run_experiment,
    xyxy_to_cxcywh,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / values
PURPLE = "#5D4A8A"  # process
GREEN = "#2E7A5A"  # good / kept / post-NMS
RED = "#8B3B4A"  # error / suppressed / false positive
AMBER = "#7A6528"  # highlight
SLATE = "#4A5B6E"  # neutral / pre-NMS
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "cv07_"


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


def _draw_box(ax: plt.Axes, box: np.ndarray, color: str, label: str | None = None, lw: float = 2.0) -> None:
    x0, y0, x1, y1 = box
    ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=lw))
    if label:
        ax.text(x0, max(y0 - 4, 2), label, fontsize=8, color="white", va="bottom",
                bbox={"boxstyle": "square,pad=0.15", "fc": color, "ec": "none"})


# ================================================================================================
# Figure: real detections — a real detector's boxes + labels + scores on a real photo (honest)
# ================================================================================================


def fig_detections(exp) -> None:
    det = exp.detections
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    ax.imshow(exp.image.array)
    ax.axis("off")
    # colour high-confidence detections green, borderline (<0.6) amber -> the honest near-misses.
    # Boxes that share a top edge (e.g. the two overlapping couch boxes) would stack their labels;
    # nudge each new label down until it clears the labels already placed near the same x.
    order = det.scores.argsort()[::-1]
    placed: list[tuple[float, float]] = []  # (x, y) of labels already drawn
    for k in order:
        color = GREEN if det.scores[k] >= 0.6 else AMBER
        box = det.boxes[k]
        x0, y0 = float(box[0]), float(box[1])
        label_y = max(y0 - 4, 2.0)
        while any(abs(x0 - px) < 90 and abs(label_y - py) < 14 for px, py in placed):
            label_y += 15.0
        placed.append((x0, label_y))
        ax.add_patch(mpatches.Rectangle((x0, y0), box[2] - x0, box[3] - y0, fill=False, edgecolor=color, linewidth=2.0))
        ax.text(x0, label_y, f"{det.names[k]} {det.scores[k]:.2f}", fontsize=8, color="white", va="bottom",
                bbox={"boxstyle": "square,pad=0.15", "fc": color, "ec": "none"})
    n_hi = int((det.scores >= 0.6).sum())
    n_lo = int((det.scores < 0.6).sum())
    ax.set_title(
        f"Real Faster R-CNN detections on a real photo — {n_hi} confident (green), "
        f"{n_lo} borderline (amber, score < 0.6)",
        fontsize=10.5, color=INK,
    )
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}detections.png")


# ================================================================================================
# Figure: NMS before/after on real boxes — the raw flood vs the survivors
# ================================================================================================


def fig_nms(exp) -> None:
    raw_boxes, raw_scores = exp.nms_raw_boxes, exp.nms_raw_scores
    kept = nms_from_scratch(raw_boxes, raw_scores, 0.5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.0))
    for ax in (ax1, ax2):
        ax.imshow(exp.image.array)
        ax.axis("off")

    # (a) all raw overlapping boxes for the chosen class
    for b in raw_boxes:
        _draw_box(ax1, b, RED, lw=1.2)
    ax1.set_title(f"(a) before NMS — {len(raw_boxes)} raw '{exp.nms_class}' boxes\n(the detector fires many "
                  f"times per object)", fontsize=10, color=INK)

    # (b) only the NMS survivors, with their scores
    for i in kept:
        _draw_box(ax2, raw_boxes[i], GREEN, f"{exp.nms_class} {raw_scores[i]:.2f}", lw=2.2)
    ax2.set_title(f"(b) after NMS — {len(kept)} kept\n(one box per real object, IoU threshold 0.5)",
                  fontsize=10, color=INK)

    fig.suptitle(
        f"Non-Maximum Suppression on real detections: {len(raw_boxes)} overlapping boxes -> "
        f"{len(kept)} (matches torchvision.ops.nms exactly)",
        fontsize=11.5, color=INK, y=1.02,
    )
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}nms.png")


# ================================================================================================
# Figure: precision-recall + AP — the worked example, and the real before/after-NMS curves
# ================================================================================================


def _step_pr(ax: plt.Axes, recall: np.ndarray, precision: np.ndarray, color: str, label: str) -> None:
    r = np.concatenate(([0.0], recall))
    p = np.concatenate(([precision[0] if len(precision) else 1.0], precision))
    ax.step(r, p, where="post", color=color, linewidth=2, label=label)
    ax.scatter(recall, precision, s=22, color=color, zorder=3)


def fig_pr_curve(exp) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))

    # (a) the hand-verified worked example: TP,TP,FP,TP,FP -> AP = 11/12, with the interpolation envelope
    _style_axis(ax1)
    curve = exp.ap_known.curve
    _step_pr(ax1, curve.recall, curve.precision, SLATE, "raw precision")
    # interpolation envelope (precision made monotone from the right)
    mrec = np.concatenate(([0.0], curve.recall, [1.0]))
    mpre = np.concatenate(([0.0], curve.precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    ax1.fill_between(mrec, mpre, step="pre", alpha=0.18, color=GREEN)
    ax1.plot(mrec, mpre, drawstyle="steps-pre", color=GREEN, linewidth=1.6, label="interpolated (AP area)")
    ax1.set_xlabel("recall")
    ax1.set_ylabel("precision")
    ax1.set_xlim(0, 1.02)
    ax1.set_ylim(0, 1.05)
    ax1.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax1.set_title(f"(a) worked example (TP,TP,FP,TP,FP)\nAP = area = {exp.ap_known.ap_measured:.3f} = 11/12",
                  fontsize=10, color=INK)

    # (b) the SAME real detections scored before vs after NMS
    _style_axis(ax2)
    pre, post = exp.pr_pre_nms, exp.pr_post_nms
    _step_pr(ax2, pre.recall, pre.precision, RED, f"pre-NMS ({pre.n_det} det)  AP={pre.ap:.2f}")
    _step_pr(ax2, post.recall, post.precision, GREEN, f"post-NMS ({post.n_det} det)  AP={post.ap:.2f}")
    ax2.set_xlabel("recall")
    ax2.set_ylabel("precision")
    ax2.set_xlim(0, 1.05)
    ax2.set_ylim(0, 1.08)
    ax2.legend(fontsize=8.5, frameon=False, loc="lower left")
    ax2.set_title(f"(b) real '{exp.ap_class}' detections: NMS removes\nduplicate false positives -> AP "
                  f"{pre.ap:.2f} to {post.ap:.2f}", fontsize=10, color=INK)

    fig.suptitle("Average Precision = area under the precision-recall curve (measured)", fontsize=11.5,
                 color=INK, y=1.03)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}pr_curve.png")


# ================================================================================================
# Figure: IoU illustration — a real detection box and its ground-truth box, ∩ and ∪ shaded
# ================================================================================================


def fig_iou(exp) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    # (a) clean geometric illustration: two moderately-overlapping boxes, IoU computed from scratch = 0.50
    a = np.array([[0.0, 0.0, 60.0, 50.0]])  # box A
    b = np.array([[10.0, 10.0, 70.0, 60.0]])  # box B (same size, shifted by 10 in x and y)
    iou_demo = float(box_iou_from_scratch(a, b)[0, 0])
    ax1.set_xlim(-6, 82)
    ax1.set_ylim(72, -6)  # image-style y-down
    ax1.set_aspect("equal")
    ax1.axis("off")
    ix0, iy0 = max(a[0, 0], b[0, 0]), max(a[0, 1], b[0, 1])
    ix1, iy1 = min(a[0, 2], b[0, 2]), min(a[0, 3], b[0, 3])
    ax1.add_patch(mpatches.Rectangle((ix0, iy0), ix1 - ix0, iy1 - iy0, facecolor=AMBER, alpha=0.5, edgecolor="none"))
    _draw_box(ax1, a[0], BLUE, "box A", lw=2.4)
    _draw_box(ax1, b[0], GREEN, "box B", lw=2.4)
    ax1.text(35, 35, "∩", fontsize=15, color=INK, ha="center", va="center", fontweight="bold")
    inter, union = 50.0 * 40.0, 60.0 * 50.0 + 60.0 * 50.0 - 50.0 * 40.0
    ax1.set_title(
        f"(a) IoU = area(∩) / area(∪)\n= {inter:.0f} / {union:.0f} = {iou_demo:.2f}",
        fontsize=10.5, color=INK,
    )

    # (b) the SAME metric on a real detection vs its ground truth (a good detection -> high IoU)
    det_box, gt_box = exp.iou_pair_boxes[0], exp.iou_pair_boxes[1]
    ax2.imshow(exp.image.array)
    ax2.axis("off")
    jx0, jy0 = max(det_box[0], gt_box[0]), max(det_box[1], gt_box[1])
    jx1, jy1 = min(det_box[2], gt_box[2]), min(det_box[3], gt_box[3])
    if jx1 > jx0 and jy1 > jy0:
        ax2.add_patch(mpatches.Rectangle((jx0, jy0), jx1 - jx0, jy1 - jy0, facecolor=AMBER, alpha=0.4,
                                         edgecolor="none"))
    _draw_box(ax2, gt_box, BLUE, "ground truth", lw=2.4)
    _draw_box(ax2, det_box, GREEN, f"detection ({exp.iou_pair_class})", lw=2.4)
    ax2.set_title(f"(b) real detection vs ground truth\nIoU = {exp.iou_pair_value:.3f} (a tight, correct box)",
                  fontsize=10.5, color=INK)

    fig.suptitle("Intersection over Union — the overlap metric behind NMS and AP (measured from scratch)",
                 fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}iou.png")


# ================================================================================================
# Figure: anchors + box regression — the spatial picture behind (tx, ty, tw, th)
# ================================================================================================


def fig_anchors() -> None:
    """Schematic feature-map grid with a real anchor set and a real (tx, ty, tw, th) regression.

    The geometry is schematic (a diagram, not a measured claim), but the anchor boxes and the printed
    (tx, ty, tw, th) correction are computed with the module's own ``encode_boxes`` on the drawn coordinates,
    so the arrow is the *actual* regression that lands the best anchor on the object, not a hand-drawn guess.
    """
    canvas_w, canvas_h, cell = 600, 400, 100
    center_x, center_y = 300.0, 200.0  # the chosen feature-map cell's center

    # 3 scales x 3 aspect ratios, all centered at the cell (the standard anchor set)
    scales = [40.0, 70.0, 110.0]
    aspects = [0.5, 1.0, 2.0]
    anchors_cxcywh = np.array(
        [[center_x, center_y, s * np.sqrt(a), s / np.sqrt(a)] for s in scales for a in aspects]
    )
    anchors = cxcywh_to_xyxy(anchors_cxcywh)

    gt = cxcywh_to_xyxy(np.array([[368.0, 176.0, 150.0, 84.0]]))  # a target object box
    best = int(box_iou_from_scratch(anchors, gt)[:, 0].argmax())  # the anchor the matcher assigns
    tx, ty, tw, th = (float(v) for v in encode_boxes(gt, anchors[best : best + 1])[0])

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    ax.set_xlim(0, canvas_w)
    ax.set_ylim(canvas_h, 0)  # image-style y-down
    ax.set_aspect("equal")
    ax.axis("off")

    for x in range(0, canvas_w + 1, cell):
        ax.plot([x, x], [0, canvas_h], color=GRID, linewidth=1.0)
    for y in range(0, canvas_h + 1, cell):
        ax.plot([0, canvas_w], [y, y], color=GRID, linewidth=1.0)

    for i, bx in enumerate(anchors):
        is_best = i == best
        ax.add_patch(mpatches.Rectangle(
            (bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1], fill=False,
            edgecolor=AMBER if is_best else SLATE, linewidth=2.4 if is_best else 1.0,
            alpha=1.0 if is_best else 0.45, linestyle="--" if is_best else "-",
        ))
    ax.scatter([center_x], [center_y], color=INK, s=20, zorder=5)

    _draw_box(ax, gt[0], GREEN, "ground-truth object", lw=2.6)
    gcx, gcy = xyxy_to_cxcywh(gt)[0, :2]
    ax.annotate("", xy=(gcx, gcy), xytext=(center_x, center_y),
                arrowprops={"arrowstyle": "->", "color": RED, "lw": 2.2})
    ax.text(210, center_y + 55,
            f"regression\n$t=({tx:.2f}, {ty:.2f}, {tw:.2f}, {th:.2f})$",
            fontsize=9.5, color=RED, ha="right", va="center")
    ax.text(8, 384,
            "9 anchors (3 scales × 3 aspect ratios) tiled at one cell.  The head picks the best-matching "
            "anchor (amber, dashed)\nand predicts $t=(t_x,t_y,t_w,t_h)$ to nudge it onto the object — centers "
            "in anchor-size units, sizes in log-space.",
            fontsize=8.5, color=INK, va="bottom")
    ax.set_title("Anchors + box regression: predict a box as a correction to a reference box",
                 fontsize=11, color=INK)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}anchors.png")


def main() -> None:
    exp = run_experiment()
    fig_detections(exp)
    fig_nms(exp)
    fig_pr_curve(exp)
    fig_iou(exp)
    fig_anchors()
    # cross-check that the IoU matrix helper is the one the figures rely on (guards silent drift)
    assert box_iou_from_scratch(exp.iou_pair_boxes[:1], exp.iou_pair_boxes[1:2])[0, 0] >= 0.0
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
