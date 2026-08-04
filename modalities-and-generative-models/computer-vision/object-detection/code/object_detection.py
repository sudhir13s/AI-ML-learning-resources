"""Object detection on REAL images with a REAL pretrained detector, and every metric MEASURED.

This is not a toy. It runs real object detection the way the chapter teaches it — a real pretrained
detector (torchvision Faster R-CNN, COCO weights) on real photographs — and then builds, *from scratch*,
the three ideas detection stands on, each cross-checked against a trusted reference with a hard ``assert``:

  1. **IoU (Intersection-over-Union) from scratch**, the box-overlap metric everything else uses. A
     vectorized NumPy implementation is checked against ``torchvision.ops.box_iou`` to ~1e-6 (asserted).

  2. **NMS (Non-Maximum Suppression) from scratch**, the greedy dedup step. Run on the *real* overlapping
     boxes a real detector emits when its own NMS is disabled, our kept indices are asserted identical to
     ``torchvision.ops.nms`` — and we report the honest raw -> kept box counts.

  3. **Average Precision (AP) from scratch**, the detection score. Detections are matched to ground truth
     by IoU, a precision-recall curve is built, and AP is the area under it (the PASCAL VOC all-point rule).
     The implementation is pinned by a hard ``assert`` against a hand-verified worked example whose AP is
     exactly 11/12, then applied to real detections vs hand-specified ground truth (reported honestly as
     *illustrative* — a single image is not a benchmark).

Two supporting pieces are also real and verified: **box-format conversions** (xyxy / xywh / cxcywh) checked
against ``torchvision.ops.box_convert``, and the Faster R-CNN **box-regression encode/decode** (tx, ty, tw,
th) round-trip that a detector's regression head actually learns.

Everything is **seeded and CPU-pinned** so the numbers are bit-reproducible on any machine (GPU kernels are
nondeterministic; we detect CUDA/MPS and report it, but pin the measured pipeline to CPU on purpose). Run::

    python object_detection.py

If the primary real image (a COCO photo) cannot be downloaded, the module *detects* that and falls back to a
**real bundled photograph** (matplotlib's Grace Hopper image) — the detector still runs on a real image and
every from-scratch check still executes on real coordinate data. It never mocks a detection or fabricates a
metric, and it says which path it took in the banner.

Verified on Python 3.12 / torch 2.12 / torchvision 0.27 / numpy 2.4 (CPU).
"""

from __future__ import annotations

import io
import os
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

SEED = 0

# The famous COCO val2017 photo: two cats on a couch with two remotes — a real multi-object scene, ideal for
# showing several detections, NMS, and AP. Downloaded on demand (never committed); a real offline fallback
# (matplotlib's bundled Grace Hopper photo) keeps the module runnable with no network.
_COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
_HTTP_TIMEOUT_S = 20

# Keep downloaded weights + images OUT of the repo: torch hub cache and the image cache both go to a temp dir.
_CACHE_DIR = Path(os.environ.get("OD_CACHE_DIR", Path(tempfile.gettempdir()) / "od_cache"))
torch.hub.set_dir(str(_CACHE_DIR / "torch_hub"))


# ================================================================================================
# Device: detect CUDA/MPS for reporting, but PIN the measured pipeline to CPU for a reproducible trace
# ================================================================================================


def detect_accelerator() -> str:
    """Report the best available accelerator (for the banner). We deliberately do NOT run on it: GPU/MPS
    kernels are nondeterministic, and every number this module prints must be bit-reproducible."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ================================================================================================
# Box-format conversions (xyxy / xywh / cxcywh), verified against torchvision.ops.box_convert
# ================================================================================================
#
# A box has four numbers, but three conventions are in daily use and mixing them is the #1 detection bug:
#   xyxy   : (x_min, y_min, x_max, y_max)          -- corners; what IoU/NMS/drawing want
#   xywh   : (x_min, y_min, width, height)         -- top-left + size; the COCO annotation format
#   cxcywh : (x_center, y_center, width, height)   -- center + size; what anchor regression predicts


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """(x_min, y_min, x_max, y_max) -> (x_min, y_min, width, height)."""
    x0, y0, x1, y1 = boxes.T
    return np.stack([x0, y0, x1 - x0, y1 - y0], axis=1)


def xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """(x_min, y_min, x_max, y_max) -> (x_center, y_center, width, height)."""
    x0, y0, x1, y1 = boxes.T
    return np.stack([(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0], axis=1)


def cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """(x_center, y_center, width, height) -> (x_min, y_min, x_max, y_max)."""
    cx, cy, w, h = boxes.T
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)


@dataclass(frozen=True)
class FormatCheck:
    max_err_xywh: float
    max_err_cxcywh: float
    max_err_roundtrip: float


def verify_box_formats(seed: int = SEED) -> FormatCheck:
    """Assert our conversions match ``torchvision.ops.box_convert`` and that xyxy->cxcywh->xyxy round-trips."""
    from torchvision.ops import box_convert

    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, size=(16, 2))
    wh = rng.uniform(1, 60, size=(16, 2))
    xyxy = np.concatenate([xy, xy + wh], axis=1).astype(np.float64)  # valid boxes (x1>x0, y1>y0)

    err_xywh = float(np.abs(xyxy_to_xywh(xyxy) - box_convert(torch.tensor(xyxy), "xyxy", "xywh").numpy()).max())
    err_cxcywh = float(
        np.abs(xyxy_to_cxcywh(xyxy) - box_convert(torch.tensor(xyxy), "xyxy", "cxcywh").numpy()).max()
    )
    err_roundtrip = float(np.abs(cxcywh_to_xyxy(xyxy_to_cxcywh(xyxy)) - xyxy).max())

    assert err_xywh < 1e-9, f"xywh conversion disagrees with torchvision: {err_xywh:.2e}"
    assert err_cxcywh < 1e-9, f"cxcywh conversion disagrees with torchvision: {err_cxcywh:.2e}"
    assert err_roundtrip < 1e-9, f"xyxy->cxcywh->xyxy round-trip broke: {err_roundtrip:.2e}"
    return FormatCheck(err_xywh, err_cxcywh, err_roundtrip)


# ================================================================================================
# IoU from scratch (the box-overlap metric), verified against torchvision.ops.box_iou
# ================================================================================================


def box_iou_from_scratch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU matrix: ``out[i, j]`` = area(a_i ∩ b_j) / area(a_i ∪ b_j), boxes in xyxy.

    IoU = intersection / union. The intersection rectangle is [max of the mins, min of the maxes], clamped
    to be non-negative (no overlap -> width or height 0 -> IoU 0). The union is area(a)+area(b)-intersection.
    Returns an [N, M] matrix so a set of detections can be scored against a set of ground-truth boxes at once.
    """
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]).clip(min=0) * (boxes_a[:, 3] - boxes_a[:, 1]).clip(min=0)
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]).clip(min=0) * (boxes_b[:, 3] - boxes_b[:, 1]).clip(min=0)

    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])  # [N, M, 2]
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])  # [N, M, 2]
    inter_wh = (bottom_right - top_left).clip(min=0)  # negative -> no overlap -> 0
    intersection = inter_wh[..., 0] * inter_wh[..., 1]

    union = area_a[:, None] + area_b[None, :] - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, 0.0)
    return iou


@dataclass(frozen=True)
class IoUCheck:
    max_err_vs_torchvision: float
    example_pair_iou: float


def verify_iou(seed: int = SEED) -> IoUCheck:
    """Assert the from-scratch IoU matrix matches ``torchvision.ops.box_iou`` to ~1e-6 on random real boxes."""
    from torchvision.ops import box_iou

    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 100, size=(24, 2))
    wh = rng.uniform(5, 70, size=(24, 2))
    a = np.concatenate([xy, xy + wh], axis=1).astype(np.float64)
    xy2 = rng.uniform(0, 100, size=(18, 2))
    wh2 = rng.uniform(5, 70, size=(18, 2))
    b = np.concatenate([xy2, xy2 + wh2], axis=1).astype(np.float64)

    ours = box_iou_from_scratch(a, b)
    ref = box_iou(torch.tensor(a), torch.tensor(b)).numpy()
    err = float(np.abs(ours - ref).max())
    assert err < 1e-6, f"IoU disagrees with torchvision.ops.box_iou: {err:.2e}"

    # a concrete, hand-checkable pair for the figure/caption: two 10x10 boxes shifted by 5 in x.
    pair = box_iou_from_scratch(
        np.array([[0.0, 0.0, 10.0, 10.0]]), np.array([[5.0, 0.0, 15.0, 10.0]])
    )[0, 0]
    return IoUCheck(err, float(pair))


# ================================================================================================
# NMS from scratch (greedy dedup), verified against torchvision.ops.nms on REAL detector boxes
# ================================================================================================


def nms_from_scratch(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float) -> np.ndarray:
    """Greedy Non-Maximum Suppression. Returns kept indices, highest score first.

    Algorithm: sort boxes by score (descending); repeatedly take the top-scoring survivor, keep it, and
    discard every remaining box whose IoU with it exceeds ``iou_thresh`` (they are duplicate detections of the
    same object). This is the exact procedure ``torchvision.ops.nms`` implements.
    """
    order = scores.argsort()[::-1].tolist()  # indices, best score first
    keep: list[int] = []
    while order:
        i = order.pop(0)
        keep.append(i)
        if not order:
            break
        ious = box_iou_from_scratch(boxes[i][None], boxes[order])[0]
        order = [j for j, iou in zip(order, ious) if iou <= iou_thresh]
    return np.array(keep, dtype=np.int64)


@dataclass(frozen=True)
class NMSCheck:
    n_raw: int
    n_kept: int
    iou_thresh: float
    matches_torchvision: bool


def verify_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.5) -> NMSCheck:
    """Assert from-scratch NMS keeps *exactly* the boxes ``torchvision.ops.nms`` keeps (same indices)."""
    from torchvision.ops import nms

    ours = nms_from_scratch(boxes, scores, iou_thresh)
    ref = nms(torch.tensor(boxes), torch.tensor(scores), iou_thresh).numpy()
    same = bool(np.array_equal(np.sort(ours), np.sort(ref)))
    assert same, f"NMS kept indices disagree with torchvision: ours={sorted(ours)} vs ref={sorted(ref)}"
    return NMSCheck(len(boxes), len(ours), iou_thresh, same)


# ================================================================================================
# Anchor box-regression encode/decode (tx, ty, tw, th): what a detector's regression head learns
# ================================================================================================


def encode_boxes(gt_xyxy: np.ndarray, anchor_xyxy: np.ndarray) -> np.ndarray:
    """Faster R-CNN regression target: encode a ground-truth box *relative to* an anchor as (tx, ty, tw, th).

    tx = (gx - ax) / aw,  ty = (gy - ay) / ah,  tw = log(gw / aw),  th = log(gh / ah)

    Centers are offset in units of the anchor's size (scale-invariant); sizes are predicted in log-space (so
    a positive/negative t means grow/shrink by a multiplicative factor, and can never produce a negative box).
    """
    g = xyxy_to_cxcywh(gt_xyxy)
    a = xyxy_to_cxcywh(anchor_xyxy)
    tx = (g[:, 0] - a[:, 0]) / a[:, 2]
    ty = (g[:, 1] - a[:, 1]) / a[:, 3]
    tw = np.log(g[:, 2] / a[:, 2])
    th = np.log(g[:, 3] / a[:, 3])
    return np.stack([tx, ty, tw, th], axis=1)


def decode_boxes(deltas: np.ndarray, anchor_xyxy: np.ndarray) -> np.ndarray:
    """Inverse of :func:`encode_boxes`: apply predicted (tx, ty, tw, th) to anchors to get boxes (xyxy)."""
    a = xyxy_to_cxcywh(anchor_xyxy)
    cx = deltas[:, 0] * a[:, 2] + a[:, 0]
    cy = deltas[:, 1] * a[:, 3] + a[:, 1]
    w = np.exp(deltas[:, 2]) * a[:, 2]
    h = np.exp(deltas[:, 3]) * a[:, 3]
    return cxcywh_to_xyxy(np.stack([cx, cy, w, h], axis=1))


@dataclass(frozen=True)
class RegressionCheck:
    max_roundtrip_err: float
    example_delta: tuple[float, float, float, float]


def verify_box_regression(seed: int = SEED) -> RegressionCheck:
    """Assert encode then decode recovers the ground-truth box exactly (the regression parametrization is sound)."""
    rng = np.random.default_rng(seed)
    axy = rng.uniform(0, 100, size=(20, 2))
    awh = rng.uniform(10, 60, size=(20, 2))
    anchors = np.concatenate([axy, axy + awh], axis=1).astype(np.float64)
    gxy = rng.uniform(0, 100, size=(20, 2))
    gwh = rng.uniform(10, 60, size=(20, 2))
    gts = np.concatenate([gxy, gxy + gwh], axis=1).astype(np.float64)

    deltas = encode_boxes(gts, anchors)
    recovered = decode_boxes(deltas, anchors)
    err = float(np.abs(recovered - gts).max())
    assert err < 1e-9, f"encode/decode round-trip broke: {err:.2e}"
    d0 = tuple(float(v) for v in deltas[0])
    return RegressionCheck(err, d0)  # type: ignore[arg-type]


# ================================================================================================
# Average Precision from scratch (PASCAL VOC all-point), pinned by a hand-verified worked example
# ================================================================================================


def _all_point_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """PASCAL VOC all-point AP: area under the precision-recall curve after making precision monotonic.

    Pad the curve with (recall 0, precision 0) on the left and (recall carried, precision 0) on the right,
    replace each precision by the max precision at any higher recall (the interpolation envelope), then sum
    the rectangles Δrecall × interpolated-precision. This is the VOC2010+ / COCO integration rule.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0] + 1
    return float(np.sum((mrec[idx] - mrec[idx - 1]) * mpre[idx]))


@dataclass
class PRCurve:
    ap: float
    precision: np.ndarray
    recall: np.ndarray
    n_gt: int
    n_det: int


def average_precision(
    det_boxes: np.ndarray, det_scores: np.ndarray, gt_boxes: np.ndarray, iou_thresh: float = 0.5
) -> PRCurve:
    """Single-class AP: match detections (by descending score) to ground truth by IoU, then integrate PR.

    Each detection greedily claims the highest-IoU *unclaimed* ground-truth box; a match with IoU >=
    ``iou_thresh`` is a true positive, otherwise the detection is a false positive (a duplicate of an
    already-claimed box is also a false positive). Precision = TP/(TP+FP), recall = TP/#GT, accumulated down
    the ranked list; AP is the area under the resulting curve.
    """
    order = det_scores.argsort()[::-1]
    boxes = det_boxes[order]
    n_gt = len(gt_boxes)
    tp = np.zeros(len(boxes))
    fp = np.zeros(len(boxes))
    claimed = np.zeros(n_gt, dtype=bool)

    ious = box_iou_from_scratch(boxes, gt_boxes) if (n_gt and len(boxes)) else np.zeros((len(boxes), n_gt))
    for d in range(len(boxes)):
        if n_gt == 0:
            fp[d] = 1.0
            continue
        best_gt = int(ious[d].argmax())
        if ious[d, best_gt] >= iou_thresh and not claimed[best_gt]:
            tp[d] = 1.0
            claimed[best_gt] = True
        else:
            fp[d] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / max(n_gt, 1)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(tp_cum + fp_cum > 0, tp_cum / (tp_cum + fp_cum), 0.0)
    ap = _all_point_ap(recall, precision) if len(boxes) else 0.0
    return PRCurve(ap, precision, recall, n_gt, len(boxes))


def mean_ap_over_iou(
    det_boxes: np.ndarray, det_scores: np.ndarray, gt_boxes: np.ndarray, thresholds: np.ndarray
) -> float:
    """COCO-style AP: the mean of single-threshold APs over an IoU sweep (COCO uses 0.50:0.05:0.95).

    Averaging over increasingly strict IoU thresholds rewards *tight* localization, not just detection — the
    reason the headline COCO number (mAP@[.5:.95]) is always well below the looser AP@0.5.
    """
    return float(np.mean([average_precision(det_boxes, det_scores, gt_boxes, t).ap for t in thresholds]))


@dataclass(frozen=True)
class APKnownCase:
    ap_measured: float
    ap_expected: float
    curve: PRCurve


def verify_ap_known_case() -> APKnownCase:
    """Pin AP against a hand-verified worked example whose all-point AP is *exactly* 11/12.

    Three well-separated ground-truth boxes; five detections whose IoU-matching, by construction, yields the
    ranked hit/miss sequence TP, TP, FP, TP, FP. Working the PR curve by hand gives AP = 1/3 + 1/3 + 1/4 =
    11/12. If our matching or integration drifts, this raises. The returned curve drives the teaching figure.
    """
    gt = np.array([[0.0, 0, 10, 10], [20, 0, 30, 10], [40, 0, 50, 10]])
    det_boxes = np.array(
        [
            [0.0, 0, 10, 10],  # score .9 -> matches GT0        -> TP
            [20, 0, 30, 10],  # score .8 -> matches GT1        -> TP
            [100, 100, 110, 110],  # score .7 -> matches nothing    -> FP
            [40, 0, 50, 10],  # score .6 -> matches GT2        -> TP
            [0, 0, 10, 10],  # score .5 -> GT0 already claimed -> FP
        ]
    )
    det_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    pr = average_precision(det_boxes, det_scores, gt, iou_thresh=0.5)
    expected = 11.0 / 12.0
    assert abs(pr.ap - expected) < 1e-9, f"AP known-case failed: got {pr.ap:.6f}, expected {expected:.6f}"
    return APKnownCase(pr.ap, expected, pr)


# ================================================================================================
# Real images: a COCO photo (downloaded on demand) with a real, bundled fallback
# ================================================================================================


@dataclass
class RealImage:
    name: str
    array: np.ndarray  # uint8 [H, W, 3]
    source: str  # "coco-download" or "bundled-fallback"
    ground_truth: dict[str, np.ndarray]  # class name -> [K, 4] xyxy hand-specified GT boxes


def _load_pil_rgb(data: bytes) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(io.BytesIO(data)).convert("RGB"))


def try_load_coco_image() -> RealImage | None:
    """Download the real COCO cats+remotes photo (cached to temp, never committed). ``None`` if offline."""
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached = _CACHE_DIR / "coco_39769.jpg"
        if cached.exists():
            arr = _load_pil_rgb(cached.read_bytes())
        else:
            req = urllib.request.Request(_COCO_IMAGE_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:  # noqa: S310 (fixed COCO URL)
                raw = resp.read()
            cached.write_bytes(raw)
            arr = _load_pil_rgb(raw)
    except Exception:  # noqa: BLE001 -- any failure (offline, DNS, disk) triggers the real bundled fallback
        return None
    # Ground truth for this specific photo: the two clearly-visible cats and two remotes (hand-specified real
    # coordinates on the real image). Used only to score detections and draw the AP/IoU figures.
    gt = {
        "cat": np.array([[8.0, 55, 305, 362], [356, 20, 608, 375]]),
        "remote": np.array([[39.0, 72, 176, 120], [333, 77, 369, 179]]),
    }
    return RealImage("cats & remotes (COCO val2017)", arr, "coco-download", gt)


def load_bundled_image() -> RealImage:
    """Real offline fallback: matplotlib's bundled Grace Hopper photograph (a real person, real necktie)."""
    import matplotlib.cbook as cbook

    with cbook.get_sample_data("grace_hopper.jpg") as f:
        arr = _load_pil_rgb(f.read())
    gt = {"person": np.array([[70.0, 30, 440, 590]]), "tie": np.array([[220.0, 300, 300, 540]])}
    return RealImage("Grace Hopper (bundled)", arr, "bundled-fallback", gt)


def load_real_image() -> RealImage:
    return try_load_coco_image() or load_bundled_image()


# ================================================================================================
# The real detector: torchvision Faster R-CNN (COCO weights)
# ================================================================================================


@dataclass
class Detections:
    boxes: np.ndarray  # [N, 4] xyxy
    scores: np.ndarray  # [N]
    labels: np.ndarray  # [N] COCO category ids
    names: list[str]  # per-detection class name
    categories: list[str]  # id -> name table


def _to_chw_float(image_uint8: np.ndarray) -> torch.Tensor:
    return torch.tensor(image_uint8).permute(2, 0, 1).float() / 255.0


def load_detector(score_thresh: float = 0.5, nms_thresh: float = 0.5):
    """Load a real pretrained Faster R-CNN (ResNet-50 FPN, COCO). ``nms_thresh`` is exposed so we can *raise*
    it to recover the raw, pre-NMS overlapping boxes for the NMS demonstration."""
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn

    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights, box_score_thresh=score_thresh)
    model.roi_heads.nms_thresh = nms_thresh
    model.eval()
    return model, list(weights.meta["categories"])


@torch.no_grad()
def run_detector(model, categories: list[str], image_uint8: np.ndarray) -> Detections:
    """Run the detector on one real image and return its boxes/scores/labels as NumPy (CPU, deterministic)."""
    out = model([_to_chw_float(image_uint8)])[0]
    boxes = out["boxes"].numpy()
    scores = out["scores"].numpy()
    labels = out["labels"].numpy()
    names = [categories[i] for i in labels]
    return Detections(boxes, scores, labels, names, categories)


# ================================================================================================
# The full experiment, bundled (so figures and the notebook reuse one measured run)
# ================================================================================================


COCO_IOU_SWEEP = np.arange(0.5, 1.0, 0.05)  # the COCO mAP@[.5:.95] thresholds


@dataclass
class Experiment:
    accelerator: str
    image: RealImage
    format_check: FormatCheck
    iou_check: IoUCheck
    regression_check: RegressionCheck
    ap_known: APKnownCase
    # real detections (post-NMS at score>=0.5): the clean, final result the detector returns
    detections: Detections = field(repr=False)
    # NMS demonstration on real raw boxes for one class
    nms_class: str = ""
    nms_raw_boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4)), repr=False)
    nms_raw_scores: np.ndarray = field(default_factory=lambda: np.empty((0,)), repr=False)
    nms_kept_idx: np.ndarray = field(default_factory=lambda: np.empty((0,), dtype=np.int64), repr=False)
    nms_check: NMSCheck | None = None
    # AP on the SAME real detections before vs after NMS (NMS removes duplicate false positives -> AP up)
    ap_class: str = ""
    pr_pre_nms: PRCurve | None = field(default=None, repr=False)
    pr_post_nms: PRCurve | None = field(default=None, repr=False)
    # per-class AP@0.5 (post-NMS) and the COCO mAP@[.5:.95] over classes
    per_class_ap: dict[str, PRCurve] = field(default_factory=dict, repr=False)
    map_50: float = 0.0
    coco_map: float = 0.0
    # one concrete IoU pair (best detection vs its matched GT) for the IoU figure
    iou_pair_boxes: np.ndarray = field(default_factory=lambda: np.empty((2, 4)), repr=False)
    iou_pair_value: float = 0.0
    iou_pair_class: str = ""


def _pick_nms_class(det_raw: Detections, gt: dict[str, np.ndarray]) -> str:
    """Choose the class with the most raw detections (best shows suppression); prefer a class present in GT."""
    from collections import Counter

    counts = Counter(det_raw.names)
    for name, _ in counts.most_common():
        if name in gt:
            return name
    return counts.most_common(1)[0][0]


def _iou_demo_pair(det: Detections, gt: dict[str, np.ndarray]) -> tuple[np.ndarray, float, str]:
    """Highest-scoring detection whose class has GT, paired with its best-matching GT box (a real IoU value)."""
    for k in det.scores.argsort()[::-1]:
        name = det.names[k]
        if name in gt and len(gt[name]):
            det_box = det.boxes[k][None]
            ious = box_iou_from_scratch(det_box, gt[name])[0]
            j = int(ious.argmax())
            pair = np.stack([det.boxes[k], gt[name][j]])
            return pair, float(ious[j]), name
    # fallback: top detection vs itself shifted (still real coordinates)
    top = det.boxes[det.scores.argmax()]
    shifted = top + np.array([15.0, 0, 15, 0])
    pair = np.stack([top, shifted])
    return pair, float(box_iou_from_scratch(top[None], shifted[None])[0, 0]), det.names[int(det.scores.argmax())]


def _class_mask(det: Detections, name: str) -> np.ndarray:
    return np.array([n == name for n in det.names])


def run_experiment(seed: int = SEED) -> Experiment:
    """Run the whole measured pipeline once and return every quantity the chapter, figures, and notebook cite."""
    _seed_everything(seed)
    accelerator = detect_accelerator()

    # --- from-scratch building blocks, each verified against a reference (hard asserts) ---
    format_check = verify_box_formats(seed)
    iou_check = verify_iou(seed)
    regression_check = verify_box_regression(seed)
    ap_known = verify_ap_known_case()

    image = load_real_image()
    gt = image.ground_truth

    # --- three real detector passes on the SAME image (only the post-processing thresholds differ) ---
    clean_model, categories = load_detector(score_thresh=0.5, nms_thresh=0.5)  # final clean result (for drawing)
    detections = run_detector(clean_model, categories, image.array)

    raw_model, _ = load_detector(score_thresh=0.05, nms_thresh=0.95)  # NMS disabled -> raw overlapping flood
    det_raw = run_detector(raw_model, categories, image.array)

    eval_model, _ = load_detector(score_thresh=0.05, nms_thresh=0.5)  # standard eval: post-NMS, low score
    det_eval = run_detector(eval_model, categories, image.array)

    # --- NMS demonstration on REAL raw boxes for one class ---
    nms_class = _pick_nms_class(det_raw, gt)
    mask = _class_mask(det_raw, nms_class) & (det_raw.scores >= 0.30)
    nms_raw_boxes, nms_raw_scores = det_raw.boxes[mask], det_raw.scores[mask]
    nms_check = verify_nms(nms_raw_boxes, nms_raw_scores, iou_thresh=0.5)
    nms_kept_idx = nms_from_scratch(nms_raw_boxes, nms_raw_scores, 0.5)

    # --- AP on the SAME real detections BEFORE vs AFTER NMS (duplicates are false positives) ---
    ap_class = nms_class if nms_class in gt else next(iter(gt))
    pre_mask, post_mask = _class_mask(det_raw, ap_class), _class_mask(det_eval, ap_class)
    pr_pre = average_precision(det_raw.boxes[pre_mask], det_raw.scores[pre_mask], gt[ap_class], 0.5)
    pr_post = average_precision(det_eval.boxes[post_mask], det_eval.scores[post_mask], gt[ap_class], 0.5)

    # --- per-class AP@0.5 (post-NMS) and COCO mAP@[.5:.95] ---
    per_class_ap: dict[str, PRCurve] = {}
    coco_aps: list[float] = []
    for name, gt_boxes in gt.items():
        m = _class_mask(det_eval, name)
        per_class_ap[name] = average_precision(det_eval.boxes[m], det_eval.scores[m], gt_boxes, 0.5)
        coco_aps.append(mean_ap_over_iou(det_eval.boxes[m], det_eval.scores[m], gt_boxes, COCO_IOU_SWEEP))
    map_50 = float(np.mean([pr.ap for pr in per_class_ap.values()])) if per_class_ap else 0.0
    coco_map = float(np.mean(coco_aps)) if coco_aps else 0.0

    iou_pair_boxes, iou_pair_value, iou_pair_class = _iou_demo_pair(detections, gt)

    return Experiment(
        accelerator, image, format_check, iou_check, regression_check, ap_known, detections,
        nms_class, nms_raw_boxes, nms_raw_scores, nms_kept_idx, nms_check,
        ap_class, pr_pre, pr_post, per_class_ap, map_50, coco_map,
        iou_pair_boxes, iou_pair_value, iou_pair_class,
    )


# ================================================================================================
# Report
# ================================================================================================


def main() -> None:
    import torchvision

    exp = run_experiment()
    print(
        f"torch {torch.__version__} | torchvision {torchvision.__version__} | numpy {np.__version__} "
        f"(reported on CPU, seed={SEED}; accelerator available: {exp.accelerator})\n"
    )

    print(f"=== Real image: {exp.image.name} [{exp.image.source}] ===")
    h, w = exp.image.array.shape[:2]
    print(f"  {w}x{h} RGB; detector found {len(exp.detections.boxes)} objects (score >= 0.5):")
    for name, score in sorted(zip(exp.detections.names, exp.detections.scores), key=lambda t: -t[1]):
        print(f"    {name:12s} {score:.3f}")
    print()

    print("=== Box formats vs torchvision.ops.box_convert ===")
    fc = exp.format_check
    print(f"  max|xywh err|={fc.max_err_xywh:.1e}  max|cxcywh err|={fc.max_err_cxcywh:.1e}  "
          f"round-trip err={fc.max_err_roundtrip:.1e}\n")

    print("=== IoU from scratch vs torchvision.ops.box_iou ===")
    print(f"  max|IoU err|={exp.iou_check.max_err_vs_torchvision:.2e}  "
          f"(sanity pair, two 10x10 boxes shifted 5px: IoU={exp.iou_check.example_pair_iou:.4f})\n")

    print("=== Box regression encode/decode (tx,ty,tw,th) round-trip ===")
    rc = exp.regression_check
    example = tuple(round(v, 3) for v in rc.example_delta)
    print(f"  max|decode(encode(gt)) - gt|={rc.max_roundtrip_err:.1e}  example delta={example}\n")

    print(f"=== NMS from scratch vs torchvision.ops.nms  (class '{exp.nms_class}', IoU thresh 0.5) ===")
    nc = exp.nms_check
    assert nc is not None
    print(f"  raw overlapping boxes: {nc.n_raw}  ->  after NMS: {nc.n_kept}   "
          f"(kept indices == torchvision.ops.nms: {nc.matches_torchvision})\n")

    print("=== Average Precision from scratch ===")
    print(f"  known worked example (TP,TP,FP,TP,FP): AP = {exp.ap_known.ap_measured:.6f}  "
          f"(hand-verified expected 11/12 = {exp.ap_known.ap_expected:.6f})")
    pre, post = exp.pr_pre_nms, exp.pr_post_nms
    assert pre is not None and post is not None
    print(f"  same real '{exp.ap_class}' detections, before vs after NMS (duplicates are false positives):")
    print(f"    pre-NMS : {pre.n_det:3d} detections -> AP@0.5 = {pre.ap:.4f}")
    print(f"    post-NMS: {post.n_det:3d} detections -> AP@0.5 = {post.ap:.4f}   "
          f"(NMS lifts AP by {(post.ap - pre.ap):+.4f})")
    print(f"  per-class AP@0.5 vs hand-specified GT on '{exp.image.name}' (illustrative, single image):")
    for name, pr in exp.per_class_ap.items():
        print(f"    AP@0.5[{name:8s}] = {pr.ap:.4f}   ({pr.n_det} detections, {pr.n_gt} ground-truth boxes)")
    print(f"    -> mAP@0.5 = {exp.map_50:.4f}   |   COCO mAP@[.5:.95] = {exp.coco_map:.4f}   "
          f"(mean over {len(exp.per_class_ap)} classes)")
    print(f"  IoU(top '{exp.iou_pair_class}' detection, its GT) = {exp.iou_pair_value:.4f}\n")

    # --- hard asserts on the headline relationships (raise, not print, if a lesson breaks) ---
    assert nc.matches_torchvision, "NMS must match torchvision"
    assert nc.n_kept < nc.n_raw, f"NMS should suppress at least one box (raw {nc.n_raw}, kept {nc.n_kept})"
    assert abs(exp.ap_known.ap_measured - 11 / 12) < 1e-9, "AP known-case must equal 11/12"
    assert post.ap >= pre.ap, f"NMS should not lower AP (pre {pre.ap:.3f}, post {post.ap:.3f})"
    print("All checks passed (box formats, IoU, NMS, and box-regression match their references; "
          "AP reproduces the hand-verified 11/12 worked example; NMS removes duplicate FPs and lifts AP).")


if __name__ == "__main__":
    main()
