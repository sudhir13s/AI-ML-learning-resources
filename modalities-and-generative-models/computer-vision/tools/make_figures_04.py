"""Figure generator for 04-Image-Classification — every quantitative figure from the REAL run in
``image_classification.py``.

One measured experiment (``run_experiment``) drives every figure below, so nothing quantitative is hand-typed:
the transfer-vs-from-scratch bars, the confusion matrix, per-class accuracy, the augmentation ablation, and
the honest sample grid all come from the same executed pipeline the chapter and notebook use. The augmentation
*examples* panel applies the chapter's real augmentation transform to a real image.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``cv04_``:

  cv04_sample_grid.png        -- real CIFAR-10 test images with true -> predicted labels; correct in green,
                                 wrong in red (honest: includes real misclassifications). Transfer model.
  cv04_transfer_vs_scratch.png-- top-1 (and top-5) accuracy: a frozen pretrained backbone + linear head vs a
                                 from-scratch CNN at the SAME data budget. The measured transfer win.
  cv04_confusion_matrix.png   -- 10x10 confusion matrix of the transfer model on 1000 real test images.
  cv04_per_class.png          -- per-class accuracy, sorted: vehicles classify best, fine-grained animals worst.
  cv04_augmentation.png       -- augmentation ablation: test-accuracy learning curves and the train-test
                                 generalization gap, with vs without random crop + flip (measured).
  cv04_aug_examples.png       -- one real image and several label-preserving augmentations of it.

    python make_figures_04.py

Verified on Python 3.12 / matplotlib 3.10 / torch 2.12 / torchvision 0.27 / numpy 2.4 / scikit-learn 1.9.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in ``07. Computer Vision/tools/``; the chapter module it demonstrates stays in that
# chapter's ``code/`` folder. Put that folder on sys.path so the ``image_classification`` import resolves.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "04-Image-Classification" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from image_classification import run_experiment  # noqa: E402  (resolved via the sys.path insert above)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / values
PURPLE = "#5D4A8A"  # process
GREEN = "#2E7A5A"  # good / correct / transfer
RED = "#8B3B4A"  # error / wrong
AMBER = "#7A6528"  # highlight
SLATE = "#4A5B6E"  # neutral / from-scratch
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "cv04_"


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


def _disp_image(img: np.ndarray) -> np.ndarray:
    """uint8 CIFAR image [32,32,3] (or grayscale [8,8]) -> float in [0,1] for imshow."""
    return img.astype(np.float64) / 255.0


# ================================================================================================
# Figure: honest sample grid — real images, true -> predicted, correct green / wrong red
# ================================================================================================


def fig_sample_grid(exp) -> None:
    s = exp.samples
    k = len(s.true_labels)
    cols = 4
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.1))
    for ax in axes.flat:
        ax.axis("off")
    for i in range(k):
        ax = axes.flat[i]
        img = _disp_image(s.images[i])
        ax.imshow(img, cmap=None if img.ndim == 3 else "gray")
        true_name = s.classes[s.true_labels[i]]
        pred_name = s.classes[s.pred_labels[i]]
        correct = s.true_labels[i] == s.pred_labels[i]
        color = GREEN if correct else RED
        mark = "OK" if correct else "X"
        ax.set_title(
            f"true: {true_name}\npred: {pred_name} ({s.confidences[i]:.2f}) {mark}",
            fontsize=8.5, color=color,
        )
    fig.suptitle(
        f"Real CIFAR-10 predictions ({exp.best_name}) — honest: correct in green, wrong in red",
        fontsize=11, color=INK, y=1.005,
    )
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}sample_grid.png")


# ================================================================================================
# Figure: transfer (frozen backbone + linear head) vs from-scratch CNN — the measured win
# ================================================================================================


def fig_transfer_vs_scratch(exp) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    _style_axis(ax)
    labels = ["Transfer\n(frozen ResNet-18\n+ linear head)", "From scratch\n(small CNN,\nrandom init)"]
    top1 = [exp.transfer_top1 * 100, exp.scratch_noaug.test_acc * 100]
    x = np.arange(2)
    bars = ax.bar(x, top1, width=0.55, color=[GREEN, SLATE], edgecolor=INK, linewidth=0.8)
    # top-5 marker for the transfer model (a translucent overlay bar)
    ax.bar(0, exp.transfer_top5 * 100, width=0.55, color=GREEN, alpha=0.22, edgecolor="none")
    ax.text(0, exp.transfer_top5 * 100 + 1.2, f"top-5 {exp.transfer_top5 * 100:.1f}%", ha="center", fontsize=8.5, color=GREEN)
    for b, v in zip(bars, top1):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}%", ha="center", fontsize=11, color=INK, fontweight="bold")
    ax.annotate(
        f"+{exp.transfer_delta * 100:.1f} points\nsame {exp.n_train}-image budget",
        xy=(0.5, (top1[0] + top1[1]) / 2), ha="center", va="center", fontsize=9.5, color=RED,
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": RED, "lw": 1.0},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("top-1 test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Transfer learning vs training from scratch (CIFAR-10, measured)", fontsize=11)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}transfer_vs_scratch.png")


# ================================================================================================
# Figure: confusion matrix of the transfer model
# ================================================================================================


def fig_confusion(exp) -> None:
    cm = exp.confusion
    n = len(exp.classes)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(exp.classes, rotation=45, ha="right", fontsize=8.5, color=INK)
    ax.set_yticklabels(exp.classes, fontsize=8.5, color=INK)
    ax.set_xlabel("predicted class", color=INK)
    ax.set_ylabel("true class", color=INK)
    thresh = cm.max() / 2
    for i in range(n):
        for j in range(n):
            val = cm[i, j]
            if val == 0:
                continue
            ax.text(j, i, str(val), ha="center", va="center", fontsize=7.5,
                    color="white" if val > thresh else INK)
    ax.set_title("Confusion matrix — transfer model, 1000 real test images\n(diagonal = correct; "
                 "off-diagonal = the mistakes)", fontsize=10.5, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=8, colors=INK)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}confusion_matrix.png")


# ================================================================================================
# Figure: per-class accuracy (vehicles best, fine-grained animals worst)
# ================================================================================================


def fig_per_class(exp) -> None:
    order = np.argsort(exp.per_class)
    names = [exp.classes[i] for i in order]
    accs = exp.per_class[order] * 100
    # vehicles vs animals colouring (CIFAR-10: airplane/automobile/ship/truck are vehicles)
    vehicles = {"airplane", "automobile", "ship", "truck"}
    colors = [BLUE if names[i] in vehicles else AMBER for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    _style_axis(ax)
    y = np.arange(len(names))
    ax.barh(y, accs, color=colors, edgecolor=INK, linewidth=0.6)
    for i, v in enumerate(accs):
        ax.text(v + 0.8, i, f"{v:.0f}%", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("per-class accuracy (%)")
    ax.set_xlim(0, 100)
    ax.axvline(exp.transfer_top1 * 100, color=RED, linestyle="--", linewidth=1.2)
    ax.text(exp.transfer_top1 * 100 - 1, 0.2, f"overall {exp.transfer_top1 * 100:.0f}%",
            color=RED, fontsize=8.5, ha="right")
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=AMBER)]
    ax.legend(handles, ["vehicles", "animals"], fontsize=8.5, loc="lower right", frameon=False)
    ax.set_title("Per-class accuracy hides behind the headline number", fontsize=11)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}per_class.png")


# ================================================================================================
# Figure: augmentation ablation — learning curves + the generalization gap
# ================================================================================================


def fig_augmentation(exp) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.3))

    # (a) test-accuracy learning curves
    _style_axis(ax1)
    epochs = np.arange(1, len(exp.scratch_noaug.val_curve) + 1)
    ax1.plot(epochs, np.array(exp.scratch_noaug.val_curve) * 100, color=SLATE, linewidth=2, label="no augmentation")
    ax1.plot(epochs, np.array(exp.scratch_aug.val_curve) * 100, color=GREEN, linewidth=2, label="with augmentation")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("test top-1 accuracy (%)")
    ax1.legend(fontsize=9, frameon=False, loc="lower right")
    ax1.set_title("(a) test accuracy over training", fontsize=10.5)

    # (b) generalization gap = train acc - test acc
    _style_axis(ax2)
    groups = ["no aug", "with aug"]
    train = [exp.scratch_noaug.train_acc * 100, exp.scratch_aug.train_acc * 100]
    test = [exp.scratch_noaug.test_acc * 100, exp.scratch_aug.test_acc * 100]
    x = np.arange(2)
    w = 0.36
    ax2.bar(x - w / 2, train, w, color=BLUE, edgecolor=INK, linewidth=0.7, label="train acc")
    ax2.bar(x + w / 2, test, w, color=GREEN, edgecolor=INK, linewidth=0.7, label="test acc")
    for xi, tr, te in zip(x, train, test):
        gap = tr - te
        ax2.annotate("", xy=(xi, tr), xytext=(xi, te), arrowprops={"arrowstyle": "<->", "color": RED, "lw": 1.3})
        ax2.text(xi + 0.02, (tr + te) / 2, f"gap\n{gap:.1f} pts", color=RED, fontsize=8.5, va="center")
    ax2.set_xticks(x)
    ax2.set_xticklabels(groups, fontsize=9.5)
    ax2.set_ylabel("accuracy (%)")
    ax2.set_ylim(0, 75)
    ax2.legend(fontsize=9, frameon=False, loc="upper right")
    ax2.set_title("(b) augmentation shrinks the overfitting gap", fontsize=10.5)

    fig.suptitle("Data augmentation as a regularizer (from-scratch CNN, measured)", fontsize=11.5, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}augmentation.png")


# ================================================================================================
# Figure: augmentation examples — one real image, several label-preserving transforms
# ================================================================================================


def fig_aug_examples(exp) -> None:
    import torch
    import torchvision.transforms.v2 as T

    # find one real image (a recognizable class); use the first test sample's source dataset image
    img = exp.samples.images[0]
    if img.ndim == 2:  # fallback grayscale
        img = np.stack([img] * 3, axis=-1)
    x = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    torch.manual_seed(0)
    aug = T.Compose([T.RandomCrop(x.shape[-1], padding=4), T.RandomHorizontalFlip()])
    n = 6
    fig, axes = plt.subplots(1, n, figsize=(n * 1.7, 2.1))
    axes[0].imshow(x.permute(1, 2, 0).numpy())
    axes[0].set_title("original", fontsize=9, color=INK)
    axes[0].axis("off")
    for i in range(1, n):
        aug_img = aug(x).permute(1, 2, 0).numpy().clip(0, 1)
        axes[i].imshow(aug_img)
        axes[i].set_title(f"aug #{i}", fontsize=9, color=SLATE)
        axes[i].axis("off")
    fig.suptitle("Label-preserving augmentation: same label, new pixels each epoch", fontsize=11, color=INK, y=1.04)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}aug_examples.png")


def main() -> None:
    exp = run_experiment()
    fig_sample_grid(exp)
    fig_transfer_vs_scratch(exp)
    fig_confusion(exp)
    fig_per_class(exp)
    fig_augmentation(exp)
    fig_aug_examples(exp)
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
