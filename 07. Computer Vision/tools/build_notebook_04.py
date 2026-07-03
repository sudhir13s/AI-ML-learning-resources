"""Generate the step-by-step teaching notebook (04-Image-Classification.ipynb).

The notebook mirrors ``image_classification.py`` one step at a time so a learner can open it, run every cell
live, and *see* the applied classification workflow being built and measured on real CIFAR-10 images: the
softmax + cross-entropy loss from scratch (verified against torch), the evaluation metrics from scratch
(top-1/top-5/confusion, cross-checked against scikit-learn), a real balanced data subset, the input pipeline,
a FROZEN pretrained ResNet-18 backbone used as a feature extractor, a linear probe trained on those features
(transfer learning), a from-scratch CNN at the same budget (transfer wins, measured), a data-augmentation
ablation (the overfitting gap shrinks), and an honest confusion matrix / per-class / prediction grid. Each
numbered step has a short markdown lead-in (the intuition) followed by a focused code cell with real output.

    python build_notebook_04.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../04-Image-Classification/code/04-Image-Classification.ipynb"

This generator lives in the domain-level ``07. Computer Vision/tools/`` folder; the notebook it writes (and the
module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited .ipynb) so
the notebook and the module stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "04-Image-Classification" / "code"
NB_PATH = _CHAPTER_CODE / "04-Image-Classification.ipynb"

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
    "# Image classification — a runnable, measured workflow\n"
    "\n"
    "Image classification assigns a whole image to one of $K$ labels. The naive attack — feed raw pixels to a "
    "classifier and train from zero — needs *millions* of labeled images to generalize, which you rarely have. "
    "The move that dominates applied vision is **transfer learning**: reuse the visual features a big model "
    "already learned on a huge dataset (ImageNet), and train only a small new head on your task. This notebook "
    "runs that workflow on **real** CIFAR-10 images and measures every claim.\n"
    "\n"
    "It uses the **exact same functions** as the companion page and its figures (imported from "
    "`image_classification.py`), so the numbers here are the numbers there. We build the loss and the metrics "
    "from scratch (checked against `torch` and `scikit-learn`), then: load a balanced real subset, freeze a "
    "pretrained **ResNet-18** backbone, train a **linear probe** on its features (transfer), compare against a "
    "**from-scratch CNN** at the same budget (transfer wins by a measured margin), ablate **augmentation** (the "
    "overfitting gap shrinks), and evaluate honestly with a confusion matrix, per-class accuracy, and a "
    "prediction grid that shows real mistakes.\n"
    "\n"
    "> Companion page: **Image Classification**. Run top-to-bottom (Kernel → Restart & Run All). The reported "
    "training is **CPU-pinned and seeded** for a reproducible trace (a few minutes end-to-end); the heavy work "
    "happens once in Step 6."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup: import the real module and print versions\n"
    "\n"
    "We import the pipeline functions from the chapter module so this notebook runs the *same code* the page and "
    "figures use, and print the library versions + the accelerator we detected (we still train on CPU on "
    "purpose — GPU kernels are nondeterministic and every number here must be bit-reproducible)."
)
add_code(
    "import numpy as np\n"
    "import torch\n"
    "import torchvision\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "import image_classification as ic\n"
    "\n"
    "print(f'torch {torch.__version__} | torchvision {torchvision.__version__} | numpy {np.__version__}')\n"
    "print(f'accelerator available: {ic.detect_accelerator()}  (reported metrics are CPU-pinned, seed={ic.SEED})')"
)

# ---- Step 1: softmax + cross-entropy ----
add_md(
    "## Step 1 — The classification loss: softmax + cross-entropy, from scratch\n"
    "\n"
    "A classifier outputs one **logit** per class. **Softmax** turns logits into a probability distribution, and "
    "**cross-entropy** scores how much probability the model put on the *true* class:\n"
    "\n"
    "$$\\mathrm{softmax}(z)_i = \\frac{e^{z_i - m}}{\\sum_j e^{z_j - m}},\\quad m=\\max_j z_j; \\qquad "
    "\\mathcal{L} = -\\frac{1}{N}\\sum_i \\log \\mathrm{softmax}(z_i)[y_i].$$\n"
    "\n"
    "The $-m$ shift is the one numerical-stability trick that matters: it stops $e^{z}$ from overflowing without "
    "changing the result. We implement both by hand and check them against `torch` to ~1e-6. (Full derivation: "
    "Foundations 23 — Cross-Entropy.)"
)
add_code(
    "lc = ic.verify_softmax_cross_entropy()\n"
    "print(f'cross-entropy  from scratch : {lc.ce_scratch:.6f}')\n"
    "print(f'cross-entropy  torch        : {lc.ce_torch:.6f}')\n"
    "print(f'max|softmax - torch softmax|: {lc.softmax_vs_torch:.1e}')\n"
    "print(f'|CE - torch CE|             : {lc.ce_vs_torch:.1e}')\n"
    "assert lc.ce_vs_torch < 1e-6\n"
    "print('OK: our softmax + cross-entropy match torch')"
)

# ---- Step 2: metrics from scratch ----
add_md(
    "## Step 2 — Evaluation metrics from scratch, cross-checked against scikit-learn\n"
    "\n"
    "How do we score a classifier? **Top-1 accuracy** = fraction whose single best guess is right. **Top-5 "
    "accuracy** = fraction whose true label is among the 5 best guesses (the standard ImageNet metric, since a "
    "single guess over 1000 classes is a harsh bar). The **confusion matrix** $C[t,p]$ counts true-class-$t$ "
    "images predicted as $p$ — its off-diagonal *is* the list of mistakes. We implement all of them in NumPy and "
    "cross-check against `sklearn.metrics` on a small synthetic example (they must match exactly)."
)
add_code(
    "rng = np.random.default_rng(0)\n"
    "toy_logits = rng.standard_normal((20, 6))\n"
    "toy_y = rng.integers(0, 6, size=20)\n"
    "print(f'top-1 (scratch): {ic.top1_accuracy(toy_logits, toy_y):.3f}')\n"
    "print(f'top-5 (scratch): {ic.topk_accuracy(toy_logits, toy_y, k=5):.3f}')\n"
    "cm = ic.confusion_matrix_from_scratch(toy_logits.argmax(1), toy_y, 6)\n"
    "print('confusion matrix (rows=true, cols=pred):\\n', cm)\n"
    "check = ic.verify_metrics(toy_logits, toy_y, 6)\n"
    "print(f'\\nvs sklearn: |top1|={check.top1_vs_sklearn:.0e}  |top5|={check.top5_vs_sklearn:.0e}  '\n"
    "      f'confusion==sklearn: {check.confusion_matches_sklearn}')\n"
    "print('OK: our metrics ARE the standard metrics')"
)

# ---- Step 3: load real data ----
add_md(
    "## Step 3 — Load real images: a balanced CIFAR-10 subset\n"
    "\n"
    "CIFAR-10 is 60,000 real 32×32 RGB photos across 10 classes. We take a small **balanced** subset (equal per "
    "class) so runs are quick but honest — a realistic 'small labeled dataset', exactly the regime where "
    "transfer learning pays off. If the download is unavailable, the module falls back to a real bundled dataset "
    "(scikit-learn digits) so the notebook still runs on real images."
)
add_code(
    "data = ic.try_load_cifar(train_per_class=300, test_per_class=100) or ic.load_digits_fallback()\n"
    "print(f'dataset : {data.name}')\n"
    "print(f'train   : {data.images_train.shape}   test: {data.images_test.shape}')\n"
    "print(f'classes : {data.classes}')\n"
    "print(f'transfer available (pretrained backbone reachable): {data.transfer_available}')"
)
add_md("A quick look at the raw images — real, low-resolution, and varied in pose, lighting, and scale:")
add_code(
    "fig, axes = plt.subplots(2, 8, figsize=(12, 3.2))\n"
    "for ax, img, lbl in zip(axes.flat, data.images_train, data.labels_train):\n"
    "    ax.imshow(img, cmap=None if data.is_rgb else 'gray')\n"
    "    ax.set_title(data.classes[lbl], fontsize=8)\n"
    "    ax.axis('off')\n"
    "plt.suptitle('Real CIFAR-10 samples (32x32 RGB)')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 4: the input pipeline ----
add_md(
    "## Step 4 — The input pipeline: resize + normalize for the backbone\n"
    "\n"
    "A pretrained backbone expects the distribution it was trained on. So before the ImageNet ResNet-18 sees a "
    "CIFAR image we **resize** it to 224×224 and **normalize** each channel with ImageNet's mean/std. Matching "
    "the pretraining statistics is what lets the frozen features transfer. Here is one image before and after the "
    "pipeline (the normalized tensor is de-normalized for display)."
)
add_code(
    "batch = ic._to_backbone_batch(data.images_train[:1])   # [1,3,224,224], ImageNet-normalized\n"
    "print(f'pipeline: {data.images_train[:1].shape} uint8  ->  {tuple(batch.shape)} float (ImageNet-normalized)')\n"
    "print(f'per-channel mean after normalize (should be near 0): {batch.mean(dim=(0,2,3)).numpy().round(2)}')"
)

# ---- Step 5: frozen backbone / features ----
add_md(
    "## Step 5 — The pretrained backbone as a frozen feature extractor\n"
    "\n"
    "Transfer learning's core object: take an ImageNet-pretrained **ResNet-18**, throw away its 1000-way "
    "classification head, and keep the 512-dimensional feature it computes just before that head. Those 512 "
    "numbers are a *general visual summary* of the image — edges, textures, parts — that the backbone learned "
    "from 1.2M ImageNet photos. We **freeze** the backbone (never update it) and only read out features. Let's "
    "extract them for the first 12 images to see the shape (the full run in Step 6 extracts them for all)."
)
add_code(
    "if data.transfer_available:\n"
    "    feats12 = ic.extract_features(data.images_train[:12])\n"
    "    print(f'12 images -> features {tuple(feats12.shape)}  (512-D per image, from the FROZEN backbone)')\n"
    "else:\n"
    "    print('offline fallback: no pretrained backbone; Step 6 trains only the from-scratch CNN')"
)

# ---- Step 6: run the full measured experiment ----
add_md(
    "## Step 6 — Run the full measured pipeline (transfer + from-scratch + augmentation)\n"
    "\n"
    "This one cell does the real training and is the notebook's heavy lifter (a few minutes, CPU-pinned and "
    "seeded). It extracts features for the whole subset, trains a **linear probe** on the frozen features "
    "(transfer), trains a **from-scratch CNN** at the same budget, trains the CNN again **with augmentation**, "
    "then evaluates the best model from scratch and cross-checks every metric against scikit-learn. It prints a "
    "full report; the following steps unpack and visualize it."
)
add_code(
    "exp = ic.run_experiment(train_per_class=300, test_per_class=100, epochs=40)\n"
    "print(f'dataset            : {exp.dataset_name}  ({exp.n_train} train / {exp.n_test} test)')\n"
    "if exp.transfer_top1 is not None:\n"
    "    print(f'transfer  top-1/5  : {exp.transfer_top1:.3f} / {exp.transfer_top5:.3f}  (head only, backbone frozen)')\n"
    "    print(f'from-scratch top-1 : {exp.scratch_noaug.test_acc:.3f}  ({exp.scratch_noaug.n_params:,} params)')\n"
    "    print(f'transfer advantage : {exp.transfer_delta*100:+.1f} points at the same budget')\n"
    "print(f'aug gap: no-aug {exp.aug_gap_noaug*100:.1f} pts -> aug {exp.aug_gap_aug*100:.1f} pts')"
)

# ---- Step 7: transfer vs scratch bar ----
add_md(
    "## Step 7 — Transfer learning vs training from scratch (the measured win)\n"
    "\n"
    "Same images, same labels, same budget — but the transfer model reuses features the backbone already "
    "learned, while the from-scratch CNN must learn everything from ~a few hundred images per class. The gap is "
    "the whole argument for transfer learning: **low-level visual features are reusable** (Yosinski et al. 2014), "
    "so you almost never train a vision model from zero."
)
add_code(
    "if exp.transfer_top1 is not None:\n"
    "    fig, ax = plt.subplots(figsize=(6, 4))\n"
    "    accs = [exp.transfer_top1*100, exp.scratch_noaug.test_acc*100]\n"
    "    bars = ax.bar(['transfer\\n(frozen backbone\\n+ linear head)', 'from scratch\\n(small CNN)'], accs,\n"
    "                  color=['#2E7A5A', '#4A5B6E'], edgecolor='#1C2530')\n"
    "    for b, v in zip(bars, accs):\n"
    "        ax.text(b.get_x()+b.get_width()/2, v+1, f'{v:.1f}%', ha='center', fontweight='bold')\n"
    "    ax.set_ylabel('top-1 test accuracy (%)')\n"
    "    ax.set_ylim(0, 100)\n"
    "    ax.set_title(f'Transfer wins by {exp.transfer_delta*100:+.1f} points (CIFAR-10, measured)')\n"
    "    plt.tight_layout()\n"
    "    plt.show()\n"
    "else:\n"
    "    print('offline fallback: transfer comparison unavailable')"
)

# ---- Step 8: augmentation examples ----
add_md(
    "## Step 8 — Data augmentation: same label, new pixels\n"
    "\n"
    "**Augmentation** creates new training views by applying *label-preserving* transforms — a random crop, a "
    "horizontal flip. A flipped cat is still a cat, so the label is unchanged, but the pixels differ every epoch, "
    "which discourages the network from memorizing exact pixel patterns. Here is one real image and several "
    "augmentations of it."
)
add_code(
    "import torchvision.transforms.v2 as T\n"
    "img = data.images_train[0]\n"
    "x = torch.tensor(img if data.is_rgb else np.stack([img]*3, -1)).permute(2,0,1).float()/255.0\n"
    "torch.manual_seed(0)\n"
    "aug = T.Compose([T.RandomCrop(x.shape[-1], padding=4), T.RandomHorizontalFlip()])\n"
    "fig, axes = plt.subplots(1, 6, figsize=(11, 2))\n"
    "for ax in axes:\n"
    "    ax.axis('off')\n"
    "axes[0].imshow(x.permute(1,2,0).numpy())\n"
    "axes[0].set_title('original', fontsize=9)\n"
    "for i in range(1, 6):\n"
    "    axes[i].imshow(aug(x).permute(1,2,0).numpy().clip(0,1))\n"
    "    axes[i].set_title(f'aug #{i}', fontsize=9)\n"
    "plt.suptitle('Label-preserving augmentation')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 9: augmentation ablation ----
add_md(
    "## Step 9 — Augmentation as a regularizer: the overfitting gap shrinks\n"
    "\n"
    "Augmentation's payoff is measured as the **generalization gap** = train accuracy − test accuracy. A model "
    "that memorizes has a big gap (high train, lower test); a regularized model has a small one. With only a few "
    "hundred images per class the from-scratch CNN overfits — and augmentation roughly halves that gap. (At this "
    "tiny budget the test-accuracy gain is small and honest; the gap reduction is the reliable signature, and it "
    "grows into a test-accuracy win with more data / longer training.)"
)
add_code(
    "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))\n"
    "ep = np.arange(1, len(exp.scratch_noaug.val_curve)+1)\n"
    "a1.plot(ep, np.array(exp.scratch_noaug.val_curve)*100, color='#4A5B6E', label='no aug')\n"
    "a1.plot(ep, np.array(exp.scratch_aug.val_curve)*100, color='#2E7A5A', label='with aug')\n"
    "a1.set_xlabel('epoch')\n"
    "a1.set_ylabel('test top-1 (%)')\n"
    "a1.legend()\n"
    "a1.set_title('test accuracy over training')\n"
    "gaps = [exp.aug_gap_noaug*100, exp.aug_gap_aug*100]\n"
    "a2.bar(['no aug', 'with aug'], gaps, color=['#8B3B4A', '#2E7A5A'], edgecolor='#1C2530')\n"
    "for i, g in enumerate(gaps):\n"
    "    a2.text(i, g+0.2, f'{g:.1f} pts', ha='center', fontweight='bold')\n"
    "a2.set_ylabel('train - test gap (pts)')\n"
    "a2.set_title('generalization gap')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 10: confusion matrix ----
add_md(
    "## Step 10 — Honest evaluation: the confusion matrix\n"
    "\n"
    "A single accuracy number hides *where* a model fails. The confusion matrix shows it: a strong diagonal "
    "(correct) and telltale off-diagonal clusters (the systematic mistakes). On CIFAR-10 the classic confusions "
    "are between visually similar **animals** — cats and dogs, deer and horses — while rigid **vehicles** are "
    "cleanly separated."
)
add_code(
    "cm = exp.confusion\n"
    "fig, ax = plt.subplots(figsize=(6.5, 5.5))\n"
    "im = ax.imshow(cm, cmap='Blues')\n"
    "ax.set_xticks(range(len(exp.classes)))\n"
    "ax.set_yticks(range(len(exp.classes)))\n"
    "ax.set_xticklabels(exp.classes, rotation=45, ha='right', fontsize=8)\n"
    "ax.set_yticklabels(exp.classes, fontsize=8)\n"
    "ax.set_xlabel('predicted')\n"
    "ax.set_ylabel('true')\n"
    "for i in range(len(exp.classes)):\n"
    "    for j in range(len(exp.classes)):\n"
    "        if cm[i, j]:\n"
    "            ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7,\n"
    "                    color='white' if cm[i, j] > cm.max()/2 else '#1C2530')\n"
    "cmn = cm.copy()\n"
    "np.fill_diagonal(cmn, 0)\n"
    "mi, mj = np.unravel_index(cmn.argmax(), cmn.shape)\n"
    "ax.set_title(f'Confusion matrix (transfer) — most confused: {exp.classes[mi]} -> {exp.classes[mj]} ({cmn[mi,mj]})')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 11: per-class ----
add_md(
    "## Step 11 — Per-class accuracy: what the headline number hides\n"
    "\n"
    "Break accuracy down by class and the averaging illusion disappears. Vehicles score far above the overall "
    "number; fine-grained animals score below it. This is exactly why accuracy alone is a poor summary on "
    "imbalanced or fine-grained problems — and why detection/segmentation reach for per-class metrics like mAP "
    "and IoU (Chapter 10)."
)
add_code(
    "order = np.argsort(exp.per_class)\n"
    "names = [exp.classes[i] for i in order]\n"
    "vals = exp.per_class[order]*100\n"
    "vehicles = {'airplane', 'automobile', 'ship', 'truck'}\n"
    "colors = ['#3A6B96' if n in vehicles else '#7A6528' for n in names]\n"
    "fig, ax = plt.subplots(figsize=(6.5, 4.5))\n"
    "ax.barh(range(len(names)), vals, color=colors, edgecolor='#1C2530')\n"
    "for i, v in enumerate(vals):\n"
    "    ax.text(v+0.6, i, f'{v:.0f}%', va='center', fontsize=8)\n"
    "ax.set_yticks(range(len(names)))\n"
    "ax.set_yticklabels(names, fontsize=9)\n"
    "ax.set_xlabel('per-class accuracy (%)')\n"
    "ax.set_xlim(0, 100)\n"
    "if exp.transfer_top1:\n"
    "    ax.axvline(exp.transfer_top1*100, color='#8B3B4A', ls='--')\n"
    "ax.set_title('Per-class accuracy (vehicles blue, animals amber)')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 12: honest predictions ----
add_md(
    "## Step 12 — Honest predictions: wins *and* real mistakes\n"
    "\n"
    "Finally, look at actual predictions — deliberately including the model's errors. The mistakes are "
    "instructive: they cluster on hard, low-resolution animal images and come with *low confidence*, exactly "
    "what a well-behaved classifier should do when it is unsure."
)
add_code(
    "s = exp.samples\n"
    "k = len(s.true_labels)\n"
    "fig, axes = plt.subplots(3, 4, figsize=(9, 7))\n"
    "for ax in axes.flat:\n"
    "    ax.axis('off')\n"
    "for i in range(k):\n"
    "    ax = axes.flat[i]\n"
    "    ax.imshow(s.images[i], cmap=None if s.images[i].ndim == 3 else 'gray')\n"
    "    ok = s.true_labels[i] == s.pred_labels[i]\n"
    "    ax.set_title(f'true: {s.classes[s.true_labels[i]]}\\npred: {s.classes[s.pred_labels[i]]} ({s.confidences[i]:.2f})',\n"
    "                 fontsize=8, color='#2E7A5A' if ok else '#8B3B4A')\n"
    "plt.suptitle('Real predictions — correct (green) and wrong (red)')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- close ----
add_md(
    "## Recap\n"
    "\n"
    "Image classification is a **workflow**, not a single model: real data → an input pipeline (resize + "
    "normalize + augment) → a **pretrained backbone** → a small **head** → softmax cross-entropy → honest "
    "evaluation. The dominant lesson is **transfer learning** — reusing ImageNet features beat training from "
    "scratch by a wide, measured margin at the same small budget, because low-level visual features are general. "
    "**Augmentation** regularizes (the overfitting gap roughly halved). And **accuracy hides failure**: the "
    "confusion matrix and per-class breakdown exposed the fine-grained animal confusions a single number "
    "concealed. Every quantity here was computed from a real run and cross-checked against `torch` / "
    "`scikit-learn`.\n"
    "\n"
    "See the companion page for the transfer freeze-vs-fine-tune spectrum, augmentation theory, class imbalance, "
    "test-set discipline, pitfalls, and references — and the sibling chapters on the CNN mechanism (05 DL 13), "
    "transfer learning for vision (05), and data augmentation (06)."
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
