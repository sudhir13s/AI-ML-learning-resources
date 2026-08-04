"""Generate the step-by-step teaching notebook (13-CNNs-and-Convolution.ipynb).

The notebook mirrors ``cnn.py`` one measurement at a time, so a learner can open it, run every cell live, and
*see* convolution being built and proven correct: one convolution by hand, the from-scratch forward pass
verified against im2col / scipy / torch, the output-size and parameter/FLOP arithmetic, the from-scratch
**backward** pass gradient-checked and autograd-checked, max-pooling forward+backward, translation
equivariance, a real Sobel filter on a real photo, receptive-field growth, and finally a small CNN trained
against a larger MLP on scikit-learn digits (same accuracy at ~5x fewer weights, far more shift-robust) with
its learned filters and feature maps visualized. Each numbered step has a short markdown lead-in (the
intuition) followed by ONE focused code cell with real output. This generator writes the .ipynb; a separate
nbconvert pass executes it headless so the outputs are embedded.

    python build_notebook_13.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../13-CNNs-and-Convolution/code/13-CNNs-and-Convolution.ipynb"

This generator lives in the domain-level ``05. Deep_Learning/tools/`` folder; the notebook it writes (and the
module it mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited .ipynb) so
the notebook and the module stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "13-CNNs-and-Convolution" / "code"
NB_PATH = _CHAPTER_CODE / "13-CNNs-and-Convolution.ipynb"

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
    "# CNNs & convolution — a step-by-step, runnable notebook\n"
    "\n"
    "A fully-connected layer on a real image is hopeless: a 224×224×3 image into one 1000-unit dense layer is "
    "**150 million weights in a single layer**, and it must re-learn a cat in every position. A **convolution** "
    "fixes both problems with one idea — slide a small **filter** across the image looking for a local pattern, "
    "and **share that filter's weights everywhere**. That buys locality, an ~8-order-of-magnitude parameter cut "
    "(independent of image size), and translation **equivariance**.\n"
    "\n"
    "This notebook builds 2-D convolution from scratch in NumPy and proves it correct on **real** data: the "
    "forward pass matches im2col, `scipy.signal`, and `torch` to machine tolerance; the **backward** pass is "
    "gradient-checked *and* cross-checked against `torch.autograd`; max-pooling routes its gradient to the "
    "argmax exactly as torch does; a real Sobel filter on a real photo reproduces `scipy`; and a small CNN, "
    "trained on scikit-learn digits, matches a 5× larger MLP's accuracy while being far more robust to a "
    "1-pixel shift. Every cell uses the **exact same functions** as the chapter and its figures (imported from "
    "`cnn.py`), so the numbers here are the numbers there.\n"
    "\n"
    "> Companion page: **CNNs & Convolution**. Run top-to-bottom (Kernel → Restart & Run All); it is CPU-only, "
    "seeded, and finishes in a few seconds."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup: import the real module and print versions\n"
    "\n"
    "We import the from-scratch functions from the chapter module so this notebook runs the *same code* the "
    "page and figures use, and print the library versions the results were produced on."
)
add_code(
    "import numpy as np\n"
    "import scipy\n"
    "import torch\n"
    "import sklearn\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "from cnn import (\n"
    "    worked_hand_conv, verify_forward, output_size, param_economics, verify_backward, verify_pool,\n"
    "    measure_equivariance, sobel_edges, load_sample_gray, receptive_field_growth, train_cnn_vs_mlp,\n"
    ")\n"
    "\n"
    "np.random.seed(0)\n"
    "torch.manual_seed(0)\n"
    "print(f'numpy {np.__version__} | scipy {scipy.__version__} | torch {torch.__version__} | scikit-learn {sklearn.__version__}  (CPU)')"
)

# ---- Step 1: one convolution by hand ----
add_md(
    "## Step 1 — One convolution, fully by hand\n"
    "\n"
    "A convolution lays a small **kernel** over a patch of the input, multiplies element-wise, and **sums** the "
    "products into a *single* output number — one dot product. Take the top-left 3×3 patch of a 5×5 input and a "
    "vertical-edge kernel $\\begin{bmatrix}1&0&-1\\\\1&0&-1\\\\1&0&-1\\end{bmatrix}$:\n"
    "\n"
    "$$(2\\cdot1+0\\cdot0+0\\cdot{-}1) + (2\\cdot1+1\\cdot0+0\\cdot{-}1) + (2\\cdot1+2\\cdot0+2\\cdot{-}1) = 2+2+0 = 4.$$\n"
    "\n"
    "That 4 is the top-left cell of the feature map. Slide the kernel across every valid position for the rest."
)
add_code(
    "hc = worked_hand_conv()\n"
    "print('patch =\\n', hc.patch.astype(int))\n"
    "print('kernel =\\n', hc.kernel.astype(int))\n"
    "print(f'top-left output value = {hc.top_left:.0f}')\n"
    "print('full 3x3 feature map =\\n', hc.feature_map)"
)

# ---- Step 2: forward verification ----
add_md(
    "## Step 2 — 2-D convolution from scratch, verified four ways\n"
    "\n"
    "The operation deep learning calls 'convolution' is really **cross-correlation** (no kernel flip) — "
    "$Y[i,j]=\\sum_{u,v} X[i+u,j+v]\\,K[u,v]$. We implement it three ways and check them against a fourth:\n"
    "\n"
    "- **`conv2d_naive`** — explicit multiply-and-sum loops;\n"
    "- **`conv2d_im2col`** — unfold every patch into a column, then one big matmul (**what frameworks do**);\n"
    "- **`scipy.signal.correlate2d`** — an independent signal-processing engine (single-channel case);\n"
    "- **`torch.nn.functional.conv2d`** — the reference.\n"
    "\n"
    "All four agree: the loops match torch to summation-order noise, im2col matches it *exactly* (it **is** the "
    "same matmul), and scipy matches the single-channel case. Each equality is a hard `assert` in the module."
)
add_code(
    "fc = verify_forward()\n"
    "print(f'output shape            : {fc.out_shape}')\n"
    "print(f'max|naive  - torch|     : {fc.naive_vs_torch:.2e}   (float64 summation order)')\n"
    "print(f'max|im2col - torch|     : {fc.im2col_vs_torch:.2e}   (im2col IS the same matmul)')\n"
    "print(f'max|scipy  - naive|     : {fc.scipy_vs_naive:.2e}')\n"
    "print('OK: loops == im2col == scipy == torch')"
)

# ---- Step 3: output-size formula ----
add_md(
    "## Step 3 — The output-size formula\n"
    "\n"
    "Three knobs set the output dimensions: kernel size $K$, stride $S$, padding $P$. For input width $W$:\n"
    "\n"
    "$$O = \\left\\lfloor \\frac{W - K + 2P}{S} \\right\\rfloor + 1.$$\n"
    "\n"
    "Padding widens the input to $W+2P$; a $K$-wide kernel starts anywhere in the span $W+2P-K$, stepped by $S$ "
    "(the floor drops a partial final step); $+1$ for the first position. Two patterns recur: **'same'** padding "
    "$P=(K-1)/2$ at stride 1 preserves size (why VGG stacks 3×3s), and **stride 2** roughly halves it."
)
add_code(
    "print(f\"{'W':>4}{'K':>4}{'P':>4}{'S':>4}{'O':>5}   note\")\n"
    "cases = [(7,3,0,1,'valid: shrinks by K-1'), (7,3,1,1,'same: size preserved'),\n"
    "         (7,3,0,2,'stride 2: ~halves'), (224,11,2,4,'AlexNet conv1 -> 55'),\n"
    "         (224,3,1,1,'VGG 3x3 same -> 224')]\n"
    "for W,K,P,S,note in cases:\n"
    "    print(f'{W:>4}{K:>4}{P:>4}{S:>4}{output_size(W,K,P,S):>5}   {note}')"
)

# ---- Step 4: params & FLOPs ----
add_md(
    "## Step 4 — Parameters and FLOPs: the weight-sharing economics\n"
    "\n"
    "A conv layer has $C_{out}(C_{in}K^2)+C_{out}$ parameters — **independent of image size**. A 3×3, 3→64 conv "
    "is 1,792 weights whether the image is 32×32 or 4K. The equivalent dense layer over a 224×224 image "
    "(every input pixel wired to every output pixel) needs ~$4.8\\times10^{11}$ — the conv is **~270 million "
    "times smaller**. FLOPs, though, scale with the spatial size ($\\text{MACs}=C_{out}H'W'(C_{in}K^2)$): convs "
    "are *cheap to store but expensive to run on big images*. Depthwise-separable conv factors the cost by "
    "$1/C_{out}+1/K^2$ (~8.4× for a 3×3)."
)
add_code(
    "pe = param_economics(cin=3, cout=64, k=3, img=224)\n"
    "print(f'3x3 conv 3->64      : {pe.conv_params:,} params')\n"
    "print(f'equivalent dense    : {pe.dense_params:,} params')\n"
    "print(f'ratio               : ~{pe.ratio:,.0f}x smaller  (and size-independent)')\n"
    "print(f'conv MACs (224x224) : {pe.conv_macs:,}  (~{pe.conv_gflop:.2f} GFLOP)')\n"
    "print(f'depthwise-separable : {pe.sep_standard:,} -> {pe.sep_separable:,} = {pe.sep_ratio:.2f}x  (theory {1/pe.sep_theory:.2f}x)')"
)

# ---- Step 5: conv backward ----
add_md(
    "## Step 5 — The backward pass: gradient-checked AND autograd-checked\n"
    "\n"
    "Convolution is linear, so it slots into [backprop](../../02-Backpropagation-and-Computational-Graphs/"
    "02-Backpropagation-and-Computational-Graphs.md) like any layer — and all three gradients are themselves "
    "convolutions. `conv2d_backward` returns $\\partial L/\\partial X$, $\\partial L/\\partial W$ (a "
    "cross-correlation of the input with the upstream gradient — the shared kernel's gradient *accumulates* "
    "over every position, the signature of weight sharing), and $\\partial L/\\partial b$ (a spatial sum). We "
    "prove it two independent ways: a **centred finite-difference** check on every kernel entry (median rel "
    "error ~1e-9), and a **torch autograd** cross-check on the identical op (~1e-15). Both are hard `assert`s."
)
add_code(
    "bc = verify_backward()\n"
    "print(f'kernel entries checked        : {bc.n_params}')\n"
    "print(f'dW rel error  median / max    : {bc.median_rel_error_dw:.2e} / {bc.max_rel_error_dw:.2e}  (<< 1e-4 => correct)')\n"
    "print(f'vs torch autograd  dX / dW / db : {bc.dx_vs_torch:.1e} / {bc.dw_vs_torch:.1e} / {bc.db_vs_torch:.1e}')\n"
    "print('OK: from-scratch conv backward matches finite differences AND torch autograd')"
)

# ---- Step 6: max-pool ----
add_md(
    "## Step 6 — Max-pooling: forward and backward\n"
    "\n"
    "Pooling shrinks a feature map by summarizing each window into one number — **max-pool** keeps the "
    "strongest activation. It has **no parameters**. Its backward pass is clean: route the whole upstream "
    "gradient to the single position that *won* the max (remembered via the argmax), zero to the rest. We "
    "cross-check both forward and backward against torch — they agree exactly."
)
add_code(
    "pc = verify_pool()\n"
    "print(f'pooled output shape       : {pc.out_shape}')\n"
    "print(f'forward  max|ours - torch|: {pc.fwd_vs_torch:.1e}')\n"
    "print(f'backward max|ours - torch|: {pc.bwd_vs_torch:.1e}   (gradient routed to the argmax)')\n"
    "print('OK: from-scratch max-pool matches torch forward and backward')"
)

# ---- Step 7: equivariance ----
add_md(
    "## Step 7 — Translation equivariance, measured\n"
    "\n"
    "The deep consequence of weight sharing: **shift the input, and the feature map shifts the same way** — "
    "$\\mathrm{Conv}(T(x)) = T(\\mathrm{Conv}(x))$. The detector fires wherever the feature goes, *without* "
    "relearning. This is exactly the property a dense net lacks. We verify it directly: convolve a small "
    "pattern, then convolve a 1-pixel-shifted copy, and confirm the outputs are shifts of each other."
)
add_code(
    "eq = measure_equivariance()\n"
    "print(f'max |shift(Conv(x)) - Conv(shift(x))| (interior) = {eq.max_diff:.2e}')\n"
    "print('OK: convolution is translation-equivariant (shift in -> shift out)')"
)

# ---- Step 8: Sobel on a real image ----
add_md(
    "## Step 8 — A real filter on a real image: Sobel edges\n"
    "\n"
    "A kernel *is* a learned pattern detector. Before learned kernels, the classic **Sobel** edge filter was "
    "hand-designed — and it is exactly the kind of vertical/horizontal-edge detector a CNN's *first* layer "
    "reliably learns on its own. We apply it with our own `conv2d_naive` to a real grayscale photograph (bundled "
    "with scikit-learn, no download) and verify against `scipy`. Flat regions vanish to black; edges light up."
)
add_code(
    "sob = sobel_edges(load_sample_gray())\n"
    "print(f'image shape {sob.gray.shape}   our-conv vs scipy = {sob.ours_vs_scipy:.2e}')\n"
    "fig, axes = plt.subplots(1, 4, figsize=(15, 4))\n"
    "panels = [(sob.gray, 'input photo', 'gray'), (sob.gx, '|Gx| vertical edges', 'magma'),\n"
    "          (sob.gy, '|Gy| horizontal edges', 'magma'), (sob.magnitude, 'edge magnitude', 'magma')]\n"
    "for ax, (img, title, cmap) in zip(axes, panels):\n"
    "    ax.imshow(img, cmap=cmap)\n"
    "    ax.set_title(title)\n"
    "    ax.axis('off')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 9: receptive field ----
add_md(
    "## Step 9 — Receptive-field growth with depth\n"
    "\n"
    "The **receptive field** is the region of the input a neuron can see. For stacked $K\\times K$ convs at "
    "stride 1 it grows *linearly*: $\\text{RF}_L = 1 + L(K-1)$. For $K=3$ that's $3,5,7,9,\\dots$ — which is why "
    "**two stacked 3×3 convs have the same 5×5 receptive field** as one 5×5 conv, with fewer weights and an "
    "extra nonlinearity (the VGG insight). A stride-2 pool multiplies the *jump* between neurons, so the RF then "
    "grows *multiplicatively* — a conv after the pool adds 4, not 2."
)
add_code(
    "rf = receptive_field_growth(k=3, depth=6)\n"
    "print('stacked 3x3, stride 1:  layers', rf.layers, '-> RF', rf.rf_stride1)\n"
    "print('\\nwith a stride-2 pool in the middle (name, jump, RF):')\n"
    "for name, j, r in rf.rf_with_pool:\n"
    "    print(f'  {name:<14} jump={j}  RF={r}')"
)

# ---- Step 10: train CNN vs MLP ----
add_md(
    "## Step 10 — The payoff: a CNN vs a bigger MLP on real digits\n"
    "\n"
    "Now train a small **CNN** (conv→pool→conv→pool→FC) and a deliberately *larger* fully-connected **MLP** on "
    "the scikit-learn digits (8×8 images, 10 classes). The CNN matches the MLP's clean accuracy with **~5× "
    "fewer parameters** — the weight-sharing win. And when we shift every test digit by one pixel, the CNN "
    "holds up far better than the position-sensitive MLP: equivariance, paying off on data it never saw shifted."
)
add_code(
    "tr = train_cnn_vs_mlp(epochs=30)\n"
    "print(f'CNN : {tr.cnn_params:,} params   clean acc = {tr.cnn_acc*100:.1f}%   shifted-1px acc = {tr.cnn_acc_shift*100:.1f}%')\n"
    "print(f'MLP : {tr.mlp_params:,} params   clean acc = {tr.mlp_acc*100:.1f}%   shifted-1px acc = {tr.mlp_acc_shift*100:.1f}%')\n"
    "print(f'-> CNN uses {tr.mlp_params/tr.cnn_params:.1f}x FEWER params, matches clean accuracy,')\n"
    "print(f'   and is far more shift-robust ({tr.cnn_acc_shift*100:.0f}% vs {tr.mlp_acc_shift*100:.0f}%)')"
)

# ---- Step 11: learned filters ----
add_md(
    "## Step 11 — What the CNN learned: first-layer filters and feature maps\n"
    "\n"
    "Finally, *see* what gradient descent found. The trained CNN's 8 first-layer 3×3 kernels are local pattern "
    "detectors (edges and strokes — strikingly Sobel-like, just as Step 8 foreshadowed), and each produces a "
    "**feature map** highlighting where its pattern occurs in a real test digit."
)
add_code(
    "fig, axes = plt.subplots(2, 8, figsize=(14, 3.6))\n"
    "for i in range(8):\n"
    "    axes[0, i].imshow(tr.first_filters[i], cmap='RdBu_r')\n"
    "    axes[0, i].set_title(f'k{i}', fontsize=9)\n"
    "    axes[0, i].axis('off')\n"
    "    axes[1, i].imshow(tr.feature_maps[i], cmap='magma')\n"
    "    axes[1, i].axis('off')\n"
    "fig.suptitle(f'Learned 3x3 filters (top) and their feature maps on a real digit (bottom) — CNN {tr.cnn_acc*100:.1f}% test acc')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- close ----
add_md(
    "## Recap\n"
    "\n"
    "A CNN slides small **learned filters** over the input and **shares those weights** across every position — "
    "encoding **locality**, **weight-sharing**, and **translation-equivariance**, exactly the structure images "
    "have. We built 2-D convolution from scratch and proved it correct against im2col, scipy, and torch "
    "(forward); against finite differences and torch autograd (backward); measured its equivariance; applied a "
    "real Sobel filter to a real photo; and trained a CNN that matched a 5× larger MLP with far more shift "
    "robustness. Output size is $O = \\lfloor(W-K+2P)/S\\rfloor + 1$; params are $C_{out}(C_{in}K^2)+C_{out}$ "
    "(size-independent); the conv backward pass is itself a (transposed) convolution; and pooling routes its "
    "gradient to the argmax.\n"
    "\n"
    "See the companion page for the full derivations, the im2col/Toeplitz view, 1×1 and depthwise-separable "
    "convolutions, the LeNet→ResNet lineage, the CNN-vs-ViT trade, pitfalls, and references."
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
