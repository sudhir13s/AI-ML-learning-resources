"""Figure generator for 02-Backpropagation-and-Computational-Graphs — every number from the REAL runs in
``backpropagation.py``.

All eight figures come from the same executed pipeline the chapter and notebook use (the four hand-traced
worked examples, the from-scratch MLP's gradient check and torch cross-check, the digits training, and the
deep-net vanishing-gradient measurement). Nothing quantitative is hand-typed; every point, curve, and
annotation is read off a function call in ``backpropagation.py``.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``dl02_``:

  dl02_gates.png            -- the four local-gradient gates (add distributes, multiply swaps, max/ReLU
                               routes, copy/fan-out sums) with an upstream gradient of 2 — the schematic
                               vocabulary of every backward pass (illustrative).
  dl02_compgraph.png        -- the 2->2->2 net of worked example 4 with EVERY forward value and backward
                               gradient annotated on it, all read from worked_two_layer_net().
  dl02_fwd_vs_rev.png       -- passes needed to get all gradients of one scalar loss: forward mode rises with
                               the parameter count, reverse mode (backprop) stays flat at one (illustrative
                               of the O(#inputs) vs O(#outputs) law).
  dl02_delta_recurrence.png -- the per-layer error ||delta^l|| pulled back through a 12-layer net for tanh vs
                               ReLU (measured by depth_gradient_profile).
  dl02_vanishing.png        -- the weight-gradient norm per layer, sigmoid vs ReLU: sigmoid vanishes toward
                               the input layers, ReLU stays flat (measured).
  dl02_gradcheck.png        -- analytic backprop vs numerical finite-difference gradient for every parameter
                               of a small MLP; all points on y = x (median rel err ~1.7e-10, max ~7.7e-8).
  dl02_gradcheck_eps.png    -- the finite-difference relative error vs step size eps: the truncation/round-off
                               U-curve, minimum near eps = 1e-6 (measured by epsilon_sweep).
  dl02_training_loss.png    -- the from-scratch net trained on scikit-learn digits by SGD driven by backprop:
                               the cross-entropy loss falling to near zero, annotated with the real test
                               accuracy (measured by train_digits).

    python make_figures_02.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / torch 2.12 / scikit-learn 1.9.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in ``05. Deep_Learning/tools/``; the chapter module it demonstrates stays in that
# chapter's ``code/`` folder. Put that folder on sys.path so the ``backpropagation`` import resolves
# regardless of the working directory.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "02-Backpropagation-and-Computational-Graphs" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from backpropagation import (  # noqa: E402  (resolved via the sys.path insert above)
    MLP,
    depth_gradient_profile,
    epsilon_sweep,
    gradient_check,
    load_digits_split,
    train_digits,
    worked_two_layer_net,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / values forward
PURPLE = "#5D4A8A"  # process / ops
GREEN = "#2E7A5A"  # output / good
RED = "#8B3B4A"  # error / gradients backward
AMBER = "#7A6528"  # upstream / seed
SLATE = "#4A5B6E"  # neutral / frozen
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "dl02_"


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


def _box(ax: plt.Axes, xy: tuple[float, float], text: str, colour: str, w: float = 1.5, h: float = 0.7) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor=colour, edgecolor="none", zorder=2,
        )
    )
    ax.text(x, y, text, ha="center", va="center", color="white", fontsize=9.5, zorder=3)


def _arrow(ax: plt.Axes, p0: tuple[float, float], p1: tuple[float, float], colour: str, dashed: bool = False) -> None:
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=13, linewidth=1.6, color=colour,
            linestyle="--" if dashed else "-", shrinkA=6, shrinkB=6, zorder=1,
        )
    )


# ============================ 1. the four gates (schematic) ======================================
def fig_gates() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.9))
    titles = ["add — distributor", "multiply — swapper", "max / ReLU — router", "copy / fan-out — adder"]
    for ax, title in zip(axes, titles):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")
        ax.set_title(title, color=INK, fontsize=11)
    # add: upstream 2 -> both inputs get 2
    _box(axes[0], (5, 5), "+", PURPLE, w=1.4, h=1.2)
    _arrow(axes[0], (8.5, 5), (5.9, 5), AMBER, dashed=True)
    axes[0].text(8.7, 5.9, "2", color=AMBER, fontsize=11, ha="center")
    _arrow(axes[0], (4.1, 5.4), (1.5, 7.5), RED, dashed=True)
    _arrow(axes[0], (4.1, 4.6), (1.5, 2.5), RED, dashed=True)
    axes[0].text(1.2, 8.2, "x: 2", color=RED, fontsize=10)
    axes[0].text(1.2, 1.8, "y: 2", color=RED, fontsize=10)
    axes[0].text(5, 2.4, "grad passes\nthrough", color=INK, fontsize=8.5, ha="center")
    # multiply: x=3, y=4, upstream 2 -> x gets 2*y=8, y gets 2*x=6
    _box(axes[1], (5, 5), "×", PURPLE, w=1.4, h=1.2)
    _arrow(axes[1], (8.5, 5), (5.9, 5), AMBER, dashed=True)
    axes[1].text(8.7, 5.9, "2", color=AMBER, fontsize=11, ha="center")
    _arrow(axes[1], (4.1, 5.4), (1.5, 7.5), RED, dashed=True)
    _arrow(axes[1], (4.1, 4.6), (1.5, 2.5), RED, dashed=True)
    axes[1].text(1.0, 8.2, "x=3: 2·y=8", color=RED, fontsize=9.5)
    axes[1].text(1.0, 1.8, "y=4: 2·x=6", color=RED, fontsize=9.5)
    axes[1].text(5, 2.4, "gate swaps\ninputs", color=INK, fontsize=8.5, ha="center")
    # max/relu: winner gets 2, loser 0
    _box(axes[2], (5, 5), "max", PURPLE, w=1.6, h=1.2)
    _arrow(axes[2], (8.5, 5), (5.9, 5), AMBER, dashed=True)
    axes[2].text(8.7, 5.9, "2", color=AMBER, fontsize=11, ha="center")
    _arrow(axes[2], (4.1, 5.4), (1.5, 7.5), RED, dashed=True)
    _arrow(axes[2], (4.1, 4.6), (1.5, 2.5), SLATE, dashed=True)
    axes[2].text(0.8, 8.2, "x=5 (won): 2", color=RED, fontsize=9.5)
    axes[2].text(0.8, 1.8, "y=1 (lost): 0", color=SLATE, fontsize=9.5)
    axes[2].text(5, 2.4, "routes to\nthe winner", color=INK, fontsize=8.5, ha="center")
    # copy/fan-out: two consumers 2 and 3 -> sum 5
    _box(axes[3], (5, 5), "copy", PURPLE, w=1.7, h=1.2)
    _arrow(axes[3], (8.5, 7.2), (5.9, 5.4), AMBER, dashed=True)
    _arrow(axes[3], (8.5, 2.8), (5.9, 4.6), AMBER, dashed=True)
    axes[3].text(8.9, 7.4, "2", color=AMBER, fontsize=11)
    axes[3].text(8.9, 2.6, "3", color=AMBER, fontsize=11)
    _arrow(axes[3], (4.1, 5), (1.5, 5), RED, dashed=True)
    axes[3].text(0.5, 5.6, "x: 2+3 = 5", color=RED, fontsize=10)
    axes[3].text(5, 2.4, "sums the\nbranches", color=INK, fontsize=8.5, ha="center")
    fig.suptitle("The four local-gradient gates (upstream gradient = 2) — illustrative", color=INK, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _save(fig, f"{IMG_PREFIX}gates.png")


# ============================ 2. worked-example-4 computational graph ============================
def fig_compgraph() -> None:
    net = worked_two_layer_net()
    fig, ax = plt.subplots(figsize=(13.5, 6.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    # node positions (left -> right)
    nodes = {
        "x": (1.2, 4), "z1": (4.0, 4), "a1": (6.4, 4), "z2": (9.0, 4), "p": (11.3, 4), "L": (13.0, 4),
    }
    _box(ax, nodes["x"], f"x\n{np.round([1.0, 2.0], 2)}", BLUE, w=1.7, h=1.1)
    _box(ax, nodes["z1"], f"z¹ = xW¹+b¹\n{np.round(net.z1, 3)}", PURPLE, w=2.0, h=1.1)
    _box(ax, nodes["a1"], f"a¹ = tanh(z¹)\n{np.round(net.a1, 3)}", PURPLE, w=2.0, h=1.1)
    _box(ax, nodes["z2"], f"z² = a¹W²+b²\n{np.round(net.z2, 3)}", PURPLE, w=2.0, h=1.1)
    _box(ax, nodes["p"], f"p = softmax\n{np.round(net.p, 3)}", GREEN, w=1.9, h=1.1)
    _box(ax, nodes["L"], f"L\n{net.loss:.3f}", GREEN, w=1.2, h=1.1)
    order = ["x", "z1", "a1", "z2", "p", "L"]
    for a, b in zip(order[:-1], order[1:]):
        _arrow(ax, (nodes[a][0] + 0.95, nodes[a][1] + 0.15), (nodes[b][0] - 0.95, nodes[b][1] + 0.15), BLUE)
    # backward (red, below)
    for a, b in zip(order[1:], order[:-1]):
        _arrow(ax, (nodes[a][0] - 0.95, nodes[a][1] - 0.15), (nodes[b][0] + 0.95, nodes[b][1] - 0.15), RED, dashed=True)
    ax.text(1, 7.3, "forward: values (blue, →)", color=BLUE, fontsize=11)
    ax.text(1, 6.8, "backward: gradients (red, ⇠)", color=RED, fontsize=11)
    # backward annotations under the arrows
    ax.text(nodes["z2"][0] + 0.2, 2.55, f"δ² = p−y\n{np.round(net.delta2, 3)}", color=RED, fontsize=8.8, ha="center")
    ax.text(nodes["a1"][0] + 0.1, 2.55, f"∂L/∂a¹ = W²δ²\n{np.round(net.da1, 3)}", color=RED, fontsize=8.8, ha="center")
    ax.text(nodes["z1"][0] + 0.1, 2.55, f"δ¹ = ·⊙tanh′\n{np.round(net.delta1, 3)}", color=RED, fontsize=8.8, ha="center")
    ax.text(6.6, 0.85, f"∂L/∂W² = a¹ᵀδ² =\n{np.round(net.dW2, 3).tolist()}", color=SLATE, fontsize=8.6, ha="center")
    ax.text(11.0, 0.85, f"∂L/∂W¹ = xᵀδ¹ =\n{np.round(net.dW1, 3).tolist()}", color=SLATE, fontsize=8.6, ha="center")
    ax.text(7, 7.5, "Worked example 4: forward AND backward on a 2→2→2 net "
            f"(all values verified vs autograd, max |diff| = {net.max_abs_diff_vs_torch:.0e})",
            color=INK, fontsize=11.5, ha="center")
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}compgraph.png")


# ============================ 3. forward vs reverse mode =========================================
def fig_fwd_vs_rev() -> None:
    params = np.array([1, 2, 5, 10, 100, 1000])
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    _style_axis(ax)
    ax.plot(params, params, "o-", color=RED, lw=2, label="forward mode: 1 pass / input")
    ax.plot(params, np.ones_like(params), "s-", color=GREEN, lw=2, label="reverse mode (backprop): 1 pass total")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("number of parameters (inputs to differentiate)")
    ax.set_ylabel("passes to get ALL gradients of one scalar loss")
    ax.set_title("Why reverse mode wins: one scalar loss, many parameters — illustrative", fontsize=11.5)
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")
    ax.annotate("a real net lives here\n(millions of params)", xy=(1000, 1000), xytext=(60, 3000),
                color=RED, fontsize=9, arrowprops={"arrowstyle": "->", "color": RED})
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}fwd_vs_rev.png")


# ============================ 4 & 5. delta recurrence + vanishing (measured) =====================
def fig_delta_and_vanishing() -> None:
    prof_tanh = depth_gradient_profile(activation="tanh")
    prof_relu = depth_gradient_profile(activation="relu")
    prof_sig = depth_gradient_profile(activation="sigmoid")
    layers = np.arange(1, len(prof_tanh.delta_norms) + 1)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    _style_axis(ax)
    ax.semilogy(layers, prof_tanh.delta_norms, "o-", color=PURPLE, lw=2, label="tanh")
    ax.semilogy(layers, prof_relu.delta_norms, "s-", color=GREEN, lw=2, label="ReLU")
    ax.set_xlabel("layer (1 = input side  →  L = output side)")
    ax.set_ylabel("‖δ  ‖  (error magnitude, log scale)")
    ax.set_title("The δ recurrence, measured: error pulled back through a 12-layer net", fontsize=11.5)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}delta_recurrence.png")

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    _style_axis(ax)
    ax.semilogy(layers, prof_sig.grad_norms, "o-", color=RED, lw=2, label="sigmoid (vanishes)")
    ax.semilogy(layers, prof_relu.grad_norms, "s-", color=GREEN, lw=2, label="ReLU (stays flat)")
    ax.set_xlabel("layer (1 = input side  →  L = output side)")
    ax.set_ylabel("‖∂L/∂Wˡ‖  (weight-gradient norm, log scale)")
    r = prof_sig.grad_norms[-1] / prof_sig.grad_norms[0]  # output/input: how many times smaller at the input
    ax.set_title(f"Vanishing gradients, measured: sigmoid's grad is {r:.1e}× smaller at the input layer", fontsize=11)
    ax.legend(frameon=False, fontsize=10)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}vanishing.png")


# ============================ 6. gradient check scatter (measured) ===============================
def fig_gradcheck() -> None:
    mlp = MLP([64, 16, 10], activation="tanh", seed=0)
    x_tr, _, y_tr, _ = load_digits_split()
    gc = gradient_check(mlp, x_tr[:16], np.eye(10)[y_tr[:16]], eps=1e-5)
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    _style_axis(ax)
    lim = float(np.abs(np.concatenate([gc.analytic, gc.numerical])).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "-", color=SLATE, lw=1, alpha=0.7, label="y = x")
    ax.scatter(gc.numerical, gc.analytic, s=10, color=BLUE, alpha=0.6, edgecolor="none")
    ax.set_xlabel("numerical gradient (centred finite difference)")
    ax.set_ylabel("analytic gradient (backprop)")
    ax.set_title(f"Gradient check: {gc.n_params} params, all on y = x\n"
                 f"median rel err {gc.median_rel_error:.2e}, max {gc.max_rel_error:.1e}", fontsize=11)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}gradcheck.png")


# ============================ 7. epsilon U-curve (measured) ======================================
def fig_gradcheck_eps() -> None:
    _, pts = epsilon_sweep(exps=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))
    eps = np.array([p.eps for p in pts])
    rel = np.array([p.rel_error for p in pts])
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    _style_axis(ax)
    ax.loglog(eps, rel, "o-", color=BLUE, lw=2)
    imin = int(rel.argmin())
    ax.scatter([eps[imin]], [rel[imin]], s=90, color=GREEN, zorder=5, label=f"sweet spot ε≈{eps[imin]:.0e}")
    ax.annotate("truncation error\nO(ε²)", xy=(1e-2, rel[1]), xytext=(3e-3, 1e-2), color=RED, fontsize=9,
                arrowprops={"arrowstyle": "->", "color": RED})
    ax.annotate("round-off error\n(cancellation)", xy=(1e-11, rel[-2]), xytext=(2e-9, 5e-4), color=AMBER, fontsize=9,
                arrowprops={"arrowstyle": "->", "color": AMBER})
    ax.set_xlabel("finite-difference step size ε")
    ax.set_ylabel("relative error of the numerical gradient")
    ax.set_title("The gradient-check U-curve, measured: too big ε truncates, too small ε rounds off", fontsize=11)
    ax.invert_xaxis()
    ax.legend(frameon=False, fontsize=9.5)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}gradcheck_eps.png")


# ============================ 8. training loss on digits (measured) ==============================
def fig_training_loss() -> None:
    tr = train_digits()
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    _style_axis(ax)
    ax.plot(range(len(tr.loss_history)), tr.loss_history, "-", color=PURPLE, lw=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training cross-entropy loss")
    ax.set_title("Backprop actually trains: from-scratch MLP on scikit-learn digits (SGD on our gradients)",
                 fontsize=11)
    ax.annotate(
        f"loss {tr.loss_history[0]:.2f} → {tr.loss_history[-1]:.3f}\n"
        f"test accuracy {tr.test_acc * 100:.1f}%  ({tr.n_test} held-out digits)\n"
        f"{tr.n_params} params, {tr.epochs} epochs, plain SGD",
        xy=(len(tr.loss_history) * 0.5, tr.loss_history[len(tr.loss_history) // 6]),
        xytext=(len(tr.loss_history) * 0.35, tr.loss_history[0] * 0.55),
        color=INK, fontsize=9.5, bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F2F4F6", "edgecolor": GRID},
    )
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}training_loss.png")


def main() -> None:
    fig_gates()
    fig_compgraph()
    fig_fwd_vs_rev()
    fig_delta_and_vanishing()
    fig_gradcheck()
    fig_gradcheck_eps()
    fig_training_loss()
    print(f"\nall figures written to {OUT_DIR} with prefix '{IMG_PREFIX}'")


if __name__ == "__main__":
    main()
