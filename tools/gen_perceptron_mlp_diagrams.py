"""Perceptron & MLP concept-page diagrams (muted palette, parallel matplotlib scale).

Four figures for 05. Deep_Learning/concepts/01-Perceptron-and-MLP.md:
  1. pmlp_xor_fail.png   -- a single perceptron's linear boundary cleanly separates
     AND and OR, but NO line separates XOR (the four points are diagonally paired).
     The geometric proof that a single perceptron cannot represent XOR. MEASURED:
     boundaries come from perceptrons actually trained on AND/OR.
  2. pmlp_xor_transform.png -- the 2-2-1 MLP's hidden layer maps the four XOR inputs
     into a new (h1,h2) space where a single straight line DOES separate them.
     MEASURED: hidden activations computed from the hand-set XOR weights.
  3. pmlp_uat.png        -- universal approximation: a sum of sigmoid "bumps" from a
     1-H-1 MLP approximates a target curve ever more closely as hidden width H grows.
     MEASURED: each curve is a least-squares fit of H sigmoids to the target.
  4. pmlp_arch.png       -- architecture schematic: a single perceptron (left) vs a
     2-2-1 MLP (right), drawn to scale with labelled weights/edges.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def step(z):
    return (z >= 0).astype(float)


def train_perceptron(X, y, eta=0.1, epochs=50, seed=0):
    """Classic perceptron learning rule; returns (w, b) after convergence."""
    rng = np.random.default_rng(seed)
    w = rng.standard_normal(X.shape[1]) * 0.1
    b = 0.0
    for _ in range(epochs):
        errors = 0
        for xi, yi in zip(X, y):
            yhat = step(np.array([w @ xi + b]))[0]
            err = yi - yhat
            if err != 0:
                w = w + eta * err * xi
                b = b + eta * err
                errors += 1
        if errors == 0:
            break
    return w, b


def xor_fail():
    """Single perceptron: separates AND & OR, fails on XOR (measured boundaries)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_and = np.array([0, 0, 0, 1.0])
    y_or = np.array([0, 1, 1, 1.0])
    y_xor = np.array([0, 1, 1, 0.0])

    w_and, b_and = train_perceptron(X, y_and)
    w_or, b_or = train_perceptron(X, y_or)

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6))

    def plot_panel(ax, y, w, b, title, drawline):
        for xi, yi in zip(X, y):
            c = GREEN if yi == 1 else RED
            m = "o" if yi == 1 else "X"
            ax.scatter(*xi, s=320, c=c, marker=m, edgecolors="white",
                       linewidths=2, zorder=3)
        if drawline and abs(w[1]) > 1e-6:
            xs = np.array([-0.4, 1.4])
            ys = -(w[0] * xs + b) / w[1]
            ax.plot(xs, ys, color=BLUE, lw=2.6, zorder=2,
                    label="learned boundary  $w^\\top x + b = 0$")
            ax.fill_between(xs, ys, 1.6, color=GREEN, alpha=0.07)
            ax.fill_between(xs, ys, -0.6, color=RED, alpha=0.07)
            ax.legend(frameon=False, fontsize=9.5, loc="upper center",
                      bbox_to_anchor=(0.5, -0.12))
        ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.5, 1.55)
        ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
        ax.set_title(title, fontsize=12.5, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        _despine(ax)

    plot_panel(axes[0], y_and, w_and, b_and, "AND — linearly separable", True)
    plot_panel(axes[1], y_or, w_or, b_or, "OR — linearly separable", True)

    # XOR panel: show that no line works
    ax = axes[2]
    for xi, yi in zip(X, y_xor):
        c = GREEN if yi == 1 else RED
        m = "o" if yi == 1 else "X"
        ax.scatter(*xi, s=320, c=c, marker=m, edgecolors="white",
                   linewidths=2, zorder=3)
    # draw two candidate lines, both fail
    xs = np.array([-0.4, 1.4])
    ax.plot(xs, 0.5 - 0 * xs, color=SLATE, lw=1.8, ls="--", alpha=0.7)
    ax.plot(0.5 + 0 * np.array([-0.4, 1.4]), [-0.5, 1.55], color=SLATE, lw=1.8,
            ls="--", alpha=0.7)
    ax.text(0.5, 1.42, "no straight line\ncan split ●(1) from ✕(0)", ha="center",
            va="top", fontsize=9.5, color=RED, fontweight="bold")
    ax.set_xlim(-0.4, 1.4); ax.set_ylim(-0.5, 1.55)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("XOR — NOT linearly separable", fontsize=12.5, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    _despine(ax)

    fig.suptitle("One perceptron draws ONE straight line: fine for AND/OR, impossible for XOR",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pmlp_xor_fail.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote pmlp_xor_fail.png  | AND w,b =", np.round(w_and, 2), round(b_and, 2),
          "| OR w,b =", np.round(w_or, 2), round(b_or, 2))


def xor_transform():
    """MLP hidden layer remaps XOR inputs into a linearly separable space (measured)."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y_xor = np.array([0, 1, 1, 0.0])
    # hand-set hidden weights: h1 = OR(x), h2 = AND(x)
    W1 = np.array([[20., 20.], [20., 20.]])   # rows are neurons
    b1 = np.array([-10., -30.])               # h1: OR threshold; h2: AND threshold
    H = 1 / (1 + np.exp(-(X @ W1.T + b1)))     # sigmoid hidden activations (measured)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7))

    # left: original input space (not separable)
    ax = axes[0]
    for xi, yi in zip(X, y_xor):
        c = GREEN if yi == 1 else RED
        m = "o" if yi == 1 else "X"
        ax.scatter(*xi, s=300, c=c, marker=m, edgecolors="white", linewidths=2, zorder=3)
    ax.set_xlim(-0.3, 1.3); ax.set_ylim(-0.3, 1.3)
    ax.set_xlabel("$x_1$"); ax.set_ylabel("$x_2$")
    ax.set_title("Input space $(x_1,x_2)$ — tangled", fontsize=12, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); _despine(ax)

    # right: hidden space (separable). (0,1) and (1,0) both map to (1,0): nudge
    # the labels apart and stack the two overlapping markers slightly.
    ax = axes[1]
    labels = ["x=(0,0)", "x=(0,1)", "x=(1,0)", "x=(1,1)"]
    jitter = [(0, 0), (0, 0.045), (0, -0.045), (0, 0)]   # separate the coincident pair
    lab_off = [(10, 8), (12, 10), (12, -20), (10, 8)]
    for hi, yi, lab, jit, off in zip(H, y_xor, labels, jitter, lab_off):
        c = GREEN if yi == 1 else RED
        m = "o" if yi == 1 else "X"
        pt = (hi[0], hi[1] + jit[1])
        ax.scatter(*pt, s=300, c=c, marker=m, edgecolors="white", linewidths=2, zorder=3)
        ax.annotate(lab, pt, textcoords="offset points", xytext=off,
                    fontsize=8.5, color=SLATE)
    # separating line in hidden space: h1 - h2 = 0.5.
    # class-1 points (h1=1,h2=0) satisfy h1 - h2 > 0.5 -> shade THAT side green.
    hs = np.array([-0.1, 1.1])
    ax.plot(hs, hs - 0.5, color=BLUE, lw=2.6, zorder=2,
            label="output unit's line\n$h_1 - h_2 = 0.5$")
    ax.fill_between(hs, hs - 0.5, -0.2, color=GREEN, alpha=0.08)   # below line: class 1
    ax.fill_between(hs, hs - 0.5, 1.2, color=RED, alpha=0.07)      # above line: class 0
    ax.set_xlim(-0.15, 1.2); ax.set_ylim(-0.2, 1.2)
    ax.set_xlabel("$h_1 = \\sigma(\\mathrm{OR})$"); ax.set_ylabel("$h_2 = \\sigma(\\mathrm{AND})$")
    ax.set_title("Hidden space $(h_1,h_2)$ — separable!", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9, loc="upper left"); _despine(ax)

    fig.suptitle("The hidden layer learns a new representation where XOR becomes a straight-line problem",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pmlp_xor_transform.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote pmlp_xor_transform.png  | hidden activations:\n", np.round(H, 3))


def uat():
    """Universal approximation: more sigmoid bumps -> closer fit (least-squares, measured)."""
    rng = np.random.default_rng(1)
    xs = np.linspace(-1, 1, 400)
    target = np.sin(3 * xs) + 0.4 * np.sin(7 * xs)   # a wiggly continuous target

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(H):
        # random fixed hidden layer (centers/slopes), solve output weights by least squares
        centers = np.linspace(-1, 1, H)
        slope = 14.0
        Phi = sigmoid(slope * (xs[:, None] - centers[None, :]))      # (N, H)
        Phi = np.hstack([Phi, np.ones((len(xs), 1))])               # bias
        w, *_ = np.linalg.lstsq(Phi, target, rcond=None)
        return Phi @ w

    fig, ax = plt.subplots(figsize=(9.2, 4.9))
    ax.plot(xs, target, color=SLATE, lw=3.2, label="target $f(x)$", zorder=2)
    for H, col in [(3, RED), (8, AMBER), (30, GREEN)]:
        ax.plot(xs, fit(H), color=col, lw=2.0, alpha=0.92,
                label=f"MLP, {H} hidden sigmoids")
    ax.set_xlabel("$x$"); ax.set_ylabel("$f(x)$")
    ax.set_title("Universal approximation: a sum of sigmoid bumps fits any continuous curve",
                 fontsize=13.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="lower right"); _despine(ax)
    ax.annotate("more hidden units\n→ tighter fit", (0.34, -0.55),
                fontsize=9.5, color=GREEN, fontweight="bold")
    ax.set_ylim(-1.45, 1.45)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pmlp_uat.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote pmlp_uat.png")


def architecture():
    """Schematic: single perceptron vs 2-2-1 MLP."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    def neuron(ax, xy, color, label="", r=0.16):
        ax.add_patch(Circle(xy, r, facecolor=color, edgecolor="white",
                            linewidth=2, zorder=3))
        if label:
            ax.text(*xy, label, ha="center", va="center", color="white",
                    fontsize=11, fontweight="bold", zorder=4)

    def edge(ax, a, b, color=SLATE, lab="", lw=1.8):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12,
                                     color=color, lw=lw, zorder=1,
                                     shrinkA=14, shrinkB=14))
        if lab:
            mx, my = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
            ax.text(mx, my + 0.06, lab, fontsize=8.5, color=color, ha="center")

    # --- single perceptron ---
    ax = axes[0]
    inp = [(0.2, 0.75), (0.2, 0.25)]
    out = (0.85, 0.5)
    for i, p in enumerate(inp):
        neuron(ax, p, BLUE, f"$x_{i+1}$")
        edge(ax, p, out, lab=f"$w_{i+1}$")
    neuron(ax, out, PURPLE, "$\\Sigma$\nstep", r=0.18)
    ax.text(out[0] + 0.05, out[1] - 0.3, "$\\hat y=\\mathrm{step}(w^\\top x+b)$",
            fontsize=10, color=PURPLE, ha="center")
    ax.text(0.2, 1.02, "inputs", fontsize=10, ha="center", color=BLUE, fontweight="bold")
    ax.text(0.85, 1.02, "output", fontsize=10, ha="center", color=PURPLE, fontweight="bold")
    ax.set_xlim(0, 1.1); ax.set_ylim(0, 1.12); ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Single perceptron — one linear boundary",
                 fontsize=12.5, fontweight="bold")

    # --- 2-2-1 MLP ---
    ax = axes[1]
    inp = [(0.12, 0.72), (0.12, 0.28)]
    hid = [(0.5, 0.72), (0.5, 0.28)]
    out = (0.88, 0.5)
    for i, p in enumerate(inp):
        neuron(ax, p, BLUE, f"$x_{i+1}$")
    for j, p in enumerate(hid):
        neuron(ax, p, PURPLE, f"$h_{j+1}$")
    neuron(ax, out, GREEN, "$\\hat y$")
    for p in inp:
        for q in hid:
            edge(ax, p, q, color=SLATE, lw=1.4)
    for q in hid:
        edge(ax, q, out, color=SLATE, lw=1.8)
    ax.text(0.12, 1.0, "input", fontsize=10, ha="center", color=BLUE, fontweight="bold")
    ax.text(0.5, 1.0, "hidden (+ nonlinearity)", fontsize=10, ha="center",
            color=PURPLE, fontweight="bold")
    ax.text(0.88, 1.0, "output", fontsize=10, ha="center", color=GREEN, fontweight="bold")
    ax.text(0.5, 0.04, "two hidden units → two lines → XOR is solvable",
            fontsize=9.5, ha="center", color=GREEN, fontweight="bold")
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 1.1); ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("2-2-1 MLP — hidden layer folds the space",
                 fontsize=12.5, fontweight="bold")

    fig.suptitle("From one linear unit to a multilayer perceptron",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pmlp_arch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote pmlp_arch.png")


if __name__ == "__main__":
    xor_fail()
    xor_transform()
    uat()
    architecture()
    print("all perceptron/MLP diagrams written to", OUT)
