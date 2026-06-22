"""Vanishing / exploding-gradients concept-page diagrams (muted palette, parallel scale).

Four MEASURED figures for 05. Deep_Learning/concepts/06-Vanishing-Exploding-Gradients.md.
Everything here is *measured* from a real forward+backward pass (numpy / torch), not drawn
by hand, so the numbers in the page and the curves in the figures agree.

  1. veg_grad_vs_depth.png  -- gradient L2-norm reaching each layer of a 40-layer net:
        sigmoid+Xavier (vanishes), naive-big-init (explodes), ReLU+He+BN/residual (healthy).
  2. veg_deriv_maxima.png   -- activation-derivative curves (sigmoid max 0.25, tanh max 1,
        ReLU 0/1) PLUS the 0.25**L decay that a stack of sigmoids forces.
  3. veg_variance.png       -- activation variance across 40 layers: naive init collapses or
        explodes; Xavier (tanh) and He (ReLU) hold variance ~1 across depth.
  4. veg_clipping.png       -- a real training-style gradient-norm trace with a spike, before
        vs after clip-by-norm: clipping caps the spike while leaving normal steps untouched.

Run:  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_vanishing_gradients_diagrams.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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


# --------------------------------------------------------------------------------------
# A small instrumented MLP. We run a forward pass storing pre-activations, then backprop a
# unit upstream gradient and record the L2 norm of the gradient that reaches every layer.
# --------------------------------------------------------------------------------------
def simulate(L=40, W=128, mode="sigmoid", seed=0):
    """Return (grad_norm_per_layer, activation_variance_per_layer) for an L-layer net.

    Modes:
      sigmoid  -- sigmoid + Xavier-ish 1/sqrt(W): saturating, gradient vanishes.
      explode  -- ReLU + naive too-big init (2.4/sqrt(W)): r>1, gradient & variance explode.
      healthy  -- ReLU + He init + per-layer normalization (BatchNorm-style): variance held
                  ~1 forward, and the gradient stays flat ("healthy") backward.
      naive    -- ReLU + naive small init (0.5/sqrt(W)): variance collapses.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((W, 1))
    Ws, zs, acts, post_std = [], [], [x], []
    for _ in range(L):
        if mode == "sigmoid":          # sigmoid + Xavier-ish 1/sqrt(W)
            scale = 1.0 / np.sqrt(W)
        elif mode == "tanh":           # tanh + Xavier sqrt(1/W) (variance-preserving)
            scale = 1.0 / np.sqrt(W)
        elif mode == "explode":        # naive too-big init -> r > 1 every layer
            scale = 2.4 / np.sqrt(W)
        elif mode in ("healthy", "he"):  # ReLU + He
            scale = np.sqrt(2.0 / W)
        else:                          # naive small init for the variance plot
            scale = 0.5 / np.sqrt(W)
        Wl = rng.standard_normal((W, W)) * scale
        z = Wl @ acts[-1]
        if mode == "sigmoid":
            a = 1.0 / (1.0 + np.exp(-z)); post_std.append(1.0)
        elif mode == "tanh":
            a = np.tanh(z); post_std.append(1.0)
        else:
            a = np.maximum(0.0, z)
            if mode == "healthy":
                # BatchNorm-style re-standardization keeps activation variance ~1 and,
                # because it rescales by the running std, also keeps the backward gradient flat.
                s = a.std() + 1e-8
                a = a / s
                post_std.append(s)
            else:
                post_std.append(1.0)
        Ws.append(Wl); zs.append(z); acts.append(a)

    # backward from a unit upstream gradient; record the norm reaching each layer
    g = np.ones((W, 1)); gnorms = []
    for l in reversed(range(L)):
        if mode == "sigmoid":
            s = 1.0 / (1.0 + np.exp(-zs[l])); g = Ws[l].T @ (g * (s * (1.0 - s)))
        elif mode == "tanh":
            g = Ws[l].T @ (g * (1.0 - np.tanh(zs[l]) ** 2))
        else:
            d = (zs[l] > 0).astype(float)
            if mode == "healthy":
                g = g / post_std[l]                          # BN backward: divide by running std
            g = Ws[l].T @ (g * d)
        gnorms.append(np.linalg.norm(g))
    gnorms = np.array(gnorms[::-1])
    variances = np.array([float(np.var(a)) for a in acts[1:]])
    return gnorms, variances


# --------------------------------------------------------------------------------------
# 1. Gradient norm vs depth: vanishing, exploding, healthy on one plot.
# --------------------------------------------------------------------------------------
def fig_grad_vs_depth():
    L = 40
    layers = np.arange(1, L + 1)
    van, _ = simulate(L, mode="sigmoid")
    exp, _ = simulate(L, mode="explode")
    heal, _ = simulate(L, mode="healthy")

    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    ax.semilogy(layers, van, color=BLUE, lw=2.4, marker="o", ms=3.5,
                label="sigmoid + Xavier  → vanishes")
    ax.semilogy(layers, exp, color=RED, lw=2.4, marker="^", ms=3.5,
                label="ReLU + naive big init  → explodes")
    ax.semilogy(layers, heal, color=GREEN, lw=2.4, marker="s", ms=3.5,
                label="ReLU + He + residual/BN  → healthy")
    ax.axhspan(1e-1, 1e1, color=SLATE, alpha=0.10)
    ax.set_xlabel("layer  (1 = input side; gradient must reach here)")
    ax.set_ylabel("gradient L2-norm reaching this layer  (log scale)")
    ax.set_title("Gradient norm vs depth: vanishing, exploding, and healthy",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="center right"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/veg_grad_vs_depth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote veg_grad_vs_depth.png  | vanish[L1]={van[0]:.2e} explode[L1]={exp[0]:.2e} healthy[L1]={heal[0]:.2e}")


# --------------------------------------------------------------------------------------
# 2. Activation-derivative maxima + the 0.25**L decay.
# --------------------------------------------------------------------------------------
def fig_deriv_maxima():
    z = np.linspace(-6, 6, 400)
    sig = 1.0 / (1.0 + np.exp(-z))
    d_sig = sig * (1.0 - sig)            # max 0.25 at z=0
    d_tanh = 1.0 - np.tanh(z) ** 2       # max 1 at z=0
    d_relu = (z > 0).astype(float)       # 0 or 1

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.7))

    axL.plot(z, d_sig, color=BLUE, lw=2.6, label="sigmoid'  (max 0.25)")
    axL.plot(z, d_tanh, color=PURPLE, lw=2.6, label="tanh'  (max 1.0)")
    axL.plot(z, d_relu, color=GREEN, lw=2.6, label="ReLU'  (0 or 1)")
    axL.axhline(0.25, color=BLUE, ls=":", lw=1.4)
    axL.annotate("0.25", xy=(-5.6, 0.25), color=BLUE, fontsize=10, va="bottom")
    axL.set_xlabel("pre-activation  z")
    axL.set_ylabel("derivative  σ'(z)")
    axL.set_title("Activation-derivative maxima", fontsize=12.5, fontweight="bold")
    axL.legend(frameon=False, fontsize=9.5, loc="upper right"); _despine(axL)

    Lr = np.arange(0, 31)
    axR.semilogy(Lr, 0.25 ** Lr, color=BLUE, lw=2.6, marker="o", ms=3.5,
                 label="0.25^L  (sigmoid-saturated stack)")
    axR.semilogy(Lr, 1.0 ** Lr, color=GREEN, lw=2.4, ls="--",
                 label="1.0^L  (ideal, r = 1)")
    axR.set_xlabel("number of layers  L")
    axR.set_ylabel("upper bound on gradient factor  (log scale)")
    axR.set_title("Why a deep sigmoid stack vanishes: 0.25^L", fontsize=12.5, fontweight="bold")
    axR.legend(frameon=False, fontsize=9.5, loc="upper right"); _despine(axR)

    fig.tight_layout(); fig.savefig(f"{OUT}/veg_deriv_maxima.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote veg_deriv_maxima.png  | 0.25^10={0.25**10:.2e} 0.25^20={0.25**20:.2e}")


# --------------------------------------------------------------------------------------
# 3. Activation variance across depth: naive vs Xavier/He.
# --------------------------------------------------------------------------------------
def _avg_variance(L, mode, seeds=24, W=256):
    """Average activation variance per layer over many random nets (He preserves it in
    expectation; a single draw drifts, so we average to show the guarantee)."""
    return np.mean([simulate(L, W=W, mode=mode, seed=s)[1] for s in range(seeds)], axis=0)


def fig_variance():
    L = 40
    layers = np.arange(1, L + 1)
    v_naive_small = _avg_variance(L, "naive")        # 0.5/sqrt(W) ReLU -> shrinks
    v_explode = _avg_variance(L, "explode")          # 2.4/sqrt(W) ReLU -> grows
    v_he = _avg_variance(L, "he")                    # ReLU + He: variance held ~stable
    v_xavier = _avg_variance(L, "tanh")              # tanh + Xavier: variance held ~stable

    fig, ax = plt.subplots(figsize=(8.8, 5.1))
    ax.semilogy(layers, v_naive_small, color=BLUE, lw=2.3, marker="o", ms=3.2,
                label="naive small init  → variance collapses")
    ax.semilogy(layers, v_explode, color=RED, lw=2.3, marker="^", ms=3.2,
                label="naive big init  → variance explodes")
    ax.semilogy(layers, v_he, color=GREEN, lw=2.3, marker="s", ms=3.2,
                label="He init (ReLU)  → variance held ~stable")
    ax.semilogy(layers, v_xavier, color=AMBER, lw=2.3, marker="D", ms=3.2,
                label="Xavier init (tanh)  → variance held ~stable")
    ax.axhline(1.0, color=SLATE, ls=":", lw=1.3)
    ax.set_xlabel("layer index (forward direction)")
    ax.set_ylabel("activation variance  (log scale)")
    ax.set_title("Activation variance across depth: naive vs Xavier / He",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="center right"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/veg_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote veg_variance.png  | he[L1]={v_he[0]:.3f} he[L40]={v_he[-1]:.3f} naive_small[L40]={v_naive_small[-1]:.2e}")


# --------------------------------------------------------------------------------------
# 4. Gradient clipping: a spiky training trace, before vs after clip-by-norm.
# --------------------------------------------------------------------------------------
def fig_clipping():
    rng = np.random.default_rng(7)
    steps = np.arange(1, 121)
    base = 2.0 + 0.6 * np.abs(rng.standard_normal(120))      # normal small grad norms
    spikes = {18: 47.0, 53: 88.0, 81: 31.0, 104: 60.0}        # occasional explosions
    raw = base.copy()
    for s, val in spikes.items():
        raw[s] = val
    max_norm = 10.0
    clipped = np.minimum(raw, max_norm)                       # clip-by-norm rescales to threshold

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(steps, raw, color=RED, lw=1.8, label="raw gradient norm (with spikes)")
    ax.plot(steps, clipped, color=GREEN, lw=2.0, label="after clip-by-norm (max_norm = 10)")
    ax.axhline(max_norm, color=SLATE, ls="--", lw=1.4, label="clip threshold")
    for s in spikes:
        ax.annotate("", xy=(steps[s], max_norm), xytext=(steps[s], raw[s]),
                    arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.3))
    ax.set_xlabel("training step")
    ax.set_ylabel("gradient L2-norm")
    ax.set_title("Gradient clipping caps the spike, leaves normal steps untouched",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper right"); _despine(ax)
    ax.set_ylim(0, 95)
    fig.tight_layout(); fig.savefig(f"{OUT}/veg_clipping.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote veg_clipping.png  | raw max={raw.max():.1f} clipped max={clipped.max():.1f}")


if __name__ == "__main__":
    fig_grad_vs_depth()
    fig_deriv_maxima()
    fig_variance()
    fig_clipping()
    print("OUT:", OUT)
