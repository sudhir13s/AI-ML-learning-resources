"""Activation-functions concept-page diagrams (muted palette, parallel matplotlib scale).

Measured PNGs for 05. Deep_Learning/concepts/03-Activation-Functions.md:
  1. act_functions.png   -- the family: sigmoid/tanh/ReLU/LeakyReLU/GELU/SiLU
     plotted together (measured).
  2. act_derivatives.png -- their DERIVATIVES (measured): sigmoid'/tanh' decay
     to ~0 (vanishing gradient) while ReLU' is a flat 1 for x>0; the saturation
     band is shaded.
  3. act_smooth_zoom.png -- GELU vs ReLU vs SiLU zoomed near 0, showing the
     smooth non-monotonic dip below zero (measured).
  4. act_training.png    -- MEASURED training: a small net trained with
     sigmoid vs ReLU vs GELU on a 2-spiral task; loss curve + the first-layer
     gradient-norm curve (the gradient-flow story), all measured by torch.
  5. act_softmax.png     -- logits -> softmax probabilities, plus the effect of
     temperature (T=0.5 / 1 / 2) on the same logits (measured).
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _np(t):
    return t.detach().numpy()


# All curves are produced by torch so the figures are MEASURED, not hand-drawn.
def _curves(x_np):
    x = torch.tensor(x_np, dtype=torch.float32, requires_grad=True)
    out = {
        "sigmoid": torch.sigmoid(x),
        "tanh": torch.tanh(x),
        "ReLU": F.relu(x),
        "Leaky ReLU": F.leaky_relu(x, 0.1),
        "GELU": F.gelu(x),
        "SiLU": F.silu(x),
    }
    return x, out


def _deriv(x, y):
    """Exact autograd derivative dy/dx, evaluated elementwise."""
    g, = torch.autograd.grad(y.sum(), x, retain_graph=True, create_graph=False)
    return _np(g)


# ---- 1. The activation family ----------------------------------------------
def act_functions():
    xs = np.linspace(-4, 4, 800)
    x, out = _curves(xs)
    order = [("sigmoid", BLUE), ("tanh", PURPLE), ("ReLU", GREEN),
             ("Leaky ReLU", RED), ("GELU", AMBER), ("SiLU", NAVY)]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for name, c in order:
        ax.plot(xs, _np(out[name]), color=c, lw=2.3, label=name)
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.axvline(0, color="#bbb", lw=0.8)
    ax.set_title("The activation family (all measured)", fontsize=14, fontweight="bold")
    ax.set_xlabel("input  x")
    ax.set_ylabel("output  phi(x)")
    ax.set_ylim(-1.7, 4)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.annotate("sigmoid/tanh flatten\n(saturate) at the extremes", (2.4, 1.05),
                fontsize=9, color=PURPLE, fontweight="bold", ha="center")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/act_functions.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote act_functions.png")


# ---- 2. The derivatives (vanishing-gradient story) --------------------------
def act_derivatives():
    xs = np.linspace(-4, 4, 800)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    for name, c in [("sigmoid", BLUE), ("tanh", PURPLE), ("ReLU", GREEN),
                    ("Leaky ReLU", RED), ("GELU", AMBER), ("SiLU", NAVY)]:
        x, out = _curves(xs)
        ax.plot(xs, _deriv(x, out[name]), color=c, lw=2.3, label=name + "'")
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.axvline(0, color="#bbb", lw=0.8)
    ax.axhline(0.25, color=BLUE, lw=0.9, ls=":", alpha=0.7)
    ax.text(-3.9, 0.27, "sigmoid' peaks at 0.25", fontsize=8.5, color=BLUE)
    ax.axhspan(-0.02, 0.05, color=RED, alpha=0.08)
    ax.annotate("sigmoid'/tanh' -> 0 in the tails\n= vanishing gradients", (2.5, 0.62),
                fontsize=9.5, color=RED, fontweight="bold", ha="center")
    ax.set_title("Their derivatives - the gradient that flows back", fontsize=14, fontweight="bold")
    ax.set_xlabel("input  x")
    ax.set_ylabel("derivative  phi'(x)")
    ax.set_ylim(-0.12, 1.3)
    ax.legend(frameon=False, fontsize=10, loc="upper left", ncol=2)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/act_derivatives.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote act_derivatives.png")


# ---- 3. Smooth activations zoomed near 0 ------------------------------------
def act_smooth_zoom():
    xs = np.linspace(-3, 1.5, 700)
    x, out = _curves(xs)
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ax.plot(xs, _np(out["ReLU"]), color=GREEN, lw=2.6, label="ReLU (hard kink at 0)")
    ax.plot(xs, _np(out["GELU"]), color=AMBER, lw=2.6, label="GELU = x.Phi(x)")
    ax.plot(xs, _np(out["SiLU"]), color=NAVY, lw=2.6, label="SiLU = x.sigma(x)")
    ax.axhline(0, color="#bbb", lw=0.8)
    ax.axvline(0, color="#bbb", lw=0.8)
    # the non-monotonic dip: minimum of GELU/SiLU below zero
    g = _np(out["GELU"]); s = _np(out["SiLU"])
    ax.scatter([xs[g.argmin()]], [g.min()], color=AMBER, zorder=5, s=40)
    ax.scatter([xs[s.argmin()]], [s.min()], color=NAVY, zorder=5, s=40)
    ax.annotate("non-monotonic dip:\nGELU/SiLU bend below 0\nthen come back up",
                (xs[g.argmin()], g.min()), textcoords="offset points",
                xytext=(45, -34), fontsize=9.5, color=AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER))
    ax.set_title("Smooth activations near zero: the non-monotonic dip", fontsize=13.5, fontweight="bold")
    ax.set_xlabel("input  x")
    ax.set_ylabel("output")
    ax.set_ylim(-0.35, 1.3)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/act_smooth_zoom.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote act_smooth_zoom.png")


# ---- 4. Measured training: sigmoid vs ReLU vs GELU --------------------------
def _two_spirals(n=600, seed=0):
    rng = np.random.RandomState(seed)
    n2 = n // 2
    theta = np.sqrt(rng.rand(n2)) * 3.5 * np.pi
    r = theta + 0.0
    x1 = np.stack([r * np.cos(theta), r * np.sin(theta)], 1)
    x2 = np.stack([-r * np.cos(theta), -r * np.sin(theta)], 1)
    X = np.concatenate([x1, x2], 0) / 11.0
    X += rng.randn(*X.shape) * 0.04
    y = np.concatenate([np.zeros(n2), np.ones(n2)]).astype(np.int64)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y)


class Net(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.l1 = nn.Linear(2, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 64)
        self.l4 = nn.Linear(64, 2)
        self.act = act

    def forward(self, x):
        x = self.act(self.l1(x))
        x = self.act(self.l2(x))
        x = self.act(self.l3(x))
        return self.l4(x)


def _train(act, X, y, steps=400, seed=0):
    torch.manual_seed(seed)
    net = Net(act)
    opt = torch.optim.SGD(net.parameters(), lr=0.3)
    losses, grad_norms = [], []
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(net(X), y)
        loss.backward()
        # first-layer gradient norm = how much signal reaches the EARLY layer
        gn = net.l1.weight.grad.norm().item()
        losses.append(loss.item())
        grad_norms.append(gn)
        opt.step()
    return losses, grad_norms


def act_training():
    X, y = _two_spirals()
    runs = [("sigmoid", torch.sigmoid, BLUE),
            ("ReLU", F.relu, GREEN),
            ("GELU", F.gelu, AMBER)]
    results = {name: _train(fn, X, y) for name, fn, _ in runs}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for name, _, c in runs:
        losses, _gn = results[name]
        ax1.plot(losses, color=c, lw=2.2, label=name)
    ax1.set_title("Training loss (same net, same data, same seed)", fontsize=12.5, fontweight="bold")
    ax1.set_xlabel("SGD step")
    ax1.set_ylabel("cross-entropy loss")
    ax1.legend(frameon=False, fontsize=10)
    _despine(ax1)
    for name, _, c in runs:
        _l, gn = results[name]
        ax2.plot(gn, color=c, lw=2.2, label=name)
    ax2.set_title("First-layer gradient norm (signal reaching the early layer)", fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("SGD step")
    ax2.set_ylabel("norm of layer-1 weight gradient")
    ax2.set_yscale("log")
    ax2.legend(frameon=False, fontsize=10)
    _despine(ax2)
    fig.suptitle("Measured: sigmoid starves the early layer of gradient; ReLU/GELU don't",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/act_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote act_training.png")
    # return final numbers so the page can quote measured values
    return {name: (results[name][0][-1], results[name][1][0]) for name, _, _ in runs}


# ---- 5. Softmax + temperature ----------------------------------------------
def act_softmax():
    logits = torch.tensor([2.0, 1.0, 0.1, -0.5])
    classes = ["cat", "dog", "bird", "fish"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.4))
    # left: logits -> probabilities
    probs = F.softmax(logits, dim=-1).numpy()
    ax1.bar(classes, logits.numpy(), color=SLATE, alpha=0.85, label="logit")
    ax1b = ax1.twinx()
    ax1b.plot(classes, probs, color=GREEN, marker="o", lw=2.4, label="softmax prob")
    for i, p in enumerate(probs):
        ax1b.text(i, p + 0.02, f"{p:.2f}", ha="center", fontsize=9.5, color=GREEN, fontweight="bold")
    ax1.axhline(0, color="#999", lw=0.8)
    ax1.set_title("logits -> softmax probabilities", fontsize=12, fontweight="bold")
    ax1.set_ylabel("logit (bars)")
    ax1b.set_ylabel("probability (line)", color=GREEN)
    ax1b.set_ylim(0, 0.8)
    _despine(ax1)
    # right: temperature
    for T, c in [(0.5, RED), (1.0, GREEN), (2.0, BLUE)]:
        p = F.softmax(logits / T, dim=-1).numpy()
        ax2.plot(classes, p, color=c, marker="o", lw=2.2, label=f"T={T}")
    ax2.set_title("temperature reshapes the distribution", fontsize=12, fontweight="bold")
    ax2.set_ylabel("probability")
    ax2.set_ylim(0, 0.9)
    ax2.annotate("low T -> peakier\nhigh T -> flatter", (2.0, 0.7),
                 fontsize=9.5, ha="center", color=SLATE, fontweight="bold")
    ax2.legend(frameon=False, fontsize=10)
    _despine(ax2)
    fig.suptitle("Softmax: exponentiate-and-normalize logits; temperature sets the sharpness",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/act_softmax.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote act_softmax.png")


if __name__ == "__main__":
    act_functions()
    act_derivatives()
    act_smooth_zoom()
    final = act_training()
    act_softmax()
    print("OUT:", OUT)
    print("MEASURED training finals (loss_last, grad_norm_step0):")
    for k, v in final.items():
        print(f"  {k:8s}: final_loss={v[0]:.4f}  layer1_grad_norm@step0={v[1]:.4e}")
