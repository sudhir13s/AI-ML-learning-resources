"""Activation-functions concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 05. Deep_Learning/concepts/03-Activation-Functions.md:
  1. act_functions.png -- left: sigmoid/tanh/ReLU/GELU/LeakyReLU; right: their
     DERIVATIVES, showing sigmoid/tanh saturate to ~0 (vanishing gradient) while
     ReLU's derivative stays 1 for x>0.
  2. act_softmax.png   -- a vector of logits -> softmax probabilities (exp-and-
     normalize), as paired bars.
"""
import os, matplotlib
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


def _sigmoid(x): return 1 / (1 + np.exp(-x))
def _gelu(x): return 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))


def act_functions():
    x = np.linspace(-4, 4, 600)
    funcs = [
        ("sigmoid", _sigmoid(x), BLUE),
        ("tanh", np.tanh(x), PURPLE),
        ("ReLU", np.maximum(0, x), GREEN),
        ("GELU", _gelu(x), AMBER),
        ("Leaky ReLU", np.where(x > 0, x, 0.1 * x), RED),
    ]
    s = _sigmoid(x)
    derivs = [
        ("sigmoid'", s * (1 - s), BLUE),
        ("tanh'", 1 - np.tanh(x)**2, PURPLE),
        ("ReLU'", (x > 0).astype(float), GREEN),
        ("GELU'", np.gradient(_gelu(x), x), AMBER),
        ("Leaky'", np.where(x > 0, 1.0, 0.1), RED),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for name, y, c in funcs:
        ax1.plot(x, y, color=c, lw=2.2, label=name)
    ax1.axhline(0, color="#bbb", lw=0.8); ax1.axvline(0, color="#bbb", lw=0.8)
    ax1.set_title("The activation functions", fontsize=12.5, fontweight="bold")
    ax1.set_xlabel("input"); ax1.set_ylabel("output"); ax1.set_ylim(-1.6, 4); ax1.legend(frameon=False, fontsize=9)
    _despine(ax1)
    for name, y, c in derivs:
        ax2.plot(x, y, color=c, lw=2.2, label=name)
    ax2.axhline(0, color="#bbb", lw=0.8); ax2.axvline(0, color="#bbb", lw=0.8)
    ax2.axhspan(-0.02, 0.05, color=RED, alpha=0.07)
    ax2.annotate("sigmoid/tanh saturate →\nderivative ≈ 0 →\nvanishing gradients", (2.6, 0.55),
                 fontsize=9, color=RED, fontweight="bold", ha="center")
    ax2.set_title("Their derivatives (the gradient that flows back)", fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("input"); ax2.set_ylabel("derivative"); ax2.set_ylim(-0.1, 1.25); ax2.legend(frameon=False, fontsize=9)
    _despine(ax2)
    fig.suptitle("Why ReLU won hidden layers: its gradient doesn't vanish for positive inputs",
                 fontsize=13.5, fontweight="bold", y=1.03)
    fig.tight_layout(); fig.savefig(f"{OUT}/act_functions.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote act_functions.png")


def act_softmax():
    logits = np.array([2.0, 1.0, 0.1, -0.5])
    classes = ["cat", "dog", "bird", "fish"]
    e = np.exp(logits - logits.max()); probs = e / e.sum()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax1.bar(classes, logits, color=SLATE, alpha=0.85)
    ax1.axhline(0, color="#999", lw=0.8)
    ax1.set_title("raw logits (any real number)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("logit"); _despine(ax1)
    bars = ax2.bar(classes, probs, color=GREEN, alpha=0.9)
    for b, p in zip(bars, probs):
        ax2.text(b.get_x() + b.get_width()/2, p + 0.01, f"{p:.2f}", ha="center", fontsize=10, fontweight="bold")
    ax2.set_title("softmax → probabilities (sum to 1)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("probability"); ax2.set_ylim(0, 0.75); _despine(ax2)
    fig.suptitle("Softmax: exponentiate and normalize logits into a distribution",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/act_softmax.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote act_softmax.png")


if __name__ == "__main__":
    act_functions()
    act_softmax()
    print("OUT:", OUT)
