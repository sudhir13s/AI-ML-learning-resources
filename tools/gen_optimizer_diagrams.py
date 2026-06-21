"""Optimizer concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 05. Deep_Learning/concepts/07-Optimizers.md:
  1. opt_trajectories.png -- SGD vs Momentum vs Adam on an ill-conditioned
     (ravine) quadratic: SGD zig-zags, Momentum damps it, Adam goes straight.
  2. opt_convergence.png  -- (left) loss vs step for SGD/Momentum/RMSprop/Adam;
     (right) Adam's per-parameter effective step: a steady vs a noisy parameter.
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


# ill-conditioned quadratic: steep in x, shallow in y  ->  a ravine
A = np.array([6.0, 1.0])
def grad(w): return A * w
def loss(w): return 0.5 * np.sum(A * w**2)


def run(opt, lr, steps=40, start=(-9.0, -4.5)):
    w = np.array(start); traj = [w.copy()]
    m = np.zeros(2); v = np.zeros(2); s = np.zeros(2)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        g = grad(w)
        if opt == "sgd":
            w = w - lr * g
        elif opt == "momentum":
            m = 0.9 * m + g; w = w - lr * m
        elif opt == "rmsprop":
            s = 0.9 * s + 0.1 * g**2; w = w - lr * g / (np.sqrt(s) + eps)
        elif opt == "adam":
            m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g**2
            mh, vh = m / (1 - b1**t), v / (1 - b2**t)
            w = w - lr * mh / (np.sqrt(vh) + eps)
        traj.append(w.copy())
    return np.array(traj)


def trajectories():
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    xs = np.linspace(-10, 10, 200); ys = np.linspace(-6, 6, 200)
    X, Y = np.meshgrid(xs, ys); Z = 0.5 * (A[0] * X**2 + A[1] * Y**2)
    ax.contour(X, Y, Z, levels=np.logspace(0.2, 2.6, 12), colors="#cbd2da", linewidths=0.8)
    for opt, lr, c, lab in [("sgd", 0.30, RED, "SGD"), ("momentum", 0.016, AMBER, "Momentum"),
                            ("adam", 0.45, GREEN, "Adam")]:
        T = run(opt, lr)
        ax.plot(T[:, 0], T[:, 1], color=c, lw=2.2, marker="o", ms=2.6, label=lab, alpha=0.95)
    ax.scatter([0], [0], color=NAVY, s=90, marker="*", zorder=6, label="minimum")
    ax.set_xlabel("steep direction (x)"); ax.set_ylabel("shallow direction (y)")
    ax.set_title("Same ravine, three optimizers: SGD zig-zags, Adam goes straight",
                 fontsize=13.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper right"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/opt_trajectories.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote opt_trajectories.png")


def convergence():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.6))
    # left: sensitivity to the learning rate — final loss after 60 steps vs lr.
    lrs = np.logspace(-2.3, 0.3, 40)
    for opt, c, lab in [("sgd", RED, "SGD"), ("adam", GREEN, "Adam")]:
        finals = []
        for lr in lrs:
            T = run(opt, lr, steps=60)
            fl = loss(T[-1]); finals.append(fl if np.isfinite(fl) and fl < 1e6 else 1e6)
        ax1.loglog(lrs, finals, color=c, lw=2.5, label=lab)
    ax1.axhline(1e6, color=SLATE, ls=":", lw=1.2)
    ax1.text(lrs[1], 1.4e6, "diverged", color=SLATE, fontsize=9, fontweight="bold")
    ax1.set_xlabel("learning rate"); ax1.set_ylabel("final loss after 60 steps")
    ax1.set_title("Adam is forgiving about the learning rate", fontsize=13, fontweight="bold")
    ax1.legend(frameon=False, fontsize=9.5, loc="lower left"); _despine(ax1)
    ax1.annotate("SGD diverges once\nlr is too large", (0.5, 1e6), textcoords="offset points",
                 xytext=(-104, -30), fontsize=8.8, color=RED, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=RED))

    # right: Adam per-parameter effective step — steady vs noisy gradient
    b1, b2, eps = 0.9, 0.999, 1e-8
    rng = np.random.default_rng(0)
    steady = np.ones(60) * 1.0
    noisy = rng.choice([1.0, -0.8], size=60)            # thrashing gradient
    def adam_step(g):
        m = v = 0.0; out = []
        for t in range(1, len(g) + 1):
            m = b1 * m + (1 - b1) * g[t-1]; v = b2 * v + (1 - b2) * g[t-1]**2
            mh, vh = m / (1 - b1**t), v / (1 - b2**t)
            out.append(mh / (np.sqrt(vh) + eps))
        return np.array(out)
    ax2.plot(np.abs(adam_step(steady)), color=GREEN, lw=2.4, label="steady gradient → full step")
    ax2.plot(np.abs(adam_step(noisy)), color=RED, lw=2.4, label="noisy gradient → throttled")
    ax2.set_xlabel("step"); ax2.set_ylabel("|effective step| of Adam")
    ax2.set_title("Adam adapts the step per parameter", fontsize=13, fontweight="bold")
    ax2.legend(frameon=False, fontsize=9.5); _despine(ax2); ax2.set_ylim(0, 1.2)
    fig.tight_layout(); fig.savefig(f"{OUT}/opt_convergence.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote opt_convergence.png")


if __name__ == "__main__":
    trajectories()
    convergence()
    print("OUT:", OUT)
