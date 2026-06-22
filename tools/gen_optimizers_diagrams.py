"""Optimizer concept-page diagrams (gold-standard set, muted palette).

All figures are MEASURED — every curve is produced by actually running the
optimizer, never hand-drawn. Five figures for
05. Deep_Learning/concepts/07-Optimizers.md:

  1. opt_trajectories.png   -- SGD vs Momentum vs RMSprop vs Adam on an
     ill-conditioned (ravine) quadratic loss-surface contour. Each reaches the
     minimum differently: SGD zig-zags, Momentum carries inertia, RMSprop /
     Adam rescale each axis and cut nearly straight.
  2. opt_loss_curves.png    -- loss vs iteration for the same four optimizers on
     the same ravine (log loss). Shows the convergence-rate ordering directly.
  3. opt_ravine_momentum.png-- the momentum-in-a-ravine effect: plain GD
     oscillating across the steep axis vs momentum cancelling the oscillation
     and accelerating along the valley floor (two panels, same surface).
  4. opt_bias_correction.png-- Adam's bias-correction factor 1/(1-beta^t) vs
     step t for beta1=0.9 and beta2=0.999: why the very first steps need it and
     why the v-correction (0.999) persists for hundreds of steps.
  5. opt_adagrad_decay.png  -- AdaGrad's effective learning rate eta/sqrt(G_t)
     decaying monotonically over steps (the "death" problem) vs RMSprop's EMA
     keeping the rate alive. Measured on a constant-gradient stream.

Run:  /Users/sudhirsingh/.uv/envs/ml-py312/bin/python3 tools/gen_optimizers_diagrams.py
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


# ---------------------------------------------------------------------------
# Shared ill-conditioned quadratic:  L(w) = 1/2 (a x^2 + b y^2),  a >> b.
# Curvature (Hessian eigenvalues) = [a, b]; condition number kappa = a/b.
# Steep in x, shallow in y  ->  a ravine. This is the canonical conditioning
# stress-test every first-order optimizer is judged on.
# ---------------------------------------------------------------------------
A = np.array([12.0, 1.0])           # kappa = 12: a pronounced ravine
def grad(w): return A * w
def loss(w): return 0.5 * float(np.sum(A * w**2))


def run(opt, lr, steps=70, start=(-9.0, -4.5), beta=0.9, b1=0.9, b2=0.999, eps=1e-8):
    """Run one optimizer on the ravine; return the (steps+1, 2) trajectory."""
    w = np.array(start, dtype=float)
    traj = [w.copy()]
    m = np.zeros(2); v = np.zeros(2); s = np.zeros(2)
    for t in range(1, steps + 1):
        g = grad(w)
        if opt == "sgd":
            w = w - lr * g
        elif opt == "momentum":
            m = beta * m + g                      # velocity EMA (Polyak)
            w = w - lr * m
        elif opt == "rmsprop":
            s = 0.9 * s + 0.1 * g**2              # EMA of squared grad
            w = w - lr * g / (np.sqrt(s) + eps)
        elif opt == "adam":
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g**2
            mh, vh = m / (1 - b1**t), v / (1 - b2**t)
            w = w - lr * mh / (np.sqrt(vh) + eps)
        traj.append(w.copy())
    return np.array(traj)


# Representative stable learning rates per optimizer on this ravine (kappa=12).
# Chosen so each shows its CHARACTERISTIC path shape without wild overshoot:
# SGD (just below the 2/12~0.167 limit) zig-zags across the steep axis;
# Momentum carves a smooth curved path; RMSprop / Adam rescale each axis and
# track much closer to a straight line into the minimum.
CFG = [("sgd", 0.13, RED, "SGD"),
       ("momentum", 0.012, AMBER, "Momentum"),
       ("rmsprop", 0.20, BLUE, "RMSprop"),
       ("adam", 0.30, GREEN, "Adam")]


def trajectories():
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    xs = np.linspace(-10, 10, 240); ys = np.linspace(-6, 6, 240)
    X, Y = np.meshgrid(xs, ys); Z = 0.5 * (A[0] * X**2 + A[1] * Y**2)
    ax.contour(X, Y, Z, levels=np.logspace(0.4, 3.2, 14),
               colors="#cbd2da", linewidths=0.8)
    for opt, lr, c, lab in CFG:
        T = run(opt, lr, steps=80)
        ax.plot(T[:, 0], T[:, 1], color=c, lw=2.1, marker="o", ms=2.4,
                label=lab, alpha=0.95)
    ax.scatter([0], [0], color=NAVY, s=120, marker="*", zorder=6, label="minimum")
    ax.set_xlabel("steep direction  x  (curvature 12)")
    ax.set_ylabel("shallow direction  y  (curvature 1)")
    ax.set_title("One ravine, four optimizers: SGD zig-zags across the steep axis",
                 fontsize=13.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper right"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/opt_trajectories.png", dpi=150,
                                    bbox_inches="tight")
    plt.close(fig); print("wrote opt_trajectories.png")


def loss_curves():
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for opt, lr, c, lab in CFG:
        T = run(opt, lr, steps=80)
        losses = [loss(w) for w in T]
        ax.semilogy(losses, color=c, lw=2.4, label=lab)
    ax.set_xlabel("iteration"); ax.set_ylabel("loss  (log scale)")
    ax.set_title("Loss vs iteration on the same ravine (measured)",
                 fontsize=13.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper right"); _despine(ax)
    ax.grid(True, which="both", ls=":", lw=0.5, color="#dfe4ea")
    fig.tight_layout(); fig.savefig(f"{OUT}/opt_loss_curves.png", dpi=150,
                                    bbox_inches="tight")
    plt.close(fig); print("wrote opt_loss_curves.png")


def ravine_momentum():
    """Plain GD oscillating vs momentum smoothing — two panels, same surface."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.8))
    xs = np.linspace(-10, 10, 240); ys = np.linspace(-6, 6, 240)
    X, Y = np.meshgrid(xs, ys); Z = 0.5 * (A[0] * X**2 + A[1] * Y**2)
    lr = 0.155                                   # just below GD's stability limit 2/a = 0.167
    Tg = run("sgd", lr, steps=50)
    Tm = run("momentum", 0.004, steps=50, beta=0.9)   # low LR -> smooth, damped descent
    for ax, T, c, ttl in [(ax1, Tg, RED, "Plain GD: oscillates across the steep axis"),
                          (ax2, Tm, GREEN, "Momentum: oscillation cancels, valley accelerates")]:
        ax.contour(X, Y, Z, levels=np.logspace(0.4, 3.2, 14),
                   colors="#cbd2da", linewidths=0.8)
        ax.plot(T[:, 0], T[:, 1], color=c, lw=2.0, marker="o", ms=3.0, alpha=0.95)
        ax.scatter([0], [0], color=NAVY, s=110, marker="*", zorder=6)
        ax.set_xlabel("steep direction  x"); ax.set_title(ttl, fontsize=12, fontweight="bold")
        _despine(ax)
    ax1.set_ylabel("shallow direction  y")
    fig.suptitle("Why momentum helps in a ravine (same surface, same start)",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/opt_ravine_momentum.png", dpi=150,
                                    bbox_inches="tight")
    plt.close(fig); print("wrote opt_ravine_momentum.png")


def bias_correction():
    """Adam's bias-correction factor 1/(1-beta^t) vs step t."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    t = np.arange(1, 1001)
    f1 = 1.0 / (1 - 0.9**t)          # first-moment correction (beta1)
    f2 = 1.0 / (1 - 0.999**t)        # second-moment correction (beta2)
    ax.plot(t, f1, color=PURPLE, lw=2.6, label=r"$1/(1-\beta_1^{\,t})$,  $\beta_1=0.9$  (m̂)")
    ax.plot(t, f2, color=AMBER, lw=2.6, label=r"$1/(1-\beta_2^{\,t})$,  $\beta_2=0.999$  (v̂)")
    ax.axhline(1.0, color=SLATE, ls=":", lw=1.3)
    ax.text(620, 1.05, "no correction needed (→ 1)", color=SLATE, fontsize=9.5)
    # annotate the big early values
    ax.annotate(f"step 1: ×{f1[0]:.1f}", (1, f1[0]), xytext=(60, 9.0),
                textcoords="data", color=PURPLE, fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PURPLE))
    ax.annotate(f"step 1: ×{f2[0]:.0f}", (1, f2[0]), xytext=(60, f2[0]),
                textcoords="data", color=AMBER, fontsize=9.5, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER))
    ax.set_xlabel("optimizer step  t"); ax.set_ylabel("bias-correction multiplier")
    ax.set_title("Adam bias correction: huge early, fades to 1",
                 fontsize=13.5, fontweight="bold")
    ax.set_ylim(0, 12); ax.legend(frameon=False, fontsize=10, loc="upper right")
    _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/opt_bias_correction.png", dpi=150,
                                    bbox_inches="tight")
    plt.close(fig); print("wrote opt_bias_correction.png")


def adagrad_decay():
    """AdaGrad effective LR decaying to ~0 vs RMSprop staying alive."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    steps = 300
    g = 1.0                                   # constant unit gradient stream
    eta, eps, rho = 0.1, 1e-8, 0.9
    G = 0.0; s = 0.0
    ada_eff, rms_eff = [], []
    for t in range(1, steps + 1):
        G += g**2                             # AdaGrad: running SUM of g^2
        s = rho * s + (1 - rho) * g**2        # RMSprop: EMA of g^2
        ada_eff.append(eta / (np.sqrt(G) + eps))
        rms_eff.append(eta / (np.sqrt(s) + eps))
    ax.plot(range(1, steps + 1), ada_eff, color=RED, lw=2.6,
            label=r"AdaGrad:  $\eta/\sqrt{\sum g^2}$  (decays to 0)")
    ax.plot(range(1, steps + 1), rms_eff, color=GREEN, lw=2.6,
            label=r"RMSprop:  $\eta/\sqrt{\mathrm{EMA}[g^2]}$  (stays alive)")
    ax.set_xlabel("step  t"); ax.set_ylabel("effective learning rate")
    ax.set_title("AdaGrad's rate dies, RMSprop's survives (constant gradient)",
                 fontsize=13.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="upper right"); _despine(ax)
    ax.grid(True, ls=":", lw=0.5, color="#dfe4ea")
    fig.tight_layout(); fig.savefig(f"{OUT}/opt_adagrad_decay.png", dpi=150,
                                    bbox_inches="tight")
    plt.close(fig); print("wrote opt_adagrad_decay.png")


def lr_robustness():
    """Adam's REAL headline advantage: final loss vs learning rate, swept over
    three orders of magnitude. SGD has a narrow sweet spot and diverges past it;
    Adam stays low across a wide band -- the 'forgiving' property. Measured."""
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    lrs = np.logspace(-3, 0.2, 60)
    for opt, c, lab in [("sgd", RED, "SGD"), ("momentum", AMBER, "Momentum"),
                        ("rmsprop", BLUE, "RMSprop"), ("adam", GREEN, "Adam")]:
        finals = []
        for lr in lrs:
            T = run(opt, lr, steps=80)
            fl = loss(T[-1])
            finals.append(fl if (np.isfinite(fl) and fl < 1e8) else 1e8)
        ax.loglog(lrs, finals, color=c, lw=2.5, label=lab)
    ax.axhline(1e8, color=SLATE, ls=":", lw=1.2)
    ax.text(lrs[1], 1.5e8, "diverged", color=SLATE, fontsize=9, fontweight="bold")
    ax.set_xlabel("learning rate (log)"); ax.set_ylabel("final loss after 80 steps (log)")
    ax.set_title("Adam tolerates a wide learning-rate band; SGD has a narrow sweet spot",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=10, loc="lower left"); _despine(ax)
    ax.grid(True, which="both", ls=":", lw=0.5, color="#dfe4ea")
    fig.tight_layout(); fig.savefig(f"{OUT}/opt_lr_robustness.png", dpi=150,
                                    bbox_inches="tight")
    plt.close(fig); print("wrote opt_lr_robustness.png")


if __name__ == "__main__":
    trajectories()
    loss_curves()
    ravine_momentum()
    bias_correction()
    adagrad_decay()
    lr_robustness()
    print("OUT:", OUT)
