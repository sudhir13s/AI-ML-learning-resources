"""Deep-learning regularization concept-page diagrams (muted palette, parallel matplotlib scale).

Four MEASURED figures for 05. Deep_Learning/concepts/09-Regularization.md. Every curve is
produced by actually training/measuring a small model or computing the closed-form quantity —
nothing is hand-faked. Run with the project's torch env (Python 3.12).

  1. reg_overfit_vs_regularized.png -- a real over-parameterized MLP trained on a tiny noisy
     dataset: an unregularized run drives train loss to ~0 while val loss climbs (overfitting),
     and a weight-decay+dropout run keeps the train/val gap closed. MEASURED.
  2. reg_earlystop_measured.png -- val loss from the SAME overfitting run, U-shaped, with the
     early-stopping point annotated at the measured val minimum. MEASURED.
  3. reg_label_smoothing.png -- the softmax distribution a one-hot target pushes toward (a
     near-delta spike) vs the softened target label smoothing asks for; plus the CE-loss /
     gradient consequence. COMPUTED.
  4. reg_weightdecay_shrink.png -- the geometric weight-norm decay a pure weight-decay step
     produces over iterations (multiplicative (1-eta*lambda) shrink), for several lambda. COMPUTED.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

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
# A small over-parameterized MLP trained on a tiny noisy regression problem.
# Returns (epochs, train_curve, val_curve) for a given weight_decay / dropout.
# ---------------------------------------------------------------------------
def _train_run(weight_decay=0.0, dropout=0.0, epochs=1500, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    # A few noisy training points from a smooth function; a wide MLP can memorize them
    # (drive train loss to ~0) while doing badly on the clean underlying signal.
    n_tr, n_val = 18, 300
    def target(x):
        return torch.sin(3 * x) + 0.3 * x
    g = torch.Generator().manual_seed(seed)
    x_tr = torch.linspace(-2, 2, n_tr).unsqueeze(1)
    y_tr = target(x_tr) + 0.45 * torch.randn(n_tr, 1, generator=g)
    x_val = torch.linspace(-2, 2, n_val).unsqueeze(1)
    y_val = target(x_val)  # clean validation = the true signal

    layers, prev = [], 1
    for h in (256, 256, 256):
        layers += [nn.Linear(prev, h), nn.Tanh()]
        if dropout > 0:
            layers += [nn.Dropout(dropout)]
        prev = h
    layers += [nn.Linear(prev, 1)]
    net = nn.Sequential(*layers)

    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=weight_decay)
    lossf = nn.MSELoss()
    tr_curve, val_curve = [], []
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(x_tr), y_tr)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            tr_curve.append(lossf(net(x_tr), y_tr).item())
            val_curve.append(lossf(net(x_val), y_val).item())
    return np.arange(1, epochs + 1), np.array(tr_curve), np.array(val_curve)


def overfit_vs_regularized():
    ep, tr0, val0 = _train_run(weight_decay=0.0, dropout=0.0)        # overfits
    ep, tr1, val1 = _train_run(weight_decay=8e-2, dropout=0.15)      # regularized
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), sharey=True)

    ax = axes[0]
    ax.plot(ep, tr0, color=BLUE, lw=2.3, label="train loss")
    ax.plot(ep, val0, color=RED, lw=2.3, label="val loss")
    gap0 = val0[-1] - tr0[-1]
    ax.fill_between(ep, tr0, val0, where=(val0 > tr0), color=RED, alpha=0.10)
    ax.annotate(f"generalization gap\n→ {gap0:.2f}", (ep[-1], (tr0[-1] + val0[-1]) / 2),
                textcoords="offset points", xytext=(-118, 6), fontsize=9.5,
                color=RED, fontweight="bold")
    ax.set_title(f"No regularization — overfits\n(train {tr0[-1]:.3f}, val {val0[-1]:.3f})",
                 fontsize=11.5, fontweight="bold", color=RED)
    ax.set_xlabel("epoch"); ax.set_ylabel("MSE loss"); ax.legend(frameon=False, fontsize=9.5)
    _despine(ax)

    ax = axes[1]
    ax.plot(ep, tr1, color=BLUE, lw=2.3, label="train loss")
    ax.plot(ep, val1, color=GREEN, lw=2.3, label="val loss")
    gap1 = val1[-1] - tr1[-1]
    ax.fill_between(ep, tr1, val1, where=(val1 > tr1), color=GREEN, alpha=0.10)
    ax.annotate(f"gap closed\n→ {gap1:.2f}", (ep[-1], (tr1[-1] + val1[-1]) / 2),
                textcoords="offset points", xytext=(-92, 18), fontsize=9.5,
                color=GREEN, fontweight="bold")
    ax.set_title(f"Weight decay + dropout — generalizes\n(train {tr1[-1]:.3f}, val {val1[-1]:.3f})",
                 fontsize=11.5, fontweight="bold", color=GREEN)
    ax.set_xlabel("epoch"); ax.legend(frameon=False, fontsize=9.5); _despine(ax)
    ax.set_ylim(0, max(val0.max(), 1.0) * 1.05)

    fig.suptitle("Measured: the same over-parameterized MLP overfits, then regularization closes the gap",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/reg_overfit_vs_regularized.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote reg_overfit_vs_regularized.png  (overfit gap {gap0:.3f} -> regularized gap {gap1:.3f})")
    return ep, val0


def earlystop_measured(ep, val0):
    stop = int(np.argmin(val0))
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.plot(ep, val0, color=RED, lw=2.4, label="validation loss (measured, U-shaped)")
    ax.axvline(ep[stop], color=GREEN, ls="--", lw=2)
    ax.scatter([ep[stop]], [val0[stop]], color=GREEN, s=75, zorder=6)
    ax.annotate(f"early stop @ epoch {ep[stop]}\n(val min = {val0[stop]:.3f})",
                (ep[stop], val0[stop]), textcoords="offset points", xytext=(16, 26),
                fontsize=10, color=GREEN, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.axvspan(ep[stop], ep[-1], color=RED, alpha=0.06)
    ax.text((ep[stop] + ep[-1]) / 2, 0.42, "overfitting region\n(val rises, then plateaus high)",
            ha="center", fontsize=9.5, color=RED)
    ax.set_xlabel("training epoch"); ax.set_ylabel("validation MSE loss")
    ax.set_title("Early stopping: halt at the measured validation-loss minimum",
                 fontsize=13, fontweight="bold")
    ax.set_ylim(0, 0.55)  # zoom into the U so the minimum + rise are legible
    ax.legend(frameon=False, fontsize=9.5); _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/reg_earlystop_measured.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote reg_earlystop_measured.png  (val min at epoch {ep[stop]})")


def label_smoothing():
    # A 6-class problem. Logits the net would push toward under one-hot vs the
    # distribution label smoothing actually asks it to match.
    K = 6
    logits = torch.tensor([4.0, 1.5, 0.8, 0.2, -0.5, -1.0])
    p = torch.softmax(logits, dim=0).numpy()                       # current model softmax
    onehot = np.zeros(K); onehot[0] = 1.0
    eps = 0.1
    ls = (1 - eps) * onehot + eps / K                              # smoothed target

    x = np.arange(K)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    ax = axes[0]
    w = 0.38
    ax.bar(x - w / 2, onehot, w, color=RED, alpha=0.85, label="one-hot target (ε=0)")
    ax.bar(x + w / 2, ls, w, color=GREEN, alpha=0.9, label=f"smoothed target (ε={eps})")
    ax.set_title("Target distribution: one-hot vs label-smoothed\n"
                 f"correct class {1-eps:.2f}, each other class {eps/K:.3f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("class"); ax.set_ylabel("target probability")
    ax.set_xticks(x); ax.legend(frameon=False, fontsize=9.5); _despine(ax); ax.set_ylim(0, 1.05)

    ax = axes[1]
    ax.bar(x, p, color=BLUE, alpha=0.88)
    ax.axhline(1 - eps, color=GREEN, ls="--", lw=1.6)
    ax.text(K - 1, 1 - eps + 0.02, f"label-smoothing ceiling {1-eps:.2f}",
            ha="right", fontsize=9, color=GREEN, fontweight="bold")
    ax.set_title("Why it helps: one-hot pulls the top logit → +∞ (over-confident);\n"
                 "smoothing caps the target so logits stay finite & calibrated",
                 fontsize=10.5, fontweight="bold")
    ax.set_xlabel("class"); ax.set_ylabel("model softmax  p(class)")
    ax.set_xticks(x); _despine(ax); ax.set_ylim(0, 1.05)

    fig.suptitle("Label smoothing softens the target so the network stops chasing infinite confidence",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(f"{OUT}/reg_label_smoothing.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote reg_label_smoothing.png  (smoothed correct target = {1-eps:.2f}, others = {eps/K:.4f})")


def weightdecay_shrink():
    # Pure weight-decay dynamics: with no data gradient, ||w|| decays geometrically
    # by (1 - eta*lambda) each step. Plot the measured trajectory for several lambda.
    eta = 0.1
    steps = 60
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    colors = [BLUE, PURPLE, AMBER, RED]
    for lam, c in zip([0.0, 0.2, 0.5, 1.0], colors):
        w = 2.0
        traj = [w]
        factor = 1 - eta * lam
        for _ in range(steps):
            w = factor * w               # pure weight-decay step, data gradient = 0
            traj.append(w)
        ax.plot(range(steps + 1), traj, color=c, lw=2.3,
                label=f"λ={lam}  →  shrink ×{factor:.3f}/step")
    ax.set_title("Weight decay shrinks the weight geometrically each step:  "
                 r"$w \leftarrow (1-\eta\lambda)\,w$",
                 fontsize=12.5, fontweight="bold")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("weight value  w")
    ax.axhline(0, color="#999", lw=0.8)
    ax.legend(frameon=False, fontsize=9.5); _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/reg_weightdecay_shrink.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote reg_weightdecay_shrink.png")


if __name__ == "__main__":
    ep, val0 = overfit_vs_regularized()
    earlystop_measured(ep, val0)
    label_smoothing()
    weightdecay_shrink()
    print("OUT:", OUT)
