"""Dropout concept-page diagrams (muted palette, parallel matplotlib scale).

Four figures for 05. Deep_Learning/concepts/10-Dropout.md:
  1. dropout_subnet.png   -- full network (test) vs a sampled thinned sub-network
     (training): ~half the hidden units zeroed, their edges removed. Each step
     trains a different sub-network -> an implicit ensemble of 2^n thinned nets.
  2. dropout_scaling.png  -- inverted dropout: drop a fraction p, scale survivors
     by 1/(1-p), so the EXPECTED activation is unchanged and test time stays clean.
  3. dropout_overfit.png  -- a MEASURED train/val curve on a tiny MLP: without
     dropout the val loss diverges from train (overfitting); with dropout the
     gap closes. This is computed live with torch, not drawn by hand.
  4. dropout_rate_sweep.png -- MEASURED train vs test accuracy as the rate p is
     swept 0 -> ~0.9 on the same MLP: a sweet spot in the middle, underfitting
     past it. Shows why p is a tuned hyperparameter, not a free knob.

Run with the torch-enabled env:
  ~/.uv/envs/ml-py312/bin/python3 tools/gen_dropout_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
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


def _draw_net(ax, dropped, title, tcol):
    layers = [3, 5, 5, 2]
    xs = [0, 1.4, 2.8, 4.2]
    pos = {}
    for li, (n, x) in enumerate(zip(layers, xs)):
        ys = np.linspace(0.5, 4.5, n)
        for ni, y in enumerate(ys):
            pos[(li, ni)] = (x, y)
    # edges
    for li in range(len(layers) - 1):
        for a in range(layers[li]):
            for b in range(layers[li + 1]):
                dead = (li, a) in dropped or (li + 1, b) in dropped
                ax.plot([pos[(li, a)][0], pos[(li + 1, b)][0]],
                        [pos[(li, a)][1], pos[(li + 1, b)][1]],
                        color="#dfe3e8" if dead else SLATE, lw=0.6,
                        alpha=0.35 if dead else 0.5, zorder=1)
    # nodes
    for (li, ni), (x, y) in pos.items():
        is_drop = (li, ni) in dropped
        col = "#d7dbe0" if is_drop else (BLUE if li == 0 else GREEN if li == len(layers)-1 else PURPLE)
        ax.add_patch(Circle((x, y), 0.18, facecolor=col, edgecolor="white", lw=1.4, zorder=3))
        if is_drop:
            ax.plot([x-0.13, x+0.13], [y-0.13, y+0.13], color=RED, lw=2, zorder=4)
            ax.plot([x-0.13, x+0.13], [y+0.13, y-0.13], color=RED, lw=2, zorder=4)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=tcol)
    ax.set_xlim(-0.4, 4.6); ax.set_ylim(-0.1, 5.3); ax.axis("off"); ax.set_aspect("equal")


def dropout_subnet():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.8))
    _draw_net(ax1, set(), "full network — test time (all units kept)", GREEN)
    dropped = {(1, 0), (1, 3), (2, 1), (2, 4)}        # ~half the hidden units dropped
    _draw_net(ax2, dropped, "training pass — a thinned sub-network (p ≈ 0.5)", RED)
    fig.suptitle("Dropout samples a different sub-network every forward pass (an implicit ensemble of 2^n)",
                 fontsize=13.5, fontweight="bold", y=1.0)
    fig.tight_layout(); fig.savefig(f"{OUT}/dropout_subnet.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dropout_subnet.png")


def dropout_scaling():
    rng = np.random.default_rng(3)
    a = np.abs(rng.normal(1.0, 0.3, 8)) + 0.4         # activations
    p = 0.4
    one = a * np.array([1, 0, 1, 1, 0, 1, 0, 1], float) / (1 - p)   # one inverted-dropout pass
    masks = (rng.random((20000, 8)) > p).astype(float)             # average over many masks
    avg = (masks * a / (1 - p)).mean(0)                            # E[inverted dropout] per unit
    idx = np.arange(8)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), sharey=True)
    for ax, (vals, ttl, col) in zip(axes, [
            (a, "original activations", SLATE),
            (one, f"one training pass (drop p={p},\nsurvivors ×1/(1−p))", RED),
            (avg, "average over many passes\n→ matches the original", GREEN)]):
        ax.bar(idx, vals, color=col, alpha=0.85)
        ax.set_title(ttl, fontsize=11, fontweight="bold", color=col)
        ax.set_xlabel("unit"); _despine(ax); ax.set_xticks(idx)
    axes[0].set_ylabel("activation")
    fig.suptitle("Inverted dropout: each unit's EXPECTED activation is unchanged, so test time needs no rescaling",
                 fontsize=12.5, fontweight="bold", y=1.04)
    fig.tight_layout(); fig.savefig(f"{OUT}/dropout_scaling.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dropout_scaling.png")


# ---------------------------------------------------------------------------
# MEASURED experiments (torch). A deliberately over-parameterized MLP on a tiny,
# noisy 2-class dataset: enough capacity to memorize, little enough data to overfit
# -- the canonical setting where dropout helps. Everything below is computed live.
# ---------------------------------------------------------------------------
def _two_moons(torch, n, sd, jitter):
    # the classic two-interleaving-half-moons 2-class task, generated by hand.
    gg = torch.Generator().manual_seed(sd)
    n0 = n // 2; n1 = n - n0
    t0 = torch.rand(n0, generator=gg) * np.pi                 # upper moon
    x0 = torch.stack([torch.cos(t0), torch.sin(t0)], 1)
    t1 = torch.rand(n1, generator=gg) * np.pi                 # lower moon, shifted
    x1 = torch.stack([1 - torch.cos(t1), -torch.sin(t1) + 0.4], 1)
    X = torch.cat([x0, x1], 0) + jitter * torch.randn(n, 2, generator=gg)
    y = torch.cat([torch.zeros(n0), torch.ones(n1)]).long()
    perm = torch.randperm(n, generator=gg)
    return X[perm], y[perm]


def _make_data(torch, n_train=120, n_test=4000, label_noise=0.18, seed=0):
    # Overfit regime: two moons with HEAVY jitter + flipped TRAIN labels. A wide
    # net can memorize the noisy training points (incl. the flips), so without
    # regularization it carves a jagged boundary and test accuracy drops. Dropout
    # smooths the boundary toward the true moons -> higher, cleaner test accuracy.
    Xtr, ytr = _two_moons(torch, n_train, seed + 1, jitter=0.35)
    m = torch.rand(n_train, generator=torch.Generator().manual_seed(seed + 9)) < label_noise
    ytr[m] = 1 - ytr[m]                                       # corrupt some train labels
    Xte, yte = _two_moons(torch, n_test, seed + 2, jitter=0.18)   # cleaner test set
    return Xtr, ytr, Xte, yte


def _train_mlp(torch, nn, p, epochs=400, seed=0, hidden=512, lr=3e-3, wd=0.0,
               data=None, record=False):
    torch.manual_seed(seed)
    Xtr, ytr, Xte, yte = data
    d = Xtr.shape[1]
    # a deliberately over-wide MLP; dropout on the hidden layers (classic placement)
    net = nn.Sequential(
        nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(p),
        nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(p),
        nn.Linear(hidden, 2))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss()
    tr_hist, te_hist = [], []
    for ep in range(epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(Xtr), ytr)
        loss.backward(); opt.step()
        if record:
            net.eval()
            with torch.no_grad():
                tr_hist.append(lossf(net(Xtr), ytr).item())
                te_hist.append(lossf(net(Xte), yte).item())
    net.eval()
    with torch.no_grad():
        tr_acc = (net(Xtr).argmax(1) == ytr).float().mean().item()
        te_acc = (net(Xte).argmax(1) == yte).float().mean().item()
    return net, tr_acc, te_acc, tr_hist, te_hist


def dropout_overfit():
    import torch, torch.nn as nn
    data = _make_data(torch)
    _, a0tr, a0te, h0tr, h0te = _train_mlp(torch, nn, p=0.0, epochs=600, data=data, record=True)
    _, a1tr, a1te, h1tr, h1te = _train_mlp(torch, nn, p=0.5, epochs=600, data=data, record=True)
    ep = np.arange(1, len(h0tr) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.6), sharey=True)
    ax1.plot(ep, h0tr, color=BLUE, lw=2, label="train loss")
    ax1.plot(ep, h0te, color=RED, lw=2, label="val loss")
    ax1.fill_between(ep, h0tr, h0te, color=RED, alpha=0.10)
    ax1.set_title(f"no dropout (p=0)\ntrain−test acc gap {a0tr-a0te:+.2f}, overfits",
                  fontsize=11.5, fontweight="bold", color=RED)
    ax2.plot(ep, h1tr, color=BLUE, lw=2, label="train loss")
    ax2.plot(ep, h1te, color=GREEN, lw=2, label="val loss")
    ax2.fill_between(ep, h1tr, h1te, color=GREEN, alpha=0.10)
    ax2.set_title(f"dropout p=0.5\ntrain−test acc gap {a1tr-a1te:+.2f}, gap closes",
                  fontsize=11.5, fontweight="bold", color=GREEN)
    for ax in (ax1, ax2):
        ax.set_xlabel("epoch"); _despine(ax); ax.legend(frameon=False, fontsize=10)
    ax1.set_ylabel("cross-entropy loss")
    fig.suptitle("Measured: dropout closes the train/validation gap on an over-parameterized MLP",
                 fontsize=12.5, fontweight="bold", y=1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/dropout_overfit.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote dropout_overfit.png  | p=0 train/test acc {a0tr:.3f}/{a0te:.3f}"
                          f"  | p=0.5 {a1tr:.3f}/{a1te:.3f}")
    return (a0tr, a0te, a1tr, a1te)


def dropout_rate_sweep():
    import torch, torch.nn as nn
    data = _make_data(torch)
    ps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    tr, te = [], []
    for p in ps:                                          # average a few seeds to smooth the curve
        a_trs, a_tes = [], []
        for s in range(4):
            _, a_tr, a_te, _, _ = _train_mlp(torch, nn, p=p, epochs=600, seed=s, data=data)
            a_trs.append(a_tr); a_tes.append(a_te)
        tr.append(float(np.mean(a_trs))); te.append(float(np.mean(a_tes)))
    best = int(np.argmax(te))
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(ps, tr, "o-", color=BLUE, lw=2, ms=6, label="train accuracy")
    ax.plot(ps, te, "s-", color=GREEN, lw=2, ms=6, label="test accuracy")
    ax.axvline(ps[best], color=AMBER, ls="--", lw=1.6, alpha=0.8)
    ax.annotate(f"sweet spot\np={ps[best]} (test {te[best]:.2f})",
                xy=(ps[best], te[best]), xytext=(ps[best] + 0.05, te[best] - 0.13),
                fontsize=10, color=AMBER, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=AMBER))
    ax.set_xlabel("dropout rate p"); ax.set_ylabel("accuracy")
    ax.set_title("Measured: test accuracy peaks at a middling p — too much dropout underfits",
                 fontsize=11.5, fontweight="bold")
    _despine(ax); ax.legend(frameon=False, fontsize=10); ax.grid(alpha=0.15)
    fig.tight_layout(); fig.savefig(f"{OUT}/dropout_rate_sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote dropout_rate_sweep.png | p, train, test:")
    for p, a, b in zip(ps, tr, te):
        print(f"  p={p:.1f}  train={a:.3f}  test={b:.3f}")
    return ps, tr, te


if __name__ == "__main__":
    dropout_subnet()
    dropout_scaling()
    dropout_overfit()
    dropout_rate_sweep()
    print("OUT:", OUT)
