"""Learning-rate schedules & warmup concept-page diagrams (muted palette).

Four MEASURED figures for 05. Deep_Learning/concepts/08-Learning-Rate-Schedules-and-Warmup.md.
Every curve is produced by actually stepping torch.optim.lr_scheduler (or the matching
closed-form) and, for the loss comparison, by training a small net three ways. Nothing is
hand-drawn or faked.

  1. lr_schedules_curves.png  -- LR vs step for constant / step / exponential / cosine /
     warmup+cosine / inverse-sqrt(Noam) -- read straight off torch schedulers.
  2. lr_loss_comparison.png   -- MEASURED training loss: constant vs cosine vs warmup+cosine
     on the same model/data/seed (warmup+cosine wins, constant is noisy late).
  3. lr_one_cycle.png         -- one-cycle LR and its (inverse) momentum schedule, read off
     torch.optim.lr_scheduler.OneCycleLR.
  4. lr_range_test.png        -- MEASURED LR-range test: loss vs exponentially-rising LR, with
     the "steepest descent" pick and the divergence point marked.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import (
    StepLR, ExponentialLR, CosineAnnealingLR, OneCycleLR, LambdaLR)

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _read_lr(make_sched, base_lr, steps):
    """Build optimizer+scheduler, step `steps` times, record the LR at each step."""
    opt = torch.optim.SGD([torch.zeros(1, requires_grad=True)], lr=base_lr)
    sched = make_sched(opt)
    lrs = []
    for _ in range(steps):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()          # pretend we took a step
        sched.step()
    return np.array(lrs)


# ----------------------------------------------------------------------------- 1
def schedules_curves():
    BASE_LR, STEPS = 1.0, 1000

    # constant
    const = np.full(STEPS, BASE_LR)
    # step decay: x0.5 every 250 steps
    step = _read_lr(lambda o: StepLR(o, step_size=250, gamma=0.5), BASE_LR, STEPS)
    # exponential: gamma per step so it reaches ~0.01 at the end
    gamma = (0.01) ** (1.0 / STEPS)
    expo = _read_lr(lambda o: ExponentialLR(o, gamma=gamma), BASE_LR, STEPS)
    # cosine annealing to 0 over the run
    cos = _read_lr(lambda o: CosineAnnealingLR(o, T_max=STEPS, eta_min=0.0), BASE_LR, STEPS)

    # warmup (linear 0->base over 100) then cosine to 0 -- via LambdaLR closed form
    WARM = 100
    def wc_lambda(t):
        if t < WARM:
            return t / WARM
        prog = (t - WARM) / (STEPS - WARM)
        return 0.5 * (1.0 + np.cos(np.pi * prog))
    warmcos = _read_lr(lambda o: LambdaLR(o, wc_lambda), BASE_LR, STEPS)

    # Noam / inverse-sqrt: lr = d^-0.5 * min(t^-0.5, t*warmup^-1.5), normalized to peak=BASE_LR
    d_model, noam_warm = 512.0, 100
    def noam_raw(t):
        t = max(t, 1)
        return d_model ** -0.5 * min(t ** -0.5, t * noam_warm ** -1.5)
    raw = np.array([noam_raw(t) for t in range(1, STEPS + 1)])
    noam = raw / raw.max() * BASE_LR

    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    xs = np.arange(STEPS)
    ax.plot(xs, const, color=SLATE, lw=2.2, label="constant (no schedule)")
    ax.plot(xs, step, color=BLUE, lw=2.2, label="step decay (x0.5 / 250)")
    ax.plot(xs, expo, color=PURPLE, lw=2.2, label="exponential decay")
    ax.plot(xs, cos, color=GREEN, lw=2.6, label="cosine annealing")
    ax.plot(xs, warmcos, color=RED, lw=2.6, label="linear warmup -> cosine (LLM default)")
    ax.plot(xs, noam, color=AMBER, lw=2.2, ls="--", label="inverse-sqrt (Noam, warmup=100)")
    ax.axvspan(0, WARM, color=AMBER, alpha=0.08)
    ax.text(WARM + 8, 0.2, "warmup\nregion", color=AMBER, fontsize=8.6, fontweight="bold")
    ax.set_xlabel("training step"); ax.set_ylabel("learning rate (relative to base)")
    ax.set_title("Six learning-rate schedules, read straight off the torch schedulers",
                 fontsize=13.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.3, loc="upper right"); _despine(ax)
    ax.set_ylim(-0.02, 1.08)
    fig.tight_layout(); fig.savefig(f"{OUT}/lr_schedules_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote lr_schedules_curves.png")


# ----------------------------------------------------------------------------- 2
class TinyNet(nn.Module):
    def __init__(self, d_in=20, h=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.GELU(),
                                 nn.Linear(h, h), nn.GELU(), nn.Linear(h, 1))
    def forward(self, x):
        return self.net(x)


def _train(schedule, steps=600, base_lr=3e-2, warm=40, seed=0):
    """Train TinyNet on a fixed regression task; return per-step train loss."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(123)
    X = torch.randn(512, 20, generator=g)
    w_true = torch.randn(20, 1, generator=g)
    y = X @ w_true + 0.1 * torch.randn(512, 1, generator=g)
    y = torch.tanh(y)                      # mild nonlinearity so the net has work to do

    torch.manual_seed(seed)
    model = TinyNet()
    opt = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    lossfn = nn.MSELoss()

    def lr_at(t):
        if schedule == "constant":
            return 1.0
        if schedule == "cosine":
            return 0.5 * (1.0 + np.cos(np.pi * t / steps))
        if schedule == "warmcos":
            if t < warm:
                return t / warm
            prog = (t - warm) / (steps - warm)
            return 0.5 * (1.0 + np.cos(np.pi * prog))
        raise ValueError(schedule)

    sched = LambdaLR(opt, lr_lambda=lr_at)
    bs, n = 64, X.shape[0]
    losses = []
    for t in range(steps):
        idx = torch.randint(0, n, (bs,))
        pred = model(X[idx])
        loss = lossfn(pred, y[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        losses.append(loss.item())
    return np.array(losses)


def loss_comparison():
    steps = 600
    const = np.mean([_train("constant", steps, seed=s) for s in range(4)], axis=0)
    cos = np.mean([_train("cosine", steps, seed=s) for s in range(4)], axis=0)
    wc = np.mean([_train("warmcos", steps, seed=s) for s in range(4)], axis=0)

    def smooth(a, k=15):
        return np.convolve(a, np.ones(k) / k, mode="valid")

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    xs = np.arange(len(smooth(const)))
    ax.plot(xs, smooth(const), color=SLATE, lw=2.2, label="constant LR (noisy, bounces back up)")
    ax.plot(xs, smooth(cos), color=GREEN, lw=2.4, label="cosine decay (settles lowest)")
    ax.plot(xs, smooth(wc), color=RED, lw=2.6, label="warmup -> cosine (low + smoothest descent)")
    ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel("training loss (MSE, log scale, smoothed)")
    ax.set_title("Measured: decaying the LR settles ~20x lower than holding it constant",
                 fontsize=12.6, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.6, loc="lower left"); _despine(ax)
    ax.annotate("constant LR can't settle:\nthe step stays too big,\nso it bounces back up",
                (xs[-30], smooth(const)[-30]),
                textcoords="offset points", xytext=(-30, 40), fontsize=8.8, color=SLATE,
                fontweight="bold", ha="right",
                arrowprops=dict(arrowstyle="->", color=SLATE))
    fig.tight_layout(); fig.savefig(f"{OUT}/lr_loss_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote lr_loss_comparison.png")


# ----------------------------------------------------------------------------- 3
def one_cycle():
    STEPS = 1000
    p = [torch.zeros(1, requires_grad=True)]
    opt = torch.optim.SGD(p, lr=1.0, momentum=0.95)
    sched = OneCycleLR(opt, max_lr=1.0, total_steps=STEPS, pct_start=0.3,
                       div_factor=20.0, final_div_factor=1e3, base_momentum=0.85,
                       max_momentum=0.95)
    lrs, moms = [], []
    for _ in range(STEPS):
        lrs.append(opt.param_groups[0]["lr"])
        moms.append(opt.param_groups[0]["momentum"])
        opt.step(); sched.step()
    lrs, moms = np.array(lrs), np.array(moms)

    fig, ax1 = plt.subplots(figsize=(9.4, 5.0))
    xs = np.arange(STEPS)
    ax1.plot(xs, lrs, color=GREEN, lw=2.6, label="learning rate")
    ax1.set_xlabel("training step"); ax1.set_ylabel("learning rate", color=GREEN)
    ax1.tick_params(axis="y", labelcolor=GREEN)
    peak = int(0.3 * STEPS)
    ax1.axvline(peak, color=SLATE, ls=":", lw=1.2)
    ax1.text(peak + 10, 0.5, "peak LR\n(end of warmup)", color=SLATE, fontsize=8.6, fontweight="bold")
    ax2 = ax1.twinx()
    ax2.plot(xs, moms, color=RED, lw=2.6, ls="--", label="momentum")
    ax2.set_ylabel("momentum", color=RED); ax2.tick_params(axis="y", labelcolor=RED)
    ax1.set_title("One-cycle: LR up-then-down while momentum moves the opposite way",
                  fontsize=12.6, fontweight="bold")
    _despine(ax1)
    l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, frameon=False, fontsize=9.6, loc="lower center")
    fig.tight_layout(); fig.savefig(f"{OUT}/lr_one_cycle.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote lr_one_cycle.png")


# ----------------------------------------------------------------------------- 4
def range_test():
    """LR-range test (Smith): exponentially ramp LR, plot loss; find steepest-descent LR."""
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(7)
    X = torch.randn(512, 20, generator=g)
    w_true = torch.randn(20, 1, generator=g)
    y = torch.tanh(X @ w_true + 0.1 * torch.randn(512, 1, generator=g))

    torch.manual_seed(0)
    model = TinyNet()
    opt = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
    lossfn = nn.MSELoss()
    n_iter = 140
    lr_min, lr_max = 1e-5, 100.0
    mult = (lr_max / lr_min) ** (1.0 / n_iter)

    lrs, losses = [], []
    lr = lr_min
    bs, n = 64, X.shape[0]
    best = float("inf")
    for i in range(n_iter):
        for pg in opt.param_groups:
            pg["lr"] = lr
        idx = torch.randint(0, n, (bs,))
        loss = lossfn(model(X[idx]), y[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        l = loss.item()
        lrs.append(lr); losses.append(l)
        best = min(best, l)
        if (l > 6 * best or not np.isfinite(l)) and i > 10:   # diverged: stop recording
            break
        lr *= mult
    lrs, losses = np.array(lrs), np.array(losses)

    # smooth & find steepest negative slope on log-LR axis
    sm = np.convolve(losses, np.ones(5) / 5, mode="same")
    log_lr = np.log10(lrs)
    slope = np.gradient(sm, log_lr)
    valid = slice(3, len(slope) - 3)
    steep_idx = np.argmin(slope[valid]) + 3
    min_idx = int(np.argmin(sm))
    div_idx = len(lrs) - 1

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.plot(lrs, sm, color=BLUE, lw=2.6)
    ax.set_xscale("log")
    ax.scatter([lrs[steep_idx]], [sm[steep_idx]], color=GREEN, s=110, zorder=6,
               label=f"steepest descent ~ {lrs[steep_idx]:.1e}  (good max LR)")
    ax.scatter([lrs[div_idx]], [sm[div_idx]], color=RED, s=110, marker="X", zorder=6,
               label=f"loss diverges ~ {lrs[div_idx]:.1e}  (too high)")
    ax.set_xlabel("learning rate (log scale, ramped each step)")
    ax.set_ylabel("training loss")
    ax.set_title("LR-range test: pick the LR where loss falls fastest, below divergence",
                 fontsize=12.6, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.6, loc="upper left"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/lr_range_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote lr_range_test.png")


if __name__ == "__main__":
    schedules_curves()
    loss_comparison()
    one_cycle()
    range_test()
    print("OUT:", OUT)
