"""Figure generator for 10.GenAI / 05-Diffusion-Models-DDPM — every figure is from the REAL run.

One measured experiment (``run_experiment`` in ``ddpm.py``) drives every figure below, so nothing quantitative is
hand-typed: the real 2-D forward diffusion (data cloud -> N(0, I)), the learned reverse/generation trajectory
(N(0, I) -> samples matching the target), the noise schedule (beta, alpha, alpha-bar and the signal/noise mix),
the closed-form-vs-iterative forward proof, the L_simple training curve, and the generation energy-distance
metric — all come from the same executed pipeline the chapter and notebook use.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``ga05_``:

  ga05_forward_diffusion.png  -- a real 2-D data cloud at increasing t, progressively noised to an isotropic
                                 Gaussian: SEE the closed-form forward q(x_t | x_0).
  ga05_noise_schedule.png     -- beta_t, alpha_t, alpha-bar_t (linear vs cosine) and the sqrt(abar)/sqrt(1-abar)
                                 signal-vs-noise mixing coefficients.
  ga05_forward_check.png      -- PROOF 1: the closed-form forward equals the iterative step-by-step chain (clouds
                                 coincide; the mean/cov error shrinks ~1/sqrt(N)).
  ga05_training_loss.png      -- the measured L_simple = ||eps - eps_theta||^2 falling per epoch.
  ga05_reverse_trajectory.png -- start from N(0, I) and run the learned reverse chain: the cloud denoises back into
                                 the target distribution, step by step.
  ga05_generation_metric.png  -- PROOF 2: generated-vs-target energy distance far below the N(0,I) baseline, with
                                 the generated/real overlay — the payoff.

    python make_figures_05.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / torch 2.12 / scikit-learn 1.9.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in ``10. GenAI/tools/``; the chapter module it demonstrates stays in that chapter's
# ``code/`` folder. Put that folder on sys.path so the ``ddpm`` import resolves.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "05-Diffusion-Models-DDPM" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ddpm import (  # noqa: E402
    iterative_forward,
    p_sample_loop,
    q_sample,
    run_experiment,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"    # data / target / reference lines
PURPLE = "#5D4A8A"  # process / loss / reverse
GREEN = "#2E7A5A"   # good / generated / matched
RED = "#8B3B4A"     # noise / penalty / baseline
AMBER = "#7A6528"   # highlight
SLATE = "#4A5B6E"   # neutral
INK = "#1C2530"     # labels
GRID = "#D4D9DF"    # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "ga05_"


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


def _scatter_axis(ax: plt.Axes, lim: float = 3.2) -> None:
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(GRID)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


# ================================================================================================
# Figure 1: forward diffusion — a real 2-D data cloud progressively noised to N(0, I)
# ================================================================================================


def fig_forward_diffusion(exp) -> None:
    sched = exp.sched
    x0 = exp.data.x[:900]
    ts = [0, 40, 120, 240, sched.T]  # timesteps to display (0 = clean data)
    fig, axes = plt.subplots(1, len(ts), figsize=(13.5, 2.9))
    torch.manual_seed(0)
    eps = torch.randn_like(x0)  # shared noise draw so the "same" cloud dissolves coherently
    for ax, t in zip(axes, ts):
        if t == 0:
            pts = x0
        else:
            t_idx = torch.full((x0.shape[0],), t - 1, dtype=torch.long)
            pts, _ = q_sample(x0, t_idx, sched, eps=eps)
        ax.scatter(pts[:, 0], pts[:, 1], s=5, c=BLUE if t == 0 else PURPLE, alpha=0.55, linewidths=0)
        _scatter_axis(ax)
        abar = 1.0 if t == 0 else float(sched.alpha_bars[t - 1])
        ax.set_title(f"t = {t}\n" + (r"$\bar\alpha_t$ = " + f"{abar:.2f}"), fontsize=9.5, color=INK)
    axes[0].set_title("t = 0  (real data)\n" + r"$\bar\alpha_0$ = 1.00", fontsize=9.5, color=INK)
    fig.suptitle("The forward process: add a little Gaussian noise at each step — the real make_moons cloud "
                 r"dissolves into pure noise $\mathcal{N}(0, I)$. Each panel is the closed-form "
                 r"$q(x_t\mid x_0)=\mathcal{N}(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I)$",
                 fontsize=10.4, color=INK, y=1.08)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}forward_diffusion.png")


# ================================================================================================
# Figure 2: the noise schedule — beta, alpha, alpha-bar (linear vs cosine) and the signal/noise mix
# ================================================================================================


def fig_noise_schedule(exp) -> None:
    sched, cos = exp.sched, exp.cosine_sched
    t = np.arange(1, sched.T + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.4))

    _style_axis(ax1)
    ax1.plot(t, sched.betas.numpy(), color=RED, lw=2.0, label=r"$\beta_t$ (noise added per step)")
    ax1.plot(t, sched.alphas.numpy(), color=SLATE, lw=1.6, label=r"$\alpha_t = 1-\beta_t$")
    ax1.plot(t, sched.alpha_bars.numpy(), color=BLUE, lw=2.4, label=r"$\bar\alpha_t=\prod\alpha_s$ (linear)")
    ax1.plot(t, cos.alpha_bars.numpy(), color=GREEN, lw=2.0, ls="--", label=r"$\bar\alpha_t$ (cosine)")
    ax1.set_xlabel("diffusion step t")
    ax1.set_ylabel("value")
    ax1.legend(fontsize=8.2, frameon=False, loc="center right")
    ax1.set_title(r"(a) $\bar\alpha_t$ falls from 1 to ~0: the compounded signal-survival."
                  "\ncosine decays more gently than linear", fontsize=9.7, color=INK)

    _style_axis(ax2)
    ax2.plot(t, sched.sqrt_abar.numpy(), color=BLUE, lw=2.4,
             label=r"$\sqrt{\bar\alpha_t}$ — signal weight on $x_0$")
    ax2.plot(t, sched.sqrt_one_minus_abar.numpy(), color=RED, lw=2.4,
             label=r"$\sqrt{1-\bar\alpha_t}$ — noise weight on $\varepsilon$")
    ax2.set_xlabel("diffusion step t")
    ax2.set_ylabel("mixing coefficient")
    ax2.legend(fontsize=8.6, frameon=False, loc="center right")
    ax2.set_title(r"(b) $x_t=\sqrt{\bar\alpha_t}\,x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon$:"
                  "\nsignal handed off to noise as t grows", fontsize=9.7, color=INK)

    fig.suptitle("The variance (noise) schedule: everything the forward and reverse steps need is precomputed "
                 r"from $\beta_t$", fontsize=10.6, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}noise_schedule.png")


# ================================================================================================
# Figure 3: PROOF 1 — the closed-form forward equals the iterative chain (clouds coincide; error ~1/sqrt(N))
# ================================================================================================


def fig_forward_check(exp) -> None:
    sched = exp.sched
    fp = exp.forward_proof
    t = fp.t
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    # (a) overlay: closed-form-sampled cloud vs iterative-chain cloud at time t (from one fixed x0)
    n = 3000
    x0_pt = torch.tensor([[1.2, -0.7]], dtype=torch.float32)
    g = torch.Generator().manual_seed(1)
    iter_pts = iterative_forward(x0_pt.expand(n, 2).contiguous(), t, sched, generator=g)
    abar = float(sched.alpha_bars[t - 1])
    closed_pts = np.sqrt(abar) * x0_pt.numpy() + np.sqrt(1 - abar) * np.random.default_rng(2).standard_normal((n, 2))
    ax1.scatter(iter_pts[:, 0], iter_pts[:, 1], s=6, c=RED, alpha=0.35, linewidths=0,
                label=f"iterative chain ({t} tiny steps)")
    ax1.scatter(closed_pts[:, 0], closed_pts[:, 1], s=6, c=GREEN, alpha=0.35, linewidths=0,
                label=r"closed form $q(x_t\mid x_0)$")
    ax1.scatter([fp.closed_mean[0]], [fp.closed_mean[1]], s=90, marker="x", c=INK, linewidths=2.2, zorder=5)
    ax1.annotate(r"$\sqrt{\bar\alpha_t}\,x_0$ (mean)", (fp.closed_mean[0], fp.closed_mean[1]),
                 xytext=(fp.closed_mean[0] + 0.5, fp.closed_mean[1] - 1.3), fontsize=8.4, color=INK,
                 arrowprops={"arrowstyle": "->", "color": INK, "lw": 1.0})
    lim = 3.0
    ax1.set_xlim(fp.closed_mean[0] - lim, fp.closed_mean[0] + lim)
    ax1.set_ylim(fp.closed_mean[1] - lim, fp.closed_mean[1] + lim)
    ax1.set_aspect("equal")
    _style_axis(ax1)
    ax1.legend(fontsize=7.8, frameon=False, loc="upper right")
    ax1.set_title(f"(a) at t={t}: the one-shot jump and the {t}-step chain\nproduce the SAME cloud "
                  f"(max|mean err|={fp.mean_abs_err:.1e}, max|cov err|={fp.var_abs_err:.1e})",
                  fontsize=9.4, color=INK)

    # (b) convergence of the empirical-vs-closed error ~ 1/sqrt(N)
    ns = np.array([100, 300, 1000, 3000, 10_000, 30_000, 60_000])
    errs = []
    g2 = torch.Generator().manual_seed(3)
    big = iterative_forward(x0_pt.expand(int(ns.max()), 2).contiguous(), t, sched, generator=g2).numpy()
    closed_mean = np.sqrt(abar) * x0_pt.numpy().ravel()
    for k in ns:
        errs.append(float(np.max(np.abs(big[:k].mean(axis=0) - closed_mean))))
    errs = np.array(errs)
    _style_axis(ax2)
    ax2.loglog(ns, errs, "o-", color=GREEN, lw=2.0, markersize=5, label="|iterative mean − closed mean|")
    ref = errs[1] * np.sqrt(ns[1]) / np.sqrt(ns)
    ax2.loglog(ns, ref, "--", color=SLATE, lw=1.3, label=r"$\propto 1/\sqrt{N}$ reference")
    ax2.set_xlabel("iterative rollouts N")
    ax2.set_ylabel("abs error vs closed form")
    ax2.legend(fontsize=8.4, frameon=False)
    ax2.set_title("(b) the chain's empirical mean converges to the\nclosed-form mean like 1/√N — same distribution",
                  fontsize=9.6, color=INK)

    fig.suptitle("Proof 1: the closed-form forward $q(x_t\\mid x_0)$ IS the marginal of the slow iterative chain — "
                 "the 'nice property' that lets training jump to any t in one step", fontsize=10.2, color=INK,
                 y=1.02)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}forward_check.png")


# ================================================================================================
# Figure 4: the L_simple training curve
# ================================================================================================


def fig_training_loss(exp) -> None:
    m = exp.main
    ep = np.arange(1, len(m.loss_hist) + 1)
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    _style_axis(ax)
    ax.plot(ep, m.loss_hist, color=PURPLE, lw=2.2)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$L_{\mathrm{simple}} = \mathbb{E}\,\|\varepsilon-\varepsilon_\theta(x_t,t)\|^2$")
    ax.set_title(f"Training a DDPM = teaching one network to predict the noise at every level.\n"
                 f"L_simple falls from {m.loss_hist[0]:.3f} to {m.final_loss:.3f} over {len(ep)} epochs "
                 f"({m.seconds:.0f}s on CPU, real make_moons)", fontsize=9.7, color=INK)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}training_loss.png")


# ================================================================================================
# Figure 5: reverse/generation trajectory — start from N(0, I), denoise back to the target
# ================================================================================================


def fig_reverse_trajectory(exp) -> None:
    m = exp.main
    n = 900
    _, traj = p_sample_loop(m.model, m.sched, n=n, dim=2, seed=7, record_every=1)  # record every step
    by_t = {t: cloud for t, cloud in traj}
    # end-weighted snapshots: DDPM structure emerges only in the last ~10% of steps, so sample densely near t=1
    want = [m.sched.T, m.sched.T // 4, 60, 25, 1]
    snaps = [(t, by_t[min(by_t, key=lambda k: abs(k - t))]) for t in want]
    fig, axes = plt.subplots(1, len(snaps), figsize=(13.5, 2.9))
    for ax, (t, cloud) in zip(axes, snaps):
        c = cloud.numpy()
        ax.scatter(c[:, 0], c[:, 1], s=5, c=GREEN if t == 1 else PURPLE, alpha=0.55, linewidths=0)
        _scatter_axis(ax)
        ax.set_title((f"t = {t}  " + (r"(start $\mathcal{N}(0,I)$)" if t == m.sched.T else
                                      "(generated)" if t == 1 else "(denoising)")), fontsize=9.3, color=INK)
    fig.suptitle("The reverse process (generation): start from pure noise and run the learned denoiser step by "
                 "step — the cloud reassembles into the two-moons target. The network never saw the data as a "
                 "whole; it only learned to undo one small noise step", fontsize=10.0, color=INK, y=1.08)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}reverse_trajectory.png")


# ================================================================================================
# Figure 6: PROOF 2 — generation energy distance far below baseline, with the generated/real overlay
# ================================================================================================


def fig_generation_metric(exp) -> None:
    gp = exp.generation_proof
    m = exp.main
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.6, 4.6))

    _style_axis(ax1)
    bars = ax1.bar(["N(0, I) baseline\n(the chain's start)", "generated\n(reverse chain)"],
                   [gp.ed_baseline, gp.ed_generated], color=[RED, GREEN], width=0.55)
    for bar, val in zip(bars, [gp.ed_baseline, gp.ed_generated]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom",
                 fontsize=10.5, color=INK, fontweight="bold")
    ax1.set_ylabel("energy distance to the real target")
    ax1.set_ylim(0, gp.ed_baseline * 1.25)
    ax1.set_title(f"(a) generated is {1/gp.ratio:.1f}× closer to the target\nthan the N(0,I) start "
                  f"(ratio = {gp.ratio:.2f})", fontsize=9.8, color=INK)

    # (b) overlay generated vs real
    target = exp.data.x[:1500].numpy()
    gen, _ = p_sample_loop(m.model, m.sched, n=1500, dim=2, seed=11)
    gen = gen.numpy()
    ax2.scatter(target[:, 0], target[:, 1], s=7, c=SLATE, alpha=0.35, linewidths=0, label="real target (moons)")
    ax2.scatter(gen[:, 0], gen[:, 1], s=7, c=GREEN, alpha=0.45, linewidths=0, label="generated samples")
    _scatter_axis(ax2, lim=2.6)
    ax2.legend(fontsize=8.4, frameon=False, loc="upper right")
    ax2.set_title("(b) generated samples (green) sit on the real\ntwo-moons manifold (grey) — it learned the shape",
                  fontsize=9.8, color=INK)

    fig.suptitle("Proof 2: a from-scratch DDPM generates the target distribution — the energy distance to the "
                 "real data is far below the N(0,I) baseline", fontsize=10.2, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}generation_metric.png")


def main() -> None:
    exp = run_experiment()
    fig_forward_diffusion(exp)
    fig_noise_schedule(exp)
    fig_forward_check(exp)
    fig_training_loss(exp)
    fig_reverse_trajectory(exp)
    fig_generation_metric(exp)

    # guard against silent drift: the proven relationships the figures show must hold
    assert exp.forward_proof.mean_abs_err < 0.05
    assert exp.forward_proof.var_abs_err < 0.05
    assert exp.generation_proof.ratio < 0.5
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
