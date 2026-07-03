"""Figure generator for 10.GenAI / 01-Variational-Autoencoders-VAE-ELBO — every figure is from the REAL run.

One measured experiment (``run_experiment`` in ``vae.py``) drives every figure below, so nothing quantitative is
hand-typed: the real MNIST reconstructions and prior samples, the 2-D latent manifold and a latent interpolation,
the measured per-epoch ELBO/reconstruction/KL curves, the beta-VAE trade-off, and the two proofs (the closed-form
Gaussian KL vs a Monte-Carlo estimate, and the reparameterization trick's gradient flow) all come from the same
executed pipeline the chapter and notebook use.

Writes muted-palette PNGs to the shared domain image dir (../images/) with prefix ``ga01_``:

  ga01_reconstructions.png  -- real held-out digits (top) vs the VAE's reconstructions (bottom): it learned to
                               compress and rebuild real images.
  ga01_prior_samples.png    -- decode z ~ N(0, I): NOVEL digits the model generated — the "it can generate"
                               payoff a plain autoencoder cannot give.
  ga01_latent_manifold.png  -- a grid of decoded z across the 2-D latent square: the smooth, organized latent
                               space (neighbours decode to similar digits).
  ga01_training_curves.png  -- the measured negative-ELBO and its two terms (reconstruction + KL) per epoch: the
                               two terms trading off as the VAE learns.
  ga01_kl_proof.png         -- PROOF 1: the closed-form Gaussian KL equals a Monte-Carlo estimate of KL(q||p),
                               and the estimate converges ~1/sqrt(N).
  ga01_reparam_gradient.png -- PROOF 2: the reparameterized z gives real gradients through mu/logvar; a raw
                               .sample() gives none — why the trick exists.
  ga01_interpolation.png    -- walk z in a straight line between two encoded digits: a smooth morph, showing the
                               latent space is continuous.
  ga01_beta_tradeoff.png    -- the beta-VAE trade-off: a larger KL weight buys a smaller (tighter-to-prior) KL at
                               the cost of reconstruction quality.

    python make_figures_01.py

Verified on Python 3.12 / matplotlib 3.10 / numpy 2.4 / torch 2.12 / torchvision 0.27.
"""

from __future__ import annotations

import sys
from pathlib import Path

# This generator lives in ``10. GenAI/tools/``; the chapter module it demonstrates stays in that chapter's
# ``code/`` folder. Put that folder on sys.path so the ``vae`` import resolves.
_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "01-Variational-Autoencoders-VAE-ELBO" / "code"
sys.path.insert(0, str(_CHAPTER_CODE))

import matplotlib  # noqa: E402  (imported after the sys.path insert above, which must run first)

matplotlib.use("Agg")  # headless: render straight to PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from vae import (  # noqa: E402
    gaussian_kl_to_standard_normal,
    interpolate,
    latent_manifold,
    reconstruct,
    run_experiment,
    sample_prior,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) ------------------------------------
BLUE = "#3A6B96"  # data / reference lines
PURPLE = "#5D4A8A"  # process / loss
GREEN = "#2E7A5A"  # good / reconstruction / matched
RED = "#8B3B4A"  # penalty / KL / no-gradient
AMBER = "#7A6528"  # highlight
SLATE = "#4A5B6E"  # neutral
INK = "#1C2530"  # labels
GRID = "#D4D9DF"  # gridlines

OUT_DIR = Path(__file__).resolve().parent.parent / "images"
DPI = 120
IMG_PREFIX = "ga01_"


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


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")


# ================================================================================================
# Figure 1: reconstructions — real held-out digits vs the VAE's reconstruction of each
# ================================================================================================


def fig_reconstructions(exp) -> None:
    h, w = exp.img_shape
    x = exp.data.x_test[:12]
    recon = reconstruct(exp.main.model, x)
    fig, axes = plt.subplots(2, 12, figsize=(12, 2.3))
    for j in range(12):
        axes[0, j].imshow(x[j].numpy().reshape(h, w), cmap="gray_r", vmin=0, vmax=1)
        axes[1, j].imshow(recon[j].reshape(h, w), cmap="gray_r", vmin=0, vmax=1)
        for i in range(2):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    axes[0, 0].set_ylabel("input", fontsize=10, color=INK)
    axes[1, 0].set_ylabel("VAE\nreconstruction", fontsize=10, color=INK)
    fig.suptitle(f"The VAE compresses each real {exp.data_label} digit to a {exp.main.latent_dim}-D code and "
                 f"rebuilds it — recognizable (blurry: the Gaussian-likelihood averaging effect)",
                 fontsize=10.3, color=INK, y=1.08)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}reconstructions.png")


# ================================================================================================
# Figure 2: prior samples — decode z ~ N(0, I): novel digits the model generated
# ================================================================================================


def fig_prior_samples(exp) -> None:
    h, w = exp.img_shape
    samples = sample_prior(exp.main.model, n=15, seed=1)
    fig, axes = plt.subplots(1, 15, figsize=(12, 1.2))
    for j in range(15):
        axes[j].imshow(samples[j].reshape(h, w), cmap="gray_r", vmin=0, vmax=1)
        axes[j].set_xticks([])
        axes[j].set_yticks([])
    fig.suptitle("The payoff a plain autoencoder cannot give: sample z ~ N(0, I) from the prior and decode — "
                 "each is a NEW digit the model generated, never in the training set",
                 fontsize=10.3, color=INK, y=1.35)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}prior_samples.png")


# ================================================================================================
# Figure 3: the 2-D latent manifold — a grid of decoded z showing the smooth organized latent space
# ================================================================================================


def fig_latent_manifold(exp) -> None:
    h, w = exp.img_shape
    grid = 18
    span = 2.5
    tiles = latent_manifold(exp.main.model, grid=grid, span=span)
    canvas = np.zeros((grid * h, grid * w))
    for idx in range(grid * grid):
        r, c = divmod(idx, grid)
        canvas[r * h : (r + 1) * h, c * w : (c + 1) * w] = tiles[idx].reshape(h, w)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.imshow(canvas, cmap="gray_r", vmin=0, vmax=1)
    ax.set_xticks([0, grid * w - 1])
    ax.set_xticklabels([f"$z_1=-{span}$", f"$z_1=+{span}$"], fontsize=9, color=INK)
    ax.set_yticks([0, grid * h - 1])
    ax.set_yticklabels([f"$z_2=-{span}$", f"$z_2=+{span}$"], fontsize=9, color=INK)
    ax.set_title("The 2-D latent manifold: decode a grid of z across the prior. Neighbours in z decode to "
                 "similar digits — a smooth, organized space you can sample", fontsize=9.7, color=INK)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}latent_manifold.png")


# ================================================================================================
# Figure 4: training curves — negative ELBO and its two terms (reconstruction + KL) per epoch
# ================================================================================================


def fig_training_curves(exp) -> None:
    m = exp.main
    epochs = np.arange(1, len(m.elbo) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    _style_axis(ax1)
    ax1.plot(epochs, m.elbo, color=PURPLE, linewidth=2.3, label="negative ELBO (total loss)")
    ax1.plot(epochs, m.recon, color=GREEN, linewidth=2.0, label="reconstruction term")
    ax1.plot(epochs, m.kl, color=RED, linewidth=2.0, label="KL-to-prior term")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("per-image loss (nats)")
    ax1.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax1.set_title(f"(a) the negative ELBO falls as the VAE learns\nfinal: total {m.final_elbo:.1f} = recon "
                  f"{m.final_recon:.1f} + KL {m.final_kl:.1f}", fontsize=10, color=INK)

    _style_axis(ax2)
    ax2.plot(epochs, m.kl, color=RED, linewidth=2.3, label="KL term (rises then settles)")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("KL(q(z|x) || N(0, I))  (nats)")
    ax2.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax2.set_title("(b) the KL term: the encoder moves off the prior\nonly as far as reconstruction needs",
                  fontsize=10, color=INK)

    fig.suptitle("Training a VAE = maximizing the ELBO: reconstruction and KL-to-prior traded off, measured per "
                 "epoch on real data", fontsize=10.6, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}training_curves.png")


# ================================================================================================
# Figure 5: PROOF 1 — the closed-form Gaussian KL equals a Monte-Carlo estimate, converging ~1/sqrt(N)
# ================================================================================================


def fig_kl_proof(exp) -> None:
    kp = exp.kl_proof
    # Recompute the MC estimate at several N (reproducibly) to show ~1/sqrt(N) convergence to the closed form.
    torch.manual_seed(exp.seed)
    latent_dim = 8
    mu = torch.randn(1, latent_dim) * 0.8
    logvar = torch.randn(1, latent_dim) * 0.5
    closed = float(gaussian_kl_to_standard_normal(mu, logvar).item())
    std = torch.exp(0.5 * logvar)
    ns = np.array([100, 300, 1000, 3000, 10_000, 30_000, 100_000, 400_000])
    big_eps = torch.randn(int(ns.max()), latent_dim)
    errs = []
    for n in ns:
        z = mu + std * big_eps[:n]
        var = logvar.exp()
        log_q = -0.5 * torch.sum(np.log(2 * np.pi) + logvar + (z - mu).pow(2) / var, dim=1)
        log_p = -0.5 * torch.sum(np.log(2 * np.pi) + z.pow(2), dim=1)
        errs.append(abs(float((log_q - log_p).mean()) - closed))
    errs = np.array(errs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))
    _style_axis(ax1)
    bars = ax1.bar(["closed form\n½Σ(μ²+σ²−1−logσ²)", f"Monte-Carlo\nE_q[log q − log p]\n(N={kp.n_samples:,})"],
                   [kp.closed_form, kp.monte_carlo], color=[BLUE, GREEN], width=0.5)
    for bar, val in zip(bars, [kp.closed_form, kp.monte_carlo]):
        ax1.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.4f}", ha="center", va="bottom",
                 fontsize=10, color=INK, fontweight="bold")
    ax1.set_ylabel("KL(q || p)  (nats)")
    ax1.set_ylim(0, kp.closed_form * 1.2)
    ax1.set_title(f"(a) the ELBO's KL term is the EXACT KL\n|closed − MC| = {kp.abs_error:.1e}",
                  fontsize=10, color=INK)

    _style_axis(ax2)
    ax2.loglog(ns, errs, "o-", color=GREEN, linewidth=2.0, markersize=5, label="|MC − closed form|")
    ref = errs[0] * np.sqrt(ns[0]) / np.sqrt(ns)
    ax2.loglog(ns, ref, "--", color=SLATE, linewidth=1.3, label=r"$\propto 1/\sqrt{N}$ reference")
    ax2.set_xlabel("Monte-Carlo samples N")
    ax2.set_ylabel("abs error vs closed form")
    ax2.legend(fontsize=8.5, frameon=False)
    ax2.set_title("(b) the MC estimate converges to the closed\nform like 1/√N — they are the same quantity",
                  fontsize=10, color=INK)

    fig.suptitle("Proof 1: the closed-form Gaussian KL the loss uses equals the Monte-Carlo estimate of "
                 "KL(q||p) — the term is exactly what the math claims", fontsize=10.4, color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}kl_proof.png")


# ================================================================================================
# Figure 6: PROOF 2 — the reparameterization trick's gradient flow (reparam has grad; direct sample none)
# ================================================================================================


def fig_reparam_gradient(exp) -> None:
    rp = exp.reparam_proof
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    _style_axis(ax)
    labels = ["reparameterized\n∂/∂μ", "reparameterized\n∂/∂ log σ²", "direct .sample()\n∂/∂μ"]
    vals = [rp.grad_mu_reparam, rp.grad_logvar_reparam, 0.0]
    colors = [GREEN, GREEN, RED]
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for bar, val in zip(bars, vals):
        txt = f"{val:.2f}" if val > 0 else "0 (no path)"
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.08, txt, ha="center", va="bottom",
                fontsize=10, color=INK, fontweight="bold")
    ax.set_ylabel("gradient norm reaching the encoder")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Proof 2: z = μ + σ·ε keeps the gradient flowing to μ and log σ² (green); a raw "
                 "Normal(μ,σ).sample() (red)\ndetaches z from the encoder — no gradient, so the encoder could "
                 "never be trained. This is WHY we reparameterize.", fontsize=9.4, color=INK)
    fig.text(0.5, -0.02,
             f"direct-sample z.requires_grad = {rp.direct_sample_has_grad}  →  backprop cannot reach μ, σ",
             ha="center", fontsize=9, color=RED, style="italic")
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}reparam_gradient.png")


# ================================================================================================
# Figure 7: latent interpolation — a straight walk in z between two encoded digits, decoded
# ================================================================================================


def fig_interpolation(exp) -> None:
    h, w = exp.img_shape
    y = exp.data.y_test
    # pick two visually different digit classes if available, else the first two test images
    def first_of(label: int, fallback: int) -> int:
        idx = np.flatnonzero(y == label)
        return int(idx[0]) if idx.size else fallback

    i0, i1 = first_of(7, 0), first_of(3, 1)
    frames = interpolate(exp.main.model, exp.data.x_test[i0], exp.data.x_test[i1], steps=10)
    fig, axes = plt.subplots(1, 10, figsize=(12, 1.3))
    for j in range(10):
        axes[j].imshow(frames[j].reshape(h, w), cmap="gray_r", vmin=0, vmax=1)
        axes[j].set_xticks([])
        axes[j].set_yticks([])
    axes[0].set_xlabel("start", fontsize=9, color=INK)
    axes[-1].set_xlabel("end", fontsize=9, color=INK)
    fig.suptitle("Walk a straight line in latent space between two encoded digits and decode each step — a "
                 "SMOOTH morph, proving the latent space is continuous (that is what the KL term buys)",
                 fontsize=10.2, color=INK, y=1.35)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}interpolation.png")


# ================================================================================================
# Figure 8: the beta-VAE trade-off — larger KL weight buys a smaller KL at the cost of reconstruction
# ================================================================================================


def fig_beta_tradeoff(exp) -> None:
    betas = sorted(exp.beta_runs)
    recons = [exp.beta_runs[b].final_recon for b in betas]
    kls = [exp.beta_runs[b].final_kl for b in betas]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.5))

    x = np.arange(len(betas))
    ww = 0.35
    _style_axis(ax1)
    b1 = ax1.bar(x - ww / 2, recons, width=ww, color=GREEN, label="reconstruction (lower = sharper)")
    b2 = ax1.bar(x + ww / 2, kls, width=ww, color=RED, label="KL to prior (lower = tighter)")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"β = {b:.0f}" for b in betas])
    ax1.set_ylabel("final per-image term (nats)")
    ax1.set_ylim(0, max(recons) * 1.28)
    for bars in (b1, b2):
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.1f}",
                     ha="center", va="bottom", fontsize=8.5, color=INK)
    ax1.legend(fontsize=8.5, frameon=False, loc="upper center", ncol=1)
    ax1.set_title("(a) larger β → smaller KL, larger reconstruction\nloss: the two terms trade off",
                  fontsize=10, color=INK)

    # visual: samples from the two betas side by side
    _style_axis(ax2)
    ax2.axis("off")
    h, w = exp.img_shape
    for row, b in enumerate(betas):
        samples = sample_prior(exp.beta_runs[b].model, n=6, seed=2)
        strip = np.concatenate([samples[j].reshape(h, w) for j in range(6)], axis=1)
        ax2.imshow(strip, cmap="gray_r", vmin=0, vmax=1,
                   extent=(0, 6, row, row + 0.9), aspect="auto")
        ax2.text(-0.15, row + 0.45, f"β={b:.0f}", ha="right", va="center", fontsize=10, color=INK)
    ax2.set_xlim(-0.9, 6)
    ax2.set_ylim(0, len(betas))
    ax2.set_title("(b) prior samples: β=4 is tidier but blurrier —\nthe disentanglement/sharpness trade-off",
                  fontsize=10, color=INK)

    fig.suptitle("The β-VAE knob: β weights the KL term. β>1 pulls the posterior tighter to the prior "
                 "(smaller KL, more disentangled) at the cost of reconstruction — measured", fontsize=10.1,
                 color=INK, y=1.02)
    fig.tight_layout()
    _save(fig, f"{IMG_PREFIX}beta_tradeoff.png")


def main() -> None:
    exp = run_experiment()
    fig_reconstructions(exp)
    fig_prior_samples(exp)
    fig_latent_manifold(exp)
    fig_training_curves(exp)
    fig_kl_proof(exp)
    fig_reparam_gradient(exp)
    fig_interpolation(exp)
    fig_beta_tradeoff(exp)
    # guard against silent drift: the proven relationships the figures show must hold
    assert exp.kl_proof.abs_error < 1e-2
    assert exp.reparam_proof.direct_sample_has_grad is False
    assert exp.beta_runs[4.0].final_kl < exp.beta_runs[1.0].final_kl
    print("all figures written to", OUT_DIR)


if __name__ == "__main__":
    main()
