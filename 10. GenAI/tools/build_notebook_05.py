"""Generate the step-by-step teaching notebook (05-Diffusion-Models-DDPM.ipynb).

The notebook mirrors ``ddpm.py`` one step at a time so a learner can open it, run every cell live, and *see* a
DDPM built and proven on a real 2-D distribution: the data, the noise schedule, the closed-form forward process
(and its proof that it equals the slow iterative chain), the denoiser, the simplified noise-prediction loss, real
from-scratch training, ancestral sampling (the reverse trajectory), and a measured proof that the generated
samples match the target distribution. Each numbered step has a short markdown lead-in (the intuition) followed by
a focused code cell with real output.

    python build_notebook_05.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../05-Diffusion-Models-DDPM/code/05-Diffusion-Models-DDPM.ipynb"

This generator lives in the domain-level ``10. GenAI/tools/`` folder; the notebook it writes (and the module it
mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited .ipynb) so the notebook
and the module stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "05-Diffusion-Models-DDPM" / "code"
NB_PATH = _CHAPTER_CODE / "05-Diffusion-Models-DDPM.ipynb"

_CELL_ID = 0


def _next_id() -> str:
    global _CELL_ID
    _CELL_ID += 1
    return f"cell-{_CELL_ID:02d}"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "id": _next_id(),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


CELLS: list[dict] = []


def add_md(source: str) -> None:
    CELLS.append(md(source))


def add_code(source: str) -> None:
    CELLS.append(code(source))


# ============================ Title ============================================================
add_md(
    "# Diffusion models (DDPM) — a runnable, measured, *proven* build\n"
    "\n"
    "A **diffusion model** generates by learning to **reverse a fixed noising process**. Add a little Gaussian "
    "noise to real data, over and over, until it becomes pure static $\\mathcal N(0,I)$ (the **forward** process, "
    "no learning). Then train one network to undo *one* small noising step; generate by starting from static and "
    "denoising step by step (the **reverse** process). The whole thing collapses to a network that **predicts the "
    "noise**, trained with plain MSE.\n"
    "\n"
    "This notebook builds a DDPM **from scratch** on a **real 2-D distribution** (`sklearn` `make_moons`) — chosen "
    "so you can *watch* the data dissolve into noise and reassemble — and *proves* the two claims that matter:\n"
    "\n"
    "- **The closed-form forward is exact.** We check with a hard `assert` that the one-shot jump "
    "$q(x_t\\mid x_0)=\\mathcal N(\\sqrt{\\bar\\alpha_t}x_0,(1-\\bar\\alpha_t)I)$ equals the distribution of the "
    "slow, step-by-step noising chain.\n"
    "- **The model generates the target.** We `assert` that samples drawn from the trained reverse chain match the "
    "real two-moons (energy distance far below the $\\mathcal N(0,I)$ baseline).\n"
    "\n"
    "It imports the **exact same functions** as the companion page and its figures (from `ddpm.py`), so the "
    "numbers here are the numbers there. Everything is **seeded and CPU-pinned** for a reproducible trace.\n"
    "\n"
    "> Companion page: **Diffusion Models (DDPM)**. Run top-to-bottom (Kernel → Restart & Run All). The 2-D demo "
    "trains in well under a minute on CPU."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup: import the real module and print versions\n"
    "\n"
    "We import the pipeline from the chapter module so this notebook runs the *same code* the page and figures "
    "use, and print the library versions and the device. Training pins **CPU** for a reproducible trace (a small "
    "MLP denoiser on 2-D points trains fast on CPU)."
)
add_code(
    "import numpy as np\n"
    "import torch\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "import ddpm as D\n"
    "\n"
    "print(f'torch {torch.__version__} | numpy {np.__version__}')\n"
    "print(f'best available device = {D.get_device()}  (training pinned to CPU for reproducibility, "
    "seed={D.SEED})')"
)

# ---- Step 1: the data ----
add_md(
    "## Step 1 — The data: a real 2-D distribution (two moons), standardized\n"
    "\n"
    "We sample the real `make_moons` distribution — two interleaving crescents — and **standardize** it to "
    "zero-mean / unit-variance so that $\\mathcal N(0,I)$ is the right target for the forward process. This is "
    "genuinely 2-D, non-Gaussian, and multi-modal: a distribution the model must *learn*, not one it starts from."
)
add_code(
    "data = D.load_moons(seed=0)\n"
    "print(f'dataset : {data.label}   |   {tuple(data.x.shape)} points, standardized')\n"
    "\n"
    "plt.figure(figsize=(4.2, 4.2))\n"
    "plt.scatter(data.x[:, 0], data.x[:, 1], s=6, c='#3A6B96', alpha=0.5, linewidths=0)\n"
    "plt.gca().set_aspect('equal')\n"
    "plt.xticks([])\n"
    "plt.yticks([])\n"
    "plt.title('real make_moons data (the target the DDPM will learn to generate)')\n"
    "plt.show()"
)

# ---- Step 2: the noise schedule ----
add_md(
    "## Step 2 — The noise schedule: β, α, and the all-important ᾱ\n"
    "\n"
    "The forward process adds noise on a fixed **variance schedule** $\\beta_1,\\dots,\\beta_T$. From it we "
    "precompute $\\alpha_t=1-\\beta_t$ and the **compounded signal-survival** "
    "$\\bar\\alpha_t=\\prod_{s\\le t}\\alpha_s$ — the single most important quantity: the closed-form forward has "
    "mean $\\sqrt{\\bar\\alpha_t}x_0$ and variance $1-\\bar\\alpha_t$. Watch $\\bar\\alpha_t$ fall from 1 (all "
    "signal) to $\\approx 0$ (all noise)."
)
add_code(
    "sched = D.make_schedule(kind='linear', T=D.T_STEPS)\n"
    "t = np.arange(1, sched.T + 1)\n"
    "print(f'T = {sched.T}   beta in [{sched.betas[0]:.4f}, {sched.betas[-1]:.4f}]   "
    "abar_T = {sched.alpha_bars[-1]:.4f}  (x_T is essentially N(0, I))')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(7.5, 4))\n"
    "ax.plot(t, sched.betas, color='#8B3B4A', lw=2, label=r'$\\beta_t$ (noise per step)')\n"
    "ax.plot(t, sched.alpha_bars, color='#3A6B96', lw=2.4, label=r'$\\bar\\alpha_t$ (signal survival)')\n"
    "ax.plot(t, sched.sqrt_one_minus_abar, color='#5D4A8A', lw=2, ls='--', label=r'$\\sqrt{1-\\bar\\alpha_t}$ "
    "(noise weight)')\n"
    "ax.set_xlabel('diffusion step t')\n"
    "ax.set_ylabel('value')\n"
    "ax.legend()\n"
    "ax.set_title('the noise schedule: everything the forward/reverse steps need, precomputed')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 3: the closed-form forward ----
add_md(
    "## Step 3 — The forward process: jump to any noise level in one shot\n"
    "\n"
    "The **nice property**: instead of simulating $t$ noising steps, jump straight to any $x_t$ with\n"
    "\n"
    "$$x_t=\\sqrt{\\bar\\alpha_t}\\,x_0+\\sqrt{1-\\bar\\alpha_t}\\,\\varepsilon,\\qquad \\varepsilon\\sim\\mathcal "
    "N(0,I).$$\n"
    "\n"
    "We apply `q_sample` at increasing $t$ and watch the two moons dissolve into an isotropic Gaussian."
)
add_code(
    "torch.manual_seed(0)\n"
    "x0 = data.x[:800]\n"
    "eps = torch.randn_like(x0)  # shared noise so the same cloud dissolves coherently\n"
    "ts = [0, 40, 120, 240, sched.T]\n"
    "fig, axes = plt.subplots(1, len(ts), figsize=(13, 2.7))\n"
    "for ax, tt in zip(axes, ts):\n"
    "    if tt == 0:\n"
    "        pts = x0\n"
    "    else:\n"
    "        idx = torch.full((x0.shape[0],), tt - 1, dtype=torch.long)\n"
    "        pts, _ = D.q_sample(x0, idx, sched, eps=eps)\n"
    "    ax.scatter(pts[:, 0], pts[:, 1], s=5, c='#3A6B96' if tt == 0 else '#5D4A8A', alpha=0.5, linewidths=0)\n"
    "    ax.set_aspect('equal')\n"
    "    ax.set_xlim(-3, 3)\n"
    "    ax.set_ylim(-3, 3)\n"
    "    ax.set_xticks([])\n"
    "    ax.set_yticks([])\n"
    "    abar = 1.0 if tt == 0 else float(sched.alpha_bars[tt - 1])\n"
    "    ax.set_title(f't = {tt}\\n' + r'$\\bar\\alpha_t$ = ' + f'{abar:.2f}', fontsize=9)\n"
    "plt.suptitle('forward diffusion: the real moons dissolve into pure N(0, I) noise', y=1.12)\n"
    "plt.show()"
)

# ---- Step 4: proof 1 ----
add_md(
    "## Step 4 — Proof: the closed form *is* the iterative chain\n"
    "\n"
    "The whole method rests on the one-shot jump equalling the slow, step-by-step chain. We check it: fix a single "
    "$x_0$, run the **iterative** forward ($t=240$ tiny Gaussian steps) $N=60{,}000$ times, and compare the "
    "empirical mean and covariance to the closed-form $\\sqrt{\\bar\\alpha_t}x_0$ and $(1-\\bar\\alpha_t)I$. "
    "`assert` they agree."
)
add_code(
    "fp = D.prove_forward_closed_form(sched)\n"
    "print(f'at t = {fp.t}:  abar_t = {fp.abar_t:.4f}')\n"
    "print(f'closed-form mean = {fp.closed_mean.round(4)}   iterative mean = {fp.iter_mean.round(4)}  "
    "(N={fp.n_rollouts:,})')\n"
    "print(f'closed-form var  = {fp.closed_var:.4f}         iterative cov diag = {np.diag(fp.iter_cov).round(4)}')\n"
    "print(f'max|mean err| = {fp.mean_abs_err:.2e}   max|cov err| = {fp.var_abs_err:.2e}')\n"
    "assert fp.mean_abs_err < 0.05 and fp.var_abs_err < 0.05\n"
    "print('OK: the one-shot q_sample is exactly the marginal of the compounded chain — the nice property holds.')"
)

# ---- Step 5: the denoiser ----
add_md(
    "## Step 5 — The denoiser $\\varepsilon_\\theta(x_t, t)$: predict the noise\n"
    "\n"
    "The learned part is one small network that takes a noised point $x_t$ and its timestep $t$ and predicts the "
    "**noise** that was added. The timestep is fed through a **sinusoidal embedding** (like transformer positions) "
    "so one network can behave differently at every noise level."
)
add_code(
    "model = D.Denoiser2D(dim=2)\n"
    "n_params = sum(p.numel() for p in model.parameters())\n"
    "print(f'denoiser: {n_params:,} parameters  (a small MLP + sinusoidal time embedding)')\n"
    "xb = data.x[:5]\n"
    "t_batch = torch.full((5,), 100.0)\n"
    "eps_pred = model(xb, t_batch)\n"
    "print(f'input x_t {tuple(xb.shape)}, timestep t {tuple(t_batch.shape)}  ->  predicted noise "
    "{tuple(eps_pred.shape)}  (same shape as the data)')"
)

# ---- Step 6: the loss ----
add_md(
    "## Step 6 — The loss: $L_{\\text{simple}}=\\mathbb E\\,\\|\\varepsilon-\\varepsilon_\\theta(x_t,t)\\|^2$\n"
    "\n"
    "Training is plain MSE: draw a random timestep $t$ and noise $\\varepsilon$, form $x_t$ with the closed-form "
    "jump, and regress the network's prediction onto the *true* noise. No adversary, no KL to balance — the "
    "variational bound reduces to this after the $\\varepsilon$-parameterization."
)
add_code(
    "g = torch.Generator().manual_seed(0)\n"
    "loss = D.diffusion_loss(model, data.x[:256], sched, generator=g)\n"
    "print(f'L_simple on one batch (untrained model) = {loss.item():.4f}   (~1.0: predicting noise no better than "
    "zero)')"
)

# ---- Step 7: train ----
add_md(
    "## Step 7 — Train the denoiser from scratch\n"
    "\n"
    "Now the full loop: for each minibatch, sample random timesteps and noise, form $x_t$, predict the noise, take "
    "the MSE, one Adam step. Watch $L_{\\text{simple}}$ fall. (A real training run — about 20 seconds on CPU.)"
)
add_code(
    "res = D.train_ddpm(data, sched, n_epochs=D.N_EPOCHS, seed=0)\n"
    "print(f'trained {D.N_EPOCHS} epochs in {res.seconds:.1f}s  |  L_simple: {res.loss_hist[0]:.3f} -> "
    "{res.final_loss:.3f}')\n"
    "\n"
    "ep = np.arange(1, len(res.loss_hist) + 1)\n"
    "plt.figure(figsize=(7.5, 4))\n"
    "plt.plot(ep, res.loss_hist, color='#5D4A8A', lw=2.2)\n"
    "plt.xlabel('epoch')\n"
    "plt.ylabel(r'$L_{simple}=\\mathbb{E}\\,\\|\\varepsilon-\\varepsilon_\\theta\\|^2$')\n"
    "plt.title('the noise-prediction loss falls as the denoiser learns')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 8: sample (reverse trajectory) ----
add_md(
    "## Step 8 — Generate: run the reverse chain from pure noise\n"
    "\n"
    "Generation is **ancestral sampling**: start from $x_T\\sim\\mathcal N(0,I)$ and, for $t=T,\\dots,1$, apply the "
    "learned reverse step\n"
    "\n"
    "$$x_{t-1}=\\tfrac{1}{\\sqrt{\\alpha_t}}\\big(x_t-\\tfrac{1-\\alpha_t}{\\sqrt{1-\\bar\\alpha_t}}"
    "\\,\\varepsilon_\\theta(x_t,t)\\big)+\\sigma_t z.$$\n"
    "\n"
    "We record the cloud at several timesteps — watch the two-moons condense out of the noise (it happens late!)."
)
add_code(
    "_, traj = D.p_sample_loop(res.model, sched, n=900, dim=2, seed=7, record_every=1)\n"
    "by_t = {tt: cloud for tt, cloud in traj}\n"
    "want = [sched.T, 100, 60, 25, 1]\n"
    "snaps = [(tt, by_t[min(by_t, key=lambda k: abs(k - tt))]) for tt in want]\n"
    "fig, axes = plt.subplots(1, len(snaps), figsize=(13, 2.7))\n"
    "for ax, (tt, cloud) in zip(axes, snaps):\n"
    "    c = cloud.numpy()\n"
    "    ax.scatter(c[:, 0], c[:, 1], s=5, c='#2E7A5A' if tt == 1 else '#5D4A8A', alpha=0.5, linewidths=0)\n"
    "    ax.set_aspect('equal')\n"
    "    ax.set_xlim(-3, 3)\n"
    "    ax.set_ylim(-3, 3)\n"
    "    ax.set_xticks([])\n"
    "    ax.set_yticks([])\n"
    "    ax.set_title(f't = {tt}' + ('  (start)' if tt == sched.T else '  (generated)' if tt == 1 else ''), "
    "fontsize=9)\n"
    "plt.suptitle('the reverse process: N(0, I) noise denoises step by step back into the two moons', y=1.12)\n"
    "plt.show()"
)

# ---- Step 9: proof 2 ----
add_md(
    "## Step 9 — Proof: the generated samples match the target\n"
    "\n"
    "A small loss is not the same as good generation — we test it *directly*. Draw samples from the trained chain "
    "and score them against the real target with the **energy distance** (a two-sample statistic, zero iff the "
    "distributions match). Compare to the $\\mathcal N(0,I)$ baseline (the chain's starting cloud). `assert` the "
    "generated samples are far closer to the target."
)
add_code(
    "gp = D.prove_generation(res, data)\n"
    "print(f'energy distance  generated  vs target = {gp.ed_generated:.4f}')\n"
    "print(f'energy distance  N(0,I) base vs target = {gp.ed_baseline:.4f}   (the chain started here)')\n"
    "print(f'ratio = {gp.ratio:.3f}   -> generated is {1/gp.ratio:.1f}x closer to the target than pure noise')\n"
    "assert gp.ratio < 0.5\n"
    "print('OK: the reverse chain learned the distribution, it did not just echo its Gaussian start.')"
)

# ---- Step 10: overlay ----
add_md(
    "## Step 10 — The payoff: generated points sit on the real manifold\n"
    "\n"
    "Overlay the generated samples on the real data. They should land on the two crescents — same shape, same gap "
    "— the geometry reproduced from pure noise, having only ever learned to undo one small step."
)
add_code(
    "gen, _ = D.p_sample_loop(res.model, sched, n=1500, dim=2, seed=11)\n"
    "gen = gen.numpy()\n"
    "plt.figure(figsize=(5, 5))\n"
    "plt.scatter(data.x[:1500, 0], data.x[:1500, 1], s=7, c='#4A5B6E', alpha=0.35, linewidths=0, label='real')\n"
    "plt.scatter(gen[:, 0], gen[:, 1], s=7, c='#2E7A5A', alpha=0.45, linewidths=0, label='generated')\n"
    "plt.gca().set_aspect('equal')\n"
    "plt.xlim(-2.6, 2.6)\n"
    "plt.ylim(-2.6, 2.6)\n"
    "plt.xticks([])\n"
    "plt.yticks([])\n"
    "plt.legend()\n"
    "plt.title('generated (green) on the real two-moons (grey)')\n"
    "plt.show()"
)

# ---- Step 11: schedule comparison ----
add_md(
    "## Step 11 — Linear vs cosine schedule\n"
    "\n"
    "The **cosine** schedule (Nichol & Dhariwal, 2021) adds noise more gently near the ends, keeping "
    "$\\bar\\alpha_t$ higher for longer — better likelihoods and samples on images than the linear schedule. "
    "Compare the two $\\bar\\alpha_t$ curves."
)
add_code(
    "cos = D.make_schedule(kind='cosine', T=D.T_STEPS)\n"
    "t = np.arange(1, sched.T + 1)\n"
    "plt.figure(figsize=(7.5, 4))\n"
    "plt.plot(t, sched.alpha_bars, color='#3A6B96', lw=2.4, label='linear')\n"
    "plt.plot(t, cos.alpha_bars, color='#2E7A5A', lw=2.2, ls='--', label='cosine')\n"
    "plt.xlabel('diffusion step t')\n"
    "plt.ylabel(r'$\\bar\\alpha_t$')\n"
    "plt.legend()\n"
    "plt.title('cosine keeps more signal for longer than linear')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 12: try it ----
add_md(
    "## Step 12 — Try it: predict, then check\n"
    "\n"
    "Before running, *predict the direction*. We cut $T$ from 400 to **50** and retrain. With so few steps, "
    "$\\bar\\alpha_T$ no longer reaches $\\approx 0$ — so $x_T$ still carries a lot of signal, and sampling from "
    "$\\mathcal N(0,I)$ starts from the *wrong* place. Will the generated-vs-target energy distance get **worse**? "
    "Write your guess, then check."
)
add_code(
    "sched50 = D.make_schedule(kind='linear', T=50)\n"
    "print(f'T=50:  abar_T = {sched50.alpha_bars[-1]:.3f}  (NOT near 0 -> x_T keeps signal, prior mismatch)')\n"
    "res50 = D.train_ddpm(data, sched50, n_epochs=400, seed=0)\n"
    "gp50 = D.prove_generation(res50, data)\n"
    "print(f'T=400 (trained above): generated-vs-target energy distance = {gp.ed_generated:.4f}  (ratio "
    "{gp.ratio:.2f})')\n"
    "print(f'T=50:                  generated-vs-target energy distance = {gp50.ed_generated:.4f}  (ratio "
    "{gp50.ratio:.2f})')\n"
    "print('Too few steps -> abar_T not ~0 -> worse samples. More (finer) steps = easier reverse = better "
    "samples, but slower sampling.')"
)

# ---- close ----
add_md(
    "## Recap\n"
    "\n"
    "You built a DDPM from scratch and **proved** it: the closed-form forward "
    "$q(x_t\\mid x_0)=\\mathcal N(\\sqrt{\\bar\\alpha_t}x_0,(1-\\bar\\alpha_t)I)$ equals the slow iterative "
    "noising chain (the *nice property* that lets training jump to any $t$), and a from-scratch denoiser trained "
    "by $L_{\\text{simple}}=\\mathbb E\\|\\varepsilon-\\varepsilon_\\theta\\|^2$ **generates** the real two-moons "
    "target (energy distance far below the $\\mathcal N(0,I)$ baseline). Generation is ancestral sampling down the "
    "learned reverse chain — slow, because it runs all $T$ steps.\n"
    "\n"
    "See the companion page for the full derivation (the compounding of $\\bar\\alpha_t$, the variational bound as "
    "a sum of Gaussian KLs, the $\\varepsilon$-parameterization that yields $L_{\\text{simple}}$, the sampling "
    "update), the pitfalls (slow sampling, the reweighted bound, schedule choice, the timestep embedding), and "
    "where it goes next: a DDPM is a **hierarchical VAE with a fixed encoder**, predicting noise is **learning the "
    "score**, and running this exact loop in a VAE's latent space is **Stable Diffusion**."
)


def build() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    NB_PATH.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {NB_PATH}  ({len(CELLS)} cells)")


if __name__ == "__main__":
    build()
