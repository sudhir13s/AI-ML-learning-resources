"""Generate the step-by-step teaching notebook (01-Variational-Autoencoders-VAE-ELBO.ipynb).

The notebook mirrors ``vae.py`` one step at a time so a learner can open it, run every cell live, and *see* a
VAE built and proven on real digits: the data, the Gaussian encoder, the reparameterization trick and its
gradient-flow proof, the ELBO loss with its closed-form KL (proven equal to a Monte-Carlo KL), real from-scratch
training on MNIST, reconstructions, prior samples (novel digits), the 2-D latent manifold, a latent interpolation,
and the beta-VAE trade-off. Each numbered step has a short markdown lead-in (the intuition) followed by a focused
code cell with real output.

    python build_notebook_01.py         # writes the .ipynb (unexecuted) into the chapter's code/
    python -m nbconvert --to notebook --execute --inplace \
        "../01-Variational-Autoencoders-VAE-ELBO/code/01-Variational-Autoencoders-VAE-ELBO.ipynb"

This generator lives in the domain-level ``10. GenAI/tools/`` folder; the notebook it writes (and the module it
mirrors) stay in the chapter's ``code/`` folder. Kept as a generator (not a hand-edited .ipynb) so the notebook
and the module stay in lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

_CHAPTER_CODE = Path(__file__).resolve().parent.parent / "01-Variational-Autoencoders-VAE-ELBO" / "code"
NB_PATH = _CHAPTER_CODE / "01-Variational-Autoencoders-VAE-ELBO.ipynb"

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
    "# Variational Autoencoders (VAE) — a runnable, measured, *proven* build\n"
    "\n"
    "A plain **autoencoder** compresses an image to a code and rebuilds it — but its latent space is a *bag of "
    "holes*: sample a random code and you decode garbage, so you cannot **generate**. A **VAE** fixes this by "
    "making the encoder output a *distribution* $q_\\phi(z\\mid x)=\\mathcal N(\\mu,\\sigma^2 I)$, pulling every "
    "posterior toward a smooth prior $\\mathcal N(0,I)$ with a **KL** term, and decoding a *sampled* $z$. Then any "
    "$z$ drawn from the prior decodes to something real — you can generate.\n"
    "\n"
    "This notebook builds a VAE **from scratch** on **real** MNIST digits and *proves* the two claims that "
    "matter:\n"
    "\n"
    "- **The closed-form Gaussian KL is exact.** We check with a hard `assert` that the ELBO's KL term "
    "$\\tfrac12\\sum(\\mu^2+\\sigma^2-1-\\log\\sigma^2)$ equals a Monte-Carlo estimate of "
    "$\\mathbb{E}_q[\\log q(z\\mid x)-\\log p(z)]$.\n"
    "- **The reparameterization trick is what lets gradients train the encoder.** We show, with an `assert`, "
    "that $z=\\mu+\\sigma\\odot\\varepsilon$ gives real gradients through $\\mu,\\sigma$, while a raw `.sample()` "
    "gives **none**.\n"
    "\n"
    "It imports the **exact same functions** as the companion page and its figures (from `vae.py`), so the "
    "numbers here are the numbers there. Everything is **seeded and CPU-pinned** for a reproducible trace.\n"
    "\n"
    "> Companion page: **Variational Autoencoders (VAE · ELBO)**. Run top-to-bottom (Kernel → Restart & Run "
    "All). If MNIST can't be downloaded, the module falls back to the real `sklearn` `load_digits` 8x8 set — "
    "every run still executes on real image data."
)

# ---- Step 0: setup ----
add_md(
    "## Step 0 — Setup: import the real module and print versions\n"
    "\n"
    "We import the pipeline from the chapter module so this notebook runs the *same code* the page and figures "
    "use, and print the library versions and the device. Training pins **CPU** for a reproducible trace (a small "
    "MLP VAE trains fast enough on CPU)."
)
add_code(
    "import numpy as np\n"
    "import torch\n"
    "import matplotlib.pyplot as plt\n"
    "\n"
    "import vae as V\n"
    "\n"
    "try:\n"
    "    import torchvision\n"
    "    tv_ver = torchvision.__version__\n"
    "except Exception:\n"
    "    tv_ver = 'not installed'\n"
    "print(f'torch {torch.__version__} | numpy {np.__version__} | torchvision {tv_ver}')\n"
    "print(f'best available device = {V.get_device()}  (training pinned to CPU for reproducibility, "
    "seed={V.SEED})')"
)

# ---- Step 1: the data ----
add_md(
    "## Step 1 — The data: real MNIST digits as pixel probabilities in [0, 1]\n"
    "\n"
    "We load a real subset of **MNIST** (28×28 grayscale handwritten digits), flattened to 784-dim vectors with "
    "pixels in $[0,1]$. We treat each pixel as the mean of a **Bernoulli** — the decoder will output a "
    "probability per pixel. (If MNIST can't be fetched, the module falls back to the real `sklearn` 8×8 digits; "
    "the banner says which.)"
)
add_code(
    "data = V.load_data()\n"
    "print(f'dataset : {data.label}   [{data.source}]')\n"
    "print(f'train   : {tuple(data.x_train.shape)}   test: {tuple(data.x_test.shape)}   "
    "pixels/image: {data.n_pixels}')\n"
    "\n"
    "h, w = data.img_shape\n"
    "fig, axes = plt.subplots(1, 8, figsize=(9, 1.4))\n"
    "for j in range(8):\n"
    "    axes[j].imshow(data.x_test[j].numpy().reshape(h, w), cmap='gray_r', vmin=0, vmax=1)\n"
    "    axes[j].set_title(str(int(data.y_test[j])), fontsize=9)\n"
    "    axes[j].axis('off')\n"
    "plt.suptitle('real MNIST digits (the data the VAE will learn to reconstruct and generate)', y=1.15)\n"
    "plt.show()"
)

# ---- Step 2: the model ----
add_md(
    "## Step 2 — The VAE: a Gaussian encoder and a Bernoulli decoder\n"
    "\n"
    "The **encoder** maps an image to the parameters of an approximate posterior "
    "$q_\\phi(z\\mid x)=\\mathcal N(\\mu(x),\\operatorname{diag}\\sigma^2(x))$ — a mean $\\mu$ and a "
    "*log-variance* $\\log\\sigma^2$ (predicting the log keeps it unconstrained and stable). The **decoder** maps "
    "$z$ back to per-pixel Bernoulli **logits**. Note the latent is just **2-D** here — small enough that we can "
    "later *draw* the whole latent space."
)
add_code(
    "model = V.VAE(data.n_pixels, latent_dim=V.LATENT_DIM)\n"
    "xb = data.x_test[:5]\n"
    "mu, logvar = model.encode(xb)\n"
    "print(f'latent dim      : {model.latent_dim}')\n"
    "print(f'encoder outputs : mu {tuple(mu.shape)}, logvar {tuple(logvar.shape)}  (one (mu, logsigma^2) per "
    "image)')\n"
    "print(f'mu[0]           : {mu[0].detach().numpy().round(3)}')\n"
    "print(f'logvar[0]       : {logvar[0].detach().numpy().round(3)}  (untrained -> near 0 => sigma^2 ~ 1)')"
)

# ---- Step 3: the reparameterization trick ----
add_md(
    "## Step 3 — The reparameterization trick: sample $z$ *differentiably*\n"
    "\n"
    "To train end-to-end we must sample $z\\sim\\mathcal N(\\mu,\\sigma^2)$ — but sampling is not a "
    "differentiable function of $\\mu,\\sigma$, so gradients cannot flow back to the encoder. The trick: move the "
    "randomness into a **parameter-free** $\\varepsilon\\sim\\mathcal N(0,I)$ and reconstruct $z$ by a "
    "differentiable affine map,\n"
    "\n"
    "$$z = \\mu + \\sigma\\odot\\varepsilon,\\qquad \\sigma=\\exp(\\tfrac12\\log\\sigma^2).$$\n"
    "\n"
    "Now $\\partial z/\\partial\\mu=1$ and $\\partial z/\\partial\\sigma=\\varepsilon$ are well defined."
)
add_code(
    "z = model.reparameterize(mu, logvar)\n"
    "print(f'z = mu + sigma * eps  ->  z shape {tuple(z.shape)}')\n"
    "print(f'z[0]      : {z[0].detach().numpy().round(3)}   (mu[0] plus noise scaled by sigma)')\n"
    "print(f'z requires_grad = {z.requires_grad}  (differentiable in mu, logvar -> the encoder can be trained)')"
)

# ---- Step 4: proof 2, the reparam gradient ----
add_md(
    "## Step 4 — Proof: the trick is *why* the encoder can be trained\n"
    "\n"
    "Claim: $z=\\mu+\\sigma\\varepsilon$ gives gradients through $\\mu,\\sigma$, while a raw "
    "`torch.distributions.Normal(mu, sigma).sample()` gives **none** (it detaches $z$ from the encoder). We build "
    "a tiny loss $\\lVert z\\rVert^2$ both ways and check the gradients. `assert` it."
)
add_code(
    "rp = V.prove_reparameterization()\n"
    "print('reparameterized z = mu + sigma*eps :')\n"
    "print(f'   ||d loss / d mu||     = {rp.grad_mu_reparam:.3f}   (gradient flows)')\n"
    "print(f'   ||d loss / d logvar|| = {rp.grad_logvar_reparam:.3f}   (gradient flows)')\n"
    "print(f'direct .sample() z :   z.requires_grad = {rp.direct_sample_has_grad}   (NO gradient to encoder)')\n"
    "assert rp.grad_mu_reparam > 1e-6 and rp.grad_logvar_reparam > 1e-6\n"
    "assert rp.direct_sample_has_grad is False\n"
    "print('OK: reparameterization gives gradients where a raw sample gives none — this is why we reparameterize.')"
)

# ---- Step 5: the ELBO loss ----
add_md(
    "## Step 5 — The loss: the (negative) ELBO = reconstruction + KL-to-prior\n"
    "\n"
    "We maximize the **Evidence Lower BOund**\n"
    "\n"
    "$$\\log p(x)\\ \\ge\\ \\underbrace{\\mathbb{E}_q[\\log p_\\theta(x\\mid z)]}_{\\text{reconstruction}}\\ -\\ "
    "\\underbrace{\\mathrm{KL}(q_\\phi(z\\mid x)\\,\\|\\,\\mathcal N(0,I))}_{\\text{regularizer}},$$\n"
    "\n"
    "i.e. minimize its negation. **Reconstruction** is the per-pixel binary cross-entropy (the Bernoulli "
    "negative log-likelihood). **KL** uses the closed form "
    "$\\tfrac12\\sum(\\mu^2+\\sigma^2-1-\\log\\sigma^2)$. Both are per-image means."
)
add_code(
    "logits, mu, logvar = model(xb)\n"
    "loss, recon, kl = V.elbo_loss(logits, xb, mu, logvar, beta=1.0)\n"
    "print('negative ELBO = recon + KL')\n"
    "print(f'   reconstruction (BCE, summed over pixels) = {recon.item():.2f}')\n"
    "print(f'   KL(q || N(0,I))  closed form             = {kl.item():.2f}')\n"
    "print(f'   total negative ELBO                      = {loss.item():.2f}   (untrained: high)')"
)

# ---- Step 6: proof 1, the closed-form KL ----
add_md(
    "## Step 6 — Proof: the closed-form KL *is* the KL\n"
    "\n"
    "The KL term is a *closed form*, but the **definition** of KL is an expectation, "
    "$\\mathrm{KL}(q\\|p)=\\mathbb{E}_{z\\sim q}[\\log q(z)-\\log p(z)]$. If the closed form is right, sampling "
    "$z\\sim q$ and averaging $\\log q(z\\mid x)-\\log p(z)$ must converge to it. We `assert` they agree."
)
add_code(
    "kp = V.prove_gaussian_kl(n_samples=400_000)\n"
    "print(f'closed form  1/2 sum(mu^2 + sigma^2 - 1 - logvar) = {kp.closed_form:.5f}')\n"
    "print(f'Monte-Carlo  E_q[log q(z|x) - log p(z)]  (N={kp.n_samples:,}) = {kp.monte_carlo:.5f}')\n"
    "print(f'|closed - MC| = {kp.abs_error:.2e}')\n"
    "assert kp.abs_error < 1e-2\n"
    "print('OK: the ELBO KL term is exactly KL(q || p) — the term the loss uses is the term the math claims.')"
)

# ---- Step 7: train ----
add_md(
    "## Step 7 — Train from scratch on real MNIST\n"
    "\n"
    "Now the full loop: for each minibatch, encode → reparameterize → decode → negative ELBO → one Adam step. We "
    "train the headline model ($\\beta=1$, 2-D latent) and watch the total loss fall, split into its "
    "reconstruction and KL parts. (A real training run — a couple of minutes on CPU.)"
)
add_code(
    "res = V.train_vae(data, beta=1.0, latent_dim=V.LATENT_DIM, n_epochs=V.N_EPOCHS, seed=0)\n"
    "print(f'trained {V.N_EPOCHS} epochs  |  final negative-ELBO = {res.final_elbo:.2f}  "
    "(recon {res.final_recon:.2f} + KL {res.final_kl:.2f})')\n"
    "\n"
    "ep = np.arange(1, len(res.elbo) + 1)\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "ax.plot(ep, res.elbo, color='#5D4A8A', lw=2.3, label='negative ELBO (total)')\n"
    "ax.plot(ep, res.recon, color='#2E7A5A', lw=2, label='reconstruction')\n"
    "ax.plot(ep, res.kl, color='#8B3B4A', lw=2, label='KL to prior')\n"
    "ax.set_xlabel('epoch')\n"
    "ax.set_ylabel('per-image loss (nats)')\n"
    "ax.set_title('the ELBO falls as the VAE learns')\n"
    "ax.legend()\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 8: reconstructions ----
add_md(
    "## Step 8 — Reconstruct: encode real digits and decode them back\n"
    "\n"
    "The first thing the VAE can do: compress a real held-out digit to its 2-D code and rebuild it. The "
    "reconstructions are recognizable but a little **blurry** — the Gaussian/Bernoulli likelihood averages over "
    "the posterior, an honest and characteristic VAE trait (GANs and diffusion are sharper)."
)
add_code(
    "x = data.x_test[:10]\n"
    "recon = V.reconstruct(res.model, x)\n"
    "fig, axes = plt.subplots(2, 10, figsize=(11, 2.3))\n"
    "for j in range(10):\n"
    "    axes[0, j].imshow(x[j].numpy().reshape(h, w), cmap='gray_r', vmin=0, vmax=1)\n"
    "    axes[1, j].imshow(recon[j].reshape(h, w), cmap='gray_r', vmin=0, vmax=1)\n"
    "    axes[0, j].axis('off')\n"
    "    axes[1, j].axis('off')\n"
    "axes[0, 0].set_title('input', fontsize=9, loc='left')\n"
    "axes[1, 0].set_title('reconstruction', fontsize=9, loc='left')\n"
    "plt.suptitle('top: real inputs   bottom: VAE reconstructions (blurry = the likelihood averaging effect)', "
    "y=1.05)\n"
    "plt.show()"
)

# ---- Step 9: prior samples ----
add_md(
    "## Step 9 — Generate: sample $z\\sim\\mathcal N(0,I)$ and decode\n"
    "\n"
    "This is the payoff a plain autoencoder cannot give. Because the KL term forced the encoded posteriors to "
    "fill the prior, we can sample a **fresh** $z$ from $\\mathcal N(0,I)$ and decode it into a *new* digit the "
    "model dreamed up — never in the training set. That is what makes a VAE a **generative** model."
)
add_code(
    "samples = V.sample_prior(res.model, n=15, seed=1)\n"
    "fig, axes = plt.subplots(1, 15, figsize=(12, 1.1))\n"
    "for j in range(15):\n"
    "    axes[j].imshow(samples[j].reshape(h, w), cmap='gray_r', vmin=0, vmax=1)\n"
    "    axes[j].axis('off')\n"
    "plt.suptitle('novel digits generated by decoding z ~ N(0, I) — a plain autoencoder cannot do this', y=1.3)\n"
    "plt.show()"
)

# ---- Step 10: latent manifold ----
add_md(
    "## Step 10 — See the whole latent space: the 2-D manifold\n"
    "\n"
    "Because the latent is 2-D, we can decode a **grid** of $z$ across the prior and lay the results out. "
    "Neighbours in $z$ decode to visually similar digits, and the classes are organized into contiguous regions "
    "with smooth transitions between them — a continuous, sampleable space, not a bag of holes."
)
add_code(
    "grid = 16\n"
    "tiles = V.latent_manifold(res.model, grid=grid, span=2.5)\n"
    "canvas = np.zeros((grid * h, grid * w))\n"
    "for idx in range(grid * grid):\n"
    "    r, c = divmod(idx, grid)\n"
    "    canvas[r*h:(r+1)*h, c*w:(c+1)*w] = tiles[idx].reshape(h, w)\n"
    "plt.figure(figsize=(6.5, 6.5))\n"
    "plt.imshow(canvas, cmap='gray_r', vmin=0, vmax=1)\n"
    "plt.xticks([0, grid*w-1], ['$z_1=-2.5$', '$z_1=+2.5$'])\n"
    "plt.yticks([0, grid*h-1], ['$z_2=-2.5$', '$z_2=+2.5$'])\n"
    "plt.title('the 2-D latent manifold: a smooth, organized space')\n"
    "plt.tight_layout()\n"
    "plt.show()"
)

# ---- Step 11: interpolation ----
add_md(
    "## Step 11 — Interpolate: walk a straight line between two digits\n"
    "\n"
    "One more window on the smoothness: encode two different digits, linearly interpolate between their latent "
    "codes, and decode each step. The frames **morph** gradually from one digit into the other — no abrupt cuts. "
    "That continuity is exactly what the KL-to-prior regularizer buys."
)
add_code(
    "y = data.y_test\n"
    "def first_of(label, fb):\n"
    "    idx = np.flatnonzero(y == label)\n"
    "    return int(idx[0]) if idx.size else fb\n"
    "i0, i1 = first_of(7, 0), first_of(3, 1)\n"
    "frames = V.interpolate(res.model, data.x_test[i0], data.x_test[i1], steps=10)\n"
    "fig, axes = plt.subplots(1, 10, figsize=(11, 1.3))\n"
    "for j in range(10):\n"
    "    axes[j].imshow(frames[j].reshape(h, w), cmap='gray_r', vmin=0, vmax=1)\n"
    "    axes[j].axis('off')\n"
    "plt.suptitle('a smooth morph between two encoded digits (latent interpolation)', y=1.3)\n"
    "plt.show()"
)

# ---- Step 12: beta-VAE trade-off ----
add_md(
    "## Step 12 — The $\\beta$-VAE knob: trade reconstruction against KL\n"
    "\n"
    "Weight the KL term by $\\beta$. With $\\beta>1$ the encoder is pulled **harder** toward the prior — a "
    "*smaller* KL (tighter, more disentangled latent) but at the cost of **worse reconstruction**. Train "
    "$\\beta=1$ and $\\beta=4$ and read the trade-off off the two terms. We `assert` the direction."
)
add_code(
    "r1 = res  # beta=1 already trained above\n"
    "r4 = V.train_vae(data, beta=4.0, latent_dim=V.LATENT_DIM, n_epochs=V.N_EPOCHS, seed=0)\n"
    "print(f'beta=1:  recon = {r1.final_recon:6.2f}   KL = {r1.final_kl:5.2f}')\n"
    "print(f'beta=4:  recon = {r4.final_recon:6.2f}   KL = {r4.final_kl:5.2f}')\n"
    "assert r4.final_kl < r1.final_kl        # larger beta -> tighter to prior (smaller KL)\n"
    "assert r4.final_recon > r1.final_recon  # ...at the cost of reconstruction\n"
    "print('OK: larger beta buys a smaller KL at the cost of reconstruction — the disentanglement/sharpness "
    "trade-off.')"
)

# ---- Step 13: try it ----
add_md(
    "## Step 13 — Try it: predict, then check\n"
    "\n"
    "Before running, *predict the direction*. **(1)** Push $\\beta$ to **10**: will the reconstructions get "
    "sharper or blurrier, and will the KL shrink further — possibly toward **posterior collapse** (KL → 0, the "
    "decoder ignoring $z$)? **(2)** Raise the latent dim from 2 to **10**: will reconstruction improve, and can "
    "you still draw the manifold? Write your guess, change the one line, and check."
)
add_code(
    "r10 = V.train_vae(data, beta=10.0, latent_dim=V.LATENT_DIM, n_epochs=V.N_EPOCHS, seed=0)\n"
    "print(f'beta=10: recon = {r10.final_recon:6.2f}   KL = {r10.final_kl:5.2f}  '\n"
    "      f'(KL shrinks further -> the road to posterior collapse)')\n"
    "samples10 = V.sample_prior(r10.model, n=10, seed=1)\n"
    "fig, axes = plt.subplots(1, 10, figsize=(11, 1.2))\n"
    "for j in range(10):\n"
    "    axes[j].imshow(samples10[j].reshape(h, w), cmap='gray_r', vmin=0, vmax=1)\n"
    "    axes[j].axis('off')\n"
    "plt.suptitle(f'beta=10 prior samples (KL={r10.final_kl:.1f}): tighter to prior, blurrier digits', y=1.3)\n"
    "plt.show()"
)

# ---- close ----
add_md(
    "## Recap\n"
    "\n"
    "You built a VAE from scratch and **proved** it: the closed-form Gaussian KL matches a Monte-Carlo estimate "
    "of $\\mathrm{KL}(q\\|p)$ (the loss uses the exact term), and the **reparameterization trick** "
    "$z=\\mu+\\sigma\\varepsilon$ gives gradients through the encoder where a raw `.sample()` gives none. A "
    "VAE trained by minimizing the negative ELBO learns to **reconstruct** real digits, **generate** novel "
    "ones from the prior, and organize a **smooth 2-D latent manifold** you can interpolate across — and the "
    "$\\beta$ knob trades reconstruction against a tighter latent, all the way to posterior collapse.\n"
    "\n"
    "See the companion page for the full derivation (the intractable marginal, the ELBO and its KL-gap identity, "
    "why the reparameterization lowers gradient variance versus the score-function estimator), the pitfalls "
    "(posterior collapse, blurry samples, the aggregate-posterior/prior mismatch), and where it goes next: the "
    "VAE is the **latent space of latent diffusion (Stable Diffusion)**, and its one-step latent-variable model "
    "is the conceptual seed of **diffusion**."
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
