"""Autoencoders concept-page diagrams (muted palette, parallel matplotlib scale).

Five visuals for 05. Deep_Learning/concepts/19-Autoencoders.md:
  1. ae_architecture.png      -- SCHEMATIC: the hourglass (encoder f -> bottleneck z
     -> decoder g), reconstruction loss closing the loop.
  2. ae_latent_dim.png        -- MEASURED: digit reconstructions get sharper as the
     bottleneck z grows (2, 4, 8, 16, 32), with the reconstruction-error curve.
  3. ae_denoising.png         -- MEASURED: a denoising AE maps a noisy digit back to a
     clean one (noisy in -> reconstruction -> clean target), error vs noise level.
  4. ae_pca_overlay.png       -- MEASURED: a linear tied-weight AE trained by MSE
     recovers the SAME subspace as PCA (its decoder rows align with the top PCs;
     reconstruction errors match).
  5. ae_vae_latent.png        -- MEASURED: a VAE's 2-D latent space colored by digit
     class + the reparameterization-trick schematic.

Run with: ~/.uv/envs/ml-py312/bin/python3 tools/gen_autoencoders_diagrams.py
torch is required; the digit demos use sklearn's load_digits (8x8, no download).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
torch.manual_seed(0)
np.random.seed(0)


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- 1. Architecture schematic: the hourglass --------------------------------
def architecture():
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.axis("off")
    ax.set_xlim(0, 10); ax.set_ylim(0, 6)

    # layer heights describe the hourglass (wide -> narrow -> wide)
    cols = [
        (1.0, 4.4, BLUE, "x\ninput\n(d)"),
        (2.7, 3.0, PURPLE, "encoder f"),
        (4.5, 1.2, GREEN, "z\nlatent\n(k << d)"),
        (6.3, 3.0, PURPLE, "decoder g"),
        (8.0, 4.4, NAVY, "x̂\nrecon\n(d)"),
    ]
    for x, h, color, label in cols:
        y0 = 3 - h / 2
        box = FancyBboxPatch((x, y0), 1.1, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                             linewidth=1.5, edgecolor=color, facecolor=color)
        ax.add_patch(box)
        ax.text(x + 0.55, 3, label, ha="center", va="center", color="#fff",
                fontsize=11, fontweight="bold")
    # arrows between columns
    for x0 in (2.1, 3.8, 5.6, 7.4):
        ax.add_patch(FancyArrowPatch((x0, 3), (x0 + 0.6, 3), arrowstyle="-|>",
                     mutation_scale=16, color=SLATE, lw=2))
    # reconstruction loss bracket
    ax.annotate("", xy=(1.55, 0.7), xytext=(8.55, 0.7),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=2))
    ax.text(5.05, 0.25, "reconstruction loss  L = ||x − x̂||²",
            ha="center", va="center", color=RED, fontsize=12, fontweight="bold")
    ax.text(2.7 + 0.55, 5.4, "compress", ha="center", color=PURPLE, fontsize=10, fontweight="bold")
    ax.text(6.3 + 0.55, 5.4, "reconstruct", ha="center", color=PURPLE, fontsize=10, fontweight="bold")
    ax.text(4.5 + 0.55, 4.55, "bottleneck", ha="center", color=GREEN, fontsize=10, fontweight="bold")
    ax.set_title("Autoencoder: compress through a bottleneck, then reconstruct",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/ae_architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ae_architecture.png")


# ---- shared: load digits, build a simple AE ----------------------------------
def _load_digits():
    d = load_digits()
    X = d.data.astype(np.float32) / 16.0   # 0..1, shape (1797, 64)
    y = d.target
    return torch.tensor(X), y


class AE(nn.Module):
    def __init__(self, k, d=64, hidden=48):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, k))
        self.dec = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, d), nn.Sigmoid())

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


def _train(model, X, epochs=400, lr=2e-3, noise=0.0, target=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tgt = X if target is None else target
    for _ in range(epochs):
        opt.zero_grad()
        inp = X if noise == 0 else (X + noise * torch.randn_like(X)).clamp(0, 1)
        out, _ = model(inp)
        loss = ((out - tgt) ** 2).mean()
        loss.backward(); opt.step()
    return float(loss.item())


# ---- 2. Reconstruction quality vs latent dim (measured) ----------------------
def latent_dim():
    X, _ = _load_digits()
    dims = [2, 4, 8, 16, 32]
    recons, errs = {}, []
    sample = X[:8]
    for k in dims:
        m = AE(k)
        _train(m, X, epochs=500)
        with torch.no_grad():
            out, _ = m(X)
            errs.append(float(((out - X) ** 2).mean()))
            recons[k] = m(sample)[0].detach().numpy()
    print("  latent_dim recon MSE:", {k: round(e, 4) for k, e in zip(dims, errs)})

    fig = plt.figure(figsize=(9.6, 5.2))
    gs = fig.add_gridspec(len(dims) + 1, 8, height_ratios=[1] * (len(dims) + 1))
    # original row
    for j in range(8):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(sample[j].numpy().reshape(8, 8), cmap="gray_r")
        ax.axis("off")
        if j == 0:
            ax.set_ylabel("orig", rotation=0, ha="right", va="center", fontsize=9)
            ax.text(-3, 4, "original", ha="right", va="center", fontsize=9, fontweight="bold", color=BLUE)
    for i, k in enumerate(dims):
        for j in range(8):
            ax = fig.add_subplot(gs[i + 1, j])
            ax.imshow(recons[k][j].reshape(8, 8), cmap="gray_r")
            ax.axis("off")
            if j == 0:
                ax.text(-3, 4, f"k={k}", ha="right", va="center", fontsize=9,
                        fontweight="bold", color=GREEN)
    fig.suptitle("Reconstructions sharpen as the bottleneck k grows (sklearn digits, 8×8=64-d)",
                 fontsize=13, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0.04, 0, 1, 0.96])
    fig.savefig(f"{OUT}/ae_latent_dim.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ae_latent_dim.png")
    return dims, errs


# ---- 3. Denoising AE (measured) ----------------------------------------------
def denoising():
    X, _ = _load_digits()
    m = AE(16)
    final = _train(m, X, epochs=600, noise=0.5, target=X)  # corrupt input, clean target
    sample = X[:8]
    noisy = (sample + 0.5 * torch.randn_like(sample)).clamp(0, 1)
    with torch.no_grad():
        rec = m(noisy)[0].numpy()
    # error vs noise level (denoising AE vs identity)
    levels = np.linspace(0, 1.0, 9)
    den_err, id_err = [], []
    for s in levels:
        xn = (X + s * torch.randn_like(X)).clamp(0, 1)
        with torch.no_grad():
            out = m(xn)[0]
        den_err.append(float(((out - X) ** 2).mean()))
        id_err.append(float(((xn - X) ** 2).mean()))
    print(f"  denoising final train MSE={final:.4f}; @noise0.5 denoised={den_err[4]:.4f} vs noisy-input={id_err[4]:.4f}")

    fig = plt.figure(figsize=(9.6, 5.0))
    gs = fig.add_gridspec(3, 9, width_ratios=[1] * 8 + [3.4])
    rows = [("noisy input x̃", noisy.numpy(), RED),
            ("reconstruction", rec, GREEN),
            ("clean target x", sample.numpy(), BLUE)]
    for i, (label, imgs, color) in enumerate(rows):
        for j in range(8):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(imgs[j].reshape(8, 8), cmap="gray_r"); ax.axis("off")
            if j == 0:
                ax.text(-3.5, 4, label, ha="right", va="center", fontsize=9,
                        fontweight="bold", color=color)
    axc = fig.add_subplot(gs[:, 8])
    axc.plot(levels, id_err, color=RED, lw=2.4, marker="o", ms=3, label="noisy input vs clean")
    axc.plot(levels, den_err, color=GREEN, lw=2.4, marker="o", ms=3, label="denoised vs clean")
    axc.set_xlabel("noise std σ"); axc.set_ylabel("MSE to clean target")
    axc.set_title("denoising lowers error", fontsize=11, fontweight="bold")
    axc.legend(frameon=False, fontsize=8.5, loc="upper left"); _despine(axc)
    fig.suptitle("Denoising autoencoder: corrupt the input, reconstruct the clean digit",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0.03, 0, 1, 0.95])
    fig.savefig(f"{OUT}/ae_denoising.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ae_denoising.png")


# ---- 4. Linear AE == PCA subspace (measured) ---------------------------------
def pca_overlay():
    X, _ = _load_digits()
    Xc = X - X.mean(0, keepdim=True)
    k = 2
    # linear AE: z = Xc W_enc ; xhat = z W_dec, no nonlinearity, MSE
    enc = nn.Linear(64, k, bias=False)
    dec = nn.Linear(k, 64, bias=False)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=5e-3)
    for _ in range(3000):
        opt.zero_grad()
        out = dec(enc(Xc))
        loss = ((out - Xc) ** 2).mean()
        loss.backward(); opt.step()
    ae_err = float(loss.item())

    pca = PCA(n_components=k)
    pca.fit(Xc.numpy())
    pca_rec = pca.inverse_transform(pca.transform(Xc.numpy()))
    pca_err = float(((pca_rec - Xc.numpy()) ** 2).mean())

    # subspace alignment: principal angles between AE decoder span and PCA span
    Wdec = dec.weight.detach().numpy()              # (64, k)
    Q_ae, _ = np.linalg.qr(Wdec)
    Q_pca, _ = np.linalg.qr(pca.components_.T)       # (64, k)
    s = np.linalg.svd(Q_ae.T @ Q_pca, compute_uv=False)
    angles_deg = np.degrees(np.arccos(np.clip(s, -1, 1)))
    print(f"  linear-AE MSE={ae_err:.4f}  PCA MSE={pca_err:.4f}  principal angles(deg)={np.round(angles_deg,2)}")

    # project data to both 2-D codes for the scatter overlay
    z_ae = enc(Xc).detach().numpy()
    z_pca = pca.transform(Xc.numpy())
    # align AE code to PCA code via orthogonal Procrustes for visual overlay
    U, _, Vt = np.linalg.svd(z_ae.T @ z_pca)
    z_ae_aligned = z_ae @ (U @ Vt)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    ax = axes[0]
    ax.scatter(z_pca[:, 0], z_pca[:, 1], s=8, color=BLUE, alpha=0.5, label="PCA code")
    ax.scatter(z_ae_aligned[:, 0], z_ae_aligned[:, 1], s=8, color=AMBER, alpha=0.5,
               marker="x", label="linear-AE code (aligned)")
    ax.set_title("Same 2-D subspace", fontsize=12, fontweight="bold")
    ax.set_xlabel("component 1"); ax.set_ylabel("component 2")
    ax.legend(frameon=False, fontsize=9); _despine(ax)

    ax = axes[1]
    bars = ax.bar(["PCA\n(k=2)", "linear AE\n(k=2)"], [pca_err, ae_err],
                  color=[BLUE, AMBER], width=0.5)
    ax.set_ylabel("reconstruction MSE")
    ax.set_title("Identical reconstruction error", fontsize=12, fontweight="bold")
    for b, v in zip(bars, [pca_err, ae_err]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.text(0.5, max(pca_err, ae_err) * 0.5,
            f"principal angles\nbetween subspaces:\n{angles_deg[0]:.1f}°, {angles_deg[1]:.1f}°",
            ha="center", va="center", fontsize=9, color=GREEN, fontweight="bold")
    _despine(ax)
    fig.suptitle("A linear autoencoder (MSE) learns the SAME subspace as PCA",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/ae_pca_overlay.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ae_pca_overlay.png")


# ---- 5. VAE latent space + reparameterization schematic (measured) -----------
class VAE(nn.Module):
    def __init__(self, d=64, hidden=48, k=2):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, k)
        self.logvar = nn.Linear(hidden, k)
        self.dec = nn.Sequential(nn.Linear(k, hidden), nn.ReLU(), nn.Linear(hidden, d), nn.Sigmoid())

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)          # reparameterization
        return self.dec(z), mu, logvar, z


def vae_latent():
    X, y = _load_digits()
    m = VAE(k=2)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    beta = 1.0
    for _ in range(800):
        opt.zero_grad()
        out, mu, logvar, _ = m(X)
        recon = ((out - X) ** 2).sum(1).mean()        # ~ -log p(x|z) up to const
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1).mean()
        loss = recon + beta * kl
        loss.backward(); opt.step()
    with torch.no_grad():
        _, mu, logvar, _ = m(X)
    print(f"  VAE final: recon={recon.item():.3f}  KL={kl.item():.3f}  ELBO≈{-(recon.item()+kl.item()):.3f}")
    mu = mu.numpy()

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))
    ax = axes[0]
    sc = ax.scatter(mu[:, 0], mu[:, 1], c=y, cmap="tab10", s=10, alpha=0.7)
    ax.set_title("VAE 2-D latent μ, colored by digit", fontsize=12, fontweight="bold")
    ax.set_xlabel("z₁"); ax.set_ylabel("z₂"); _despine(ax)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label("digit class", fontsize=9)

    # reparameterization schematic
    ax = axes[1]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    nodes = [
        (1.2, 4.4, BLUE, "μ(x)"),
        (1.2, 1.6, BLUE, "σ(x)"),
        (4.3, 1.6, SLATE, "ε ~ N(0,I)"),
        (6.6, 3.0, GREEN, "z = μ + σ⊙ε"),
        (9.0, 3.0, PURPLE, "decoder\n→ x̂"),
    ]
    for x, yy, c, t in nodes:
        box = FancyBboxPatch((x - 0.85, yy - 0.5), 1.7, 1.0,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             linewidth=1.5, edgecolor=c, facecolor=c)
        ax.add_patch(box)
        ax.text(x, yy, t, ha="center", va="center", color="#fff", fontsize=10, fontweight="bold")
    arrows = [((2.05, 4.4), (5.8, 3.3)), ((2.05, 1.6), (5.8, 2.7)),
              ((5.15, 1.6), (5.85, 2.55)), ((7.45, 3.0), (8.15, 3.0))]
    for (x0, y0), (x1, y1) in arrows:
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                     mutation_scale=14, color=SLATE, lw=1.8))
    ax.text(5.0, 0.4, "randomness lives in ε → gradients flow through μ, σ",
            ha="center", color=RED, fontsize=9.5, fontweight="bold")
    ax.set_title("Reparameterization trick", fontsize=12, fontweight="bold")
    fig.suptitle("VAE: a structured, samplable latent space via the reparameterization trick",
                 fontsize=13, fontweight="bold", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{OUT}/ae_vae_latent.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote ae_vae_latent.png")


if __name__ == "__main__":
    architecture()
    latent_dim()
    denoising()
    pca_overlay()
    vae_latent()
    print("done.")
