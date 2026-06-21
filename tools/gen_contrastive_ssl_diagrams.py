"""Contrastive / Self-Supervised Learning concept-page diagrams.

Muted palette, color:#fff on fills (parallel matplotlib scale to the rest of the
platform). Four visuals for
  04. Unsupervised_Learning/concepts/12-Contrastive-Self-Supervised-Learning.md:

  1. cssl_simclr_pipeline.png   -- SCHEMATIC: one image -> two augmentations ->
     shared encoder f -> projection head g -> NT-Xent pull(positive)/push(negatives).
  2. cssl_temperature.png       -- MEASURED: how temperature tau reshapes the
     NT-Xent softmax over one positive + several negatives, and the resulting
     loss + gradient-on-the-hardest-negative as tau varies.
  3. cssl_align_uniform.png     -- MEASURED: alignment (positives close) and
     uniformity (negatives spread on the hypersphere) on a 2-D unit circle, for
     a collapsed embedding vs a well-trained one, with the two metric values.
  4. cssl_methods.png           -- SCHEMATIC table-figure: SimCLR / MoCo / BYOL /
     SimSiam / Barlow Twins -- negatives? momentum? predictor? stop-grad?
     PLUS a MEASURED toy NT-Xent loss curve descending over training steps.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "04. Unsupervised_Learning",
                   "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _box(ax, x, y, w, h, text, fc, fs=10, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.04",
                                facecolor=fc, edgecolor="white", linewidth=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold")


def _arrow(ax, x0, y0, x1, y1, color="#444", style="->", lw=1.8, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=14, color=color, lw=lw, linestyle=ls))


# ---- 1. SimCLR pipeline schematic -------------------------------------------
def simclr_pipeline():
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.set_xlim(0, 12.4); ax.set_ylim(0, 7.2); ax.axis("off")

    # source image
    _box(ax, 0.2, 2.9, 1.5, 1.2, "image\nx", SLATE, fs=11)
    # two augmentations
    _box(ax, 2.4, 4.5, 1.7, 1.0, "augment t\n(crop+jitter)", AMBER, fs=9)
    _box(ax, 2.4, 1.4, 1.7, 1.0, "augment t'\n(crop+blur)", AMBER, fs=9)
    _arrow(ax, 1.7, 3.7, 2.4, 5.0, NAVY)
    _arrow(ax, 1.7, 3.4, 2.4, 1.9, NAVY)
    ax.text(1.55, 4.95, "two views", fontsize=8.5, color=NAVY, fontweight="bold")

    # views
    _box(ax, 4.5, 4.5, 1.0, 1.0, "x̃ᵢ", BLUE, fs=12)
    _box(ax, 4.5, 1.4, 1.0, 1.0, "x̃ⱼ", BLUE, fs=12)
    _arrow(ax, 4.1, 5.0, 4.5, 5.0, NAVY)
    _arrow(ax, 4.1, 1.9, 4.5, 1.9, NAVY)

    # encoder (shared)
    _box(ax, 5.9, 4.4, 1.5, 1.2, "encoder f\n(ResNet)", PURPLE, fs=9)
    _box(ax, 5.9, 1.3, 1.5, 1.2, "encoder f\n(shared θ)", PURPLE, fs=9)
    _arrow(ax, 5.5, 5.0, 5.9, 5.0, NAVY)
    _arrow(ax, 5.5, 1.9, 5.9, 1.9, NAVY)
    # representations h (kept for downstream)
    ax.text(6.65, 3.9, "h = f(x̃)\nkeep this\nfor downstream", fontsize=7.6,
            color=PURPLE, fontweight="bold", ha="center")

    # projection head
    _box(ax, 7.7, 4.4, 1.5, 1.2, "proj head g\n(MLP)", GREEN, fs=9)
    _box(ax, 7.7, 1.3, 1.5, 1.2, "proj head g\n(discard@test)", GREEN, fs=8.2)
    _arrow(ax, 7.4, 5.0, 7.7, 5.0, NAVY)
    _arrow(ax, 7.4, 1.9, 7.7, 1.9, NAVY)

    # embeddings z on the hypersphere
    _box(ax, 9.5, 4.5, 1.0, 1.0, "zᵢ", BLUE, fs=12)
    _box(ax, 9.5, 1.4, 1.0, 1.0, "zⱼ", BLUE, fs=12)
    _arrow(ax, 9.2, 5.0, 9.5, 5.0, NAVY)
    _arrow(ax, 9.2, 1.9, 9.5, 1.9, NAVY)

    # NT-Xent: pull positives together, push negatives apart
    _arrow(ax, 10.0, 4.5, 10.0, 2.4, GREEN, style="<->", lw=2.6)
    ax.text(10.15, 3.45, "PULL\n(positive\npair)", fontsize=8.6, color=GREEN,
            fontweight="bold", va="center")
    ax.text(10.0, 6.5, "NT-Xent / InfoNCE loss", fontsize=10.5, color="#222",
            fontweight="bold", ha="center")
    # push to negatives (other batch items, schematic)
    for yy in (5.2, 0.5):
        ax.add_patch(Rectangle((11.4, yy), 0.55, 0.55, facecolor=RED,
                     edgecolor="white"))
        ax.text(11.67, yy + 0.27, "z₋", ha="center", va="center", color="white",
                fontsize=9, fontweight="bold")
    _arrow(ax, 10.5, 4.7, 11.4, 5.4, RED, style="->", lw=1.6, ls="--")
    _arrow(ax, 10.5, 1.7, 11.4, 0.75, RED, style="->", lw=1.6, ls="--")
    ax.text(11.5, 3.3, "PUSH\nnegatives\napart", fontsize=8.0, color=RED,
            fontweight="bold", ha="center")

    ax.set_title("SimCLR: two augmented views → shared encoder → projection head → NT-Xent",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{OUT}/cssl_simclr_pipeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cssl_simclr_pipeline.png")


# ---- 2. Temperature effect (MEASURED) ---------------------------------------
def temperature_effect():
    # One anchor z_i. Cosine sims to: its positive (0.8) and four negatives.
    pos = 0.80
    negs = np.array([0.65, 0.40, 0.10, -0.30])   # one hard negative (0.65)
    sims = np.concatenate([[pos], negs])
    labels = ["positive\n(0.80)", "hard neg\n(0.65)", "neg\n(0.40)",
              "neg\n(0.10)", "neg\n(-0.30)"]
    taus = [0.5, 0.2, 0.07]
    colors = [SLATE, BLUE, RED]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))

    # (a) softmax distribution under each tau
    ax = axes[0]
    x = np.arange(len(sims)); w = 0.26
    for k, (tau, c) in enumerate(zip(taus, colors)):
        p = np.exp(sims / tau); p = p / p.sum()
        ax.bar(x + (k - 1) * w, p, w, color=c, edgecolor="white",
               label=f"τ = {tau}")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.6)
    ax.set_ylabel("softmax weight  p (mass on each key)")
    ax.set_title("Temperature sharpens the contrastive softmax", fontsize=12.5,
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5)
    ax.annotate("small τ → almost all mass on the\npositive + the hard negative\n(hard-negative mining)",
                (1, 0.30), xytext=(1.6, 0.55), fontsize=8.6, color=RED,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=RED))
    _despine(ax)

    # (b) loss and gradient magnitude on the hardest negative vs tau
    ax = axes[1]
    tau_grid = np.linspace(0.02, 0.8, 200)
    loss = []
    grad_hard = []
    for tau in tau_grid:
        logits = sims / tau
        m = logits.max()
        p = np.exp(logits - m); p = p / p.sum()
        loss.append(-np.log(p[0]))            # -log softmax of the positive
        # d(loss)/d(sim_hardneg) for InfoNCE = p_hardneg / tau
        grad_hard.append(p[1] / tau)
    ax.plot(tau_grid, loss, color=PURPLE, lw=2.6, label="NT-Xent loss  ℓ")
    ax.set_xlabel("temperature τ"); ax.set_ylabel("loss ℓ", color=PURPLE)
    ax.tick_params(axis="y", labelcolor=PURPLE)
    ax2 = ax.twinx()
    ax2.plot(tau_grid, grad_hard, color=RED, lw=2.6, ls="--",
             label="|∂ℓ/∂ sim| on hard neg")
    ax2.set_ylabel("gradient on hard negative", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    ax.axvline(0.07, color=SLATE, ls=":", lw=1.4)
    ax.text(0.085, max(loss) * 0.82, "τ=0.07\n(SimCLR)", fontsize=8.6,
            color=SLATE, fontweight="bold")
    ax.set_title("Small τ → larger gradient on hard negatives", fontsize=12.5,
                 fontweight="bold")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cssl_temperature.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cssl_temperature.png")


# ---- 3. Alignment & Uniformity on the hypersphere (MEASURED) ----------------
def _align_loss(z_a, z_b):
    # E || f(x) - f(x+) ||^2  over positive pairs (rows aligned)
    return float(np.mean(np.sum((z_a - z_b) ** 2, axis=1)))


def _uniform_loss(z, t=2.0):
    # log E exp(-t || zi - zj ||^2)  over all pairs (Wang & Isola)
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=-1)
    iu = np.triu_indices(len(z), k=1)
    return float(np.log(np.mean(np.exp(-t * d2[iu]))))


def align_uniform():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))

    def draw(ax, angles_anchor, angles_pos, title, c_anchor):
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color=SLATE, lw=1.0, alpha=0.5)
        za = np.c_[np.cos(angles_anchor), np.sin(angles_anchor)]
        zp = np.c_[np.cos(angles_pos), np.sin(angles_pos)]
        # positive pair links
        for a, p in zip(za, zp):
            ax.plot([a[0], p[0]], [a[1], p[1]], color=GREEN, lw=1.4, alpha=0.7)
        ax.scatter(za[:, 0], za[:, 1], color=c_anchor, s=70, edgecolor="white",
                   zorder=5, label="view 1")
        ax.scatter(zp[:, 0], zp[:, 1], color=AMBER, s=70, edgecolor="white",
                   zorder=5, label="view 2 (positive)")
        z_all = np.vstack([za, zp])
        al = _align_loss(za, zp); un = _uniform_loss(z_all)
        ax.set_title(f"{title}\nalign={al:.3f}   uniform={un:.2f}",
                     fontsize=11.5, fontweight="bold")
        ax.set_aspect("equal"); ax.axis("off")
        ax.legend(loc="lower center", frameon=False, fontsize=8.6,
                  ncol=2, bbox_to_anchor=(0.5, -0.12))

    # Collapsed: everything bunched near one point -> tiny align, BAD uniform (high)
    base = 0.5
    anchor_c = base + rng.normal(0, 0.12, 8)
    pos_c = anchor_c + rng.normal(0, 0.05, 8)
    draw(axes[0], anchor_c, pos_c, "Collapsed embedding\n(all points bunched)", RED)

    # Good: positives close, anchors spread uniformly around the circle
    anchor_g = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    pos_g = anchor_g + rng.normal(0, 0.10, 8)
    draw(axes[1], anchor_g, pos_g, "Well-trained embedding\n(aligned + uniform)", BLUE)

    fig.suptitle("Alignment + Uniformity (Wang & Isola): positives close, negatives spread",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cssl_align_uniform.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cssl_align_uniform.png")


# ---- 4. Methods comparison + MEASURED toy NT-Xent loss curve ----------------
def _ntxent_numpy(z, tau=0.2):
    # z: (2N, d) where rows 2k, 2k+1 are a positive pair (SimCLR layout).
    z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)
    S = z @ z.T / tau
    np.fill_diagonal(S, -1e9)
    n2 = z.shape[0]
    # positive index for row i: its partner
    pos = np.arange(n2) ^ 1
    logZ = np.log(np.exp(S).sum(axis=1) + 1e-12)
    loss = -(S[np.arange(n2), pos] - logZ)
    return float(loss.mean())


def methods_and_curve():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0),
                             gridspec_kw={"width_ratios": [1.35, 1.0]})

    # (a) comparison table-figure
    ax = axes[0]; ax.axis("off")
    rows = ["SimCLR", "MoCo", "BYOL", "SimSiam", "Barlow Twins"]
    cols = ["negatives?", "momentum\nencoder?", "predictor?", "stop-grad?",
            "anti-collapse\nmechanism"]
    data = [
        ["yes\n(in-batch)", "no", "no", "no", "negatives"],
        ["yes\n(queue)", "yes (EMA)", "no", "no", "negatives"],
        ["no", "yes (EMA)", "yes", "yes", "predictor+EMA+SG"],
        ["no", "no", "yes", "yes", "stop-gradient"],
        ["no", "no", "no", "no", "decorrelation"],
    ]
    ccol = [GREEN, NAVY, PURPLE, AMBER, RED]
    nr, nc = len(rows), len(cols)
    cw, ch = 1.0 / (nc + 1), 1.0 / (nr + 1)
    for j, c in enumerate(cols):
        ax.text((j + 1.5) * cw, 1 - 0.5 * ch, c, ha="center", va="center",
                fontsize=8.4, fontweight="bold")
    for i, r in enumerate(rows):
        y = 1 - (i + 1.5) * ch
        ax.add_patch(Rectangle((0.02, y - 0.5 * ch), cw - 0.02, ch * 0.92,
                     facecolor=ccol[i], edgecolor="white"))
        ax.text(0.02 + (cw - 0.02) / 2, y, r, ha="center", va="center",
                color="white", fontsize=8.8, fontweight="bold")
        for j in range(nc):
            txt = data[i][j]
            face = "#eef1f4" if (i % 2 == 0) else "#e2e7ec"
            ax.add_patch(Rectangle(((j + 1) * cw, y - 0.5 * ch), cw, ch * 0.92,
                         facecolor=face, edgecolor="white"))
            ax.text((j + 1.5) * cw, y, txt, ha="center", va="center",
                    fontsize=7.6, color="#222")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Contrastive vs non-contrastive SSL methods",
                 fontsize=12.5, fontweight="bold")

    # (b) MEASURED toy NT-Xent loss curve: optimize 6 embeddings to lower NT-Xent
    ax = axes[1]
    rng = np.random.default_rng(1)
    N = 6                      # 6 instances -> 12 views
    z = rng.normal(0, 1.0, (2 * N, 8))
    lr, steps = 0.5, 120
    losses = []
    for _ in range(steps):
        # numerical gradient via the analytic NT-Xent on normalized z (autograd-free)
        zc = z.copy()
        eps = 1e-4
        base = _ntxent_numpy(zc)
        losses.append(base)
        g = np.zeros_like(zc)
        for a in range(zc.shape[0]):       # cheap finite-diff on a tiny problem
            for b in range(zc.shape[1]):
                zc[a, b] += eps
                g[a, b] = (_ntxent_numpy(zc) - base) / eps
                zc[a, b] -= eps
        z = z - lr * g
    ax.plot(range(steps), losses, color=PURPLE, lw=2.6)
    ax.scatter([0], [losses[0]], color=RED, s=55, zorder=5, edgecolor="white")
    ax.scatter([steps - 1], [losses[-1]], color=GREEN, s=55, zorder=5,
               edgecolor="white")
    ax.annotate(f"start {losses[0]:.2f}", (0, losses[0]),
                xytext=(14, -2), textcoords="offset points", fontsize=9,
                color=RED, fontweight="bold")
    ax.annotate(f"end {losses[-1]:.2f}\n(positives pulled together)",
                (steps - 1, losses[-1]), xytext=(-130, 30),
                textcoords="offset points", fontsize=8.6, color=GREEN,
                fontweight="bold", arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.set_xlabel("optimization step"); ax.set_ylabel("NT-Xent loss")
    ax.set_title("Measured: NT-Xent descends as views align",
                 fontsize=12.0, fontweight="bold")
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/cssl_methods.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cssl_methods.png")
    print(f"  toy NT-Xent: {losses[0]:.3f} -> {losses[-1]:.3f}")


if __name__ == "__main__":
    simclr_pipeline()
    temperature_effect()
    align_uniform()
    methods_and_curve()
    print("OUT:", OUT)
