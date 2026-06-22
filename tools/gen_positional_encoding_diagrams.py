"""Positional-Encoding concept-page diagrams (muted palette, parallel matplotlib scale).

Four MEASURED visuals for 05. Deep_Learning/concepts/17-Positional-Encoding.md:
  1. pe_sinusoidal_heatmap.png -- the sinusoidal PE matrix as a heatmap over
     (position x dimension): the wavelength bands. MEASURED from the formula.
  2. pe_sinusoid_curves.png    -- a few PE dimensions plotted vs position:
     low dims oscillate fast, high dims slowly (geometric wavelengths). MEASURED.
  3. pe_rope_relative.png      -- RoPE: the attention dot-product q_m . k_n as a
     function of offset (m - n), showing it depends only on relative distance.
     MEASURED from a real RoPE rotation.
  4. pe_alibi_bias.png         -- ALiBi: the linear distance penalty -m*|i-j|
     added to attention scores, per head slope. MEASURED/schematic.
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


def sinusoidal_pe(n_pos, d_model, base=10000.0):
    """The Vaswani et al. 2017 sinusoidal positional encoding matrix (n_pos x d_model)."""
    pos = np.arange(n_pos)[:, None]                 # (n_pos, 1)
    i = np.arange(d_model)[None, :]                 # (1, d_model)
    # angle = pos / base^(2*floor(i/2)/d_model); even dims -> sin, odd dims -> cos
    div = base ** (2 * (i // 2) / d_model)
    angle = pos / div
    pe = np.zeros((n_pos, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


# ---- 1. Sinusoidal PE matrix heatmap ---------------------------------------
def sinusoidal_heatmap():
    n_pos, d_model = 100, 128
    pe = sinusoidal_pe(n_pos, d_model)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    im = ax.imshow(pe, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   origin="lower", interpolation="nearest")
    ax.set_xlabel("Embedding dimension  i  (0 … d-1)")
    ax.set_ylabel("Token position  pos")
    ax.set_title("Sinusoidal positional encoding  PE[pos, i]  (d = 128)",
                 fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("encoding value", fontsize=10)
    bbox = dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.82)
    ax.annotate("low dims: short\nwavelength\n(fast stripes)", xy=(16, 50), xytext=(34, 88),
                color=AMBER, fontsize=9.5, fontweight="bold", ha="center", bbox=bbox,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.6))
    ax.annotate("high dims: long\nwavelength\n(slow bands)", xy=(108, 50), xytext=(96, 88),
                color=NAVY, fontsize=9.5, fontweight="bold", ha="center", bbox=bbox,
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.6))
    fig.tight_layout()
    fig.savefig(f"{OUT}/pe_sinusoidal_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote pe_sinusoidal_heatmap.png")


# ---- 2. Individual sinusoid curves vs position ------------------------------
def sinusoid_curves():
    n_pos, d_model = 64, 128
    pe = sinusoidal_pe(n_pos, d_model)
    pos = np.arange(n_pos)
    dims = [0, 4, 20, 60]                       # increasing dimension => longer wavelength
    colors = [RED, AMBER, GREEN, BLUE]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for d, c in zip(dims, colors):
        wl = 2 * np.pi * (10000.0 ** (2 * (d // 2) / d_model))
        ax.plot(pos, pe[:, d], color=c, lw=2.4,
                label=f"dim i={d}  (wavelength ≈ {wl:,.0f})")
    ax.axhline(0, color=SLATE, lw=0.8, alpha=0.5)
    ax.set_xlabel("Token position  pos")
    ax.set_ylabel("PE[pos, i]")
    ax.set_title("Each dimension is a sinusoid of geometrically increasing wavelength",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=9.0)
    ax.set_xlim(0, n_pos - 1)
    ax.set_ylim(-1.25, 1.25)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pe_sinusoid_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote pe_sinusoid_curves.png")


# ---- 3. RoPE: dot product depends only on relative offset -------------------
def rope_relative():
    """Apply a real RoPE rotation to a fixed q and k across positions and plot the
    attention score q_m . k_n as a function of the relative offset (m - n)."""
    d = 64                                          # rotation operates on d/2 pairs
    rng = np.random.default_rng(0)
    q = rng.standard_normal(d)
    k = rng.standard_normal(d)
    # RoPE frequencies: theta_j = base^(-2j/d) for j = 0..d/2-1
    j = np.arange(d // 2)
    theta = 10000.0 ** (-2 * j / d)

    def rope(x, pos):
        x1, x2 = x[0::2], x[1::2]                    # interleaved pair layout
        ang = pos * theta
        c, s = np.cos(ang), np.sin(ang)
        out = np.empty_like(x)
        out[0::2] = x1 * c - x2 * s
        out[1::2] = x1 * s + x2 * c
        return out

    # Fix n at a reference position; vary m so offset = m - n spans a range.
    n = 64
    offsets = np.arange(-48, 49)
    scores = np.array([rope(q, n + off) @ rope(k, n) for off in offsets])
    # Show shift-invariance: redo with a different absolute n, same offsets -> identical curve
    n2 = 200
    scores2 = np.array([rope(q, n2 + off) @ rope(k, n2) for off in offsets])

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(offsets, scores, color=PURPLE, lw=2.6, label="anchor n = 64")
    ax.plot(offsets, scores2, color=GREEN, lw=1.6, ls="--",
            label="anchor n = 200 (identical → relative only)")
    ax.axvline(0, color=SLATE, lw=0.8, alpha=0.6)
    ax.set_xlabel("Relative offset  (m − n)")
    ax.set_ylabel("RoPE attention score  ⟨R_m q, R_n k⟩")
    ax.set_title("RoPE: the q·k score depends ONLY on the relative offset m − n",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax.set_xlim(-48, 48)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pe_rope_relative.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote pe_rope_relative.png  (max|score(n=64)-score(n=200)| =",
          f"{np.abs(scores - scores2).max():.2e})")


# ---- 4. ALiBi linear distance bias -----------------------------------------
def alibi_bias():
    """ALiBi adds -m * (distance) to each attention score; m is a per-head slope
    taken from a geometric sequence. Plot the bias vs distance for several heads,
    and (inset) a query row of the bias matrix."""
    dist = np.arange(0, 40)
    # 8-head slopes: geometric, ratio 1/2 starting at 2^-1 (the ALiBi recipe for n=8)
    slopes = [2 ** (-(i + 1)) for i in range(4)]        # show 4 representative heads
    colors = [RED, AMBER, GREEN, BLUE]
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for m, c in zip(slopes, colors):
        ax.plot(dist, -m * dist, color=c, lw=2.4, label=f"head slope m = {m:g}")
    ax.set_xlabel("Distance between query i and key j   |i − j|")
    ax.set_ylabel("ALiBi bias added to score   −m·|i − j|")
    ax.set_title("ALiBi: a linear distance penalty added straight to attention scores",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    ax.set_xlim(0, 39)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/pe_alibi_bias.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote pe_alibi_bias.png")


if __name__ == "__main__":
    sinusoidal_heatmap()
    sinusoid_curves()
    rope_relative()
    alibi_bias()
    print("all positional-encoding diagrams written to", OUT)
