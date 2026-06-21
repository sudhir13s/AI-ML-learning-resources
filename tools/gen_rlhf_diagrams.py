"""RLHF & DPO concept-page diagrams (muted palette, parallel matplotlib scale).

Two figures for 09. LLMs/concepts/15-RLHF-and-DPO.md:
  1. rlhf_overoptimization.png -- reward over-optimization (Goodhart): as the policy
     drifts (KL) from the reference, the proxy reward keeps rising but the true/gold
     reward peaks and falls. Why RLHF needs a KL leash.
  2. dpo_margin.png -- DPO directly optimizes the preference: as the implicit-reward
     margin grows, the DPO loss falls and P(chosen > rejected) = sigmoid(margin) rises.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "09. LLMs", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def overoptimization():
    kl = np.linspace(0, 10, 200)
    proxy = 3.4 * (1 - np.exp(-0.45 * kl))                 # RM score: keeps rising, saturating
    gold = 3.0 * (1 - np.exp(-0.6 * kl)) - 0.16 * kl       # true quality: rises then falls
    peak = int(np.argmax(gold))
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.plot(kl, proxy, color=RED, lw=2.6, label="Proxy reward (what the reward model says)")
    ax.plot(kl, gold, color=GREEN, lw=2.6, label="True quality (actual human preference)")
    ax.axvline(kl[peak], color=SLATE, ls="--", lw=1.6)
    ax.scatter([kl[peak]], [gold[peak]], color=AMBER, s=80, zorder=5, edgecolor="white")
    ax.annotate("sweet spot:\nstop here (KL leash)", (kl[peak], gold[peak]),
                textcoords="offset points", xytext=(8, -46), fontsize=9.5, fontweight="bold", color="#222",
                arrowprops=dict(arrowstyle="->", color=SLATE))
    ax.annotate("reward hacking:\nproxy ↑ but quality ↓", (8.2, proxy[160]),
                textcoords="offset points", xytext=(-150, -8), fontsize=9.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.fill_between(kl[peak:], gold[peak:], proxy[peak:], color=RED, alpha=0.10)
    ax.set_xlabel("KL divergence from the reference model  (how far the policy drifts)")
    ax.set_ylabel("reward")
    ax.set_title("Reward over-optimization: why RLHF needs a KL leash", fontsize=13.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right"); _despine(ax); ax.set_ylim(0, 3.6)
    fig.tight_layout(); fig.savefig(f"{OUT}/rlhf_overoptimization.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote rlhf_overoptimization.png")


def dpo_margin():
    m = np.linspace(-4, 4, 200)
    loss = np.log1p(np.exp(-m))           # -log sigmoid(m), numerically safe
    win = 1 / (1 + np.exp(-m))            # sigmoid(m) = P(chosen > rejected)
    fig, ax1 = plt.subplots(figsize=(8.6, 5.0))
    ax1.plot(m, loss, color=PURPLE, lw=2.8, label="DPO loss = −log σ(margin)")
    ax1.set_xlabel("implicit-reward margin  β·[ (logπ/π_ref)$_{chosen}$ − (logπ/π_ref)$_{rejected}$ ]")
    ax1.set_ylabel("DPO loss", color=PURPLE)
    ax1.tick_params(axis="y", labelcolor=PURPLE); _despine(ax1)
    ax2 = ax1.twinx()
    ax2.plot(m, win, color=GREEN, lw=2.8, ls="--", label="P(chosen ≻ rejected) = σ(margin)")
    ax2.set_ylabel("P(chosen ≻ rejected)", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    for s in ("top",): ax2.spines[s].set_visible(False)
    ax1.axvline(0, color=SLATE, ls=":", lw=1.2)
    ax1.text(0.1, 2.3, "init: policy = reference\nmargin 0, loss = log 2", fontsize=9, color=SLATE)
    ax1.set_title("DPO directly raises the preferred response's probability", fontsize=13.5, fontweight="bold")
    l1, lab1 = ax1.get_legend_handles_labels(); l2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lab1 + lab2, frameon=False, fontsize=9.5, loc="upper center")
    fig.tight_layout(); fig.savefig(f"{OUT}/dpo_margin.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote dpo_margin.png")


if __name__ == "__main__":
    overoptimization()
    dpo_margin()
    print("OUT:", OUT)
