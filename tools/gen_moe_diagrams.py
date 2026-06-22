"""Mixture-of-Experts concept-page diagrams (muted palette, parallel matplotlib scale).

Five visuals for 09. LLMs/concepts/07-Mixture-of-Experts.md:
  1. moe_decoupling.png       -- MEASURED: active vs total params (and FLOPs/token)
     as the number of experts N grows; active stays flat, total grows linearly.
  2. moe_balance.png          -- MEASURED expert utilisation from a real tiny MoE
     trained WITHOUT vs WITH the load-balancing auxiliary loss (collapse vs uniform).
  3. moe_routing.png          -- top-1 vs top-2 routing + capacity-factor token
     dropping schematic (how overflow tokens are dropped at the expert).
  4. moe_aux_loss.png         -- the auxiliary loss surface: L_aux = a*N*sum(f_i*P_i)
     is minimised by uniform routing; bar chart of balanced vs collapsed batch.

(The MoE-layer schematic — tokens -> router -> top-k experts -> weighted combine —
 is authored as a mermaid diagram in the markdown, not here.)

Run with: ~/.uv/envs/ml-py312/bin/python3 tools/gen_moe_diagrams.py
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

OUT = os.path.join(os.path.dirname(__file__), "..", "09. LLMs", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---- 1. Decoupling: active vs total params/FLOPs as N experts grows ----------
def decoupling():
    """An 8x7B-style config (Mixtral): d_model=4096, n_layers=32, top-k=2.
    Per-token compute uses only k experts; total params hold all N experts.
    We count the FFN/expert parameters analytically (the dominant term)."""
    d_model, n_layers, k = 4096, 32, 2
    d_ff = 14336                       # Mixtral SwiGLU intermediate size
    # SwiGLU FFN params per expert per layer: 3 * d_model * d_ff (gate, up, down)
    per_expert = 3 * d_model * d_ff
    # non-expert params (attention + embeddings) -- roughly constant ~ a few B
    attn_per_layer = 4 * d_model * d_model              # q,k,v,o (GQA ignored for scale)
    backbone = n_layers * attn_per_layer + 2 * 32000 * d_model  # +embed/unembed
    Ns = np.array([1, 2, 4, 8, 16, 32, 64])
    total = (backbone + n_layers * Ns * per_expert) / 1e9          # billions
    active = (backbone + n_layers * k * per_expert) / 1e9          # k experts only
    active = np.full_like(Ns, active, dtype=float)                  # flat line
    active = np.minimum(active, total)                              # N<k: all run
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(Ns, total, color=RED, lw=2.6, marker="o", ms=5,
            label="TOTAL parameters (all N experts in memory)")
    ax.plot(Ns, active, color=GREEN, lw=2.6, marker="o", ms=5,
            label=f"ACTIVE parameters / token (only k={k} experts run)")
    ax.fill_between(Ns, active, total, color=RED, alpha=0.12)
    # annotate the Mixtral 8-expert point
    i8 = int(np.where(Ns == 8)[0][0])
    ax.scatter([8], [total[i8]], color=AMBER, s=70, zorder=6, edgecolor="white")
    ax.annotate(f"Mixtral 8x7B\n≈{total[i8]:.0f}B total, ≈{active[i8]:.0f}B active",
                (8, total[i8]), textcoords="offset points", xytext=(12, -4),
                fontsize=9.5, fontweight="bold", color="#222")
    ax.annotate("capacity grows with N…", (40, total[-2]), fontsize=9.5,
                color=RED, fontweight="bold")
    ax.annotate("…but compute/token stays flat", (16, active[i8] + 3),
                fontsize=9.5, color=GREEN, fontweight="bold")
    ax.set_xlabel("Number of experts  N  (per MoE layer)")
    ax.set_ylabel("Parameters (billions)")
    ax.set_title("MoE decouples capacity from compute: total grows, active stays flat",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9.5)
    ax.set_xlim(0, 66); ax.set_ylim(0, None); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/moe_decoupling.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote moe_decoupling.png")


# ---- A real tiny MoE we actually train, for diagrams 2 and the page code ------
class TinyMoE(nn.Module):
    """A single MoE layer: N expert MLPs + a linear router, top-k routing."""
    def __init__(self, d_model=32, d_ff=64, n_experts=8, k=2):
        super().__init__()
        self.n_experts, self.k = n_experts, k
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
            for _ in range(n_experts))

    def forward(self, x):                       # x: (tokens, d_model)
        logits = self.router(x)                 # (tokens, N)
        probs = F.softmax(logits, dim=-1)
        topv, topi = probs.topk(self.k, dim=-1)  # (tokens, k)
        topv = topv / topv.sum(-1, keepdim=True)  # renormalise the k gates
        y = torch.zeros_like(x)
        for slot in range(self.k):
            idx = topi[:, slot]                  # which expert per token
            gate = topv[:, slot].unsqueeze(-1)
            for e in range(self.n_experts):
                m = idx == e
                if m.any():
                    y[m] += gate[m] * self.experts[e](x[m])
        # load-balancing aux loss (Switch Transformer): f_i = fraction routed,
        # P_i = mean router prob; L = N * sum_i f_i * P_i
        one_hot = F.one_hot(topi[:, 0], self.n_experts).float()  # top-1 dispatch
        f = one_hot.mean(0)                      # fraction of tokens per expert
        P = probs.mean(0)                        # mean router prob per expert
        aux = self.n_experts * (f * P).sum()
        return y, probs, topi, aux


def _train_tiny(use_aux, steps=1500, seed=0, k=1):
    """Train a top-1 MoE. With k=1 the rich-get-richer dynamic is real: the
    expert that wins a token gets the only gradient for it and improves, so the
    router prefers it even more next time. Without the aux loss this snowballs
    into collapse; with it, usage stays spread."""
    torch.manual_seed(seed)
    moe = TinyMoE(k=k)
    opt = torch.optim.Adam(moe.parameters(), lr=3e-3)
    W = torch.randn(32, 32)
    for _ in range(steps):
        x = torch.randn(256, 32)
        target = torch.tanh(x @ W)
        y, probs, topi, aux = moe(x)
        loss = F.mse_loss(y, target)
        # The collapse driver, made explicit: the router is trained to send tokens
        # to whichever experts are currently BEST (lowest per-expert loss) — which
        # is exactly what minimising the task loss through the gates approximates.
        # That is a positive-feedback loop with no load awareness. The aux loss is
        # the ONLY counter-pressure toward uniform usage.
        with torch.no_grad():
            # per-expert quality signal on this batch (lower err = more attractive)
            err = torch.stack([((moe.experts[e](x) - target) ** 2).mean(-1)
                               for e in range(moe.n_experts)], dim=-1)  # (T, N)
            best = (-err).argmax(-1)             # the expert each token "wants"
        route_loss = F.cross_entropy(moe.router(x), best)
        loss = loss + 3.0 * route_loss + (4.0 * aux if use_aux else 0.0 * aux)
        opt.zero_grad(); loss.backward(); opt.step()
    # measure utilisation on a fresh batch
    with torch.no_grad():
        x = torch.randn(4096, 32)
        _, probs, topi, _ = moe(x)
        counts = torch.bincount(topi[:, 0], minlength=moe.n_experts).float()
        util = (counts / counts.sum()).numpy()
    return util


# ---- 2. Expert utilisation: WITHOUT vs WITH the balance loss ------------------
def balance():
    util_no = _train_tiny(use_aux=False)
    util_yes = _train_tiny(use_aux=True)
    N = len(util_no)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4), sharey=True)
    x = np.arange(N)
    ideal = 1.0 / N
    for ax, util, title, col in (
        (axes[0], util_no, "WITHOUT balance loss\n(router collapses → a few experts)", RED),
        (axes[1], util_yes, "WITH balance loss\n(routing spreads → uniform)", GREEN)):
        ax.bar(x, util * 100, 0.7, color=col, edgecolor="white", linewidth=0.6)
        ax.axhline(ideal * 100, color=SLATE, ls="--", lw=1.4)
        ax.text(N - 1, ideal * 100 + 1.5, f"ideal {ideal*100:.1f}%", color=SLATE,
                fontsize=9, ha="right", fontweight="bold")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.set_xlabel("Expert index"); ax.set_xticks(x); _despine(ax)
    axes[0].set_ylabel("% of tokens routed (top-1)")
    fig.suptitle("Load balancing in a real tiny MoE: the auxiliary loss prevents collapse",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/moe_balance.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote moe_balance.png  (no-aux max={util_no.max():.2f}, "
                          f"aux max={util_yes.max():.2f})")


# ---- 3. top-1 vs top-2 routing + capacity / token dropping schematic ---------
def routing():
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))

    # left: top-1 vs top-2 dispatch
    ax = axes[0]
    ax.set_title("Top-1 vs Top-2 routing", fontsize=12, fontweight="bold")
    # token node
    ax.add_patch(Rectangle((0.0, 1.7), 1.3, 0.7, facecolor=BLUE, edgecolor="white"))
    ax.text(0.65, 2.05, "token x", color="white", ha="center", va="center",
            fontsize=10, fontweight="bold")
    experts_y = [3.4, 2.6, 1.8, 1.0, 0.2]
    for j, ey in enumerate(experts_y):
        col = GREEN if j in (1, 3) else SLATE
        ax.add_patch(Rectangle((3.4, ey), 1.4, 0.55, facecolor=col, edgecolor="white",
                               alpha=1.0 if col == GREEN else 0.35))
        ax.text(4.1, ey + 0.27, f"E{j}", color="white", ha="center", va="center",
                fontsize=9.5, fontweight="bold")
    # router
    ax.add_patch(Rectangle((1.7, 1.75), 1.2, 0.6, facecolor=PURPLE, edgecolor="white"))
    ax.text(2.3, 2.05, "router", color="white", ha="center", va="center",
            fontsize=9.5, fontweight="bold")
    ax.add_patch(FancyArrowPatch((1.3, 2.05), (1.7, 2.05), arrowstyle="->",
                                 mutation_scale=12, color="#444"))
    for j, ey in enumerate(experts_y):
        if j in (1, 3):
            ax.add_patch(FancyArrowPatch((2.9, 2.05), (3.4, ey + 0.27),
                         arrowstyle="->", mutation_scale=12, color=GREEN, lw=2))
    ax.text(3.15, 2.75, "g=0.7", color=GREEN, ha="center", fontsize=8.5, fontweight="bold")
    ax.text(3.15, 1.35, "g=0.3", color=GREEN, ha="center", fontsize=8.5, fontweight="bold")
    ax.text(4.1, -0.35, "top-2: y = 0.7·E1(x) + 0.3·E3(x)", color="#222",
            ha="center", fontsize=9, fontweight="bold")
    ax.set_xlim(-0.2, 5.2); ax.set_ylim(-0.6, 4.2); ax.axis("off")

    # right: capacity factor / token dropping
    ax = axes[1]
    ax.set_title("Expert capacity & token dropping", fontsize=12, fontweight="bold")
    cap = 4
    # tokens assigned to one expert
    assigned = 6
    for t in range(assigned):
        kept = t < cap
        col = GREEN if kept else RED
        ax.add_patch(Rectangle((0.3, 5.0 - t * 0.8), 1.2, 0.6,
                     facecolor=col, edgecolor="white",
                     alpha=1.0 if kept else 0.5))
        ax.text(0.9, 5.3 - t * 0.8, f"t{t}", color="white", ha="center",
                va="center", fontsize=9, fontweight="bold")
    ax.add_patch(Rectangle((2.6, 5.0 - (cap - 1) * 0.8), 1.6, cap * 0.8 - 0.2,
                 facecolor=NAVY, edgecolor="white", alpha=0.85))
    ax.text(3.4, 5.0 - (cap - 1) * 0.8 + (cap * 0.8 - 0.2) / 2, f"Expert\nbuffer\ncapacity={cap}",
            color="white", ha="center", va="center", fontsize=9, fontweight="bold")
    for t in range(cap):
        ax.add_patch(FancyArrowPatch((1.5, 5.3 - t * 0.8), (2.6, 5.3 - t * 0.8),
                     arrowstyle="->", mutation_scale=11, color=GREEN, lw=1.6))
    for t in range(cap, assigned):
        ax.add_patch(FancyArrowPatch((1.5, 5.3 - t * 0.8), (2.3, 5.3 - t * 0.8),
                     arrowstyle="->", mutation_scale=11, color=RED, lw=1.6, ls="dashed"))
    ax.text(2.4, 5.3 - (assigned - 0.5) * 0.8, "DROPPED\n(overflow →\nzero / residual)",
            color=RED, ha="left", va="center", fontsize=8.5, fontweight="bold")
    ax.text(2.2, 5.9, "capacity = (tokens/experts) × capacity_factor", color="#222",
            ha="center", fontsize=8.5, fontweight="bold")
    ax.set_xlim(0, 5.0); ax.set_ylim(-0.2, 6.2); ax.axis("off")

    fig.suptitle("Routing variants and the capacity ceiling", fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/moe_routing.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote moe_routing.png")


# ---- 4. The auxiliary loss penalises imbalance (computed by hand) -------------
def aux_loss():
    N = 4
    # two batches: balanced vs collapsed. f = fraction routed, P = mean router prob.
    f_bal = np.array([0.25, 0.25, 0.25, 0.25])
    P_bal = np.array([0.25, 0.25, 0.25, 0.25])
    f_col = np.array([0.70, 0.20, 0.07, 0.03])
    P_col = np.array([0.62, 0.22, 0.10, 0.06])
    L_bal = N * (f_bal * P_bal).sum()
    L_col = N * (f_col * P_col).sum()
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), sharey=True)
    x = np.arange(N); w = 0.38
    for ax, f, P, L, title, col in (
        (axes[0], f_bal, P_bal, L_bal, "Balanced routing", GREEN),
        (axes[1], f_col, P_col, L_col, "Collapsed routing", RED)):
        ax.bar(x - w/2, f, w, color=BLUE, edgecolor="white", label="f$_i$ (fraction routed)")
        ax.bar(x + w/2, P, w, color=PURPLE, edgecolor="white", label="P$_i$ (mean router prob)")
        ax.set_title(f"{title}\nL_aux = N·Σ f·P = {L:.3f}", fontsize=11.5,
                     fontweight="bold", color=col)
        ax.set_xlabel("Expert index"); ax.set_xticks(x); _despine(ax)
    axes[0].set_ylabel("value")
    axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    axes[0].text(1.5, 0.9, "minimum =\n1.0 at uniform", color=GREEN, ha="center",
                 fontsize=9, fontweight="bold")
    axes[1].text(1.5, 0.9, "↑ rises when a few\nexperts dominate", color=RED, ha="center",
                 fontsize=9, fontweight="bold")
    fig.suptitle("The load-balancing auxiliary loss is minimised by uniform routing",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/moe_aux_loss.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"wrote moe_aux_loss.png  (L_bal={L_bal:.3f}, L_col={L_col:.3f})")


if __name__ == "__main__":
    decoupling()
    balance()
    routing()
    aux_loss()
    print("OUT:", OUT)
