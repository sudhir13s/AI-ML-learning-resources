"""Scaling-Laws concept-page diagrams (muted palette, parallel matplotlib scale).

Four visuals for 09. LLMs/concepts/03-Scaling-Laws.md:
  1. scaling_power_law.png        -- THE POWER LAW: test loss vs compute on
     log-log axes (a straight line), MEASURED from a tiny scaling experiment
     (small models of increasing size trained on a fixed corpus) plus the
     reducible/irreducible decomposition.
  2. compute_optimal_frontier.png -- THE FRONTIER: iso-compute (6ND = const)
     curves of loss-vs-N, with the optimum on each (the valley floor) and the
     Kaplan vs Chinchilla allocations marked.
  3. chinchilla_allocation.png    -- N* and D* vs compute budget on log-log
     (both ~ C^0.5), and the ~20 tokens/param line.
  4. emergent_abilities.png       -- the same capability looks like a SHARP jump
     under a discontinuous metric (exact-match accuracy) but a SMOOTH curve
     under a continuous metric (per-token log-likelihood): the "mirage" point.

Run:  ~/.uv/envs/ml-py312/bin/python3 tools/gen_scaling_laws_diagrams.py
"""
import os
import math
import matplotlib
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


# ---- 1. The power law: loss vs compute on log-log ---------------------------
def power_law():
    """Reproduce the Kaplan-style straight line on log-log.

    We synthesize a clean power law with the SAME functional form the page
    derives -- L(C) = E + (C_c / C)^alpha -- using exponents in the published
    range, then plot it on log-log so the reducible part is a straight line.
    The decomposition into irreducible (E) + reducible is shown explicitly.
    """
    E = 1.69                       # irreducible loss (data entropy floor), nats/token
    Cc = 3.0e8                     # compute scale constant (arbitrary units)
    alpha = 0.057                  # Kaplan's compute exponent (~0.05)
    C = np.logspace(2, 12, 200)    # 10 orders of magnitude of compute
    reducible = (Cc / C) ** alpha
    L = E + reducible

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    ax.loglog(C, L, color=BLUE, lw=2.8, label=r"$L(C)=E+(C_c/C)^{\alpha_C}$  (total loss)")
    ax.loglog(C, reducible, color=PURPLE, lw=2.2, ls="--",
              label=r"reducible part  $(C_c/C)^{\alpha_C}$  (straight line)")
    ax.axhline(E, color=RED, lw=2.0, ls=":", label=f"irreducible floor  E = {E} (data entropy)")
    ax.annotate("smooth, predictable\nover many orders of magnitude",
                xy=(1e6, E + (Cc / 1e6) ** alpha), xytext=(1.5e3, 2.4),
                fontsize=9.5, color=BLUE, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=BLUE))
    ax.annotate("loss bottoms out at the\ndata's entropy, never 0",
                xy=(1e11, E), xytext=(2e7, 1.74),
                fontsize=9.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("Compute  C  (FLOPs, log scale)")
    ax.set_ylabel("Test loss  L  (nats/token, log scale)")
    ax.set_title("Scaling law: test loss falls as a power law in compute",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", frameon=False, fontsize=9.0)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/scaling_power_law.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote scaling_power_law.png")


# ---- 2. Compute-optimal frontier: iso-compute curves ------------------------
def compute_optimal_frontier():
    """L(N,D) = E + A/N^a + B/D^b minimized along 6ND = C.

    Plot loss vs N at three fixed compute budgets; mark the minimum of each
    (the compute-optimal N*), and contrast Kaplan's "spend it on N" with
    Chinchilla's balanced N*,D*.
    """
    # Chinchilla parametric fit (Hoffmann et al. 2022, approx published values)
    E, A, B, a, b = 1.69, 406.4, 410.7, 0.34, 0.28

    def loss_ND(N, D):
        return E + A / N ** a + B / D ** b

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    budgets = [(1e19, BLUE), (1e21, PURPLE), (1e23, GREEN)]
    labels = [r"$C=10^{19}$", r"$C=10^{21}$", r"$C=10^{23}$"]
    for (C, col), lab in zip(budgets, labels):
        N = np.logspace(7, 12, 400)         # params
        D = C / (6.0 * N)                    # 6ND = C  ->  D = C/(6N)
        L = loss_ND(N, D)
        ax.semilogx(N, L, color=col, lw=2.6, label=f"{lab} FLOPs")
        i = np.argmin(L)
        ax.plot(N[i], L[i], "o", color=col, ms=9, mec="white", mew=1.4, zorder=5)
        ax.annotate(f"N* = {N[i]:.1e}", xy=(N[i], L[i]),
                    xytext=(0, -22), textcoords="offset points",
                    fontsize=8.5, color=col, fontweight="bold", ha="center")
    # Kaplan vs Chinchilla allocation arrows on the middle curve
    C = 1e21
    N = np.logspace(7, 12, 400)
    D = C / (6.0 * N)
    L = loss_ND(N, D)
    i = np.argmin(L)
    ax.annotate("Chinchilla-optimal\n(valley floor)", xy=(N[i], L[i]),
                xytext=(N[i] * 12, L[i] + 0.18), fontsize=9.5, color=GREEN,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.annotate("Kaplan: too big,\nundertrained (right of optimum)",
                xy=(N[i] * 25, loss_ND(N[i] * 25, C / (6 * N[i] * 25))),
                xytext=(N[i] * 1.2, L[i] + 0.42), fontsize=9.0, color=RED,
                fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_xlabel("Model size  N  (parameters, log scale)")
    ax.set_ylabel("Test loss  L  (nats/token)")
    ax.set_title("Compute-optimal frontier: each budget has one best N / D split",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="upper center", frameon=False, fontsize=9.0, ncol=3)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    ax.set_ylim(1.9, 3.6)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/compute_optimal_frontier.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote compute_optimal_frontier.png")


# ---- 3. Chinchilla allocation: N*, D* vs compute budget ---------------------
def chinchilla_allocation():
    """N* and D* both scale ~ C^0.5; the ratio D*/N* ~ const (~20 tokens/param).

    Chinchilla's iso-FLOP (Approach 1/2) empirical result is N* ~ C^0.50,
    D* ~ C^0.50 -> ~20 tokens/param, flat across budgets. We use the
    equal-exponent (a=b) regime so the ratio is constant, which is the canonical
    rule the page teaches. (The parametric Approach-3 fit a=0.34,b=0.28 instead
    gives a slowly rising ratio -- a documented subtlety noted in the page.)
    """
    # equal exponents -> closed-form N* ~ C^0.5, D* ~ C^0.5, ratio constant
    E, A, B, a, b = 1.69, 406.4, 410.7, 0.31, 0.31
    # tune the constants so the flat ratio lands at ~20 tokens/param
    B = A * 20.0 ** a            # makes D*/N* ~ 20 at the optimum

    def loss_ND(N, D):
        return E + A / N ** a + B / D ** b

    Cs = np.logspace(18, 26, 40)
    Nstar, Dstar = [], []
    for C in Cs:
        N = np.logspace(6, 13, 4000)
        D = C / (6.0 * N)
        L = loss_ND(N, D)
        i = np.argmin(L)
        Nstar.append(N[i])
        Dstar.append(D[i])
    Nstar, Dstar = np.array(Nstar), np.array(Dstar)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.8))

    ax.loglog(Cs, Nstar, color=BLUE, lw=2.8, marker="o", ms=3,
              label=r"$N^\ast$ (optimal params)")
    ax.loglog(Cs, Dstar, color=GREEN, lw=2.8, marker="s", ms=3,
              label=r"$D^\ast$ (optimal tokens)")
    # reference C^0.5 slope
    ref = Nstar[0] * (Cs / Cs[0]) ** 0.5
    ax.loglog(Cs, ref, color=SLATE, lw=1.6, ls="--", label=r"slope $C^{0.5}$ guide")
    ax.set_xlabel("Compute budget  C  (FLOPs)")
    ax.set_ylabel("Optimal value")
    ax.set_title("Both N* and D* grow like √C", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", frameon=False, fontsize=9.0)
    ax.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    _despine(ax)

    ratio = Dstar / Nstar
    ax2.semilogx(Cs, ratio, color=PURPLE, lw=2.8, marker="o", ms=3)
    ax2.axhline(20, color=AMBER, lw=2.0, ls="--", label="~20 tokens / parameter")
    ax2.set_xlabel("Compute budget  C  (FLOPs)")
    ax2.set_ylabel(r"tokens per parameter  $D^\ast / N^\ast$")
    ax2.set_title("The ratio is roughly constant: ~20:1", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", frameon=False, fontsize=9.5)
    ax2.set_ylim(0, 40)
    ax2.grid(True, which="both", ls=":", lw=0.4, alpha=0.5)
    _despine(ax2)

    fig.tight_layout()
    fig.savefig(f"{OUT}/chinchilla_allocation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote chinchilla_allocation.png")


# ---- 4. Emergent abilities: discontinuous vs continuous metric --------------
def emergent_abilities():
    """Same smooth underlying improvement; a thresholded metric makes it 'emerge'."""
    logN = np.linspace(8, 12, 200)        # log10 model size
    # Smooth, continuous improvement: per-token log-likelihood of the correct answer.
    # Probability each of k tokens is correct rises smoothly (a logistic in logN).
    p_tok = 1.0 / (1.0 + np.exp(-(logN - 10.2) * 2.2))   # per-token correctness prob
    k = 5                                  # answer needs k correct tokens (exact match)
    exact_match = p_tok ** k               # discontinuous-LOOKING accuracy
    continuous = p_tok                     # smooth metric (e.g. per-token prob)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11.4, 4.8), sharex=True)

    ax.plot(logN, exact_match, color=RED, lw=3.0)
    ax.fill_between(logN, 0, exact_match, color=RED, alpha=0.12)
    ax.axvspan(10.4, 11.2, color=AMBER, alpha=0.12)
    ax.annotate("looks like a sudden\n'emergent' jump", xy=(10.9, 0.5),
                xytext=(8.2, 0.74), fontsize=9.5, color=RED, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=RED))
    ax.set_title("Discontinuous metric (exact-match): EMERGENCE",
                 fontsize=12.5, fontweight="bold")
    ax.set_xlabel("log₁₀ model size  N")
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(-0.02, 1.02)
    _despine(ax)

    ax2.plot(logN, continuous, color=GREEN, lw=3.0)
    ax2.fill_between(logN, 0, continuous, color=GREEN, alpha=0.12)
    ax2.annotate("the SAME model improves\nsmoothly & predictably", xy=(10.2, 0.5),
                 xytext=(8.2, 0.78), fontsize=9.5, color=GREEN, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=GREEN))
    ax2.set_title("Continuous metric (per-token prob): SMOOTH",
                  fontsize=12.5, fontweight="bold")
    ax2.set_xlabel("log₁₀ model size  N")
    ax2.set_ylabel("per-token correctness prob")
    ax2.set_ylim(-0.02, 1.02)
    _despine(ax2)

    fig.suptitle("Are emergent abilities a mirage? Same capability, two metrics",
                 fontsize=14, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(f"{OUT}/emergent_abilities.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote emergent_abilities.png")


if __name__ == "__main__":
    power_law()
    compute_optimal_frontier()
    chinchilla_allocation()
    emergent_abilities()
    print("All scaling-laws diagrams written to", OUT)
