"""Hyperparameter-tuning concept-page diagrams (muted palette, parallel scale).

Four figures for 05. Deep_Learning/concepts/12-Hyperparameter-Tuning.md, all from
REAL runs/computation (no fabricated curves):

  1. hpt_grid_vs_random.png -- grid (9x9) vs random (81 pts) projected onto the ONE
     dimension that matters: grid tries only 9 distinct values of the important knob;
     random tries ~81 distinct values. The Bergstra-Bengio picture.
  2. hpt_bayes_opt.png -- a toy 1-D objective with 4 observed points, a Gaussian-process
     posterior (mean +/- 2 sigma), and the Expected-Improvement acquisition below it,
     pointing at the next query. Computed exactly (RBF GP closed form + EI formula).
  3. hpt_hyperband.png -- successive halving / Hyperband: rungs of trials, each rung
     keeps the top 1/eta and multiplies their budget by eta. Bars show survivors per rung.
  4. hpt_random_vs_tpe.png -- MEASURED best-so-far validation error vs trial number,
     random search vs Optuna TPE, tuning a small MLP on a real dataset (averaged seeds).
"""
import os, math, warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

OUT = os.path.join(os.path.dirname(__file__), "..", "05. Deep_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- 1. grid vs random
def grid_vs_random():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))

    # left: GRID 9x9 = 81 trials
    g = np.linspace(0.04, 0.96, 9)
    gx, gy = np.meshgrid(g, g)
    axL = axes[0]
    axL.scatter(gx.ravel(), gy.ravel(), s=42, color=BLUE, edgecolor="#1A4B70", zorder=3)
    # marginal "histograms" on the important axis (x): grid hits only 9 distinct values
    for v in g:
        axL.plot([v, v], [-0.06, -0.02], color=RED, lw=2.4, solid_capstyle="butt")
    axL.set_title("Grid search: 81 trials, only 9 distinct\nvalues of the important knob",
                  fontsize=12.5, fontweight="bold")

    # right: RANDOM 81 trials
    rx = rng.uniform(0.02, 0.98, 81)
    ry = rng.uniform(0.02, 0.98, 81)
    axR = axes[1]
    axR.scatter(rx, ry, s=42, color=GREEN, edgecolor="#1E6A4A", zorder=3)
    for v in rx:
        axR.plot([v, v], [-0.06, -0.02], color=RED, lw=1.2, alpha=0.8, solid_capstyle="butt")
    axR.set_title("Random search: 81 trials, ~81 distinct\nvalues of the important knob",
                  fontsize=12.5, fontweight="bold")

    for ax in axes:
        # the "important" dimension is x; the unimportant one is y (flat ridge)
        ax.axhspan(-0.02, 1.0, xmin=0, xmax=1, color="none")
        ax.set_xlim(-0.02, 1.0); ax.set_ylim(-0.09, 1.0)
        ax.set_xlabel("important hyperparameter (e.g. learning rate)")
        ax.set_ylabel("unimportant hyperparameter")
        _despine(ax)
        ax.text(0.5, 1.03, "", transform=ax.transAxes)
    fig.text(0.5, 0.005, "Red ticks = distinct values actually tried on the important axis "
             "(Bergstra & Bengio, 2012)", ha="center", fontsize=9.5, color="#444")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(f"{OUT}/hpt_grid_vs_random.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote hpt_grid_vs_random.png")


# ---------------------------------------------------------------- 2. Bayesian opt (GP + EI)
def _rbf(a, b, ell=0.12, sf=1.0):
    a = a.reshape(-1, 1); b = b.reshape(-1, 1)
    d2 = (a - b.T) ** 2
    return sf ** 2 * np.exp(-0.5 * d2 / ell ** 2)


def bayes_opt():
    # toy objective on [0,1] we are MINIMIZING
    def f(x):
        return np.sin(3 * np.pi * x) * 0.5 + (x - 0.5) ** 2 * 2.2 + 0.15

    Xobs = np.array([0.07, 0.27, 0.62, 0.88])
    yobs = f(Xobs)
    noise = 1e-4

    Xs = np.linspace(0, 1, 400)
    K = _rbf(Xobs, Xobs) + noise * np.eye(len(Xobs))
    Kinv = np.linalg.inv(K)
    Ks = _rbf(Xs, Xobs)
    mu = Ks @ Kinv @ yobs
    var = np.clip(np.diag(_rbf(Xs, Xs)) - np.einsum("ij,jk,ik->i", Ks, Kinv, Ks), 1e-9, None)
    sd = np.sqrt(var)

    # Expected Improvement (minimization): improvement over current best f*
    fbest = yobs.min()
    from math import erf
    def Phi(z): return 0.5 * (1 + np.vectorize(erf)(z / np.sqrt(2)))
    def phi(z): return np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    imp = fbest - mu
    Z = imp / sd
    EI = imp * Phi(Z) + sd * phi(Z)
    EI[sd < 1e-6] = 0.0
    xnext = Xs[np.argmax(EI)]

    fig, (axT, axB) = plt.subplots(2, 1, figsize=(9.2, 6.6), height_ratios=[2.3, 1],
                                   sharex=True)
    axT.plot(Xs, f(Xs), color=SLATE, lw=1.6, ls="--", label="true objective (hidden)")
    axT.plot(Xs, mu, color=PURPLE, lw=2.4, label="GP posterior mean")
    axT.fill_between(Xs, mu - 2 * sd, mu + 2 * sd, color=PURPLE, alpha=0.18,
                     label="±2σ uncertainty")
    axT.scatter(Xobs, yobs, s=60, color=BLUE, edgecolor="#1A4B70", zorder=5,
                label="observed trials")
    axT.axhline(fbest, color=GREEN, lw=1.2, ls=":", label="best so far  f*")
    axT.set_ylabel("validation loss (minimize)")
    axT.set_title("Bayesian optimization: a GP surrogate + Expected Improvement choose the next trial",
                  fontsize=12.5, fontweight="bold")
    axT.legend(frameon=False, fontsize=9, ncol=2, loc="upper center")
    _despine(axT)

    axB.fill_between(Xs, 0, EI, color=AMBER, alpha=0.45)
    axB.plot(Xs, EI, color=AMBER, lw=2.0)
    axB.axvline(xnext, color=RED, lw=2.0, ls="-")
    axB.annotate(f"next trial\nx ≈ {xnext:.2f}", xy=(xnext, EI.max()),
                 xytext=(xnext + 0.04, EI.max() * 0.7), fontsize=9.5, color=RED,
                 fontweight="bold")
    axB.set_ylabel("EI(x)")
    axB.set_xlabel("hyperparameter x")
    axB.set_title("Acquisition = Expected Improvement (explore high σ + exploit low μ)",
                  fontsize=11, fontweight="bold")
    _despine(axB)
    fig.tight_layout()
    fig.savefig(f"{OUT}/hpt_bayes_opt.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote hpt_bayes_opt.png  (EI picks x≈{xnext:.3f})")


# ---------------------------------------------------------------- 3. Hyperband / SHA rungs
def hyperband():
    eta = 3
    n0 = 27          # trials at rung 0
    r0 = 1           # budget (epochs) at rung 0
    rungs = []
    n, r = n0, r0
    while n >= 1:
        rungs.append((n, r))
        if n == 1:
            break
        n = max(1, n // eta)
        r = r * eta
    # rungs: [(27,1),(9,3),(3,9),(1,27)]

    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    cols = [BLUE, PURPLE, AMBER, GREEN]
    xpos = np.arange(len(rungs))
    survivors = [n for n, _ in rungs]
    budgets = [r for _, r in rungs]
    bars = ax.bar(xpos, survivors, color=cols[:len(rungs)], edgecolor="#222", width=0.6)
    for i, (n, r) in enumerate(rungs):
        ax.text(i, n + 0.6, f"{n} trials\n× {r} epoch{'s' if r > 1 else ''}",
                ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"rung {i}\n(keep top 1/{eta})" if i < len(rungs) - 1 else
                        f"rung {i}\n(winner)" for i in xpos])
    ax.set_ylabel("surviving trials (log-spaced cull)")
    ax.set_ylim(0, n0 * 1.18)
    ax.set_title("Successive halving (η=3): cull the bottom, give survivors more budget",
                 fontsize=12.5, fontweight="bold")
    total = sum(n * r for n, r in rungs)
    ax.text(0.98, 0.92, f"total budget = Σ nᵢ·rᵢ = {total} epoch-units\n"
            f"(vs {n0}×{budgets[-1]}={n0*budgets[-1]} to fully train all 27)",
            transform=ax.transAxes, ha="right", fontsize=9.5, color="#444",
            bbox=dict(boxstyle="round", fc="#f3f3f3", ec="#ccc"))
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/hpt_hyperband.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote hpt_hyperband.png  (rungs={rungs}, total={total})")


# ---------------------------------------------------------------- 4. MEASURED random vs TPE
def random_vs_tpe():
    import optuna
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = load_digits(return_X_y=True)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        alpha = trial.suggest_float("alpha", 1e-6, 1e-1, log=True)
        h = trial.suggest_int("hidden", 16, 256, log=True)
        clf = MLPClassifier(hidden_layer_sizes=(h,), alpha=alpha,
                            learning_rate_init=lr, max_iter=120, random_state=0)
        clf.fit(Xtr, ytr)
        return 1.0 - clf.score(Xva, yva)        # validation error (minimize)

    N_TRIALS = 40
    SEEDS = [0, 1, 2]

    def run(sampler_factory):
        curves = []
        for s in SEEDS:
            study = optuna.create_study(direction="minimize",
                                        sampler=sampler_factory(s))
            study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
            vals = [t.value for t in study.trials]
            best = np.minimum.accumulate(vals)
            curves.append(best)
        return np.mean(curves, axis=0)

    rand = run(lambda s: optuna.samplers.RandomSampler(seed=s))
    tpe = run(lambda s: optuna.samplers.TPESampler(seed=s, n_startup_trials=8))

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    t = np.arange(1, N_TRIALS + 1)
    ax.plot(t, rand * 100, color=BLUE, lw=2.4, marker="o", ms=3,
            label="random search")
    ax.plot(t, tpe * 100, color=GREEN, lw=2.4, marker="s", ms=3,
            label="TPE (Bayesian, Optuna)")
    ax.set_xlabel("trial number")
    ax.set_ylabel("best validation error so far (%)")
    ax.set_title("Measured: TPE reaches a lower error in fewer trials than random search\n"
                 "(MLP on digits, mean of 3 seeds)", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=10.5)
    _despine(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/hpt_random_vs_tpe.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote hpt_random_vs_tpe.png  (random→{rand[-1]*100:.2f}%  tpe→{tpe[-1]*100:.2f}%)")


if __name__ == "__main__":
    grid_vs_random()
    bayes_opt()
    hyperband()
    random_vs_tpe()
    print("ALL DIAGRAMS WRITTEN ->", OUT)
