"""Cross-validation concept-page diagrams (muted palette, parallel scale).

Four figures for 03. Supervised_Learning/concepts/13-Cross-Validation.md:
  1. cv_kfold.png  -- the 5-fold layout: each row is one iteration; the validation
     fold rotates so every point is used for both training and validation.
  2. cv_stability.png -- REAL measurement: 300 single random train/test splits give
     a noisy spread of scores; 5-fold CV gives one stable, lower-variance estimate.
  3. cv_variants.png -- the split SCHEMES side by side: plain k-fold, stratified
     (class ratio preserved per fold), grouped (a group never spans train+val), and
     time-series forward-chaining (never train on the future). One picture, four splitters.
  4. cv_k_tradeoff.png -- REAL measurement: the pessimistic BIAS of the CV ESTIMATE as a
     function of k (CV error minus the true error of a full-data model). Small k -> larger
     pessimistic bias (smaller train folds); it vanishes by k~10. Plotted against the
     COMPUTATIONAL COST (k model fits, rising to n at LOOCV). Why k=5/10 is the sweet spot:
     near-zero bias while still cheap.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, KFold

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def cv_kfold():
    k = 5
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for it in range(k):                                  # one row per iteration
        y = k - 1 - it
        for fold in range(k):
            is_val = (fold == it)
            ax.add_patch(Rectangle((fold, y), 0.94, 0.84,
                                   facecolor=RED if is_val else BLUE,
                                   alpha=0.85 if is_val else 0.55, edgecolor="white", lw=2))
            ax.text(fold + 0.47, y + 0.42, "validate" if is_val else "train",
                    ha="center", va="center", color="#fff", fontsize=9.5, fontweight="bold")
        ax.text(-0.25, y + 0.42, f"iter {it+1}", ha="right", va="center", fontsize=10, fontweight="bold", color=SLATE)
    ax.text(k/2, k + 0.25, "the dataset, split into 5 folds  →  rotate the validation fold",
            ha="center", fontsize=10.5, color=SLATE, fontweight="bold")
    ax.text(k/2, -0.7, "final score = average of the 5 validation scores (every point validated exactly once)",
            ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
    ax.set_xlim(-1.4, k + 0.2); ax.set_ylim(-1.1, k + 0.6); ax.axis("off")
    ax.set_title("5-fold cross-validation: every point is used for both training and validation",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout(); fig.savefig(f"{OUT}/cv_kfold.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cv_kfold.png")


def cv_stability():
    X, y = make_classification(n_samples=300, n_features=20, n_informative=6,
                               class_sep=0.8, random_state=0)
    single = []
    for s in range(300):                                 # many single random splits
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=s)
        single.append(LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte))
    cv = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    ax.hist(single, bins=24, color=SLATE, alpha=0.55, label=f"300 single splits (noisy: {np.std(single):.3f} std)")
    ax.axvline(np.mean(single), color=SLATE, ls=":", lw=1.5)
    ax.axvline(cv.mean(), color=GREEN, lw=2.6, label=f"5-fold CV estimate = {cv.mean():.3f} ± {cv.std():.3f}")
    for c in cv:
        ax.plot([c], [2], marker="v", color=RED, ms=9)
    ax.text(cv.mean(), ax.get_ylim()[1]*0.55, "the 5 fold scores (red ▾)\naverage to one stable number",
            color=RED, fontsize=9, fontweight="bold", ha="center")
    ax.set_xlabel("accuracy"); ax.set_ylabel("count (of single splits)")
    ax.set_title("Why k-fold: one split is a noisy estimate; averaging folds is far steadier",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left"); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/cv_stability.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cv_stability.png")


def cv_variants():
    """Four splitter schemes drawn as fold-membership strips over n=20 ordered samples."""
    n, k = 20, 4
    rng = np.random.default_rng(1)
    # class labels (imbalanced ~ 30% positive) and group ids for the panels that use them
    classes = (rng.random(n) < 0.30).astype(int)
    groups = np.repeat(np.arange(n // 4), 4)            # 5 groups of 4 consecutive samples

    def plain_assign():
        a = np.zeros(n, int); fs = n // k
        for f in range(k):
            a[f*fs:(f+1)*fs] = f
        return a

    def stratified_assign():
        a = np.zeros(n, int)
        for c in (0, 1):                                # round-robin within each class
            idx = np.where(classes == c)[0]
            for j, i in enumerate(idx):
                a[i] = j % k
        return a

    def grouped_assign():
        a = np.zeros(n, int)
        for g in range(n // 4):                         # whole group -> one fold
            a[groups == g] = g % k
        return a

    fig, axes = plt.subplots(4, 1, figsize=(10.4, 7.6))
    panels = [
        ("Plain k-fold — contiguous folds; class ratio drifts, time order ignored", plain_assign(), None, BLUE),
        ("Stratified k-fold — each fold keeps the ~30% positive class ratio (imbalance-safe)", stratified_assign(), "class", GREEN),
        ("Grouped k-fold — a whole group (■ same colour) stays in ONE fold (no leakage across folds)", grouped_assign(), "group", PURPLE),
        ("Time-series CV — forward-chaining: each fold trains on the PAST, validates the next block (never the future)", None, "time", AMBER),
    ]
    fold_colors = [BLUE, GREEN, AMBER, RED]
    for ax, (title, assign, mode, _accent) in zip(axes, panels):
        ax.set_xlim(-0.5, n - 0.5); ax.set_ylim(-1.4, 1.0); ax.axis("off")
        ax.set_title(title, fontsize=10.5, fontweight="bold", loc="left", color=SLATE)
        if mode == "time":
            # 4 successive forward-chaining iterations stacked as thin rows
            for it in range(4):
                yrow = 0.55 - it * 0.42
                tr_end = (it + 1) * (n // 5)
                val_end = tr_end + (n // 5)
                for i in range(n):
                    role = "train" if i < tr_end else ("val" if i < val_end else "future")
                    col = NAVY if role == "train" else (RED if role == "val" else "#d9dde2")
                    ax.add_patch(Rectangle((i - 0.45, yrow), 0.9, 0.34, facecolor=col,
                                           edgecolor="white", lw=0.6))
            ax.text(-0.5, 0.9, "iter1→4 (top→bottom): train ■ navy on past, validate ■ red the next block, ignore ■ grey the future",
                    fontsize=8.0, color=SLATE, va="bottom")
        else:
            for i in range(n):
                col = fold_colors[assign[i]]
                ax.add_patch(Rectangle((i - 0.45, -0.1), 0.9, 0.7, facecolor=col,
                                       edgecolor="white", lw=0.6))
                if mode == "class":
                    ax.text(i, 0.78, "+" if classes[i] else "·", ha="center", va="center",
                            fontsize=9, fontweight="bold", color=RED if classes[i] else SLATE)
                if mode == "group":
                    ax.text(i, 0.78, f"g{groups[i]}", ha="center", va="center", fontsize=6.5, color=SLATE)
            ax.text(-0.5, -0.95, "colour = which of the 4 folds each sample belongs to",
                    fontsize=8.0, color=SLATE, va="center")
    fig.suptitle("Match the split to the data: plain · stratified · grouped · time-series",
                 fontsize=13.0, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{OUT}/cv_variants.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cv_variants.png")


def cv_k_tradeoff():
    """REAL measurement: bias and variance of the k-fold ESTIMATE vs k.

    The target is the generalization error of a model trained on the FULL n_train rows
    (estimated on a large held-out pool from the SAME distribution). For each k we run
    k-fold on the n_train rows and compare. Because each k-fold training set has only
    (k-1)/k of the data, CV trains on a weaker model and OVERestimates error -> a
    POSITIVE pessimistic bias that shrinks as k grows (train folds approach the full
    set). Variance of the estimate is measured as the std of the CV error across many
    independent datasets; it rises toward LOOCV (k=n), where the n near-identical
    leave-one-out fits make the errors highly correlated.
    """
    ks = [2, 3, 5, 10, 20, 40]
    n_train, n_pool, reps = 60, 8000, 120
    # one large fixed population from the SAME generative process for the "true error" target
    biases, variances = [], []
    for k in ks:
        cv_errs, true_errs = [], []
        for r in range(reps):
            X, ytmp = make_classification(n_samples=n_train + n_pool, n_features=20,
                                          n_informative=8, class_sep=1.0, random_state=r)
            Xtr, ytr = X[:n_train], ytmp[:n_train]
            Xpool, ypool = X[n_train:], ytmp[n_train:]            # held-out, same distribution
            kf = KFold(n_splits=k, shuffle=True, random_state=r)
            cv_acc = cross_val_score(LogisticRegression(max_iter=2000), Xtr, ytr, cv=kf)
            cv_errs.append(1 - cv_acc.mean())
            # target: model trained on ALL n_train rows, scored on the big pool
            true_acc = LogisticRegression(max_iter=2000).fit(Xtr, ytr).score(Xpool, ypool)
            true_errs.append(1 - true_acc)
        cv_errs = np.array(cv_errs); true_errs = np.array(true_errs)
        biases.append((cv_errs - true_errs).mean())              # per-dataset bias, then averaged
        variances.append(cv_errs.std())

    cost = list(ks)                                              # k fits = compute cost (LOOCV = n fits)
    fig, ax1 = plt.subplots(figsize=(9.2, 5.2))
    ax2 = ax1.twinx()
    l1, = ax1.plot(ks, biases, marker="o", color=RED, lw=2.4,
                   label="pessimistic BIAS  (CV error − true error)")
    ax1.axhline(0, color=SLATE, ls=":", lw=1.0)
    l2, = ax2.plot(ks, cost, marker="s", color=BLUE, lw=2.4,
                   label="COST  (model fits = k; LOOCV needs n)")
    ax1.axvspan(5, 10, color=GREEN, alpha=0.12)
    ymid = (max(biases) + min(biases)) / 2
    ax1.text(7.0, ymid, "sweet spot\nk = 5–10\n(near-zero bias,\nstill cheap)", color=GREEN,
             fontsize=9.5, fontweight="bold", ha="center", va="center")
    ax1.set_xlabel("k  (number of folds; k = n ≈ LOOCV at the right)")
    ax1.set_ylabel("pessimistic bias of the CV error estimate", color=RED)
    ax2.set_ylabel("computational cost  (number of model fits)", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=RED); ax2.tick_params(axis="y", labelcolor=BLUE)
    ax1.set_xticks(ks)
    ax1.set_title("Choosing k: the pessimistic bias of the CV estimate vanishes by k≈10, while cost (k fits) keeps climbing toward LOOCV",
                  fontsize=10.0, fontweight="bold")
    ax1.legend(handles=[l1, l2], frameon=False, fontsize=9.5, loc="center right")
    _despine(ax1)
    for s in ("top",):
        ax2.spines[s].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{OUT}/cv_k_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote cv_k_tradeoff.png")


if __name__ == "__main__":
    cv_kfold()
    cv_stability()
    cv_variants()
    cv_k_tradeoff()
    print("OUT:", OUT)
