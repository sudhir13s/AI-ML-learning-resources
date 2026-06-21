"""Naive-Bayes concept-page diagrams (muted palette, parallel scale).

Two figures for 03. Supervised_Learning/concepts/05-Naive-Bayes.md:
  1. nb_gaussian.png -- Gaussian Naive Bayes on 2D data: each class is modelled as
     a Gaussian per feature (independent axes); the resulting decision boundary. REAL sklearn.
  2. nb_spam.png -- the multinomial/text mechanic: reading a message word by word,
     the running log-odds (sum of per-word log P(w|spam)/P(w|ham)) crosses into spam.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_blobs
from matplotlib.colors import ListedColormap

OUT = os.path.join(os.path.dirname(__file__), "..", "03. Supervised_Learning", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def nb_gaussian():
    X, y = make_blobs(n_samples=240, centers=[(-2, -1), (2, 1.5)], cluster_std=[1.5, 1.3], random_state=2)
    clf = GaussianNB().fit(X, y)
    xx, yy = np.meshgrid(np.linspace(-7, 7, 400), np.linspace(-6, 6, 400))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    ax.contourf(xx, yy, Z, alpha=0.16, cmap=ListedColormap([BLUE, RED]))
    ax.contour(xx, yy, Z, levels=[0.5], colors=[NAVY], linewidths=2.2)
    # per-class Gaussian density contours (the model NB fits)
    for k, col in [(0, BLUE), (1, RED)]:
        mu, var = clf.theta_[k], clf.var_[k]
        d = ((xx - mu[0])**2 / var[0] + (yy - mu[1])**2 / var[1])
        ax.contour(xx, yy, d, levels=[1, 4], colors=[col], linewidths=1.0, alpha=0.7)
    ax.scatter(X[y == 0, 0], X[y == 0, 1], color=BLUE, s=22, edgecolor="white", lw=0.4, label="class 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], color=RED, s=22, edgecolor="white", lw=0.4, label="class 1")
    ax.set_xlabel("feature 1"); ax.set_ylabel("feature 2")
    ax.set_title("Gaussian Naive Bayes: model each class as a Gaussian, classify by the posterior",
                 fontsize=11.5, fontweight="bold")
    ax.legend(loc="upper left", frameon=True, fontsize=9); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/nb_gaussian.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote nb_gaussian.png")


def nb_spam():
    # per-word log-odds log P(word|spam) - log P(word|ham); a spammy message
    words = ["hi", "free", "meeting", "winner", "click", "prize", "tomorrow", "now"]
    llr = [-0.4, 1.6, -0.8, 1.9, 1.3, 1.7, -0.6, 0.9]      # >0 leans spam, <0 leans ham
    prior = np.log(0.4 / 0.6)                               # P(spam)=0.4 prior log-odds
    cum = np.cumsum([prior] + llr)
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.axhline(0, color=SLATE, lw=1.2, ls="--")
    ax.fill_between(range(len(cum)), 0, 6, color=RED, alpha=0.05)
    ax.fill_between(range(len(cum)), -6, 0, color=BLUE, alpha=0.05)
    ax.plot(range(len(cum)), cum, "o-", color=PURPLE, lw=2.4, ms=6)
    labels = ["(prior)"] + words
    ax.set_xticks(range(len(cum))); ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=9)
    for i, w in enumerate(words):
        ax.annotate(f"{llr[i]:+.1f}", (i+1, cum[i+1]), textcoords="offset points", xytext=(0, 8),
                    fontsize=8, color=PURPLE, ha="center")
    ax.text(0.3, 4.6, "log-odds > 0 → classify SPAM", color=RED, fontsize=10, fontweight="bold")
    ax.text(0.3, -4.8, "log-odds < 0 → classify HAM", color=BLUE, fontsize=10, fontweight="bold")
    ax.set_ylabel("running log-odds  (spam vs ham)")
    ax.set_title("Naive Bayes decides by adding per-word log-probabilities",
                 fontsize=12.5, fontweight="bold")
    ax.set_ylim(-6, 6); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/nb_spam.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote nb_spam.png")


def nb_calibration():
    """NB's independence assumption double-counts correlated features -> over-confident
    probabilities. Reliability diagram: predicted prob vs actual frequency."""
    from sklearn.naive_bayes import GaussianNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import calibration_curve
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    # correlated, redundant features -> NB double-counts them
    X, y = make_classification(n_samples=4000, n_features=20, n_informative=5, n_redundant=12,
                               random_state=0)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=0)
    fig, ax = plt.subplots(figsize=(7.6, 6.0))
    ax.plot([0, 1], [0, 1], color=SLATE, ls="--", lw=1.5, label="perfectly calibrated")
    for model, col, lab in [(GaussianNB(), RED, "Naive Bayes (over-confident)"),
                            (LogisticRegression(max_iter=1000), GREEN, "Logistic Regression (well-calibrated)")]:
        p = model.fit(Xtr, ytr).predict_proba(Xte)[:, 1]
        frac, mean_pred = calibration_curve(yte, p, n_bins=10)
        ax.plot(mean_pred, frac, "o-", color=col, lw=2.2, ms=5, label=lab)
    ax.text(0.27, 0.05, "NB pushes probabilities toward 0 and 1:\ncorrelated features are counted as\nindependent evidence (double-counting)",
            fontsize=9, color=RED, fontweight="bold")
    ax.set_xlabel("predicted probability"); ax.set_ylabel("actual fraction positive")
    ax.set_title("Naive Bayes is accurate but mis-calibrated (over-confident)",
                 fontsize=12.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left"); _despine(ax)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(f"{OUT}/nb_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote nb_calibration.png")


if __name__ == "__main__":
    nb_gaussian()
    nb_spam()
    nb_calibration()
    print("OUT:", OUT)
