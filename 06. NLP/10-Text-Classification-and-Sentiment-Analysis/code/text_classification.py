"""Text classification & sentiment analysis: the single seeded source of truth.

Every number in the chapter page, every figure in `make_figures_10.py`, and every cell of
`10-Text-Classification-and-Sentiment-Analysis.ipynb` is computed by the functions in THIS
file, so the prose, the pictures, and the notebook cannot silently drift apart. Nothing here
touches the network or downloads a dataset: the sentiment corpus is *generated deterministically*
from polarity-bearing word pools (seeded), so the whole demo runs on CPU in a couple of seconds
and reproduces bit-for-bit on any machine.

What it provides:
  * `nb_by_hand` / `nb_sklearn_joint_log_lik` -- the multinomial-NB log-posterior worked by hand,
    proven against scikit-learn's `MultinomialNB` on the tiny 4-document corpus.
  * `make_sentiment_corpus` -- a deterministic synthetic review generator (balanced or skewed).
  * `train_nb` / `train_logreg` / `train_svm` -- the three classical baselines on TF-IDF.
  * `evaluate` -- accuracy, macro-/micro-F1, the confusion matrix, PR-AUC and ROC-AUC.
  * `precision_recall_f1_from_counts` -- precision/recall/F1 derived straight from TP/FP/FN.
  * `threshold_sweep` -- precision/recall/F1 as the decision threshold moves off 0.5.
  * `stable_token_hash` -- an md5-based hash (NEVER Python's salted `hash()`), so any hashing
    trick is reproducible across processes.

Determinism: every stochastic step is seeded (numpy default_rng, sklearn `random_state=0`).
Verified on Python 3.12 / scikit-learn 1.9 / numpy 2.x, CPU.

Run:
    python text_classification.py
"""

from __future__ import annotations

import hashlib
import math
import platform
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# --------------------------------------------------------------------------------------------
# Reproducibility knobs -- one place, imported everywhere.
# --------------------------------------------------------------------------------------------
SEED = 0
LAPLACE_ALPHA = 1.0  # add-1 (Laplace) smoothing for multinomial NB

# The tiny worked-by-hand corpus (matches Worked Example 1 in the page, exactly).
TINY_TRAIN: list[tuple[str, int]] = [
    ("great fun great", 1),
    ("great great movie", 1),
    ("boring boring film", 0),
    ("boring dull movie", 0),
]
TINY_TEST_DOC = "great fun"  # the document we classify by hand


# --------------------------------------------------------------------------------------------
# Stable hashing -- md5, never Python's per-process-salted hash().
# --------------------------------------------------------------------------------------------
def stable_token_hash(token: str, n_buckets: int) -> int:
    """Map a token to a bucket in [0, n_buckets) with a STABLE hash.

    Python's built-in ``hash()`` is salted per process (PYTHONHASHSEED), so a hashing-trick
    vectorizer built on it would bucket words differently on every run -- silently breaking
    reproducibility. md5 is content-addressed and identical across processes/machines.
    """
    digest = hashlib.md5(token.encode("utf-8")).hexdigest()
    return int(digest, 16) % n_buckets


# --------------------------------------------------------------------------------------------
# Multinomial Naive Bayes, worked by hand (and the matching sklearn joint log-likelihood).
# --------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class NaiveBayesByHand:
    """The two class log-posteriors and the predicted label for one test document."""

    logp_pos: float
    logp_neg: float

    @property
    def predicted(self) -> int:
        return 1 if self.logp_pos > self.logp_neg else 0


def nb_by_hand(
    train: list[tuple[str, int]] = TINY_TRAIN,
    test_doc: str = TINY_TEST_DOC,
    alpha: float = LAPLACE_ALPHA,
) -> NaiveBayesByHand:
    """Compute the multinomial-NB log-posterior for `test_doc`, fully by hand.

    score(c) = log P(c) + sum_w  count_w(test_doc) * log P(w | c),
    with P(w|c) = (count(w,c) + alpha) / (sum_w' count(w',c) + alpha*|V|)  (Laplace smoothing).
    """
    vocab = sorted({w for doc, _ in train for w in doc.split()})
    vocab_size = len(vocab)

    # Per-class word counts and per-class total word count.
    class_counts: dict[int, dict[str, int]] = {1: {}, 0: {}}
    class_total: dict[int, int] = {1: 0, 0: 0}
    n_docs = {1: 0, 0: 0}
    for doc, label in train:
        n_docs[label] += 1
        for w in doc.split():
            class_counts[label][w] = class_counts[label].get(w, 0) + 1
            class_total[label] += 1

    n_total = len(train)
    log_prior = {c: math.log(n_docs[c] / n_total) for c in (0, 1)}

    def log_likelihood(word: str, c: int) -> float:
        num = class_counts[c].get(word, 0) + alpha
        den = class_total[c] + alpha * vocab_size
        return math.log(num / den)

    test_tokens = test_doc.split()
    score: dict[int, float] = {}
    for c in (0, 1):
        score[c] = log_prior[c] + sum(log_likelihood(w, c) for w in test_tokens)
    return NaiveBayesByHand(logp_pos=score[1], logp_neg=score[0])


def nb_sklearn_joint_log_lik(
    train: list[tuple[str, int]] = TINY_TRAIN,
    test_doc: str = TINY_TEST_DOC,
    alpha: float = LAPLACE_ALPHA,
) -> tuple[float, float]:
    """Return scikit-learn's (logp_neg, logp_pos) joint log-likelihood for `test_doc`.

    Uses ``MultinomialNB._joint_log_likelihood`` so we read the SAME quantity the by-hand
    derivation computes (prior + sum of count*log-likelihood), not the renormalised posterior.
    """
    docs = [d for d, _ in train]
    labels = [c for _, c in train]
    vectorizer = CountVectorizer()
    x_counts = vectorizer.fit_transform(docs)
    model = MultinomialNB(alpha=alpha).fit(x_counts, labels)
    jll = model._joint_log_likelihood(vectorizer.transform([test_doc]))[0]  # noqa: SLF001
    # classes_ is sorted: [0, 1] -> [neg, pos]
    neg_idx = list(model.classes_).index(0)
    pos_idx = list(model.classes_).index(1)
    return float(jll[neg_idx]), float(jll[pos_idx])


def logreg_tiny_pos_proba(
    train: list[tuple[str, int]] = TINY_TRAIN, test_doc: str = TINY_TEST_DOC
) -> float:
    """Positive-class probability LogReg assigns to `test_doc` on the tiny by-hand corpus.

    The discriminative counterpart to `nb_by_hand` on the *same* four-document corpus: fit a
    TF-IDF + LogisticRegression and read P(pos | test_doc). Logistic regression's probabilities
    are typically *calmer* (better calibrated) than Naive Bayes', whose false independence
    assumption pushes its effective probability toward 0/1 -- this number makes that concrete.
    """
    docs = [d for d, _ in train]
    labels = [c for _, c in train]
    vectorizer = TfidfVectorizer()
    clf = LogisticRegression(max_iter=1000, random_state=SEED).fit(
        vectorizer.fit_transform(docs), labels
    )
    pos_idx = list(clf.classes_).index(1)
    return float(clf.predict_proba(vectorizer.transform([test_doc]))[0, pos_idx])


# --------------------------------------------------------------------------------------------
# A deterministic synthetic sentiment corpus (no downloads).
# --------------------------------------------------------------------------------------------
# Polarity-bearing word pools. Positive reviews draw mostly from POS_WORDS, negative from
# NEG_WORDS; both mix in NEUTRAL_WORDS. A controlled fraction of "confusable" words leak across
# classes so the task is NOT linearly trivial -- that head-room is what lets LogReg edge out NB.
POS_WORDS = [
    "brilliant", "wonderful", "loved", "excellent", "superb", "gripping", "heartwarming",
    "masterpiece", "delightful", "fantastic", "moving", "charming", "stellar", "enjoyable",
]
NEG_WORDS = [
    "boring", "awful", "terrible", "dull", "waste", "disappointing", "predictable",
    "tedious", "bland", "forgettable", "clumsy", "lifeless", "mediocre", "painful",
]
NEUTRAL_WORDS = [
    "movie", "film", "plot", "scene", "actor", "director", "story", "cast", "runtime",
    "the", "a", "was", "is", "it", "this", "and", "with", "of", "really", "quite",
]

# Correlated "burst" clusters: words that travel together. When a review uses one, it tends to
# use the whole cluster -- so the features are HIGHLY correlated, not independent. This is the
# knob that makes the data violate Naive Bayes' conditional-independence assumption: NB treats
# each correlated word as fresh evidence and double-counts it, while LogReg learns to down-weight
# the redundant cluster. That is exactly why LogReg edges out NB on realistic text.
POS_CLUSTERS = [
    ["brilliant", "wonderful", "loved"],
    ["excellent", "superb", "stellar"],
    ["gripping", "moving", "heartwarming"],
]
NEG_CLUSTERS = [
    ["boring", "dull", "tedious"],
    ["awful", "terrible", "painful"],
    ["waste", "disappointing", "forgettable"],
]


def make_sentiment_corpus(
    n_per_class: int = 600,
    pos_fraction: float = 0.5,
    seed: int = SEED,
    doc_len: int = 14,
    leak: float = 0.15,
    burst: float = 0.6,
) -> tuple[list[str], np.ndarray]:
    """Generate a deterministic synthetic sentiment corpus.

    Each document is `doc_len` words: ~half neutral filler, the rest polarity-bearing. The polar
    words are drawn either as a CORRELATED BURST (a full cluster of co-occurring words, with
    probability `burst`) or as independent single words; a `leak` fraction of the polar slots is
    drawn from the OTHER class's pool (the confusable signal that makes the problem non-trivial
    and keeps accuracy realistic, ~0.85). The burst correlation is what makes LogReg beat NB.
    With `pos_fraction != 0.5` the positive class becomes rare -- the imbalance demo.

    Returns (texts, y) with y in {0=neg, 1=pos}.
    """
    rng = np.random.default_rng(seed)
    n_total = int(round(2 * n_per_class))
    n_pos = int(round(n_total * pos_fraction))
    n_neg = n_total - n_pos
    texts: list[str] = []
    labels: list[int] = []
    for label, count in ((1, n_pos), (0, n_neg)):
        own_words = POS_WORDS if label == 1 else NEG_WORDS
        other_words = NEG_WORDS if label == 1 else POS_WORDS
        own_clusters = POS_CLUSTERS if label == 1 else NEG_CLUSTERS
        other_clusters = NEG_CLUSTERS if label == 1 else POS_CLUSTERS
        for _ in range(count):
            n_neutral = doc_len // 2
            n_polar = doc_len - n_neutral
            words = list(rng.choice(NEUTRAL_WORDS, size=n_neutral))
            filled = 0
            while filled < n_polar:
                flip_class = rng.random() < leak
                if rng.random() < burst:
                    clusters = other_clusters if flip_class else own_clusters
                    cluster = clusters[rng.integers(len(clusters))]
                    words.extend(cluster)  # the whole correlated burst lands together
                    filled += len(cluster)
                else:
                    pool = other_words if flip_class else own_words
                    words.append(str(rng.choice(pool)))
                    filled += 1
            rng.shuffle(words)
            texts.append(" ".join(words))
            labels.append(label)
    # Shuffle documents so class order is not a confound for any model.
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    y = np.array(labels)[order]
    return texts, y


def train_test_split_texts(
    texts: list[str], y: np.ndarray, test_fraction: float = 0.4, seed: int = SEED
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    """Deterministic split of (texts, y) into train/test."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(texts))
    n_test = int(round(len(texts) * test_fraction))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    x_tr = [texts[i] for i in train_idx]
    x_te = [texts[i] for i in test_idx]
    return x_tr, x_te, y[train_idx], y[test_idx]


# --------------------------------------------------------------------------------------------
# The three classical baselines on TF-IDF features.
# --------------------------------------------------------------------------------------------
def _vectorizer() -> TfidfVectorizer:
    # 1-2 grams + sublinear tf is the standard strong text-baseline recipe.
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)


def _count_vectorizer() -> CountVectorizer:
    return CountVectorizer(ngram_range=(1, 2))


@dataclass
class FittedModel:
    name: str
    vectorizer: object
    clf: object

    def predict(self, texts: list[str]) -> np.ndarray:
        return self.clf.predict(self.vectorizer.transform(texts))

    def scores(self, texts: list[str]) -> np.ndarray:
        """Positive-class score in [0,1] where available, else a decision-function value.

        Used for PR/ROC curves and the threshold sweep.
        """
        x = self.vectorizer.transform(texts)
        if hasattr(self.clf, "predict_proba"):
            return self.clf.predict_proba(x)[:, 1]
        # LinearSVC has no predict_proba; use the signed margin as a ranking score.
        return self.clf.decision_function(x)


def train_nb(x_train: list[str], y_train: np.ndarray) -> FittedModel:
    # NB wants counts, not TF-IDF (it models a multinomial over term frequencies).
    vec = _count_vectorizer()
    clf = MultinomialNB(alpha=LAPLACE_ALPHA).fit(vec.fit_transform(x_train), y_train)
    return FittedModel("Multinomial NB", vec, clf)


def train_logreg(
    x_train: list[str], y_train: np.ndarray, class_weight: str | None = None
) -> FittedModel:
    vec = _vectorizer()
    clf = LogisticRegression(
        max_iter=2000, random_state=SEED, class_weight=class_weight
    ).fit(vec.fit_transform(x_train), y_train)
    return FittedModel("TF-IDF + LogReg", vec, clf)


def train_svm(x_train: list[str], y_train: np.ndarray) -> FittedModel:
    vec = _vectorizer()
    clf = LinearSVC(random_state=SEED).fit(vec.fit_transform(x_train), y_train)
    return FittedModel("Linear SVM", vec, clf)


# --------------------------------------------------------------------------------------------
# Evaluation done right.
# --------------------------------------------------------------------------------------------
@dataclass
class EvalResult:
    name: str
    accuracy: float
    macro_f1: float
    micro_f1: float
    cm: np.ndarray  # confusion_matrix, rows=actual [neg,pos], cols=predicted [neg,pos]
    pr_auc: float | None  # average precision for the positive class (None if no scores)
    roc_auc: float | None


def evaluate(
    model: FittedModel, x_test: list[str], y_test: np.ndarray, with_curves: bool = True
) -> EvalResult:
    """Accuracy, macro-/micro-F1, confusion matrix, PR-AUC and ROC-AUC for `model`."""
    y_pred = model.predict(x_test)
    acc = float((y_pred == y_test).mean())
    macro = float(f1_score(y_test, y_pred, average="macro"))
    micro = float(f1_score(y_test, y_pred, average="micro"))
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    pr_auc = roc = None
    if with_curves:
        s = model.scores(x_test)
        pr_auc = float(average_precision_score(y_test, s))
        roc = float(roc_auc_score(y_test, s))
    return EvalResult(model.name, acc, macro, micro, cm, pr_auc, roc)


def precision_recall_f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Precision, recall, F1 derived straight from the confusion-matrix counts.

    precision = TP / (TP+FP)   "of what I flagged, how much was right"
    recall    = TP / (TP+FN)   "of what was truly positive, how much I caught"
    F1        = harmonic mean of the two.
    """
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def threshold_sweep(
    scores: np.ndarray, y_true: np.ndarray, thresholds: np.ndarray | None = None
) -> dict[str, np.ndarray]:
    """Precision, recall, and F1 for the positive class as the decision threshold moves.

    The default 0.5 is just one column of this table. Sweeping it is how you trade precision
    against recall to hit a product target.
    """
    if thresholds is None:
        thresholds = np.linspace(0.02, 0.98, 49)
    precisions, recalls, f1s = [], [], []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        p, r, f = precision_recall_f1_from_counts(tp, fp, fn)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f)
    return {
        "thresholds": thresholds,
        "precision": np.array(precisions),
        "recall": np.array(recalls),
        "f1": np.array(f1s),
    }


# --------------------------------------------------------------------------------------------
# Convenience: the full balanced ladder comparison (NB / LogReg / SVM), measured.
# --------------------------------------------------------------------------------------------
def run_ladder_comparison(
    n_per_class: int = 600, seed: int = SEED
) -> tuple[list[EvalResult], FittedModel, list[str], np.ndarray]:
    """Train NB, LogReg, SVM on a balanced synthetic corpus and evaluate all three.

    Returns (results, logreg_model, x_test, y_test) -- the LogReg model and held-out test set
    are handed back so the confusion-matrix / PR / threshold figures all use the SAME split.
    """
    texts, y = make_sentiment_corpus(n_per_class=n_per_class, pos_fraction=0.5, seed=seed)
    x_tr, x_te, y_tr, y_te = train_test_split_texts(texts, y, seed=seed)
    nb = train_nb(x_tr, y_tr)
    lr = train_logreg(x_tr, y_tr)
    svm = train_svm(x_tr, y_tr)
    results = [
        evaluate(nb, x_te, y_te),
        evaluate(lr, x_te, y_te),
        evaluate(svm, x_te, y_te),
    ]
    return results, lr, x_te, y_te


def run_imbalance_demo(
    n_per_class: int = 600, pos_fraction: float = 0.08, seed: int = SEED
) -> dict[str, object]:
    """Show accuracy lying under imbalance, while F1 / PR-AUC tell the truth.

    Builds a skewed corpus (positive class rare), then compares the trivial 'always predict the
    majority' classifier against a real LogReg model on accuracy vs macro-F1 vs PR-AUC.
    """
    texts, y = make_sentiment_corpus(
        n_per_class=n_per_class, pos_fraction=pos_fraction, seed=seed
    )
    x_tr, x_te, y_tr, y_te = train_test_split_texts(texts, y, seed=seed)
    pos_rate = float(y_te.mean())

    # Trivial baseline: always predict the majority class (negative).
    majority_pred = np.zeros_like(y_te)
    majority_acc = float((majority_pred == y_te).mean())
    majority_f1 = float(f1_score(y_te, majority_pred, average="macro"))
    # PR-AUC of a no-skill classifier equals the positive prevalence.
    majority_pr_auc = pos_rate

    # A real model trained with class weighting to counter the skew.
    lr = train_logreg(x_tr, y_tr, class_weight="balanced")
    lr_eval = evaluate(lr, x_te, y_te)

    return {
        "pos_rate": pos_rate,
        "majority_acc": majority_acc,
        "majority_f1": majority_f1,
        "majority_pr_auc": majority_pr_auc,
        "model_acc": lr_eval.accuracy,
        "model_f1": lr_eval.macro_f1,
        "model_pr_auc": lr_eval.pr_auc,
        "model_roc_auc": lr_eval.roc_auc,
        "model": lr,
        "x_test": x_te,
        "y_test": y_te,
    }


def run_leakage_demo(
    n_per_class: int = 400, seed: int = SEED, n_select: int = 40
) -> dict[str, float]:
    """Show how a preprocessing step fit on the WHOLE dataset leaks the test labels.

    This is the textbook 'preprocess/select-before-the-split' leakage -- the same family as
    fitting a TF-IDF vectorizer on train+test. To make the effect unmistakable we use a corpus
    of PURE NOISE (random rare tokens, label independent of the text): the honest answer is
    chance (~0.50). The honest pipeline selects features using TRAIN labels only and correctly
    stays at chance. The leaky pipeline ranks features by their correlation with ALL labels
    (train + test), so it 'discovers' tokens that match the test labels by accident, then scores
    on those same labels -- and reports a confidently-wrong accuracy well above chance. That gap
    is fabricated by the leak and would vanish in production.
    """
    import scipy.sparse as sp
    from sklearn.feature_selection import SelectKBest, chi2

    rng = np.random.default_rng(seed + 7)
    n_total = 2 * n_per_class
    labels = np.array([1] * n_per_class + [0] * n_per_class)
    texts = [
        " ".join(f"z{int(t)}" for t in rng.integers(0, 2000, size=30)) for _ in range(n_total)
    ]
    order = rng.permutation(n_total)
    texts = [texts[i] for i in order]
    y = labels[order]
    x_tr, x_te, y_tr, y_te = train_test_split_texts(texts, y, seed=seed)

    vec = CountVectorizer().fit(x_tr + x_te)  # shared vocabulary (not the leak by itself)
    xtr_counts = vec.transform(x_tr)
    xte_counts = vec.transform(x_te)

    def score(fit_on_all: bool) -> float:
        selector = SelectKBest(chi2, k=n_select)
        if fit_on_all:
            # LEAK: rank features using train+test labels, then score on those same test labels.
            x_all = sp.vstack([xtr_counts, xte_counts])
            y_all = np.concatenate([y_tr, y_te])
            selector.fit(x_all, y_all)
        else:
            selector.fit(xtr_counts, y_tr)  # honest: train labels only
        clf = LogisticRegression(max_iter=2000, random_state=seed).fit(
            selector.transform(xtr_counts), y_tr
        )
        return float((clf.predict(selector.transform(xte_counts)) == y_te).mean())

    honest_acc = score(fit_on_all=False)
    leaky_acc = score(fit_on_all=True)
    return {"honest_acc": honest_acc, "leaky_acc": leaky_acc, "gap": leaky_acc - honest_acc}


def device_line() -> str:
    """An honest one-line environment string for the notebook header."""
    parts = [
        f"python {sys.version.split()[0]}",
        f"numpy {np.__version__}",
        f"platform {platform.system()} (CPU)",
    ]
    try:
        import sklearn

        parts.insert(1, f"sklearn {sklearn.__version__}")
    except ImportError:  # pragma: no cover - sklearn is a hard dependency here
        pass
    return " · ".join(parts)


def main() -> None:
    print(device_line())
    print()

    # --- 1. Naive Bayes by hand, proven against scikit-learn --------------------------------
    by_hand = nb_by_hand()
    neg_sk, pos_sk = nb_sklearn_joint_log_lik()
    print("Worked example 1 -- multinomial NB by hand vs sklearn (doc = 'great fun'):")
    print(f"  by hand : logp_pos={by_hand.logp_pos:.4f}  logp_neg={by_hand.logp_neg:.4f}"
          f"  -> {'pos' if by_hand.predicted else 'neg'}")
    print(f"  sklearn : jll_pos ={pos_sk:.4f}  jll_neg ={neg_sk:.4f}"
          f"  -> {'pos' if pos_sk > neg_sk else 'neg'}")
    match = math.isclose(by_hand.logp_pos, pos_sk, abs_tol=1e-4) and math.isclose(
        by_hand.logp_neg, neg_sk, abs_tol=1e-4
    )
    assert match, "by-hand NB must equal sklearn's joint log-likelihood"
    print(f"  match: {match}\n")

    # --- 1b. LogReg on the same tiny corpus: a calmer probability than NB -------------------
    lr_pos = logreg_tiny_pos_proba()
    print("Worked example 2 -- TF-IDF + LogReg on the same tiny corpus (doc = 'great fun'):")
    assert lr_pos > 0.5, "LogReg should also call 'great fun' positive"
    print(f"  P(pos | 'great fun') = {lr_pos:.3f}  -> pos "
          f"(calmer than NB's effective ~0.9)\n")

    # --- 2. The measured ladder: NB vs LogReg vs SVM ----------------------------------------
    results, lr, x_te, y_te = run_ladder_comparison()
    print("Worked example 2/3 -- balanced synthetic sentiment (720 train / 480 test):")
    assert results[0].name == "Multinomial NB"
    nb_acc, lr_acc, svm_acc = (r.accuracy for r in results)
    assert lr_acc > nb_acc, "LogReg should edge out NB on this corpus"
    for r in results:
        line = (f"  {r.name:<18} acc={r.accuracy:.3f}  macroF1={r.macro_f1:.3f}"
                f"  microF1={r.micro_f1:.3f}")
        if r.pr_auc is not None:
            line += f"  PR-AUC={r.pr_auc:.3f}  ROC-AUC={r.roc_auc:.3f}"
        print(line)
    print(f"  confusion (LogReg) [rows=actual, cols=pred]:\n{results[1].cm}\n")

    # --- 3. Precision/recall/F1 from the confusion-matrix counts ----------------------------
    cm = results[1].cm
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    p, r, f = precision_recall_f1_from_counts(tp, fp, fn)
    print("Worked example -- precision/recall/F1 from counts (LogReg, positive class):")
    print(f"  TP={tp} FP={fp} FN={fn} TN={tn}  ->  P={p:.3f}  R={r:.3f}  F1={f:.3f}\n")

    # --- 4. Threshold sweep ------------------------------------------------------------------
    s = lr.scores(x_te)
    sweep = threshold_sweep(s, y_te)
    best_idx = int(np.argmax(sweep["f1"]))
    default_idx = int(np.argmin(np.abs(sweep["thresholds"] - 0.5)))
    print("Worked example -- threshold sweep (precision/recall trade-off):")
    print(f"  at thr=0.50 : P={sweep['precision'][default_idx]:.3f}"
          f"  R={sweep['recall'][default_idx]:.3f}  F1={sweep['f1'][default_idx]:.3f}")
    print(f"  best F1 thr ={sweep['thresholds'][best_idx]:.2f} : "
          f"P={sweep['precision'][best_idx]:.3f}  R={sweep['recall'][best_idx]:.3f}"
          f"  F1={sweep['f1'][best_idx]:.3f}\n")

    # --- 5. Imbalance: accuracy lies, F1/PR-AUC don't ---------------------------------------
    imb = run_imbalance_demo()
    print("Worked example 4 -- imbalance (8% positive): accuracy lies, F1/PR-AUC don't")
    print(f"  positive rate in test: {imb['pos_rate']:.3f}")
    print(f"  ALWAYS-NEGATIVE  acc={imb['majority_acc']:.3f}  macroF1={imb['majority_f1']:.3f}"
          f"  PR-AUC={imb['majority_pr_auc']:.3f}   <- high accuracy, useless model")
    print(f"  LogReg(balanced) acc={imb['model_acc']:.3f}  macroF1={imb['model_f1']:.3f}"
          f"  PR-AUC={imb['model_pr_auc']:.3f}")
    assert imb["majority_acc"] > 0.85, "majority classifier should look deceptively accurate"
    assert imb["model_f1"] > imb["majority_f1"], "real model must win on macro-F1"
    assert imb["model_pr_auc"] > imb["majority_pr_auc"], "real model must win on PR-AUC"
    print("  -> accuracy ranks the useless model on top; macro-F1 and PR-AUC correctly don't.")


if __name__ == "__main__":
    main()
