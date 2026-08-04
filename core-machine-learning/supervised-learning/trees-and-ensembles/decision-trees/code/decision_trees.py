"""Decision trees on REAL datasets, built from scratch and VERIFIED against scikit-learn — the module.

This is not a toy. Every number the chapter, the figures, and the notebook show is produced here from
a real pipeline (``numpy`` + ``scikit-learn``) on datasets that ship with scikit-learn (no download):

  * **Iris** (150 flowers, 4 measurements, 3 species) — for the from-scratch tree, the sklearn match,
    the learned-tree picture, and the 2-feature decision boundary. Its two petal measurements separate
    the species almost perfectly with a handful of axis-aligned cuts, so a shallow tree is both accurate
    and readable — ideal for *seeing* what a tree computes.
  * **Breast Cancer Wisconsin** (569 tumours, 30 measurements, benign/malignant) — noisier and
    higher-dimensional, so a deep tree memorizes the training split (accuracy → 1.0) while held-out
    accuracy plateaus. That gap *is* overfitting, and it is where depth control and pruning earn their
    keep. Also the substrate for the feature-importance and MDI-vs-permutation-bias demos.
  * **Diabetes** (442 patients, 10 measurements, continuous disease-progression target) — one real
    feature (``bmi``) fit by a regression tree, to show the piecewise-constant staircase and why a deep
    regression tree overfits the noise.

What this module measures (all real, all reproducible from the seed):

  * **A decision tree grown from scratch.** ``DecisionTreeScratch`` implements Gini impurity, the greedy
    best-split search over every feature and every candidate threshold, recursive ``fit``, and ``predict``,
    with a ``max_depth`` stop — the actual CART growing algorithm, ~120 lines, no shortcuts.

  * **The from-scratch tree VERIFIED against scikit-learn.** Grown with the same criterion (Gini) and the
    same ``max_depth`` on Iris, our tree and ``DecisionTreeClassifier`` reach the **same test accuracy** and
    **agree on the vast majority of predictions**. We report both, and are honest that trees can tie-break a
    tie in impurity differently (sklearn picks among equal-gain splits by its own rule), so a perfect
    prediction match is not guaranteed — equal accuracy plus near-total prediction agreement is the proof.

  * **The overfitting curve, in real numbers.** A depth sweep on Breast Cancer: training accuracy climbs to
    1.0 while validation accuracy peaks at a shallow depth and then flattens/declines. We pick the depth
    with the best *validation* accuracy — the same sweet spot the U-curve of the overfitting chapter shows.

  * **Feature importance, and its high-cardinality bias.** Mean-decrease-in-impurity (MDI) importances on
    Breast Cancer, then the trap: inject a pure-noise, near-unique-valued column and watch MDI rank it
    highly (more thresholds ⇒ more chances to luck into a spurious split) while permutation importance
    correctly pins it at ~0.

  * **Regression as variance reduction.** A regression tree on one real feature (diabetes ``bmi``): a shallow
    tree is a coarse staircase that generalizes; a deep tree is a fine staircase that chases noise.

Everything is seeded and CPU-only; runs standalone in a couple of seconds::

    python decision_trees.py
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# ---- Constants (hoisted; no magic numbers inline) ----------------------------------------------
RANDOM_STATE = 42  # the one seed fixing every split and every estimator's randomness
TEST_SIZE = 0.30  # fraction held out as a test/validation set, stratified where it applies
IRIS_2D_FEATURES = ("petal length (cm)", "petal width (cm)")  # the clean 2-feature slice for the boundary
MATCH_MAX_DEPTH = 3  # depth at which we compare from-scratch vs scikit-learn (readable + accurate on Iris)
DEPTH_SWEEP = (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)  # tree depths probed for the overfitting curve
NOISE_COL_NAME = "random_id"  # the injected pure-noise, near-unique-valued column (MDI trap)
PERM_REPEATS = 20  # permutation-importance shuffles per feature (more = smoother estimate)
REG_FEATURE = "bmi"  # the single real diabetes feature fit by the regression-tree staircase demo
REG_SHALLOW_DEPTH = 2  # a coarse regression tree that generalizes
REG_DEEP_DEPTH = 8  # a fine regression tree that overfits the noise
PREDICTION_AGREEMENT_FLOOR = 0.90  # from-scratch vs sklearn must agree on at least this share of predictions


# ============================ 1. real data =====================================================
@dataclass
class Dataset:
    """A real, stratified train/test split plus feature and class names."""

    x_train: NDArray[np.float64]
    x_test: NDArray[np.float64]
    y_train: NDArray[np.int64]
    y_test: NDArray[np.int64]
    feature_names: list[str]
    class_names: list[str]
    name: str


def load_iris_2d(*, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE) -> Dataset:
    """Iris restricted to its two petal measurements — the canonical 2-D, human-readable tree problem.

    Petal length and petal width separate the three species almost perfectly with a few axis-aligned
    cuts, so the learned tree is small enough to draw and the decision boundary is a clean staircase of
    rectangles. Stratifying keeps the 50/50/50 class balance in both splits.
    """
    data = load_iris()
    idx = [list(data.feature_names).index(f) for f in IRIS_2D_FEATURES]
    x = data.data[:, idx]
    x_train, x_test, y_train, y_test = train_test_split(
        x, data.target, test_size=test_size, random_state=seed, stratify=data.target
    )
    return Dataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=list(IRIS_2D_FEATURES),
        class_names=list(data.target_names),
        name="Iris (petal length, petal width)",
    )


def load_iris_full(*, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE) -> Dataset:
    """The full 4-feature Iris — for the learned-tree picture with real thresholds on all measurements."""
    data = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=seed, stratify=data.target
    )
    return Dataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=list(data.feature_names),
        class_names=list(data.target_names),
        name="Iris (all 4 features)",
    )


def load_cancer(*, test_size: float = TEST_SIZE, seed: int = RANDOM_STATE) -> Dataset:
    """Breast Cancer Wisconsin — noisier, 30-D; where deep trees overfit and importance matters."""
    data = load_breast_cancer()
    x_train, x_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=test_size, random_state=seed, stratify=data.target
    )
    return Dataset(
        x_train=x_train,
        x_test=x_test,
        y_train=y_train.astype(np.int64),
        y_test=y_test.astype(np.int64),
        feature_names=list(data.feature_names),
        class_names=list(data.target_names),
        name="Breast Cancer Wisconsin",
    )


# ============================ 2. impurity: Gini and entropy =====================================
def gini(y: NDArray[np.int64]) -> float:
    """Gini impurity ``1 - sum_c p_c^2`` — the probability two random draws from the node disagree.

    0 when the node is pure (one class has ``p=1``); maximal when classes are evenly mixed (0.5 for two
    classes, at 50/50). This is scikit-learn's CART default and needs no logarithms.
    """
    if y.size == 0:
        return 0.0
    counts = np.bincount(y)
    p = counts / y.size
    return float(1.0 - np.sum(p**2))


def entropy(y: NDArray[np.int64]) -> float:
    """Shannon entropy ``-sum_c p_c log2 p_c`` (bits) — the average surprise of the node's label.

    0 when pure, 1 bit at 50/50 for two classes. Quinlan's ID3/C4.5 criterion; near-identical splits to
    Gini in practice, at the cost of a logarithm per class.
    """
    if y.size == 0:
        return 0.0
    counts = np.bincount(y)
    p = counts[counts > 0] / y.size  # drop empty classes: 0 * log 0 = 0
    h = float(-np.sum(p * np.log2(p)))
    return h if h > 0.0 else 0.0  # guard the -0.0 that log2(1)=0 produces on a pure node


def information_gain(
    y: NDArray[np.int64], mask: NDArray[np.bool_], *, criterion: str = "gini"
) -> float:
    """Impurity reduction of a split: ``I(parent) - (n_L/n) I(left) - (n_R/n) I(right)``.

    ``mask`` is the boolean "goes left" indicator (a candidate ``feature <= threshold`` test). The
    weighting by child size is what stops the tree from "improving" a node by carving off a tiny pure
    sliver: a child only helps in proportion to how many samples it actually cleans up. Maximizing this
    gain is identical to minimizing the weighted child impurity (the parent term is a constant).
    """
    impurity = gini if criterion == "gini" else entropy
    n = y.size
    y_left, y_right = y[mask], y[~mask]
    if y_left.size == 0 or y_right.size == 0:
        return 0.0  # a split that sends everyone one way separates nothing
    w_left = y_left.size / n
    w_right = y_right.size / n
    return impurity(y) - (w_left * impurity(y_left) + w_right * impurity(y_right))


# ============================ 3. the greedy best-split search ===================================
@dataclass
class Split:
    """The best (feature, threshold) test found at a node, and the impurity gain it buys."""

    feature: int  # column index of the feature being tested
    threshold: float  # the ``feature <= threshold`` cut point (a midpoint between sorted values)
    gain: float  # impurity reduction this split achieves (0 if no useful split exists)


def _candidate_thresholds(column: NDArray[np.float64]) -> NDArray[np.float64]:
    """Midpoints between adjacent *distinct* sorted values — the only thresholds worth testing.

    Between two consecutive distinct values every cut gives the identical partition, so one midpoint per
    gap suffices: ``O(n)`` candidates per feature instead of infinitely many. This is exactly how CART
    enumerates numeric splits.
    """
    uniq = np.unique(column)
    if uniq.size < 2:
        return np.empty(0)  # a constant column cannot be split
    return (uniq[:-1] + uniq[1:]) / 2.0


def best_split(
    x: NDArray[np.float64], y: NDArray[np.int64], *, criterion: str = "gini"
) -> Split | None:
    """Search every feature and every candidate threshold; return the split with the highest gain.

    This is the greedy heart of tree growing: it looks one step ahead, scores each ``feature <= t`` test by
    its impurity gain, and keeps the single best. It is ``O(n_features * n_samples)`` gain evaluations per
    node (sorting to get the thresholds dominates). Returns ``None`` if no split yields positive gain (the
    node is pure, constant, or genuinely unsplittable) — the signal to make a leaf.
    """
    best = Split(feature=-1, threshold=0.0, gain=0.0)
    for feature in range(x.shape[1]):
        column = x[:, feature]
        for threshold in _candidate_thresholds(column):
            mask = column <= threshold
            gain = information_gain(y, mask, criterion=criterion)
            if gain > best.gain:
                best = Split(feature=feature, threshold=float(threshold), gain=gain)
    return best if best.feature >= 0 else None


# ============================ 4. the tree, grown from scratch ===================================
@dataclass
class Node:
    """One node of the tree: either an internal test (``feature``/``threshold``/children) or a leaf.

    A leaf carries the predicted class (majority of the training samples that reached it). An internal
    node carries the split test and its left/right subtrees. ``n_samples`` and ``impurity`` are kept for
    the readable print-out and to mirror what scikit-learn stores per node.
    """

    prediction: int  # majority class of the training samples in this node (used at a leaf)
    n_samples: int  # how many training samples reached this node
    impurity: float  # the node's impurity (Gini/entropy) before any split
    feature: int | None = None  # split feature (None at a leaf)
    threshold: float | None = None  # split threshold (None at a leaf)
    left: Node | None = None  # subtree for ``feature <= threshold``
    right: Node | None = None  # subtree for ``feature >  threshold``

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


class DecisionTreeScratch:
    """A from-scratch CART classification tree: Gini/entropy, greedy binary splits, ``max_depth`` stop.

    Grown top-down and greedily — at each node find the best ``feature <= threshold`` test, send samples
    left/right, and recurse — exactly the algorithm scikit-learn's ``DecisionTreeClassifier`` implements
    (minus the C-speed and the extra stopping knobs). Stops when a node is pure, too small, cannot be
    split, or hits ``max_depth``. Verified against scikit-learn in :func:`verify_against_sklearn`.
    """

    def __init__(self, *, max_depth: int = 3, min_samples_split: int = 2, criterion: str = "gini") -> None:
        if criterion not in {"gini", "entropy"}:
            raise ValueError(f"criterion must be 'gini' or 'entropy', got {criterion!r}")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root: Node | None = None
        self._n_classes = 0

    def _impurity(self, y: NDArray[np.int64]) -> float:
        return gini(y) if self.criterion == "gini" else entropy(y)

    def _majority(self, y: NDArray[np.int64]) -> int:
        """Most frequent class, ties broken by lowest class index (matches numpy.bincount.argmax)."""
        return int(np.bincount(y, minlength=self._n_classes).argmax())

    def _build(self, x: NDArray[np.float64], y: NDArray[np.int64], depth: int) -> Node:
        node = Node(prediction=self._majority(y), n_samples=y.size, impurity=self._impurity(y))
        # Stopping rules: pure node, depth cap, or too few samples to split.
        if node.impurity == 0.0 or depth >= self.max_depth or y.size < self.min_samples_split:
            return node
        split = best_split(x, y, criterion=self.criterion)
        if split is None:  # no split yields positive gain -> leaf
            return node
        mask = x[:, split.feature] <= split.threshold
        node.feature = split.feature
        node.threshold = split.threshold
        node.left = self._build(x[mask], y[mask], depth + 1)
        node.right = self._build(x[~mask], y[~mask], depth + 1)
        return node

    def fit(self, x: NDArray[np.float64], y: NDArray[np.int64]) -> DecisionTreeScratch:
        """Grow the tree from training data. ``y`` must be integer class labels ``0..K-1``."""
        self._n_classes = int(y.max()) + 1
        self.root = self._build(x, y, depth=0)
        return self

    def _predict_one(self, row: NDArray[np.float64]) -> int:
        node = self.root
        assert node is not None, "call fit before predict"
        while not node.is_leaf:
            assert node.feature is not None and node.threshold is not None
            node = node.left if row[node.feature] <= node.threshold else node.right  # type: ignore[assignment]
            assert node is not None
        return node.prediction

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.int64]:
        """Route every row down the tree to its leaf and return the leaf's majority-class prediction."""
        return np.array([self._predict_one(row) for row in x], dtype=np.int64)

    def score(self, x: NDArray[np.float64], y: NDArray[np.int64]) -> float:
        """Accuracy = fraction of rows whose predicted class matches the true label."""
        return float(np.mean(self.predict(x) == y))

    def n_leaves(self) -> int:
        """Count the leaves (for reporting tree size)."""

        def _count(node: Node | None) -> int:
            if node is None:
                return 0
            return 1 if node.is_leaf else _count(node.left) + _count(node.right)

        return _count(self.root)

    def describe(self, feature_names: list[str], class_names: list[str]) -> str:
        """Render the tree as indented ``feature <= threshold`` text — the readable flowchart."""
        lines: list[str] = []

        def _walk(node: Node | None, depth: int, label: str) -> None:
            if node is None:
                return
            pad = "    " * depth
            if node.is_leaf:
                lines.append(f"{pad}{label}predict '{class_names[node.prediction]}'  "
                             f"(n={node.n_samples}, {self.criterion}={node.impurity:.3f})")
                return
            assert node.feature is not None and node.threshold is not None
            lines.append(f"{pad}{label}{feature_names[node.feature]} <= {node.threshold:.3f}?  "
                         f"({self.criterion}={node.impurity:.3f}, n={node.n_samples})")
            _walk(node.left, depth + 1, "├─ yes: ")
            _walk(node.right, depth + 1, "└─ no:  ")

        _walk(self.root, 0, "")
        return "\n".join(lines)


# ============================ 5. verify from-scratch == scikit-learn ============================
@dataclass
class MatchReport:
    """The from-scratch vs scikit-learn comparison on identical data at identical depth."""

    scratch_test_acc: float
    sklearn_test_acc: float
    prediction_agreement: float  # fraction of test rows on which the two trees agree
    max_depth: int


def verify_against_sklearn(data: Dataset, *, max_depth: int = MATCH_MAX_DEPTH) -> MatchReport:
    """Grow our tree and scikit-learn's on the same data at the same depth; compare acc + predictions.

    Both use Gini and the same ``max_depth``. We report test accuracy for each and the fraction of test
    predictions they agree on. We do **not** demand an exact prediction match: when two candidate splits
    tie on gain, CART implementations break the tie by their own rule, so a different-but-equally-good
    tree is legitimate. Equal test accuracy plus near-total prediction agreement is the honest proof that
    our from-scratch growing algorithm is the real thing.
    """
    scratch = DecisionTreeScratch(max_depth=max_depth, criterion="gini").fit(data.x_train, data.y_train)
    sk = DecisionTreeClassifier(max_depth=max_depth, criterion="gini", random_state=RANDOM_STATE)
    sk.fit(data.x_train, data.y_train)
    scratch_pred = scratch.predict(data.x_test)
    sk_pred = sk.predict(data.x_test)
    return MatchReport(
        scratch_test_acc=scratch.score(data.x_test, data.y_test),
        sklearn_test_acc=float(sk.score(data.x_test, data.y_test)),
        prediction_agreement=float(np.mean(scratch_pred == sk_pred)),
        max_depth=max_depth,
    )


# ============================ 6. the overfitting depth sweep ====================================
@dataclass
class DepthSweep:
    """Train and validation accuracy at each probed depth, and the depth with the best validation acc."""

    depths: list[int]
    train_acc: list[float]
    val_acc: list[float]
    best_depth: int  # the depth maximizing validation accuracy — the sweet spot
    best_val_acc: float


def depth_sweep(data: Dataset, *, depths: tuple[int, ...] = DEPTH_SWEEP) -> DepthSweep:
    """Grow scikit-learn trees of increasing depth and record train vs validation accuracy.

    The signature of overfitting: training accuracy marches to 1.0 as the tree is allowed to memorize,
    while validation accuracy rises, peaks at a shallow depth, then flattens or declines. We pick the
    depth with the best *validation* accuracy — the honest way to choose complexity, and the same peak the
    bias-variance U-curve shows.
    """
    train_acc: list[float] = []
    val_acc: list[float] = []
    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, criterion="gini", random_state=RANDOM_STATE)
        clf.fit(data.x_train, data.y_train)
        train_acc.append(float(clf.score(data.x_train, data.y_train)))
        val_acc.append(float(clf.score(data.x_test, data.y_test)))
    best_i = int(np.argmax(val_acc))
    return DepthSweep(
        depths=list(depths),
        train_acc=train_acc,
        val_acc=val_acc,
        best_depth=depths[best_i],
        best_val_acc=val_acc[best_i],
    )


# ============================ 7. feature importance + the MDI bias ==============================
@dataclass
class ImportanceReport:
    """MDI importances on the real features, plus the MDI-vs-permutation result on an injected noise column."""

    feature_names: list[str]
    mdi: NDArray[np.float64]  # mean-decrease-in-impurity importance per real feature (sums to 1)
    top_features: list[tuple[str, float]]  # the highest-MDI real features, name + importance
    noise_mdi: float  # MDI importance the tree gives the pure-noise, near-unique column
    noise_permutation: float  # permutation importance of that same noise column (should be ~0)
    signal_mdi: float  # MDI of the weak-but-real informative feature in the same trap experiment
    signal_permutation: float  # permutation importance of that informative feature


def feature_importance(data: Dataset, *, max_depth: int = 5) -> ImportanceReport:
    """MDI importances on the real dataset, then the high-cardinality trap measured on held-out data.

    Part 1: fit a tree and read scikit-learn's ``feature_importances_`` (MDI) — the total impurity
    decrease each real feature contributes, normalized to sum to 1.

    Part 2: the trap. We build a tiny controlled experiment — one weak-but-real informative binary feature
    and one pure-noise, near-unique-valued column — and show MDI ranking the noise column *above* the real
    signal (its many distinct values give it more thresholds to luck into a spurious split), while
    permutation importance, measured on a held-out set, correctly scores the noise ~0.
    """
    clf = DecisionTreeClassifier(max_depth=max_depth, criterion="gini", random_state=RANDOM_STATE)
    clf.fit(data.x_train, data.y_train)
    mdi = clf.feature_importances_
    order = np.argsort(mdi)[::-1][:6]
    top = [(data.feature_names[i], float(mdi[i])) for i in order]

    rng = np.random.default_rng(RANDOM_STATE)
    n = 2000
    informative = rng.integers(0, 2, size=n).astype(float)  # 2 values -> few thresholds
    noise_flip = rng.random(n) < 0.30  # a weak, noisy signal: informative XOR 30%-noise
    y = np.logical_xor(informative == 1, noise_flip).astype(np.int64)
    random_id = rng.uniform(size=n)  # near-unique -> many thresholds -> PURE NOISE
    x_trap = np.column_stack([informative, random_id])
    xtr, xte, ytr, yte = train_test_split(x_trap, y, test_size=0.5, random_state=RANDOM_STATE)
    trap = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(xtr, ytr)
    perm = permutation_importance(trap, xte, yte, n_repeats=PERM_REPEATS, random_state=RANDOM_STATE)
    return ImportanceReport(
        feature_names=data.feature_names,
        mdi=mdi,
        top_features=top,
        noise_mdi=float(trap.feature_importances_[1]),
        noise_permutation=float(perm.importances_mean[1]),
        signal_mdi=float(trap.feature_importances_[0]),
        signal_permutation=float(perm.importances_mean[0]),
    )


# ============================ 8. regression as variance reduction ===============================
@dataclass
class RegressionStaircase:
    """A regression tree on one real feature: the shallow (generalizing) and deep (overfitting) fits."""

    feature_name: str
    x: NDArray[np.float64]  # the single feature, sorted (for a clean staircase plot)
    y: NDArray[np.float64]  # the continuous target aligned to ``x``
    grid: NDArray[np.float64]  # dense x-grid the fits are evaluated on
    shallow_pred: NDArray[np.float64]  # depth-2 tree prediction on the grid (coarse staircase)
    deep_pred: NDArray[np.float64]  # depth-8 tree prediction on the grid (fine, noise-chasing staircase)
    shallow_depth: int
    deep_depth: int


def regression_staircase(*, feature: str = REG_FEATURE) -> RegressionStaircase:
    """Fit shallow vs deep regression trees to one real diabetes feature and evaluate on a dense grid.

    A regression tree splits to maximize *variance reduction* and predicts the *mean* target in each leaf,
    so its fit is piecewise-constant — a staircase. The shallow tree (depth 2, four steps) captures the
    broad trend; the deep tree (depth 8) chases noise with many tiny steps. Because it can only output
    training-leaf means, a regression tree cannot extrapolate beyond the training range.
    """
    data = load_diabetes()
    j = list(data.feature_names).index(feature)
    x = data.data[:, j]
    y = data.target.astype(np.float64)
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    grid = np.linspace(x_sorted.min(), x_sorted.max(), 400)
    fits: dict[str, NDArray[np.float64]] = {}
    for label, depth in (("shallow", REG_SHALLOW_DEPTH), ("deep", REG_DEEP_DEPTH)):
        reg = DecisionTreeRegressor(max_depth=depth, random_state=RANDOM_STATE)
        reg.fit(x_sorted.reshape(-1, 1), y_sorted)
        fits[label] = reg.predict(grid.reshape(-1, 1))
    return RegressionStaircase(
        feature_name=feature,
        x=x_sorted,
        y=y_sorted,
        grid=grid,
        shallow_pred=fits["shallow"],
        deep_pred=fits["deep"],
        shallow_depth=REG_SHALLOW_DEPTH,
        deep_depth=REG_DEEP_DEPTH,
    )


# ============================ 9. run it all: the printed proof ==================================
def main() -> None:
    """Run every measured experiment and cross-check, printing the results the chapter cites."""
    import sklearn

    print(f"numpy {np.__version__} | scikit-learn {sklearn.__version__}\n")

    # ---- 0. impurity anchors: the numbers the worked examples derive by hand ----
    print("=== 0. Impurity anchors (the worked-example numbers, computed) ===")
    fifty_fifty = np.array([0] * 5 + [1] * 5)
    pure = np.array([0] * 10)
    print(f"  50/50 node : Gini={gini(fifty_fifty):.3f}  entropy={entropy(fifty_fifty):.3f} bits  (max impurity)")
    print(f"  pure node  : Gini={gini(pure):.3f}  entropy={entropy(pure):.3f} bits  (zero impurity)")
    # Worked example 2: parent 9 'yes' / 5 'no' -> children (6/2) and (3/3)
    parent = np.array([1] * 9 + [0] * 5)
    mask_ex2 = np.array([True] * 8 + [False] * 6)  # left = first 8 (6 yes, 2 no), right = last 6 (3/3)
    parent_reordered = np.array([1] * 6 + [0] * 2 + [1] * 3 + [0] * 3)
    gain_ex2 = information_gain(parent_reordered, mask_ex2, criterion="gini")
    print(f"  worked ex.2: parent Gini={gini(parent):.3f} -> split (6/2)&(3/3) -> Gini gain={gain_ex2:.3f}\n")

    # ---- 1. from-scratch tree, and the readable flowchart ----
    iris2d = load_iris_2d()
    scratch = DecisionTreeScratch(max_depth=MATCH_MAX_DEPTH, criterion="gini").fit(iris2d.x_train, iris2d.y_train)
    print(f"=== 1. A decision tree grown FROM SCRATCH on {iris2d.name} (max_depth={MATCH_MAX_DEPTH}) ===")
    print(scratch.describe(iris2d.feature_names, iris2d.class_names))
    print(f"  leaves={scratch.n_leaves()}  train_acc={scratch.score(iris2d.x_train, iris2d.y_train):.3f}  "
          f"test_acc={scratch.score(iris2d.x_test, iris2d.y_test):.3f}\n")

    # ---- 2. verify against scikit-learn ----
    match = verify_against_sklearn(iris2d, max_depth=MATCH_MAX_DEPTH)
    print("=== 2. Verify: from-scratch tree == scikit-learn (same Gini, same max_depth) ===")
    print(f"  from-scratch test accuracy : {match.scratch_test_acc:.3f}")
    print(f"  scikit-learn test accuracy : {match.sklearn_test_acc:.3f}")
    print(f"  they agree on {match.prediction_agreement * 100:.1f}% of test predictions")
    if abs(match.scratch_test_acc - match.sklearn_test_acc) > 1e-9:
        raise AssertionError("from-scratch and scikit-learn test accuracy must match on Iris")
    if match.prediction_agreement < PREDICTION_AGREEMENT_FLOOR:
        raise AssertionError("from-scratch and scikit-learn should agree on the vast majority of predictions")
    print("  -> same accuracy, near-total prediction agreement: the growing algorithm is the real thing.\n")

    # ---- 3. the overfitting depth sweep ----
    cancer = load_cancer()
    sweep = depth_sweep(cancer)
    print(f"=== 3. The overfitting curve on {cancer.name} (depth sweep) ===")
    print(f"  {'depth':>6}{'train acc':>12}{'val acc':>10}")
    for d, tr, va in zip(sweep.depths, sweep.train_acc, sweep.val_acc):
        mark = "   <- best val" if d == sweep.best_depth else ""
        print(f"  {d:>6}{tr:>12.3f}{va:>10.3f}{mark}")
    print(f"  best validation accuracy {sweep.best_val_acc:.3f} at max_depth={sweep.best_depth}; "
          f"deepest tree memorizes train to {sweep.train_acc[-1]:.3f}")
    if sweep.train_acc[-1] < 0.999:
        raise AssertionError("the deepest tree should memorize the training set (~1.0 train accuracy)")
    print("  -> train -> 1.0 while val plateaus: the gap IS overfitting. Limit depth (or prune).\n")

    # ---- 4. feature importance and the high-cardinality bias ----
    imp = feature_importance(cancer)
    print(f"=== 4. Feature importance on {cancer.name}, and the MDI high-cardinality trap ===")
    print("  top MDI features (impurity decrease):")
    for name, val in imp.top_features:
        print(f"     {name:<26}{val:.3f}")
    print("\n  the trap (weak real signal vs a pure-noise near-unique column):")
    print(f"     MDI         : signal={imp.signal_mdi:.3f}   random_id={imp.noise_mdi:.3f}  <- noise ranked HIGHER")
    print(f"     permutation : signal={imp.signal_permutation:.3f}   random_id={imp.noise_permutation:.3f}  "
          f"<- noise correctly ~0")
    if imp.noise_mdi <= imp.signal_mdi:
        raise AssertionError("MDI should be fooled: the noise column should outrank the real signal")
    if imp.noise_permutation > 0.02:
        raise AssertionError("permutation importance should score the pure-noise column ~0")
    print("  -> never feature-select from raw MDI with high-cardinality features; use permutation/SHAP.\n")

    # ---- 5. regression as variance reduction ----
    reg = regression_staircase()
    print(f"=== 5. Regression tree on diabetes '{reg.feature_name}' (piecewise-constant staircase) ===")
    print(f"  shallow tree (depth {reg.shallow_depth}) -> {len(np.unique(reg.shallow_pred))} distinct leaf values (coarse steps)")
    print(f"  deep tree    (depth {reg.deep_depth}) -> {len(np.unique(reg.deep_pred))} distinct leaf values (fine, noise-chasing steps)")
    print("  each step is one leaf predicting the MEAN target of its interval; deep = overfit, shallow = generalizes.")


if __name__ == "__main__":
    main()
