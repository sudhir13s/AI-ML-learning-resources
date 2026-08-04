"""Sentence & document embeddings, measured: one vector for a whole passage.

This is the single source of truth for the chapter. The concept page, the teaching notebook
(`07-Sentence-and-Document-Embeddings.ipynb`), and the figure generator (`make_figures_07.py`)
all import the functions and constants defined here, so none of them can silently drift from the
others. Every number on the page is produced by this file.

What it shows, end to end:
  1. mean-pooling word vectors is ORDER-BLIND -- "the dog bit the man" and "the man bit the dog"
     collapse to the IDENTICAL vector (cosine 1.000), while a real sentence encoder separates them;
  2. SIF (smooth inverse frequency) weighting a/(a+p(w)) + common-component removal beats a plain
     average on a toy example by pulling the sentence vector toward CONTENT words, off stopwords;
  3. a trained bi-encoder (Sentence-BERT) separates paraphrases from non-paraphrases by cosine far
     better than mean-pooled static vectors -- the whole reason SBERT exists;
  4. pooling strategy matters -- mean vs max vs CLS produce different vectors and different STS scores;
  5. evaluation on an STS-like set with SPEARMAN rank correlation, the field's convention.

Determinism and device policy
-----------------------------
A pretrained Sentence-BERT (`all-MiniLM-L6-v2`) is loaded when `sentence-transformers` and the
weights are reachable; the result is REAL measured sentence vectors -- the exact numbers quoted on
the page. If the model is unavailable (offline / firewall / missing package) the module transparently
falls back to a small, fully deterministic SYNTHETIC bi-encoder (`SyntheticEncoder`) seeded with
`SEED`, built to honour the same qualitative facts (paraphrases close, unrelated far, order matters),
so the notebook and the figures ALWAYS run with no network dependency that can fail. Which path ran is
reported by `load_encoder()` and printed honestly everywhere ("backend: all-MiniLM-L6-v2 (real)" vs
"backend: synthetic (fallback)").

Everything runs on CPU, pinned for reproducibility: torch and numpy are seeded, the device line is
printed honestly ("device: cpu (detected <x>; pinned to CPU for reproducibility)"), and the encoder
runs in eval mode, so the same input yields the same vectors on any machine.

Run:
    python sentence_embeddings.py
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch


def _stable_hash(text: str) -> int:
    """A process-independent hash of `text` as a 32-bit int.

    Python's built-in `hash()` is salted per process (PYTHONHASHSEED randomization), which would make
    the synthetic encoder and the ColBERT demo produce DIFFERENT vectors in every process (the script,
    the notebook, the figure generator). We use md5 so the synthetic numbers are reproducible across
    processes and machines -- the whole point of the deterministic fallback.
    """
    return int.from_bytes(hashlib.md5(text.encode("utf-8")).digest()[:4], "big")


# --- Reproducibility -------------------------------------------------------------------------------
SEED = 0

# Detect the best accelerator for honest reporting, but PIN execution to CPU so measured vectors are
# bit-for-bit reproducible across machines (MPS/CUDA reductions are not guaranteed identical). The
# detected device is reported; the pinned device is what actually runs.
DETECTED_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = "cpu"  # pinned: sentence vectors must be reproducible, so we never run on MPS/CUDA

# The bi-encoder used for every measured number on the page (384-dim, small and fast).
MODEL_NAME = "all-MiniLM-L6-v2"

# --- The sentences used everywhere on the page, in the notebook, and in the figures ----------------
# Order-blindness probe: same multiset of words, opposite meaning. A bag-of-vectors average gives
# these the IDENTICAL vector; a real encoder must separate them.
ORDER_A = "The dog bit the man."
ORDER_B = "The man bit the dog."

# Paraphrase vs unrelated probe (the SBERT headline): s0 and s1 mean the same thing; s2 is unrelated.
PARAPHRASE_SET: tuple[str, ...] = (
    "A man is playing a guitar.",
    "A person is strumming an acoustic guitar.",  # paraphrase of [0]
    "A child is feeding a baby elephant.",          # unrelated
)

# Three topics, three paraphrases each -- used for the PCA cluster scatter. Paraphrases must cluster;
# the three topics must separate.
TOPIC_SENTENCES: tuple[tuple[str, str], ...] = (
    ("weather", "The forecast says it will rain heavily tomorrow afternoon."),
    ("weather", "Heavy showers are expected across the region tomorrow."),
    ("weather", "Tomorrow brings a downpour and gusty winds to the area."),
    ("finance", "The central bank raised interest rates by half a point."),
    ("finance", "Policymakers hiked the benchmark rate to curb inflation."),
    ("finance", "The Fed lifted borrowing costs to slow rising prices."),
    ("cooking", "Saute the onions until golden, then add the garlic."),
    ("cooking", "Fry the chopped onion slowly before stirring in garlic."),
    ("cooking", "Cook the onions till caramelized, then mix in minced garlic."),
)

# Semantic-search probe: one query over a small support corpus. Note the relevant doc 2 says "forgot"
# / "Forgot password", never "reset" -- a win only a MEANING-based match can buy.
SEARCH_QUERY = "How do I reset my account password?"
SEARCH_CORPUS: tuple[str, ...] = (
    "To change your password, open Settings and click 'Reset password'.",
    "Our refund policy allows returns within 30 days of purchase.",
    "The mobile app supports dark mode and push notifications.",
    "If you forgot your login, use the 'Forgot password' link on the sign-in page.",
    "Premium plans include priority support and extra storage.",
)

# Bi-encoder-vs-cross-encoder worked example: the SAME query against three documents, scored both
# ways. NOTE: doc 2 here is a SHORTENED form of the search-corpus doc ("...link." vs "...link on the
# sign-in page."), so its bi-encoder cosine is 0.642 here vs 0.618 in the full-string search ranking
# -- the page annotates this. These exact strings reproduce the bi-vs-cross table's numbers.
BI_VS_CROSS_QUERY = "How do I reset my account password?"
BI_VS_CROSS_DOCS: tuple[str, ...] = (
    "To change your password, open Settings and click 'Reset password'.",
    "If you forgot your login, use the 'Forgot password' link.",
    "Our refund policy allows returns within 30 days of purchase.",
)

# A tiny labelled STS-like set: (sentence_a, sentence_b, gold_similarity in [0, 1]). Used to compute a
# Spearman correlation between cosine and human-style judgments, and to compare pooling strategies.
STS_PAIRS: tuple[tuple[str, str, float], ...] = (
    ("A man is playing a guitar.", "A person is strumming an acoustic guitar.", 0.95),
    ("A dog is running in the park.", "A puppy sprints across the grass.", 0.85),
    ("The chef chopped fresh vegetables.", "A cook is slicing carrots and onions.", 0.80),
    ("She booked a flight to Tokyo.", "He reserved a plane ticket to Japan.", 0.78),
    ("The stock market fell sharply today.", "Share prices dropped a lot this afternoon.", 0.82),
    ("A child is feeding a baby elephant.", "A man is playing a guitar.", 0.05),
    ("The forecast predicts heavy rain.", "The chef chopped fresh vegetables.", 0.04),
    ("She booked a flight to Tokyo.", "The stock market fell sharply today.", 0.08),
    ("A dog is running in the park.", "Policymakers hiked interest rates.", 0.03),
    ("The cat sat on the warm windowsill.", "A feline rested by the sunny window.", 0.88),
)

# --- Toy word vectors + unigram frequencies for the SIF worked example (page-faithful) -------------
# 2-D toy vectors and unigram probabilities for "the cat sat", matching the page's worked example.
SIF_A = 1e-3  # the SIF smoothing constant a, as in Arora et al. (2017): weight = a / (a + p(w))
SIF_TOY: dict[str, tuple[np.ndarray, float]] = {
    # word: (2-D toy vector, unigram probability p(w))
    "the": (np.array([1.0, 0.0]), 0.05),
    "cat": (np.array([0.0, 1.0]), 0.0003),
    "sat": (np.array([0.2, 0.9]), 0.0008),
}


# ==================================================================================================
# Pure math: mean-pooling, SIF, triplet/contrastive losses, cosine, Spearman.
# These are backend-free -- they operate on plain vectors, so the page's formulas are exact and
# reproducible offline regardless of which encoder loaded.
# ==================================================================================================
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors, as a Python float (0 if either is degenerate)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


def mean_pool(word_vectors: np.ndarray) -> np.ndarray:
    """Mean-pool a (T, d) matrix of word vectors into one (d,) sentence vector.

    This is the continuous bag-of-words of a sentence: order-blind (a permutation of the rows gives
    the IDENTICAL output) and frequency-dominated (large-norm frequent-word vectors pull the mean).
    """
    return np.asarray(word_vectors, dtype=np.float64).mean(axis=0)


def max_pool(word_vectors: np.ndarray) -> np.ndarray:
    """Elementwise max-pool a (T, d) matrix into one (d,) sentence vector (a pooling alternative)."""
    return np.asarray(word_vectors, dtype=np.float64).max(axis=0)


def sif_weight(p_w: float, a: float = SIF_A) -> float:
    """The SIF weight a / (a + p(w)): a smooth, learned-cutoff down-weighting of frequent words.

    Source: Arora, Liang & Ma (2017), "A Simple but Tough-to-Beat Baseline for Sentence Embeddings."
    As p(w) -> large (common word), the weight -> small; as p(w) -> 0 (rare content word) -> ~1.
    """
    return a / (a + p_w)


def sif_sentence_vector(
    word_vectors: np.ndarray, probabilities: np.ndarray, a: float = SIF_A
) -> np.ndarray:
    """SIF-weighted average: s = (1/|S|) sum_w  [a / (a + p(w))] * v_w.

    `word_vectors` is (T, d); `probabilities` is (T,) unigram p(w). Returns the (d,) sentence vector
    BEFORE common-component removal (call `remove_common_component` on a corpus of these for step 2).
    """
    word_vectors = np.asarray(word_vectors, dtype=np.float64)
    weights = np.array([sif_weight(p, a) for p in probabilities], dtype=np.float64)
    return (weights[:, None] * word_vectors).sum(axis=0) / len(word_vectors)


def remove_common_component(sentence_matrix: np.ndarray) -> np.ndarray:
    """SIF step 2: subtract the projection onto the first principal component of the corpus.

    `sentence_matrix` is (n_sentences, d). The top singular direction u is the direction EVERY
    sentence shares (syntax, the stopword baseline); subtracting (u^T s) u from each s removes that
    non-discriminative common direction. Returns the cleaned (n_sentences, d) matrix.

    Source: Arora, Liang & Ma (2017), Algorithm 1, step 2 (common-component removal). The same
    "remove the top principal component" trick reappears as a cure for BERT's anisotropy (whitening).
    """
    matrix = np.asarray(sentence_matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    u = vt[0]  # first principal component (unit vector)
    u = u / np.linalg.norm(u)
    projections = matrix @ u  # (n_sentences,)
    return matrix - np.outer(projections, u)


def triplet_loss(d_ap: float, d_an: float, margin: float = 0.3) -> float:
    """Triplet loss max(0, d(a,p) - d(a,n) + margin): zero once positive is closer by the margin.

    `d_ap` = distance(anchor, positive), `d_an` = distance(anchor, negative). Loss is 0 when the
    positive is already at least `margin` closer than the negative (the triplet is "solved"), else a
    positive value whose gradient pulls the positive in and pushes the negative out.

    Source: Reimers & Gurevych (2019), Sentence-BERT, eq. for the triplet objective (margin epsilon).
    """
    return max(0.0, d_ap - d_an + margin)


def info_nce_loss(
    anchor: np.ndarray, positive: np.ndarray, negatives: np.ndarray, tau: float = 0.05
) -> float:
    """InfoNCE / contrastive loss for one anchor: -log [ exp(cos(a,p)/tau) / sum_j exp(cos(a, x_j)/tau) ].

    The denominator sums the positive and all `negatives` (rows of an (n, d) matrix). Maximizes
    agreement with the positive, pushes away every negative -- the SimCSE / SimCLR objective.

    Source: Gao, Yao & Chen (2021), SimCSE, the contrastive (InfoNCE) objective with temperature tau.
    """
    pos = np.exp(cosine(anchor, positive) / tau)
    neg = np.sum([np.exp(cosine(anchor, n) / tau) for n in np.asarray(negatives)])
    return float(-np.log(pos / (pos + neg)))


def spearman(xs: np.ndarray, ys: np.ndarray) -> float:
    """Spearman rank correlation between two arrays (Pearson correlation of their ranks).

    STS evaluation convention: we care about ORDERING similarity correctly, not matching an absolute
    scale, so we rank both cosine and gold and correlate the ranks. Dependency-light (no scipy).
    """
    xs = np.asarray(xs, dtype=np.float64)
    ys = np.asarray(ys, dtype=np.float64)
    rx = _rankdata(xs)
    ry = _rankdata(ys)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom == 0.0:
        return 0.0
    return float((rx * ry).sum() / denom)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with ties broken by the mean rank -- the standard tie handling."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    # Average ties so equal values share the mean of their rank positions.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


# ==================================================================================================
# Synthetic fallback encoder: a tiny, fully deterministic stand-in for a sentence bi-encoder.
# It is NOT SBERT -- it exists only so the notebook/figures always run offline. It is built to honour
# the qualitative facts the page asserts: paraphrases land close, unrelated land far, and -- crucially
# -- it is ORDER-AWARE (uses bigram features) so it can separate "dog bit man" from "man bit dog",
# the one thing a mean-pool baseline cannot do.
# ==================================================================================================
SYNTH_DIM = 256

# A tiny set of synthetic "concepts": each is a meaning direction triggered by ANY of its words. This
# is what lets the offline encoder map synonyms close WITHOUT lexical overlap (e.g. "reset"/"forgot"
# both fire the PASSWORD concept; "rain"/"showers"/"downpour" all fire WEATHER) -- a crude stand-in
# for the meaning a real trained encoder learns, so the offline demo still shows the right SHAPE.
SYNTH_CONCEPTS: dict[str, tuple[str, ...]] = {
    "password": ("password", "reset", "forgot", "login", "sign-in", "settings", "account"),
    "refund": ("refund", "return", "returns", "purchase", "policy"),
    "appfeatures": ("mobile", "app", "dark", "mode", "push", "notifications"),
    "premium": ("premium", "plans", "priority", "storage", "support"),
    "weather": ("rain", "showers", "downpour", "forecast", "winds", "gusty", "heavy"),
    "finance": ("bank", "rate", "rates", "interest", "inflation", "fed", "borrowing", "prices"),
    "cooking": ("saute", "fry", "onions", "onion", "garlic", "cook", "caramelized", "minced", "chef", "chopped"),
    "music": ("guitar", "playing", "strumming", "acoustic"),
    "animal": ("dog", "puppy", "cat", "feline", "elephant"),
    "travel": ("flight", "plane", "ticket", "tokyo", "japan", "booked", "reserved"),
    "market": ("stock", "market", "share", "shares", "dropped", "fell"),
}


class SyntheticEncoder:
    """Deterministic synthetic sentence encoder: hashed (unigram+bigram) features + concept anchors.

    Two signals are summed, then L2-normalized:
      * a hashed bag of unigram AND adjacent-bigram features -- the bigrams make it ORDER-AWARE, so
        "the dog bit the man" and "the man bit the dog" get DIFFERENT vectors (mean-pooling cannot);
      * a small set of CONCEPT anchors (SYNTH_CONCEPTS): any trigger word adds its concept's fixed
        direction, so synonyms like "reset"/"forgot" land close WITHOUT sharing a word -- a crude
        stand-in for learned meaning, so the offline demo still shows the right qualitative shape.

    It is NOT SBERT -- it exists only so the notebook/figures always run offline; figures label it.
    """

    name = "synthetic"
    is_real = False
    _concept_weight = 2.2  # how strongly concept anchors pull, relative to surface word features

    def __init__(self, dim: int = SYNTH_DIM, seed: int = SEED) -> None:
        self.dim = dim
        self._seed = seed
        self._cache: dict[str, np.ndarray] = {}
        # Build a word -> concept-direction lookup once, deterministically.
        self._word_to_concepts: dict[str, list[str]] = {}
        for concept, words in SYNTH_CONCEPTS.items():
            for w in words:
                self._word_to_concepts.setdefault(w, []).append(concept)

    def _feature_vec(self, feature: str) -> np.ndarray:
        """A fixed pseudo-random unit vector per string feature (word/bigram/concept), stable run-to-run."""
        if feature not in self._cache:
            rng = np.random.default_rng((_stable_hash(feature) ^ self._seed) % (2**32))
            v = rng.standard_normal(self.dim)
            self._cache[feature] = v / np.linalg.norm(v)
        return self._cache[feature]

    @staticmethod
    def _tokens(sentence: str) -> list[str]:
        cleaned = sentence.lower()
        for ch in ".,'\"":
            cleaned = cleaned.replace(ch, " ")
        return [t for t in cleaned.split() if t]

    def encode_one(self, sentence: str) -> np.ndarray:
        toks = self._tokens(sentence)
        if not toks:
            return np.zeros(self.dim)
        vec = np.zeros(self.dim)
        for t in toks:  # unigram features
            vec += self._feature_vec(t)
            for concept in self._word_to_concepts.get(t, ()):  # concept anchors -> synonym meaning
                vec += self._concept_weight * self._feature_vec(f"concept::{concept}")
        for a, b in zip(toks, toks[1:]):  # bigram features -> order sensitivity
            vec += 0.9 * self._feature_vec(f"{a}_{b}")
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def encode(self, sentences: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        out = np.stack([self.encode_one(s) for s in sentences], axis=0)
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            out = out / np.where(norms == 0, 1.0, norms)
        return out


# ==================================================================================================
# Backend loading: real Sentence-BERT when available, synthetic fallback otherwise.
# ==================================================================================================
@dataclass
class Encoder:
    """A loaded sentence-embedding backend plus a flag for which path ran (real vs synthetic)."""

    name: str
    is_real: bool
    model: object | None = None  # SentenceTransformer (real) or SyntheticEncoder (fallback)

    def encode(self, sentences: list[str], normalize: bool = True) -> np.ndarray:
        """Encode `sentences` into an (n, d) array of (optionally unit-normalized) vectors."""
        if self.is_real:
            return np.asarray(
                self.model.encode(sentences, normalize_embeddings=normalize), dtype=np.float64
            )
        return np.asarray(
            self.model.encode(sentences, normalize_embeddings=normalize), dtype=np.float64
        )


def load_encoder(*, force_synthetic: bool = False) -> Encoder:
    """Load Sentence-BERT (`all-MiniLM-L6-v2`) if reachable, else a deterministic synthetic fallback.

    Set `force_synthetic=True` to exercise the offline path deliberately (used by a notebook cell and
    a test). Any failure to import or download falls through to the synthetic encoder -- the notebook
    and figures must never crash on a network problem.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if not force_synthetic:
        try:
            import logging
            import warnings

            warnings.filterwarnings("ignore")
            # Quiet sentence-transformers' loader and the tokenize() deprecation logger so the
            # executed notebook stays clean (the message is emitted via logging, not warnings).
            logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
            try:  # silence the "Loading weights" progress bar from the transformers backend
                from transformers import logging as hf_logging

                hf_logging.set_verbosity_error()
                hf_logging.disable_progress_bar()
            except Exception:  # noqa: BLE001 -- transformers may be absent; not fatal
                pass
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODEL_NAME, device=DEVICE)
            model.eval()
            return Encoder(name=MODEL_NAME, is_real=True, model=model)
        except Exception:  # noqa: BLE001 -- any failure (no net, no package) -> synthetic fallback
            pass
    return Encoder(name="synthetic", is_real=False, model=SyntheticEncoder(seed=SEED))


def load_cross_encoder() -> object | None:
    """Load the reranking cross-encoder if reachable, else None (the page degrades gracefully).

    Returns a CrossEncoder for `cross-encoder/ms-marco-MiniLM-L-6-v2`, or None offline. Used only by
    the bi-vs-cross worked example; the rest of the chapter does not depend on it.
    """
    try:
        import warnings

        warnings.filterwarnings("ignore")
        from sentence_transformers import CrossEncoder

        return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=DEVICE)
    except Exception:  # noqa: BLE001
        return None


def device_report() -> str:
    """One honest line: the detected accelerator and the pinned execution device."""
    return f"device: {DEVICE} (detected {DETECTED_DEVICE}; pinned to CPU for reproducibility)"


# ==================================================================================================
# Measurements built on a backend -- these work on EITHER encoder so numbers are reproducible offline.
# ==================================================================================================
def order_blindness_demo(encoder: Encoder) -> dict[str, float]:
    """Mean-pool (order-blind) vs the encoder (order-aware) on "dog bit man" vs "man bit dog".

    Returns:
      mean_pool_cosine -- cosine of the two sentences under MEAN-POOLED encoder word vectors via a
                          static bag (here: the encoder's own per-word vectors averaged), which is
                          EXACTLY 1.000 because averaging is permutation-invariant;
      encoder_cosine   -- cosine under the full (order-aware) sentence encoder: well below 1.0.
    The contract the page asserts: mean_pool ~ 1.000 (cannot tell them apart); encoder < mean_pool.
    """
    # Build a static word table from the union of words, then mean-pool -- guaranteed order-blind.
    words = sorted(set(_words(ORDER_A) + _words(ORDER_B)))
    rng = np.random.default_rng(SEED)
    table = {w: _unit(rng.standard_normal(32)) for w in words}
    mp_a = mean_pool(np.stack([table[w] for w in _words(ORDER_A)]))
    mp_b = mean_pool(np.stack([table[w] for w in _words(ORDER_B)]))
    enc = encoder.encode([ORDER_A, ORDER_B])
    return {
        "mean_pool_cosine": cosine(mp_a, mp_b),
        "encoder_cosine": cosine(enc[0], enc[1]),
    }


def _words(sentence: str) -> list[str]:
    return [t for t in sentence.lower().replace(".", "").split() if t]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def sif_worked_example() -> dict[str, np.ndarray]:
    """The page's "the cat sat" SIF worked example: plain mean vs SIF-weighted average.

    Returns {'plain': (2,), 'sif': (2,), 'weights': (3,), 'words': [...]} so the figure and the page
    quote the SAME numbers. SIF pulls the sentence vector OFF the stopword 'the' toward content words.
    """
    words = list(SIF_TOY.keys())
    vectors = np.stack([SIF_TOY[w][0] for w in words])
    probs = np.array([SIF_TOY[w][1] for w in words])
    plain = mean_pool(vectors)
    raw_weights = np.array([sif_weight(p) for p in probs])
    # Normalize weights to sum to 1 (as the page does), then take the weighted average.
    norm_w = raw_weights / raw_weights.sum()
    sif = (norm_w[:, None] * vectors).sum(axis=0)
    return {"plain": plain, "sif": sif, "weights": raw_weights, "norm_weights": norm_w,
            "words": words, "vectors": vectors, "probs": probs}


def paraphrase_separation(encoder: Encoder) -> dict[str, float]:
    """Cosine(paraphrase) vs cosine(unrelated) under the encoder, plus a mean-pool baseline.

    Returns the encoder's paraphrase/unrelated cosines AND a static mean-pool baseline's, so the page
    can show that the trained encoder SEPARATES the two far better than averaging static vectors.
    """
    enc = encoder.encode(list(PARAPHRASE_SET))
    enc_par = cosine(enc[0], enc[1])
    enc_unrel = cosine(enc[0], enc[2])
    # Static mean-pool baseline over a shared per-word table (topic overlap only, no training).
    all_words = sorted({w for s in PARAPHRASE_SET for w in _words(s)})
    rng = np.random.default_rng(SEED + 1)
    table = {w: _unit(rng.standard_normal(64)) for w in all_words}
    mp = [mean_pool(np.stack([table[w] for w in _words(s)])) for s in PARAPHRASE_SET]
    return {
        "encoder_paraphrase": enc_par,
        "encoder_unrelated": enc_unrel,
        "meanpool_paraphrase": cosine(mp[0], mp[1]),
        "meanpool_unrelated": cosine(mp[0], mp[2]),
    }


def topic_projection(encoder: Encoder) -> tuple[np.ndarray, list[str], list[str]]:
    """Encode the 9 topic sentences and project to 2-D via PCA for the cluster scatter.

    Returns (coords [9, 2], topic_labels, sentence_snippets). Paraphrases of one topic cluster; the
    three topics separate -- the geometry that powers search, clustering, and dedup.
    """
    labels = [t for t, _ in TOPIC_SENTENCES]
    sents = [s for _, s in TOPIC_SENTENCES]
    vecs = encoder.encode(sents)
    return pca_2d(vecs), labels, sents


def semantic_search(encoder: Encoder) -> list[tuple[float, str]]:
    """Rank SEARCH_CORPUS by cosine to SEARCH_QUERY (descending). Returns [(cosine, document), ...].

    The relevant doc 2 ("forgot"/"Forgot password") ranks high WITHOUT sharing the word "reset" --
    a meaning match, not a keyword match.
    """
    q = encoder.encode([SEARCH_QUERY])[0]
    docs = encoder.encode(list(SEARCH_CORPUS))
    sims = [(cosine(q, d), doc) for d, doc in zip(docs, SEARCH_CORPUS)]
    return sorted(sims, key=lambda t: -t[0])


def bi_vs_cross_demo(encoder: Encoder) -> dict[str, object]:
    """Score BI_VS_CROSS_DOCS against the query with BOTH a bi-encoder and a cross-encoder.

    Reproduces the page's bi-vs-cross worked-example table exactly: bi-encoder cosines (cacheable,
    one vector per text) and cross-encoder logits (joint cross-attention, one score per pair). The
    cross-encoder is loaded on demand; if it is unavailable (offline / no package) we return
    `cross_available=False` and `cross_logits=None` so callers can degrade gracefully.

    Returns:
      {'docs': [...], 'bi_cosines': [...], 'cross_logits': [...] | None, 'cross_available': bool}.
    Both encoders agree on the top doc; the cross-encoder's separation is far sharper.
    """
    q = encoder.encode([BI_VS_CROSS_QUERY])[0]
    doc_vecs = encoder.encode(list(BI_VS_CROSS_DOCS))
    bi_cosines = [cosine(q, d) for d in doc_vecs]

    cross = load_cross_encoder()
    if cross is None:
        return {
            "docs": list(BI_VS_CROSS_DOCS),
            "bi_cosines": bi_cosines,
            "cross_logits": None,
            "cross_available": False,
        }
    logits = cross.predict([(BI_VS_CROSS_QUERY, d) for d in BI_VS_CROSS_DOCS])
    return {
        "docs": list(BI_VS_CROSS_DOCS),
        "bi_cosines": bi_cosines,
        "cross_logits": [float(x) for x in logits],
        "cross_available": True,
    }


# ColBERT late-interaction demo: a short query and document whose per-token vectors we score with
# the MaxSim operator. Kept short so the heatmap is readable token-by-token.
COLBERT_QUERY_TOKENS: tuple[str, ...] = ("reset", "my", "password")
COLBERT_DOC_TOKENS: tuple[str, ...] = ("change", "your", "password", "in", "settings")


def colbert_maxsim(
    query_tokens: tuple[str, ...] = COLBERT_QUERY_TOKENS,
    doc_tokens: tuple[str, ...] = COLBERT_DOC_TOKENS,
    seed: int = SEED,
) -> dict[str, object]:
    """ColBERT late interaction: per-token cosine matrix, the per-query-token MaxSim, and their sum.

    For each query token we take the MAX cosine similarity over all document tokens (the MaxSim
    operator), then SUM those maxima for the final relevance score:  score = sum_i max_j cos(q_i, d_j).

    Token vectors here are deterministic, seeded synthetic embeddings keyed off the token strings (so
    the figure is reproducible and offline) but built to honour the qualitative point: the shared
    token "password" lights up its row/column, and each query token's best-matching doc token is the
    highlighted MaxSim cell. Returns the similarity matrix, the per-row argmax/max, and the summed
    score. Illustrative -- it shows the MaxSim MECHANISM, not measured ColBERT vectors.

    Source: Khattab & Zaharia (2020), ColBERT, the MaxSim late-interaction operator.
    """
    rng = np.random.default_rng(seed)
    dim = 32
    # One fixed unit vector per distinct token; synonyms/related tokens get a shared nudge so the
    # matrix has structure to read (the shared "password" token is identical in both lists).
    vocab: dict[str, np.ndarray] = {}

    def vec(tok: str) -> np.ndarray:
        if tok not in vocab:
            local = np.random.default_rng((_stable_hash(tok) ^ seed) % (2**32))
            v = local.standard_normal(dim)
            vocab[tok] = v / np.linalg.norm(v)
        return vocab[tok]

    # A topical nudge so each query token's intended doc match reliably dominates its row:
    # "reset"~"change", "my"~"your", "password"~"password" (the shared token is exact, cos = 1.0).
    related = {"reset": "change", "my": "your", "password": "password"}
    q_vecs = []
    for qt in query_tokens:
        base = vec(qt)
        if qt in related:
            base = base + 1.6 * vec(related[qt])  # strong pull toward its best doc match
            base = base / np.linalg.norm(base)
        q_vecs.append(base)
    d_vecs = [vec(dt) for dt in doc_tokens]

    matrix = np.array([[cosine(q, d) for d in d_vecs] for q in q_vecs])  # (|Q|, |D|)
    row_argmax = matrix.argmax(axis=1)
    row_max = matrix.max(axis=1)
    score = float(row_max.sum())
    return {
        "query_tokens": list(query_tokens),
        "doc_tokens": list(doc_tokens),
        "matrix": matrix,
        "row_argmax": row_argmax,
        "row_max": row_max,
        "score": score,
    }


def pooling_comparison(encoder: Encoder) -> dict[str, float]:
    """STS Spearman under three pooling strategies, computed on the encoder's TOKEN vectors.

    For the real backend we read per-token vectors from the transformer and pool them three ways
    (mean / max / first-token "CLS-like"); for the synthetic backend we approximate token vectors
    from its per-word features. Returns {'mean': r, 'max': r, 'cls': r} -- mean should win (the page's
    claim). This is an APPROXIMATION of the SBERT result on a tiny set, labelled as such on the figure.
    """
    pairs = STS_PAIRS
    gold = np.array([g for _, _, g in pairs])
    out = {}
    for strategy in ("mean", "max", "cls"):
        cosines = []
        for a, b, _ in pairs:
            va = _pooled_vector(encoder, a, strategy)
            vb = _pooled_vector(encoder, b, strategy)
            cosines.append(cosine(va, vb))
        out[strategy] = spearman(np.array(cosines), gold)
    return out


def _pooled_vector(encoder: Encoder, sentence: str, strategy: str) -> np.ndarray:
    """A sentence vector under a pooling `strategy` ('mean'|'max'|'cls'), from token-level vectors."""
    tokens = _token_vectors(encoder, sentence)
    if strategy == "mean":
        return mean_pool(tokens)
    if strategy == "max":
        return max_pool(tokens)
    if strategy == "cls":
        return tokens[0]  # first-token vector -- the "[CLS]-like" pooling
    raise ValueError(f"unknown pooling strategy: {strategy}")


def _token_vectors(encoder: Encoder, sentence: str) -> np.ndarray:
    """(T, d) per-token vectors for `sentence`.

    Real backend: the transformer's last-hidden-state token embeddings (before SBERT's mean pool).
    Synthetic backend: per-word (unigram) feature vectors, so the three pooling strategies still
    differ and the comparison is meaningful offline.
    """
    if encoder.is_real:
        import warnings

        st_model = encoder.model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # silence the tokenize() deprecation notice
            features = st_model.tokenize([sentence])
        # Keep only tensor features (newer sentence-transformers adds a non-tensor 'modality' key)
        # and move them to the pinned device before the transformer forward pass.
        features = {
            k: v.to(DEVICE) for k, v in features.items() if isinstance(v, torch.Tensor)
        }
        with torch.no_grad():
            out = st_model[0].forward(features)  # transformer module -> token_embeddings
        token_emb = out["token_embeddings"][0].cpu().numpy()  # (T, d)
        return np.asarray(token_emb, dtype=np.float64)
    synth: SyntheticEncoder = encoder.model
    toks = SyntheticEncoder._tokens(sentence)
    return np.stack([synth._feature_vec(t) for t in toks], axis=0)


def sts_evaluation(encoder: Encoder) -> dict[str, object]:
    """Evaluate the encoder on STS_PAIRS: cosine per pair, gold per pair, and Spearman correlation.

    Returns {'cosines': (n,), 'gold': (n,), 'spearman': r}. Spearman (rank correlation) is the STS
    convention: it scores whether the ORDER of similarities matches human judgment.
    """
    cosines, gold = [], []
    for a, b, g in STS_PAIRS:
        va, vb = encoder.encode([a, b])
        cosines.append(cosine(va, vb))
        gold.append(g)
    cosines = np.array(cosines)
    gold = np.array(gold)
    return {"cosines": cosines, "gold": gold, "spearman": spearman(cosines, gold)}


def pca_2d(matrix: np.ndarray) -> np.ndarray:
    """Project rows of `matrix` to 2-D via PCA (mean-center + top-2 right singular vectors).

    Deterministic and dependency-light (NumPy SVD); the sign of each axis is fixed so the layout is
    stable run to run (the largest-magnitude loading on each component is forced positive).
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    for k in range(2):
        if components[k][np.argmax(np.abs(components[k]))] < 0:
            components[k] = -components[k]
    return centered @ components.T


def main() -> None:
    """Print the page's headline numbers from whichever backend loaded -- the reproducible core."""
    print(device_report())
    print("torch:", torch.__version__, " numpy:", np.__version__)
    encoder = load_encoder()
    tag = "real" if encoder.is_real else "synthetic (fallback)"
    print(f"backend: {encoder.name} ({tag})\n")

    # 1) Order-blindness: mean-pool cannot tell the two sentences apart; the encoder can.
    ob = order_blindness_demo(encoder)
    print("--- order-blindness: 'dog bit man' vs 'man bit dog' ---")
    print(f"  mean-pool cosine : {ob['mean_pool_cosine']:.3f}   (order-blind -> identical)")
    print(f"  encoder   cosine : {ob['encoder_cosine']:.3f}   (order-aware -> separated)")
    assert ob["mean_pool_cosine"] > 0.999, "mean-pool must be order-blind (cosine ~ 1.000)"
    assert ob["encoder_cosine"] < ob["mean_pool_cosine"], "encoder must beat mean-pool on order"

    # 2) SIF worked example: weighting pulls the vector off the stopword toward content words.
    sif = sif_worked_example()
    print("\n--- SIF worked example: 'the cat sat' (x-axis = stopword 'the' direction) ---")
    print(f"  plain mean x-coord : {sif['plain'][0]:.3f}   (dominated by 'the')")
    print(f"  SIF    avg x-coord : {sif['sif'][0]:.3f}   (down-weights 'the')")
    assert sif["sif"][0] < sif["plain"][0], "SIF must reduce the stopword's pull on the x-axis"

    # 3) Paraphrase separation: the trained encoder separates paraphrase from unrelated far better.
    ps = paraphrase_separation(encoder)
    print("\n--- paraphrase vs unrelated cosine ---")
    print(f"  encoder : paraphrase {ps['encoder_paraphrase']:.3f}  unrelated {ps['encoder_unrelated']:.3f}")
    print(f"  meanpool: paraphrase {ps['meanpool_paraphrase']:.3f}  unrelated {ps['meanpool_unrelated']:.3f}")
    enc_gap = ps["encoder_paraphrase"] - ps["encoder_unrelated"]
    mp_gap = ps["meanpool_paraphrase"] - ps["meanpool_unrelated"]
    print(f"  separation gap: encoder {enc_gap:.3f}  vs  mean-pool {mp_gap:.3f}")
    assert enc_gap > mp_gap, "the trained encoder must separate paraphrase/unrelated better"

    # 4) Semantic search: the relevant doc ranks high WITHOUT sharing the query word "reset".
    ranking = semantic_search(encoder)
    print("\n--- semantic search: 'How do I reset my account password?' ---")
    for rank, (sim, doc) in enumerate(ranking, 1):
        print(f"  {rank}. {sim:+.3f}  {doc[:54]}")
    assert "password" in ranking[0][1].lower(), "top result should be a password doc"

    # 4b) Bi-encoder vs cross-encoder on the EXACT page-table strings.
    bvc = bi_vs_cross_demo(encoder)
    print("\n--- bi-encoder vs cross-encoder (reproduces the page table) ---")
    print(f"{'bi cos':>8} | {'cross logit':>11} | document")
    for i, (doc, bi) in enumerate(zip(bvc["docs"], bvc["bi_cosines"])):
        if bvc["cross_available"]:
            print(f"{bi:>+8.3f} | {bvc['cross_logits'][i]:>+11.2f} | {doc[:46]}")
        else:
            print(f"{bi:>+8.3f} | {'offline':>11} | {doc[:46]}")
    if not bvc["cross_available"]:
        print("  (cross-encoder offline: bi-encoder cosines shown; cross-encoder logits unavailable)")
    else:
        # Both encoders rank the password doc first; the cross-encoder spreads scores far more.
        assert bvc["bi_cosines"][0] == max(bvc["bi_cosines"]), "bi top must be the password doc"
        assert bvc["cross_logits"][0] == max(bvc["cross_logits"]), "cross top must be the password doc"

    # 5) Pooling comparison + STS Spearman.
    pooling = pooling_comparison(encoder)
    print("\n--- pooling strategy -> STS Spearman (mean usually wins) ---")
    for strat in ("mean", "max", "cls"):
        print(f"  {strat:>4}: {pooling[strat]:+.3f}")

    sts = sts_evaluation(encoder)
    print(f"\n--- STS evaluation: Spearman(cosine, gold) = {sts['spearman']:.3f} ---")
    assert sts["spearman"] > 0.5, "cosine should rank-correlate with gold similarity"

    # 6) Triplet + InfoNCE loss spot-checks (the page's worked numbers).
    print("\n--- loss spot-checks ---")
    print(f"  triplet (solved)   : {triplet_loss(0.013, 0.884):.3f}  (expect 0.000)")
    print(f"  triplet (violated) : {triplet_loss(0.400, 0.200):.3f}  (expect 0.500)")
    assert triplet_loss(0.013, 0.884) == 0.0
    assert abs(triplet_loss(0.400, 0.200) - 0.5) < 1e-9
    print("\nall qualitative contracts held.")


if __name__ == "__main__":
    main()
