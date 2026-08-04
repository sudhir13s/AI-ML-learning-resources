"""Word embeddings from scratch — the single source of truth for the chapter.

The concept page, the teaching notebook, and the figure generator (`make_figures_05.py`) all
import the functions and constants defined here, so none of them can silently drift from the
others. Every number quoted on the page is produced by this file.

What lives here:
  * a small but STRUCTURED toy corpus (royalty words share contexts; animal words share contexts);
  * skip-gram with negative sampling, from scratch in PyTorch — DETERMINISTIC and DEVICE-AGNOSTIC
    (CUDA / MPS / CPU; the reproducible trace is pinned to CPU so the printed numbers are stable);
  * CBOW context-pooling and the skip-gram-vs-CBOW pair counts, for the contrast figure;
  * the freq^0.75 negative-sampling distribution (the exact table on the page);
  * one (center, true-context, negative) triple worked end to end — softmax p(o|c), the
    negative-sampling loss, and its gradient w.r.t. v_c — matching the by-hand algebra;
  * the GloVe ice/steam co-occurrence-ratio table.

Determinism: every torch / numpy RNG is seeded from SEED, so the same corpus yields the same
vectors on a given device. Absolute losses are device-dependent at the last digits; the
qualitative checks (ordering of cosines, loss falling, ratio > 1) hold on any device and are
asserted before anything is printed.

Run:
    python word_embeddings.py
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Reproducibility ---------------------------------------------------------------------------
SEED = 0

# Run on the best available accelerator; CPU is the universal fallback. The reproducible trace in
# main() pins training to CPU so the printed losses/cosines are stable across machines (tiny
# single-layer tensors are dominated by kernel-launch overhead on MPS/CUDA, and the last digits of
# the loss drift between backends — the ORDERING of cosines, which is the teaching point, does not).
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# --- The toy corpus used everywhere on the page, in the notebook, and in the figures -----------
# Two semantic clusters with disjoint contexts: royalty words ("king", "queen") always appear with
# {ruled, kingdom, wore, crown, sat, throne}; animal words ("dog", "cat") always appear with
# {chased, ball, furry, pet, slept}. Nothing in the data says king~queen directly — the model must
# INFER it from shared company, which is exactly the distributional hypothesis under test.
ROYALTY: tuple[str, ...] = ("king", "queen")
ANIMALS: tuple[str, ...] = ("dog", "cat")
ROYALTY_FRAMES: tuple[list[str], ...] = (
    ["the", "_", "ruled", "the", "kingdom"],
    ["the", "_", "wore", "a", "crown"],
    ["the", "_", "sat", "on", "the", "throne"],
)
ANIMAL_FRAMES: tuple[list[str], ...] = (
    ["the", "_", "chased", "the", "ball"],
    ["the", "_", "was", "a", "furry", "pet"],
    ["the", "_", "slept", "all", "day"],
)
CORPUS_REPEATS = 6  # repeat each frame so SGD sees every context enough times to separate clusters

# --- Skip-gram-with-negative-sampling hyper-parameters -----------------------------------------
WINDOW = 2  # half-width: a center word predicts neighbours up to 2 positions away
EMBED_DIM = 16  # embedding dimensionality (tiny — this is a mechanism demo, not a real model)
N_NEGATIVES = 5  # k random negatives per true (center, context) pair
N_STEPS = 300  # full-batch SGD steps
LEARNING_RATE = 0.01

# --- The freq^0.75 negative-sampling distribution ----------------------------------------------
# Five illustrative unigram counts spanning a stop-word ("the") down to a rare word ("kingdom").
# Counts chosen so the raw probabilities are round (0.602, 0.301, 0.060, 0.030, 0.006) and the
# 0.75-power lift on the rarest word is a clean ~2.75x.
NEG_SAMPLING_COUNTS: tuple[int, ...] = (1000, 500, 100, 50, 10)
NEG_SAMPLING_WORDS: tuple[str, ...] = ("the", "of", "king", "queen", "kingdom")
NEG_SAMPLING_POWER = 0.75  # Mikolov et al. (2013b): damps frequent words, lifts rare ones

# --- One worked (center, true-context, negative) triple, for the by-hand gradient check --------
HAND_VC = (0.5, -0.2, 0.1)  # center vector v_c
HAND_UO = (0.4, 0.1, 0.3)  # true context  u_o
HAND_UN = (-0.3, 0.5, -0.2)  # one negative   u_n
HAND_U3 = (0.0, 0.2, -0.1)  # a third vocab word, only for the full-softmax denominator

# --- GloVe ice/steam co-occurrence demo --------------------------------------------------------
# Toy co-occurrence counts X[i, probe] for i in {ice, steam} against four probe words. The STRUCTURE
# (solid≫ for ice, gas≫ for steam, water/fashion ≈1) is the point, not the exact integers.
ICE_STEAM_PROBES: tuple[str, ...] = ("solid", "gas", "water", "fashion")
ICE_COUNTS: tuple[int, ...] = (600, 80, 300, 10)  # X[ice, probe]
STEAM_COUNTS: tuple[int, ...] = (60, 550, 280, 10)  # X[steam, probe]


# ====================================================================================================
# Corpus construction
# ====================================================================================================
def build_corpus() -> list[list[str]]:
    """Materialize the structured toy corpus as a list of tokenized sentences.

    The repeat structure matches the page's inline snippet exactly -- each word's three frames are
    grouped and the whole group is repeated CORPUS_REPEATS times ([f1, f2, f3] * R) -- so the inline
    code, this source of truth, the notebook, and the figures all produce the SAME pair ordering and
    therefore the SAME learned vectors.
    """
    sentences: list[list[str]] = []
    for word in ROYALTY:
        group = [[word if tok == "_" else tok for tok in frame] for frame in ROYALTY_FRAMES]
        sentences += group * CORPUS_REPEATS
    for word in ANIMALS:
        group = [[word if tok == "_" else tok for tok in frame] for frame in ANIMAL_FRAMES]
        sentences += group * CORPUS_REPEATS
    return sentences


def fasttext_corpus(repeats: int = 8) -> list[list[str]]:
    """The toy corpus for the FastText demo, repeated `repeats` times (default 8).

    FastText learns per-n-gram vectors, which need a bit more signal than the skip-gram demo to
    settle, so the OOV demo uses 8 repeats (vs CORPUS_REPEATS=6 for skip-gram). The page's Code 3,
    the notebook's FastText cell, and the FastText figure all call this, so they share one corpus
    and one set of numbers (norm ~0.60, cos(kingdom, kingdoms) ~0.9997).
    """
    sentences: list[list[str]] = []
    for word in ROYALTY:
        group = [[word if tok == "_" else tok for tok in frame] for frame in ROYALTY_FRAMES]
        sentences += group * repeats
    for word in ANIMALS:
        group = [[word if tok == "_" else tok for tok in frame] for frame in ANIMAL_FRAMES]
        sentences += group * repeats
    return sentences


def build_vocab(sentences: list[list[str]]) -> tuple[list[str], dict[str, int]]:
    """Return the sorted vocabulary and a word->index map (sorted => deterministic indices)."""
    vocab = sorted({w for s in sentences for w in s})
    return vocab, {w: i for i, w in enumerate(vocab)}


def skipgram_pairs(
    sentences: list[list[str]], word_to_index: dict[str, int], window: int = WINDOW
) -> torch.Tensor:
    """All (center, context) index pairs within `window` — the skip-gram training set.

    Skip-gram emits one pair per (center, neighbour): a center at position i pairs with every
    j in [i-window, i+window], j != i. Returns an (n_pairs, 2) long tensor of [center, context].
    """
    pairs: list[tuple[int, int]] = []
    for sentence in sentences:
        idx = [word_to_index[w] for w in sentence]
        for i, center in enumerate(idx):
            lo, hi = max(0, i - window), min(len(idx), i + window + 1)
            for j in range(lo, hi):
                if j != i:
                    pairs.append((center, idx[j]))
    return torch.tensor(pairs, dtype=torch.long)


def cbow_windows(
    sentences: list[list[str]], word_to_index: dict[str, int], window: int = WINDOW
) -> int:
    """Count CBOW training examples (one pooled prediction per center) — for the contrast figure.

    Where skip-gram emits 2c pairs per center, CBOW pools the context into ONE example per center,
    so this returns the number of centers that have at least one neighbour in `window`.
    """
    examples = 0
    for sentence in sentences:
        for i in range(len(sentence)):
            lo, hi = max(0, i - window), min(len(sentence), i + window + 1)
            if (hi - lo) > 1:  # at least one neighbour besides the center
                examples += 1
    return examples


# ====================================================================================================
# Skip-gram with negative sampling — the from-scratch model
# ====================================================================================================
class SkipGramNS(nn.Module):
    """Skip-gram with negative sampling: two embedding tables (center v, context u).

    The forward pass returns the negative-sampling LOSS for a batch of (center, context) pairs and
    their sampled negatives — exactly the boxed objective on the page:
        L = -[ log sigma(u_o . v_c) + sum_i log sigma(-u_{n_i} . v_c) ]
    """

    def __init__(self, vocab_size: int, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.center = nn.Embedding(vocab_size, embed_dim)  # v_c  (input / center vectors)
        self.context = nn.Embedding(vocab_size, embed_dim)  # u_o  (output / context vectors)

    def forward(
        self, centers: torch.Tensor, contexts: torch.Tensor, negatives: torch.Tensor
    ) -> torch.Tensor:
        v_c = self.center(centers)  # (B, d)
        pos_score = (v_c * self.context(contexts)).sum(-1)  # u_o . v_c   -> (B,)
        neg_score = torch.bmm(self.context(negatives), v_c.unsqueeze(-1)).squeeze(-1)  # (B, k)
        # logsigmoid(pos): push the TRUE pair's score up (sigma -> 1)
        # logsigmoid(-neg): push each negative's score down (sigma(-.) -> 1)
        loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score).sum(-1)).mean()
        return loss


def train_skipgram(
    *,
    device: str = "cpu",
    n_steps: int = N_STEPS,
    seed: int = SEED,
) -> tuple[SkipGramNS, list[str], dict[str, int], list[float]]:
    """Train skip-gram+NS on the toy corpus; return (model, vocab, word_to_index, loss_history).

    Device-agnostic: every tensor is created on `device` and the model is moved with `.to(device)`.
    The page pins `device="cpu"` for a reproducible loss/cosine trace.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    sentences = build_corpus()
    vocab, word_to_index = build_vocab(sentences)
    vocab_size = len(vocab)
    pairs = skipgram_pairs(sentences, word_to_index).to(device)

    model = SkipGramNS(vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    loss_history: list[float] = []
    n_pairs = pairs.shape[0]
    for _ in range(n_steps):
        perm = pairs[torch.randperm(n_pairs, device=device)]
        centers, contexts = perm[:, 0], perm[:, 1]
        negatives = torch.randint(0, vocab_size, (n_pairs, N_NEGATIVES), device=device)
        loss = model(centers, contexts, negatives)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))
    return model, vocab, word_to_index, loss_history


def unit_embeddings(model: SkipGramNS) -> torch.Tensor:
    """L2-normalized CENTER embeddings (so a dot product IS the cosine similarity)."""
    return F.normalize(model.center.weight.detach(), dim=1)


def cosine(embeddings: torch.Tensor, word_to_index: dict[str, int], a: str, b: str) -> float:
    """Cosine similarity between two words, given UNIT embeddings (dot product = cosine)."""
    return float(embeddings[word_to_index[a]] @ embeddings[word_to_index[b]])


# ====================================================================================================
# The freq^0.75 negative-sampling distribution
# ====================================================================================================
def negative_sampling_distribution(
    counts: tuple[int, ...] = NEG_SAMPLING_COUNTS, power: float = NEG_SAMPLING_POWER
) -> tuple[np.ndarray, np.ndarray]:
    """Return (raw unigram p(w), smoothed p(w) ∝ count^power). The 0.75 damps frequent words."""
    counts_arr = np.asarray(counts, dtype=float)
    raw = counts_arr / counts_arr.sum()
    smoothed = counts_arr**power
    smoothed /= smoothed.sum()
    return raw, smoothed


# ====================================================================================================
# One (center, context, negative) triple, worked end to end
# ====================================================================================================
def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def hand_worked_triple() -> dict[str, float | np.ndarray]:
    """Softmax p(o|c), negative-sampling loss, and gradient w.r.t. v_c for one fixed triple.

    Reproduces the by-hand algebra on the page: the gradient is (sigma(s_o)-1)*u_o + sigma(s_n)*u_n,
    pulling v_c toward the true context u_o and pushing it away from the negative u_n.
    """
    v_c = np.array(HAND_VC)
    u_o = np.array(HAND_UO)
    u_n = np.array(HAND_UN)
    u_3 = np.array(HAND_U3)

    s_o, s_n, s_3 = v_c @ u_o, v_c @ u_n, v_c @ u_3
    # (a) full softmax over the 3-word vocab {o, n, w3}
    z = np.exp(s_o) + np.exp(s_n) + np.exp(s_3)
    p_softmax = float(np.exp(s_o) / z)
    # (b) negative-sampling loss with k=1 negative
    ns_loss = float(-(np.log(_sigmoid(s_o)) + np.log(_sigmoid(-s_n))))
    # (c) gradient of the NS loss w.r.t. v_c
    grad_vc = (_sigmoid(s_o) - 1.0) * u_o + _sigmoid(s_n) * u_n
    return {
        "s_o": float(s_o),
        "s_n": float(s_n),
        "s_3": float(s_3),
        "p_softmax": p_softmax,
        "sig_so": float(_sigmoid(s_o)),
        "sig_neg_sn": float(_sigmoid(-s_n)),
        "ns_loss": ns_loss,
        "grad_vc": np.round(grad_vc, 4),
    }


# ====================================================================================================
# GloVe ice/steam co-occurrence ratio
# ====================================================================================================
def ice_steam_ratios() -> dict[str, np.ndarray]:
    """Conditional probabilities P(probe | ice/steam) and their ratio — the GloVe intuition.

    The ratio cleanly isolates meaning: ≫1 belongs to ice (solid), ≪1 to steam (gas), ≈1 means the
    probe relates to both (water) or neither (fashion). GloVe is engineered so vector DIFFERENCES
    reproduce the LOG of these ratios.
    """
    ice = np.asarray(ICE_COUNTS, dtype=float)
    steam = np.asarray(STEAM_COUNTS, dtype=float)
    p_ice = ice / ice.sum()
    p_steam = steam / steam.sum()
    return {"p_ice": p_ice, "p_steam": p_steam, "ratio": p_ice / p_steam}


# ====================================================================================================
# Demo entry point
# ====================================================================================================
def main() -> None:
    # Pin the trace to CPU so the printed numbers are reproducible on any machine.
    trace_device = "cpu"
    detected = DEVICE
    print(
        f"device: {trace_device} (detected {detected}; pinned to CPU for reproducibility)  "
        f"|  torch: {torch.__version__}  |  numpy: {np.__version__}"
    )

    # 1) Train skip-gram + negative sampling and show the cluster cosines + loss fall.
    model, vocab, word_to_index, loss_history = train_skipgram(device=trace_device)
    embeddings = unit_embeddings(model)
    cos_royalty = cosine(embeddings, word_to_index, "king", "queen")
    cos_animal = cosine(embeddings, word_to_index, "dog", "cat")
    cos_cross = cosine(embeddings, word_to_index, "king", "dog")
    # The whole teaching point: context-sharing words end up MORE similar than cross-cluster words.
    assert cos_royalty > cos_cross, "royalty pair should beat the cross-cluster pair"
    assert cos_animal > cos_cross, "animal pair should beat the cross-cluster pair"
    assert loss_history[-1] < loss_history[0], "loss must fall over training"
    print(
        f"\nskip-gram+NS  |  loss {loss_history[0]:.3f} -> {loss_history[-1]:.3f}  "
        f"({len(vocab)}-word vocab, {EMBED_DIM}-d)"
    )
    print(f"  cos(king, queen) = {cos_royalty:+.3f}  (royalty pair  -> HIGH)")
    print(f"  cos(dog,  cat)   = {cos_animal:+.3f}  (animal pair   -> HIGH)")
    print(f"  cos(king, dog)   = {cos_cross:+.3f}  (cross-cluster -> LOWER)")

    # 2) The freq^0.75 negative-sampling distribution.
    raw, smoothed = negative_sampling_distribution()
    rare_lift = smoothed[-1] / raw[-1]
    assert smoothed[-1] > raw[-1], "the 0.75 power must LIFT the rarest word"
    assert smoothed[0] < raw[0], "the 0.75 power must DAMP the most frequent word"
    print("\nfreq^0.75 negative-sampling distribution:")
    print(f"  {'word':>8} | {'raw p(w)':>9} | {'p(w)^0.75':>10}")
    for word, r, s in zip(NEG_SAMPLING_WORDS, raw, smoothed):
        print(f"  {word:>8} | {r:>9.4f} | {s:>10.4f}")
    print(f"  -> rarest word 'kingdom' sampled {rare_lift:.2f}x more often under the 0.75 power")

    # 3) One worked triple: softmax p(o|c), NS loss, gradient.
    triple = hand_worked_triple()
    assert triple["grad_vc"][0] < 0, "gradient should move v_c toward the true context u_o"
    print("\none (center, context, negative) triple, by hand:")
    print(
        f"  scores: u_o.v_c={triple['s_o']:.4f}  u_n.v_c={triple['s_n']:.4f}  u3.v_c={triple['s_3']:.4f}"
    )
    print(f"  p(o|c) full softmax = {triple['p_softmax']:.4f}")
    print(
        f"  sigma(u_o.v_c)={triple['sig_so']:.4f}  sigma(-u_n.v_c)={triple['sig_neg_sn']:.4f}  "
        f"NS loss={triple['ns_loss']:.4f}"
    )
    print(f"  grad_vc = {triple['grad_vc']}")

    # 4) GloVe ice/steam ratio.
    ratios = ice_steam_ratios()
    assert ratios["ratio"][0] > 3.0, "solid should belong to ICE (ratio >> 1)"
    assert ratios["ratio"][1] < 0.5, "gas should belong to STEAM (ratio << 1)"
    print("\nGloVe ice/steam co-occurrence ratios:")
    print(f"  {'probe':>8} | {'P(.|ice)':>9} | {'P(.|steam)':>11} | {'ratio':>7}")
    for probe, pi, ps, rr in zip(
        ICE_STEAM_PROBES, ratios["p_ice"], ratios["p_steam"], ratios["ratio"]
    ):
        print(f"  {probe:>8} | {pi:>9.3f} | {ps:>11.3f} | {rr:>7.2f}")


if __name__ == "__main__":
    main()
