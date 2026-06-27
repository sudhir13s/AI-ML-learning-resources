"""From-scratch embedding models for retrieval: why dense beats sparse on paraphrases.

Retrieval is bounded by the EMBEDDER: if the right passage isn't near the query in vector space,
no chunking or search recovers it. This script makes that concrete and CPU-runnable.

It builds two embedders from primitives so the difference is inspectable:
  1. a SPARSE bag-of-words embedder (a vector slot per vocabulary word) — it can only match text
     that shares literal words, so it FAILS on paraphrases ("reset my password" vs "forgot my
     login credentials") that share almost no words;
  2. a DENSE bi-encoder trained from scratch with the contrastive InfoNCE loss on toy paraphrase
     pairs — it learns to place paraphrases NEAR each other even with no shared words.

The whole thing is deterministic (seeded) and trains in a fraction of a second on CPU. An optional
bridge to a real pretrained model (all-MiniLM-L6-v2 via sentence-transformers) confirms the lesson
on a production embedder *if it's available*, but the from-scratch demo is the load-bearing lesson
and never depends on a download.

Verified on Python 3.12 / torch 2.x / numpy 2.x. Device-agnostic (CUDA / MPS / CPU); the contrastive
math and similarities are deterministic across devices.

Run:
    python embedding_models.py
"""

from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Hyperparameters (hoisted; no magic numbers inline) ------------------------------------------
DENSE_DIM = 16  # dense embedding width — tiny on purpose so the toy trains instantly and is readable
TEMPERATURE = 0.07  # InfoNCE temperature τ — sharpens the softmax over in-batch negatives (a standard value)
TRAIN_STEPS = 800  # gradient steps for the toy contrastive trainer
LEARNING_RATE = 0.03  # Adam step size for the toy trainer
SEED = 0  # seed for torch — makes the trained embedder and every printed number reproducible
TOKEN_RE = re.compile(r"[a-z]+")  # lowercase alphabetic tokens; drops digits/punctuation for the toy

# Toy paraphrase pairs: (query, paraphrased passage) — same meaning, deliberately FEW shared words.
# These are the positives the dense bi-encoder learns to pull together.
PARAPHRASE_PAIRS: tuple[tuple[str, str], ...] = (
    ("how do I reset my password", "I forgot my login credentials"),
    ("the car will not start", "my automobile won't turn on"),
    ("what is the refund policy", "how do I get my money back"),
    ("the flight was delayed", "my plane departed late"),
)

# Run on the best available accelerator; CPU is the universal fallback.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphabetic tokens — the unit both embedders count over."""
    return TOKEN_RE.findall(text.lower())


def build_vocab(texts: list[str]) -> dict[str, int]:
    """Map each distinct token to a fixed column index — the sparse embedder's coordinate system."""
    vocab = sorted({tok for text in texts for tok in tokenize(text)})  # sorted -> deterministic order
    return {tok: i for i, tok in enumerate(vocab)}


def sparse_embed(text: str, vocab: dict[str, int]) -> np.ndarray:
    """Bag-of-words: one slot per vocabulary word, count of that word, then L2-normalized.

    This is the essence of a lexical/sparse embedder (TF-style): the vector has a nonzero entry ONLY
    for words literally present. Two texts are similar only if they SHARE words — so a paraphrase with
    different wording lands far away, which is exactly the failure dense models fix.
    """
    vec = np.zeros(len(vocab), dtype=np.float64)
    for tok in tokenize(text):
        if tok in vocab:  # out-of-vocab words contribute nothing — another sparse limitation
            vec[vocab[tok]] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec  # unit-norm so cosine == dot product


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two vectors (== dot product when both are already unit-norm)."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


class DenseBiEncoder(nn.Module):
    """A minimal dense bi-encoder: bag-of-words -> learned linear projection -> L2-normalized vector.

    A real bi-encoder is a transformer; here the encoder is a single learned linear map so the
    *mechanism* — encode query and passage INDEPENDENTLY to fixed vectors, compare by cosine, train
    so paraphrases align — is visible end to end. The projection is what training shapes: it learns to
    send different-but-synonymous words to nearby directions.
    """

    def __init__(self, vocab_size: int, dim: int = DENSE_DIM) -> None:
        super().__init__()
        self.proj = nn.Linear(vocab_size, dim, bias=False)  # the learned embedding map (no bias: pure direction)

    def forward(self, bow: torch.Tensor) -> torch.Tensor:
        """(batch, vocab) bag-of-words -> (batch, dim) L2-normalized embeddings."""
        return F.normalize(self.proj(bow), dim=-1)  # normalize so cosine similarity is a plain dot product


def bow_tensor(text: str, vocab: dict[str, int]) -> torch.Tensor:
    """Bag-of-words as a torch row vector (the dense encoder's input)."""
    vec = torch.zeros(len(vocab))
    for tok in tokenize(text):
        if tok in vocab:
            vec[vocab[tok]] += 1.0
    return vec


def info_nce_loss(query_emb: torch.Tensor, passage_emb: torch.Tensor, temperature: float = TEMPERATURE) -> torch.Tensor:
    """InfoNCE with in-batch negatives: pull each query to its paired passage, push it from the rest.

    For a batch of B (query, positive-passage) pairs, we score every query against every passage:
    S[i, j] = cos(q_i, p_j) / τ. Row i's correct passage is j = i (the diagonal); every other column
    is an in-batch negative. Cross-entropy with the identity labels maximizes the diagonal and
    minimizes the off-diagonal — i.e. positives together, negatives apart. τ (temperature) sharpens
    the softmax: smaller τ = harsher penalty for a negative scoring near the positive.
    """
    scores = query_emb @ passage_emb.t() / temperature  # (B, B): every query vs every passage, scaled by 1/τ
    labels = torch.arange(query_emb.shape[0], device=query_emb.device)  # correct passage for row i is column i
    # symmetric loss (query->passage and passage->query) is standard; averaging both directions stabilizes training
    return 0.5 * (F.cross_entropy(scores, labels) + F.cross_entropy(scores.t(), labels))


def train_bi_encoder(
    pairs: tuple[tuple[str, str], ...],
    vocab: dict[str, int],
    steps: int = TRAIN_STEPS,
    seed: int = SEED,
    temperature: float = TEMPERATURE,
) -> tuple[DenseBiEncoder, list[float]]:
    """Train the toy bi-encoder with InfoNCE; return the model and the per-step loss curve."""
    torch.manual_seed(seed)  # deterministic init + training -> identical numbers every run
    model = DenseBiEncoder(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    queries = torch.stack([bow_tensor(q, vocab) for q, _ in pairs])  # (B, vocab) fixed inputs
    passages = torch.stack([bow_tensor(p, vocab) for _, p in pairs])
    losses: list[float] = []
    for _ in range(steps):
        q_emb, p_emb = model(queries), model(passages)  # encode both sides independently (the bi-encoder)
        loss = info_nce_loss(q_emb, p_emb, temperature=temperature)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    return model, losses


def tau_sweep(
    pairs: tuple[tuple[str, str], ...],
    vocab: dict[str, int],
    temperatures: tuple[float, ...] = (0.07, 1.0, 5.0),
) -> dict[float, float]:
    """Train at each temperature; return {temperature: final InfoNCE loss}.

    Demonstrates WHY temperature matters: a large tau flattens the softmax over in-batch negatives,
    so the gradient barely distinguishes the positive from the negatives and the loss stays high —
    training makes far less progress. (On this tiny, easy 4-pair toy the positives still separate
    even at large tau; on real data with hard negatives, a too-large tau visibly fails to separate
    them. The loss-stays-high signal is the honest, reproducible part.)
    """
    return {tau: train_bi_encoder(pairs, vocab, temperature=tau)[1][-1] for tau in temperatures}


def similarity_matrix(
    queries: list[str], passages: list[str], embed_fn
) -> np.ndarray:
    """(len(queries), len(passages)) cosine matrix under a given embedding function."""
    q_vecs = [embed_fn(q) for q in queries]
    p_vecs = [embed_fn(p) for p in passages]
    return np.array([[cosine(q, p) for p in p_vecs] for q in q_vecs])


def diagonal_vs_offdiagonal(matrix: np.ndarray) -> tuple[float, float]:
    """Mean of the diagonal (paraphrase pairs) vs mean off-diagonal (unrelated pairs)."""
    n = matrix.shape[0]
    diag = float(np.mean(np.diag(matrix)))
    off = float((matrix.sum() - np.trace(matrix)) / (matrix.size - n))
    return diag, off


def try_pretrained_demo(queries: list[str], passages: list[str]) -> np.ndarray | None:
    """Confirm the lesson on a REAL pretrained bi-encoder, if sentence-transformers + model load.

    Returns the (queries x passages) cosine matrix from all-MiniLM-L6-v2, or None if the model
    isn't available — so the notebook/script never hard-fail on a missing download. The from-scratch
    demo above is the load-bearing lesson; this is confirmation on a production embedder.
    """
    try:
        import contextlib
        import io
        import logging
        import os

        # Silence the HF/transformers load chatter (the "Loading weights" bar writes via a logger that
        # ignores a plain stderr redirect under nbconvert) BEFORE importing, so notebook output is clean.
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        for logger_name in ("sentence_transformers", "transformers", "transformers.modeling_utils"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)
        with contextlib.redirect_stderr(io.StringIO()):  # belt-and-suspenders for any residual stderr
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        q = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)  # (n, 384) unit-norm
        p = model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(q @ p.T)
    except Exception as exc:  # missing package, no network, or model not cached — degrade gracefully
        print(f"  (pretrained demo skipped: {type(exc).__name__}: {exc})")
        return None


def main() -> None:
    torch.manual_seed(SEED)
    print("torch:", torch.__version__, "| numpy:", np.__version__, "| device:", DEVICE)
    print(f"paraphrase pairs: {len(PARAPHRASE_PAIRS)} | dense_dim: {DENSE_DIM} | temperature: {TEMPERATURE}\n")

    queries = [q for q, _ in PARAPHRASE_PAIRS]
    passages = [p for _, p in PARAPHRASE_PAIRS]
    all_texts = queries + passages
    vocab = build_vocab(all_texts)
    print(f"vocabulary size: {len(vocab)} distinct tokens")

    # --- Show the shared-word problem: paraphrase pairs share almost no words ---
    print("\nWord overlap between each query and its paraphrase passage:")
    for q, p in PARAPHRASE_PAIRS:
        shared = set(tokenize(q)) & set(tokenize(p))
        print(f"  {q!r}\n  {p!r}\n  shared content words: {sorted(shared) or 'NONE'}\n")

    # --- SPARSE embedder: fails the paraphrases (no shared words -> ~0 similarity) ---
    sparse_fn = lambda t: sparse_embed(t, vocab)  # noqa: E731 - tiny local adapter for clarity
    sparse_sim = similarity_matrix(queries, passages, sparse_fn)
    sparse_diag, sparse_off = diagonal_vs_offdiagonal(sparse_sim)
    print(f"SPARSE bag-of-words — paraphrase (diagonal) similarity: {sparse_diag:.3f}")
    print(f"SPARSE bag-of-words — unrelated (off-diagonal) similarity: {sparse_off:.3f}")
    # the sparse model cannot tell paraphrases from unrelated text: both are near 0
    assert sparse_diag < 0.2, "sparse embedder should NOT match paraphrases (few shared words)"

    # --- DENSE bi-encoder: train from scratch, then it separates paraphrases from the rest ---
    model, losses = train_bi_encoder(PARAPHRASE_PAIRS, vocab)
    print(f"\nDENSE bi-encoder — trained {TRAIN_STEPS} steps, loss {losses[0]:.3f} -> {losses[-1]:.4f}")
    dense_fn = lambda t: model(bow_tensor(t, vocab).unsqueeze(0)).squeeze(0).detach().numpy()  # noqa: E731
    dense_sim = similarity_matrix(queries, passages, dense_fn)
    dense_diag, dense_off = diagonal_vs_offdiagonal(dense_sim)
    print(f"DENSE bi-encoder — paraphrase (diagonal) similarity: {dense_diag:.3f}")
    print(f"DENSE bi-encoder — unrelated (off-diagonal) similarity: {dense_off:.3f}")

    # --- Correctness BEFORE any claim: dense must separate paraphrases from unrelated, sparse must not ---
    assert dense_diag > 0.4, "dense bi-encoder must pull paraphrases together after training"
    assert dense_diag - dense_off > 0.5, "dense must put a clear GAP between paraphrase and unrelated"
    assert (dense_diag - dense_off) > (sparse_diag - sparse_off), "dense must beat sparse on the paraphrase gap"
    print("\ndense separates paraphrases from unrelated; sparse cannot: True")
    print(f"  paraphrase-vs-unrelated gap — sparse: {sparse_diag - sparse_off:+.3f}  dense: {dense_diag - dense_off:+.3f}")

    # --- Temperature matters: a too-large tau flattens the softmax, so the loss stays high ---
    print("\nTemperature sweep — final InfoNCE loss after training (lower = learned more):")
    tau_losses = tau_sweep(PARAPHRASE_PAIRS, vocab)
    for tau, final_loss in tau_losses.items():
        print(f"  tau={tau:>4}: final loss {final_loss:.4f}")
    # a larger temperature leaves the loss higher: the gradient can't push the positive past negatives
    assert tau_losses[0.07] < tau_losses[1.0] < tau_losses[5.0], "larger tau must leave loss higher"

    # --- Optional: confirm on a real pretrained model (cached -> runs offline) ---
    print("\nPretrained confirmation (all-MiniLM-L6-v2, 384-dim) — if available:")
    pre_sim = try_pretrained_demo(queries, passages)
    if pre_sim is not None:
        pre_diag, pre_off = diagonal_vs_offdiagonal(pre_sim)
        print(f"  pretrained — paraphrase similarity: {pre_diag:.3f} | unrelated: {pre_off:.3f}")
        print(f"  pretrained paraphrase-vs-unrelated gap: {pre_diag - pre_off:+.3f}")


if __name__ == "__main__":
    main()
