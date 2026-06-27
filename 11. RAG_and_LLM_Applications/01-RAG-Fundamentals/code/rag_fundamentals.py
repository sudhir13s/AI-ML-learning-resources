"""From-scratch RAG: retrieve-then-generate on a tiny corpus, with the retrieval visible.

Builds a complete Retrieval-Augmented Generation pipeline from primitives so every step is
inspectable: a transparent hashing + TF-IDF embedder (no model download, runs on CPU in
milliseconds), cosine-similarity top-k retrieval, prompt augmentation, and a deterministic
extractive "generator". The whole point is to *see* the mechanics — embedding -> index ->
top-k -> augment -> ground — not to call a black-box library.

The headline demonstration: ask a question whose answer is NOT in the model's frozen
parametric knowledge (a private fact), show the ungrounded path inventing/refusing, then show
the grounded path answering correctly because the right passage was retrieved into the prompt.

This is the same verified code embedded in the concept page and the teaching notebook.
Verified on Python 3.12 / numpy 2.x. Device-agnostic for the torch report line (CUDA / MPS /
CPU); the retrieval math itself is pure numpy and identical on any machine.

Run:
    python rag_fundamentals.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# ---- Corpus: a tiny "private knowledge base" the LLM was never trained on -------------------
# Each string is one chunk (one passage). In a real system these come from splitting documents;
# here we keep them short and self-contained so retrieval is easy to read by eye.
CORPUS: tuple[str, ...] = (
    "The Helios-7 satellite was launched on March 3rd, 2024 from the Kourou spaceport.",
    "Helios-7 carries a hyperspectral imager with a ground resolution of 4 meters.",
    "The project lead for Helios-7 is Dr. Amara Okoye, based in the Nairobi office.",
    "Helios-7 completes one orbit of Earth every 97 minutes in a sun-synchronous orbit.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight.",
    "The Eiffel Tower in Paris was completed in 1889 for the World's Fair.",
    "A standard chessboard has 64 squares arranged in an 8 by 8 grid.",
    "Water boils at 100 degrees Celsius at standard atmospheric pressure.",
)

# ---- Embedding hyperparameters (hoisted; no magic numbers inline) ---------------------------
EMBED_DIM = 256  # hashing dimensionality: big enough that token collisions are rare on this corpus
TOP_K = 3  # how many passages to retrieve and stuff into the prompt
RANDOM_SEED = 0  # seed for any stochastic step, so runs are reproducible
TOKEN_RE = re.compile(r"[a-z0-9]+")  # lowercase alphanumeric tokens; drops punctuation/case

# Two test questions: one answerable only from the private corpus, one a generic fact.
PRIVATE_QUESTION = "When was the Helios-7 satellite launched?"
GENERIC_QUESTION = "How many squares are on a chessboard?"

# The single passage that actually answers PRIVATE_QUESTION (index into CORPUS), used to assert
# the retriever surfaces the right evidence rather than something merely word-adjacent.
PRIVATE_GOLD_INDEX = 0


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens — the unit both the embedder and IDF count over."""
    return TOKEN_RE.findall(text.lower())  # findall over the precompiled regex = fast, punctuation-free


def compute_idf(corpus: tuple[str, ...]) -> dict[str, float]:
    """Inverse document frequency per token: log(N / df) — rare words weigh more than common ones.

    IDF is what makes "Helios" matter more than "the": a token appearing in every passage carries
    no discriminative signal, so its weight collapses toward zero; a token in one passage is a
    strong fingerprint for that passage.
    """
    n_docs = len(corpus)
    doc_freq: dict[str, int] = {}
    for doc in corpus:
        # set(...) so a token repeated within one passage still counts as ONE document, not many
        for token in set(tokenize(doc)):
            doc_freq[token] = doc_freq.get(token, 0) + 1
    # +1 inside the log keeps IDF strictly positive even for a token present in every document
    return {tok: float(np.log(n_docs / df) + 1.0) for tok, df in doc_freq.items()}


def embed(text: str, idf: dict[str, float], dim: int = EMBED_DIM) -> np.ndarray:
    """Map text -> a fixed-length L2-normalized vector via IDF-weighted feature hashing.

    Feature hashing (the "hashing trick") gives every token a deterministic slot in a `dim`-wide
    vector via `hash(token) % dim`, with no vocabulary to store. Each token adds its IDF weight to
    its slot, so the vector is a weighted bag-of-words. We L2-normalize at the end so that cosine
    similarity reduces to a plain dot product — the geometry the retriever relies on.
    """
    vec = np.zeros(dim, dtype=np.float64)
    for token in tokenize(text):
        # a stable per-token slot; we fold the token's own hash to pick a +/- sign so unrelated
        # tokens colliding in the same slot tend to cancel rather than spuriously reinforce
        slot = _stable_hash(token) % dim
        sign = 1.0 if (_stable_hash(token + "#sign") % 2 == 0) else -1.0
        vec[slot] += sign * idf.get(token, 1.0)  # unseen tokens (e.g. in a query) default to weight 1
    norm = np.linalg.norm(vec)
    # guard the all-zero vector (text with no known tokens) so we never divide by zero
    return vec / norm if norm > 0 else vec


def _stable_hash(token: str) -> int:
    """A deterministic hash that does NOT depend on Python's per-process hash randomization.

    Python salts the built-in `hash()` of strings per process (PYTHONHASHSEED), which would make
    embeddings — and therefore every downstream result — change run to run. A small FNV-1a hash
    keeps the whole pipeline reproducible.
    """
    h = 0x811C9DC5  # FNV offset basis (32-bit)
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF  # FNV prime, masked to 32 bits
    return h


def build_index(corpus: tuple[str, ...], idf: dict[str, float]) -> np.ndarray:
    """Embed every passage once -> a (n_docs, dim) matrix. This is the 'index' we retrieve against."""
    # np.stack over a list comprehension: one row per passage, all rows the same width
    return np.stack([embed(doc, idf) for doc in corpus])


def cosine_top_k(
    query_vec: np.ndarray, index: np.ndarray, k: int = TOP_K
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of the k passages most similar to the query, best first.

    Because every row of `index` and the query are L2-normalized, the matrix-vector product
    `index @ query_vec` is exactly the vector of cosine similarities — one dot product per passage.
    """
    scores = index @ query_vec  # (n_docs, dim) @ (dim,) -> (n_docs,) cosine score per passage
    # argsort is ascending; take the last k and reverse to get the top-k in descending score order
    top = np.argsort(scores)[-k:][::-1]
    return top, scores[top]


def build_prompt(question: str, passages: list[str]) -> str:
    """Stitch retrieved passages + the question into the augmented prompt the generator sees.

    This is the 'augment' step: the model's input is no longer just the question, it's the question
    grounded in evidence. The instruction line tells the generator to answer *from the context*.
    """
    context = "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(passages))  # numbered so answers can cite
    return (
        "Answer the question using ONLY the context below. "
        "If the context does not contain the answer, say you don't know.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
    )


@dataclass
class Answer:
    """A generator's reply plus the passage it grounded on (None when it had no evidence)."""

    text: str
    grounded_on: str | None


def generate_ungrounded(question: str) -> Answer:
    """Stand in for a frozen LLM answering from parametric memory alone — no retrieval.

    A real LLM has no Helios-7 facts in its weights, so it would refuse or hallucinate. We model the
    *honest* version of that failure: it knows the generic fact (chessboard) but not the private one.
    The lesson is the contrast with the grounded path, not this toy's cleverness.
    """
    parametric_memory = {
        "chessboard": "A chessboard has 64 squares.",  # a fact plausibly in pretraining data
    }
    for key, fact in parametric_memory.items():
        if key in question.lower():
            return Answer(fact, grounded_on=None)
    # the private fact isn't in memory -> the honest outcome is "I don't know" (vs. a hallucination)
    return Answer("I don't have information about that.", grounded_on=None)


def generate_grounded(question: str, passages: list[str]) -> Answer:
    """Deterministic extractive 'generator': answer from the retrieved passages, not from memory.

    Real RAG feeds the augmented prompt to an LLM. To keep this runnable with no model, we extract
    the best-matching passage as the answer — which is enough to *prove the mechanism*: the answer
    now comes from retrieved evidence. The selection mirrors retrieval: score each passage's token
    overlap with the question and return the strongest, so the 'answer' is grounded and citable.
    """
    q_tokens = set(tokenize(question)) - _STOPWORDS  # drop filler so overlap reflects content words
    best_passage, best_overlap = None, -1
    for passage in passages:
        overlap = len(q_tokens & set(tokenize(passage)))  # shared content tokens = grounding strength
        if overlap > best_overlap:
            best_passage, best_overlap = passage, overlap
    if best_passage is None or best_overlap == 0:
        return Answer("I don't have information about that.", grounded_on=None)
    return Answer(best_passage, grounded_on=best_passage)


# Stopwords kept tiny and explicit — just enough that question/passage overlap reflects content,
# not grammar. A production system would use a real list or learned weights.
_STOPWORDS = frozenset(
    {"the", "a", "an", "is", "was", "were", "are", "of", "on", "in", "to", "how", "when", "what", "many"}
)


def rag_answer(
    question: str, corpus: tuple[str, ...], idf: dict[str, float], index: np.ndarray, k: int = TOP_K
) -> tuple[Answer, np.ndarray, np.ndarray]:
    """End-to-end RAG: embed query -> top-k retrieve -> augment -> grounded generate.

    Returns the grounded answer plus the retrieved (indices, scores) so callers can inspect exactly
    what evidence the answer stood on.
    """
    query_vec = embed(question, idf)  # same embedder as the corpus -> query and passages share a space
    top_idx, top_scores = cosine_top_k(query_vec, index, k=k)
    retrieved = [corpus[i] for i in top_idx]  # the actual passage strings, best first
    answer = generate_grounded(question, retrieved)  # generate FROM the retrieved evidence
    return answer, top_idx, top_scores


def _report_device() -> str:
    """Pick the best torch device for the version banner; the retrieval math itself is pure numpy."""
    try:
        import torch

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print("torch:", torch.__version__, "| device:", device)
        return device
    except ImportError:
        print("torch: not installed (retrieval is pure numpy — unaffected)")
        return "cpu"


def main() -> None:
    np.random.seed(RANDOM_SEED)
    _report_device()
    print("numpy:", np.__version__)
    print(f"corpus: {len(CORPUS)} passages | embed_dim: {EMBED_DIM} | top_k: {TOP_K}\n")

    idf = compute_idf(CORPUS)
    index = build_index(CORPUS, idf)
    # Shapes first — prove the index is what we think before trusting any retrieval result.
    print(f"index shape (n_docs, dim): {index.shape}")
    assert index.shape == (len(CORPUS), EMBED_DIM), "index shape must be (n_docs, embed_dim)"
    # Every row L2-normalized -> cosine == dot product; check a couple of norms are ~1.
    norms = np.linalg.norm(index, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-9), "every passage embedding must be unit-norm"
    print(f"all passage norms == 1.0: {np.allclose(norms, 1.0, atol=1e-9)}\n")

    # --- Retrieval, visible: show the top-k for the private question, scores and all ---
    print(f"QUESTION (private): {PRIVATE_QUESTION}")
    q_vec = embed(PRIVATE_QUESTION, idf)
    top_idx, top_scores = cosine_top_k(q_vec, index, k=TOP_K)
    print(f"top-{TOP_K} retrieved (index, cosine):")
    for rank, (idx, score) in enumerate(zip(top_idx, top_scores), start=1):
        print(f"  {rank}. doc[{idx}] cos={score:.3f}  | {CORPUS[idx]}")
    # Correctness BEFORE any claim: the gold passage must be retrieved, and rank #1.
    assert PRIVATE_GOLD_INDEX in top_idx, "the answering passage must be in the top-k"
    assert top_idx[0] == PRIVATE_GOLD_INDEX, "the answering passage must rank #1 for this query"
    print(f"gold passage doc[{PRIVATE_GOLD_INDEX}] retrieved at rank 1: True\n")

    # --- The headline contrast: ungrounded vs grounded on the SAME private question ---
    ungrounded = generate_ungrounded(PRIVATE_QUESTION)
    grounded, _, _ = rag_answer(PRIVATE_QUESTION, CORPUS, idf, index)
    print("UNGROUNDED (parametric memory only):")
    print(f"  -> {ungrounded.text}")
    print("GROUNDED (retrieve-then-generate):")
    print(f"  -> {grounded.text}")
    assert grounded.grounded_on is not None, "the grounded answer must cite a retrieved passage"
    assert "March 3rd, 2024" in grounded.text, "grounded answer must contain the launch date"
    assert "March 3rd, 2024" not in ungrounded.text, "ungrounded path cannot know the private date"
    print("grounded answer contains the correct date; ungrounded does not: True\n")

    # --- The augmented prompt the generator actually sees ---
    retrieved_passages = [CORPUS[i] for i in top_idx]
    print("AUGMENTED PROMPT (what the generator receives):")
    print("-" * 70)
    print(build_prompt(PRIVATE_QUESTION, retrieved_passages))
    print("-" * 70)

    # --- A generic question the parametric model already knows -> RAG agrees ---
    print(f"\nQUESTION (generic): {GENERIC_QUESTION}")
    generic_ans, _, _ = rag_answer(GENERIC_QUESTION, CORPUS, idf, index)
    print(f"  grounded -> {generic_ans.text}")
    assert "64 squares" in generic_ans.text, "generic answer must retrieve the chessboard fact"


if __name__ == "__main__":
    main()
