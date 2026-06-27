"""From-scratch document chunking: three strategies, measured on a toy RAG corpus.

How you split a document decides what retrieval can ever find. This script builds three chunkers
from primitives so every cut is inspectable — fixed-size-with-overlap, recursive (structure-aware),
and semantic (cut where adjacent-sentence embedding similarity drops) — then *measures* the
retrieval-quality difference between them on probe questions, reusing the chapter-01 hashing
embedder + cosine retrieval so the whole thing runs on CPU in milliseconds (no model download).

The headline demonstration: a naive fixed-size split cuts a fact in half (the date is separated from
its subject) so neither fragment is retrievable whole; a recursive/semantic split keeps the fact
intact and retrieval finds it. The lesson is the *cut points*, not the toy's cleverness.

This is the same verified code embedded in the concept page and the teaching notebook.
Verified on Python 3.12 / numpy 2.x. Device-agnostic for the torch version banner (CUDA / MPS / CPU);
the chunking and retrieval math is pure numpy and identical on any machine. Hash-deterministic
(fixed FNV-1a embedder, no random step) — every number is identical on every run.

Run:
    python document_chunking.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# ---- The toy document: a short multi-section "spec" with headings and paragraphs ----------------
# Realistic structure (headings + blank-line-separated paragraphs + a sentence carrying a key fact)
# so the three chunkers visibly differ. The launch-date sentence is the fact retrieval must keep whole.
DOCUMENT = """# Helios-7 Mission Overview

The Helios-7 satellite is an Earth-observation platform operated by the Nairobi office.
It was launched on March 3rd, 2024 from the Kourou spaceport aboard an Ariane 6 rocket.

# Instruments

Helios-7 carries a hyperspectral imager with a ground resolution of 4 meters.
The imager captures 200 spectral bands across the visible and near-infrared range.

# Orbit

Helios-7 completes one orbit of Earth every 97 minutes in a sun-synchronous orbit.
This orbit keeps the local solar time of each pass roughly constant for stable imaging."""

# ---- Chunking hyperparameters (hoisted; no magic numbers inline) --------------------------------
FIXED_CHUNK_CHARS = 140  # fixed-size chunk width in chars — chosen so a boundary falls INSIDE the date
FIXED_OVERLAP_CHARS = 0  # overlap for the *naive* fixed split (0 = the worst case that splits facts)
FIXED_OVERLAP_FIX_CHARS = 40  # overlap for the *fixed* version — enough to make a straddling fact whole
RECURSIVE_MAX_CHARS = 240  # recursive splitter target: pack whole paragraphs up to this width
SEMANTIC_PERCENTILE = 35  # cut at adjacent-sentence similarity dips below this percentile of the trace
EMBED_DIM = 256  # hashing embedder width (same as chapter 01)
TOP_K = 1  # retrieve the single best chunk so "did the right fact survive chunking?" is unambiguous
TOKEN_RE = re.compile(r"[a-z0-9]+")  # lowercase alphanumeric tokens; drops punctuation/case
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")  # split on sentence-ending punctuation followed by space

# The probe question and the fact that answers it — the fact retrieval must keep whole to find.
PROBE_QUESTION = "When was the Helios-7 satellite launched?"
ANSWER_PHRASE = "March 3rd, 2024"


# ============================ embedding + retrieval (reused from chapter 01) ======================
def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens — the unit both the embedder and IDF count over."""
    return TOKEN_RE.findall(text.lower())


def _stable_hash(token: str) -> int:
    """Deterministic FNV-1a hash, independent of Python's per-process hash randomization.

    Built-in hash() is salted per process (PYTHONHASHSEED), which would make embeddings change run to
    run; FNV-1a keeps the whole pipeline reproducible.
    """
    h = 0x811C9DC5  # FNV offset basis (32-bit)
    for byte in token.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & 0xFFFFFFFF  # FNV prime, masked to 32 bits
    return h


def compute_idf(docs: list[str]) -> dict[str, float]:
    """Inverse document frequency log(N/df)+1 per token — rare words weigh more than common ones."""
    n_docs = len(docs)
    doc_freq: dict[str, int] = {}
    for doc in docs:
        for token in set(tokenize(doc)):  # set: a token repeated in one doc counts once
            doc_freq[token] = doc_freq.get(token, 0) + 1
    return {tok: float(np.log(n_docs / df) + 1.0) for tok, df in doc_freq.items()}


def embed(text: str, idf: dict[str, float], dim: int = EMBED_DIM) -> np.ndarray:
    """Map text -> a fixed-length L2-normalized vector via IDF-weighted feature hashing.

    Each token adds its IDF weight (with a hash-derived sign so collisions tend to cancel) to its slot
    `hash(token) % dim`; L2-normalizing makes cosine similarity a plain dot product downstream.
    """
    vec = np.zeros(dim, dtype=np.float64)
    for token in tokenize(text):
        slot = _stable_hash(token) % dim
        sign = 1.0 if (_stable_hash(token + "#sign") % 2 == 0) else -1.0
        vec[slot] += sign * idf.get(token, 1.0)  # unseen query tokens default to weight 1
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec  # guard the all-zero vector (no known tokens)


def build_index(chunks: list[str], idf: dict[str, float]) -> np.ndarray:
    """Embed every chunk once -> a (n_chunks, dim) unit-norm matrix to retrieve against."""
    return np.stack([embed(c, idf) for c in chunks])


def cosine_top_k(
    query_vec: np.ndarray, index: np.ndarray, k: int = TOP_K
) -> tuple[np.ndarray, np.ndarray]:
    """Return (indices, scores) of the k chunks most similar to the query, best first.

    Rows of `index` and the query are unit-norm, so `index @ query_vec` is the cosine score vector.
    """
    scores = index @ query_vec
    top = np.argsort(scores)[-k:][::-1]  # top-k descending
    return top, scores[top]


# ================================ the three chunking strategies ===================================
def chunk_fixed(text: str, size: int = FIXED_CHUNK_CHARS, overlap: int = FIXED_OVERLAP_CHARS) -> list[str]:
    """Fixed-size character chunks with optional overlap — the simplest, content-blind strategy.

    Slides a window of `size` characters across the raw text, stepping by `size - overlap`. Cuts fall
    wherever the counter lands — mid-word, mid-sentence, mid-fact — which is exactly the failure the
    page demonstrates. Overlap re-includes the last `overlap` chars of each chunk at the start of the
    next, so a fact straddling a boundary appears whole in at least one chunk.
    """
    assert overlap < size, "overlap must be smaller than chunk size, else the window never advances"
    step = size - overlap  # how far the window moves each step
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunk = text[start : start + size].strip()  # strip so leading/trailing whitespace isn't 'content'
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def chunk_recursive(text: str, max_chars: int = RECURSIVE_MAX_CHARS) -> list[str]:
    """Structure-aware recursive splitter: prefer paragraph, then sentence, then hard cut.

    Mirrors LangChain's RecursiveCharacterTextSplitter idea: try the coarsest natural boundary first
    (paragraphs split on a blank line), and only descend to finer boundaries (sentences) when a piece
    still exceeds `max_chars`. This keeps whole ideas together instead of cutting at a blind offset.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]  # coarsest boundary: blank line
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)  # whole paragraph fits -> keep it intact
            continue
        # paragraph too big: fall back to sentence boundaries, packing greedily up to max_chars
        sentences = SENT_SPLIT_RE.split(para)
        current = ""
        for sent in sentences:
            candidate = f"{current} {sent}".strip()
            if len(candidate) <= max_chars:
                current = candidate  # still fits -> accumulate
            else:
                if current:
                    chunks.append(current)
                current = sent  # start a new chunk with the overflowing sentence
        if current:
            chunks.append(current)
    return chunks


def split_sentences(text: str) -> list[str]:
    """Flatten a document to a list of non-empty sentences (drops heading-only lines' blank tails)."""
    # join lines so sentences that wrap headings still parse, then split on sentence punctuation
    flat = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in SENT_SPLIT_RE.split(flat) if s.strip()]


def adjacent_similarities(sentences: list[str], idf: dict[str, float]) -> np.ndarray:
    """Cosine similarity between each consecutive sentence pair — the signal semantic chunking cuts on."""
    vecs = np.stack([embed(s, idf) for s in sentences])  # (n_sents, dim) unit-norm
    return np.array([float(vecs[i] @ vecs[i + 1]) for i in range(len(sentences) - 1)])  # adjacent cosines


def semantic_boundaries(
    sentences: list[str], idf: dict[str, float], percentile: float = SEMANTIC_PERCENTILE
) -> list[int]:
    """Return indices where a new chunk should START — at the largest adjacent-similarity dips.

    Embed each sentence, measure cosine similarity between consecutive sentences, and declare a
    boundary wherever that similarity falls below the `percentile`-th percentile of the trace — the
    breakpoint-percentile method LlamaIndex's SemanticSplitterNodeParser uses. A topic shift (one
    section to the next) shows up as a similarity dip, exactly where a human would cut. Returns the
    start indices of each segment (always includes 0).
    """
    sims = adjacent_similarities(sentences, idf)
    threshold = float(np.percentile(sims, percentile))  # the lowest `percentile`% of gaps become cuts
    starts = [0]
    for i, sim in enumerate(sims):
        if sim <= threshold:  # similarity dipped into the low tail -> sentence i+1 starts a new chunk
            starts.append(i + 1)
    return starts


def chunk_semantic(
    text: str, idf: dict[str, float], percentile: float = SEMANTIC_PERCENTILE
) -> list[str]:
    """Semantic chunking: group consecutive sentences, cutting where their embeddings diverge."""
    sentences = split_sentences(text)
    starts = semantic_boundaries(sentences, idf, percentile)
    bounds = starts + [len(sentences)]  # append the end so the last segment closes
    chunks = []
    for a, b in zip(bounds, bounds[1:]):
        chunks.append(" ".join(sentences[a:b]))  # join each segment's sentences into one chunk
    return chunks


# ====================================== measuring quality =========================================
@dataclass
class StrategyResult:
    """Retrieval outcome for one chunking strategy on the probe question."""

    name: str
    n_chunks: int
    retrieved_chunk: str
    answer_intact: bool  # did the single retrieved chunk contain the full answer phrase?


def evaluate_strategy(name: str, chunks: list[str], idf: dict[str, float]) -> StrategyResult:
    """Index the chunks, retrieve the top-1 for the probe, and check the answer survived the split."""
    index = build_index(chunks, idf)
    q_vec = embed(PROBE_QUESTION, idf)
    top_idx, _ = cosine_top_k(q_vec, index, k=TOP_K)
    retrieved = chunks[int(top_idx[0])]
    return StrategyResult(
        name=name,
        n_chunks=len(chunks),
        retrieved_chunk=retrieved,
        answer_intact=ANSWER_PHRASE in retrieved,  # the fact is usable only if it's whole in the chunk
    )


def answer_survives_any_chunk(chunks: list[str]) -> bool:
    """Whether the full answer phrase appears intact in at least one chunk (recall ceiling)."""
    return any(ANSWER_PHRASE in c for c in chunks)


def _report_device() -> str:
    """Pick the best torch device for the version banner; the chunking/retrieval math is pure numpy."""
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
        print("torch: not installed (chunking + retrieval are pure numpy — unaffected)")
        return "cpu"


def main() -> None:
    _report_device()
    print("numpy:", np.__version__)
    print(f"document: {len(DOCUMENT)} chars | embed_dim: {EMBED_DIM} | probe: {PROBE_QUESTION!r}\n")

    # Build the three chunkings. IDF is computed over each strategy's own chunks (its retrieval corpus).
    naive = chunk_fixed(DOCUMENT, overlap=FIXED_OVERLAP_CHARS)
    fixed_overlap = chunk_fixed(DOCUMENT, overlap=FIXED_OVERLAP_FIX_CHARS)
    recursive = chunk_recursive(DOCUMENT)
    idf_for_semantic = compute_idf(split_sentences(DOCUMENT))  # sentence-level idf for boundary detection
    semantic = chunk_semantic(DOCUMENT, idf_for_semantic)

    strategies = {
        "fixed (no overlap)": naive,
        "fixed (+overlap)": fixed_overlap,
        "recursive": recursive,
        "semantic": semantic,
    }

    # --- Show the naive fixed split shattering the launch-date fact ---
    print(f"NAIVE FIXED SPLIT (size={FIXED_CHUNK_CHARS}, overlap=0) — watch the date get cut from its subject:")
    for i, c in enumerate(naive):
        marker = "  <-- launch date here" if ANSWER_PHRASE in c else ""
        print(f"  chunk[{i}] ({len(c):>3} chars): {c[:70]!r}{marker}")
    naive_intact = answer_survives_any_chunk(naive)
    print(f"answer phrase {ANSWER_PHRASE!r} intact in some chunk: {naive_intact}\n")

    # --- Per-strategy retrieval quality ---
    print(f"{'strategy':<20} | {'#chunks':>7} | {'answer in top-1':>15} | answer intact anywhere")
    print("-" * 78)
    results = []
    for name, chunks in strategies.items():
        res = evaluate_strategy(name, chunks, compute_idf(chunks))
        intact_anywhere = answer_survives_any_chunk(chunks)
        results.append((res, intact_anywhere))
        print(f"{name:<20} | {res.n_chunks:>7} | {str(res.answer_intact):>15} | {intact_anywhere}")

    # --- Correctness BEFORE any claim: the naive split must FAIL and the better ones must SUCCEED ---
    by_name = {r.name: (r, intact) for r, intact in results}
    assert not by_name["fixed (no overlap)"][0].answer_intact, "naive split should NOT retrieve the whole fact"
    assert by_name["fixed (+overlap)"][1], "adding overlap must make the fact survive in some chunk"
    assert by_name["recursive"][0].answer_intact, "recursive split must keep the fact whole and retrievable"
    assert by_name["semantic"][0].answer_intact, "semantic split must keep the fact whole and retrievable"
    print("\nnaive split loses the fact; overlap/recursive/semantic keep it: True")

    # --- The recursive chunks, to show structure-aware boundaries land on paragraphs ---
    print("\nRECURSIVE CHUNKS (cut on paragraph/sentence boundaries — ideas stay whole):")
    for i, c in enumerate(recursive):
        print(f"  chunk[{i}] ({len(c):>3} chars): {c[:70]!r}")


if __name__ == "__main__":
    main()
