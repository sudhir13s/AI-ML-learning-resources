"""Stage 1 of the ANN pipeline: build a REAL passage corpus and embed it.

This script does the *heavy, one-off* work that the teaching notebook depends on:

  1. Stream real articles from Simple English Wikipedia (`wikimedia/wikipedia`, a freely
     licensed CC-BY-SA dump on the Hugging Face Hub).
  2. Chunk each article into paragraph-sized *passages* — exactly what a RAG pipeline
     indexes (chapter 2's chunking + chapter 3's embeddings feed this chapter's index).
  3. Embed every passage with a real sentence-transformer (`BAAI/bge-small-en-v1.5`,
     384-dim, L2-normalized) so cosine similarity is a plain dot product.
  4. Hold out a set of passages as *queries* (their text is a natural in-distribution query),
     re-embedded the same way.
  5. Save everything to `data/` as `.npy` + `.jsonl` so the notebook and `vector_indexes.py`
     can load real vectors WITHOUT importing torch.

Why the split? On this machine `faiss` and `torch` both link `libomp` and **crash the
process if co-loaded** (OpenMP double-initialisation). So embedding (torch) and indexing
(faiss) must live in *separate processes*. This script owns the torch half; it writes plain
numpy arrays to disk; `vector_indexes.py` and the notebook own the faiss half and never touch
torch. That separation is also good production hygiene: embedding is a batch job, indexing is
a serving job.

Run once (a few minutes, downloads the model + a Wikipedia slice on first run):

    python embed_corpus.py                 # default ~30k passages
    N_TARGET_PASSAGES=60000 python embed_corpus.py

Deterministic: fixed seed for the query sample; embeddings are a deterministic function of the
model + text. Real-world timing varies run to run (reported, not asserted).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

# ---- Configuration (env-overridable; no magic numbers inline) -----------------------------------
DATA_DIR = Path(__file__).resolve().parent / "data"
WIKI_DATASET = "wikimedia/wikipedia"  # freely licensed (CC-BY-SA) real Wikipedia on the HF Hub
WIKI_CONFIG = "20231101.simple"  # Simple English Wikipedia — real articles, manageable size
EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # a real, strong small retrieval model (384-dim)
N_TARGET_PASSAGES = int(os.environ.get("N_TARGET_PASSAGES", "30000"))  # real corpus size
N_QUERIES = int(os.environ.get("N_QUERIES", "500"))  # held-out passages reused as queries
MIN_PASSAGE_CHARS = 200  # drop stubs/navigation lines — keep real, substantive paragraphs
MAX_PASSAGE_CHARS = 700  # cap paragraph length (chapter 2's chunk-size discipline; faster encode)
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "64"))  # encode batch size (smaller = faster here)
SEED = 0  # reproducible query hold-out


def _paragraphs(text: str) -> list[str]:
    """Split an article into paragraph passages, keeping only substantive ones.

    Real RAG chunking is more elaborate (chapter 2), but paragraph splitting on a double
    newline is a faithful, honest baseline: it yields coherent, self-contained passages of
    the right size to embed and retrieve.
    """
    out: list[str] = []
    for para in text.split("\n\n"):
        para = " ".join(para.split())  # collapse internal whitespace/newlines
        if len(para) < MIN_PASSAGE_CHARS:
            continue
        out.append(para[:MAX_PASSAGE_CHARS])
    return out


def build_passages(n_target: int) -> list[dict[str, str]]:
    """Stream real Wikipedia articles and chunk them into ~`n_target` passages."""
    from datasets import load_dataset

    print(f"streaming {WIKI_DATASET} [{WIKI_CONFIG}] ...")
    ds = load_dataset(WIKI_DATASET, WIKI_CONFIG, split="train", streaming=True)
    passages: list[dict[str, str]] = []
    for article in ds:
        title = article["title"]
        for j, para in enumerate(_paragraphs(article["text"])):
            passages.append({"id": f"{article['id']}#{j}", "title": title, "text": para})
        if len(passages) >= n_target:
            break
    passages = passages[:n_target]
    print(f"built {len(passages):,} real passages from Simple English Wikipedia")
    return passages


def embed_texts(texts: list[str], model) -> np.ndarray:
    """Embed a list of texts to L2-normalized float32 vectors (cosine == dot product)."""
    emb = model.encode(
        texts,
        batch_size=EMBED_BATCH,
        show_progress_bar=True,
        normalize_embeddings=True,  # unit vectors -> inner product IS cosine similarity
        convert_to_numpy=True,
    )
    return np.ascontiguousarray(emb.astype("float32"))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    passages = build_passages(N_TARGET_PASSAGES)
    texts = [p["text"] for p in passages]

    from sentence_transformers import SentenceTransformer

    print(f"loading embedder {EMBED_MODEL} ...")
    model = SentenceTransformer(EMBED_MODEL)
    dim = model.get_embedding_dimension()
    print(f"embedding {len(texts):,} passages (dim={dim}) ...")

    t0 = time.perf_counter()
    corpus_emb = embed_texts(texts, model)
    embed_secs = time.perf_counter() - t0
    print(f"embedded corpus in {embed_secs:.1f}s ({len(texts) / embed_secs:.0f} passages/s)")

    # Hold out real passages as queries: their own text is a natural in-distribution query,
    # and we KNOW a strong true neighbour exists (the passage itself + its topical siblings).
    q_idx = rng.choice(len(passages), size=min(N_QUERIES, len(passages)), replace=False)
    query_texts = [passages[i]["text"] for i in q_idx]
    query_emb = embed_texts(query_texts, model)

    # Persist real vectors + text + metadata. numpy for vectors, jsonl for human-readable text.
    np.save(DATA_DIR / "corpus_emb.npy", corpus_emb)
    np.save(DATA_DIR / "query_emb.npy", query_emb)
    np.save(DATA_DIR / "query_idx.npy", q_idx.astype(np.int64))
    with (DATA_DIR / "passages.jsonl").open("w", encoding="utf-8") as fh:
        for p in passages:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    meta = {
        "dataset": f"{WIKI_DATASET}:{WIKI_CONFIG}",
        "embed_model": EMBED_MODEL,
        "n_passages": len(passages),
        "n_queries": int(len(q_idx)),
        "dim": int(dim),
        "normalized": True,
        "embed_seconds": round(embed_secs, 2),
        "embed_passages_per_sec": round(len(texts) / embed_secs, 1),
        "seed": SEED,
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\nsaved to", DATA_DIR)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
