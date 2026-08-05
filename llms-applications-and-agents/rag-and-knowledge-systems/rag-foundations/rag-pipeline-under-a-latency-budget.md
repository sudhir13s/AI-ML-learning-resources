---
id: rag-pipeline-under-a-latency-budget
title: RAG Pipeline Under a Latency Budget
minutes: 15
category: rag-and-knowledge-systems
---
# RAG basics under latency budget

> Curated concept note, migrated from the in-app practice library into the canonical repo so the content lives once at the source. Original wording.

## Problem statement

Build a retrieval-augmented chatbot for an enterprise knowledge base: 10M documents, ~2 kB average, updated daily. The bot must answer factual questions grounded in the corpus, refuse when unsure, and respond within a 2-second p95 latency budget. The interviewer will push on chunking, retrieval quality, grounding, and the failure modes when the LLM hallucinates despite retrieved context.

## Core concepts

- **Retrieval-Augmented Generation** = LLM answers a question using context fetched from an external store at query time, rather than relying only on parametric memory.
- Pipeline: **chunk → embed → index → retrieve → (optionally rerank) → assemble prompt → generate**.
- Retrieval quality dominates end-to-end answer quality. A great LLM cannot compensate for bad retrieval.
- Grounding = instructing the LLM to only answer from retrieved context and to cite sources. Still not a hallucination guarantee.
- Latency budget compounds: embed the query (50 ms) + vector search (20 ms) + rerank (100 ms) + LLM generate (1-1.5 s) = close to the 2-second ceiling.

## Deep dive

### Chunking

Split each document into passages of ~300-800 tokens with 10-20% overlap. Smaller chunks = higher precision retrieval but miss cross-chunk context; larger = better coherence but noisier top-K.

- **Fixed-size** chunks are the cheap baseline.
- **Semantic** chunking respects paragraph / section boundaries.
- **Hierarchical** chunking stores multiple granularities (section, paragraph, sentence) and retrieves at the right level.

### Embedding

Pass each chunk through an embedding model (e.g. `text-embedding-3-small`, `bge-large`, `nomic-embed`) to produce a fixed-length vector (384-1536 dims).

- Same model for ingestion and query. Mixing models is the single most common bug.
- Normalize embeddings to unit length if using cosine similarity; many ANN libs assume this.
- Embedding cost = one-time per document (+ delta on updates). Query embed is on the hot path — cache query embeddings if traffic patterns are repetitive.

### Indexing

Store vectors in an ANN index (HNSW, IVF-PQ, ScaNN). See the vector-index gate for tradeoffs. 10M vectors at 768 dims × 4 bytes = ~30 GB; HNSW metadata ~2×. Fits in RAM on a single node at this scale.

### Retrieval + reranking

1. **First-stage retrieval**: approximate nearest neighbor returns top-K (e.g. K=50). Fast (~5-20 ms) but noisy.
2. **Rerank** (optional, high-impact): a cross-encoder scores (query, chunk) pairs and reorders. Top-K becomes top-N (e.g. N=5). Cross-encoders are ~10× more accurate than bi-encoders but ~100× slower — rerank only 50 candidates, not 10k.
3. **Hybrid retrieval**: combine BM25 (keyword) + vector (semantic) with reciprocal rank fusion. Catches both exact-match queries (product SKUs, code identifiers) and paraphrases.

### Prompt assembly

```text
System: Answer the user question using ONLY the provided context.
        If the context does not contain the answer, say "I don't know".
        Cite sources as [chunk_id].

Context:
[1] (chunk text)
[2] (chunk text)
...

User: <original question>
```

- Put the question both before and after the context if the context is long — LLMs have lost-in-the-middle attention bias.
- Include enough chunks to cover the answer but not so many that you dilute with irrelevant passages (the "needle in a haystack" degrades past ~5-10 chunks).

### Grounding + refusal

Instruct the model to cite chunk ids for every factual claim. After generation, verify citations exist in the retrieved set; if a claim lacks citation, either discard it or flag the response.

Refusal: explicit "say I don't know if unsure" in the system prompt plus a confidence threshold on the retrieval score. If top-1 similarity is below threshold, skip generation and return "no answer found".

### Latency decomposition

| Stage | Typical cost |
|---|---|
| Query embedding | 20-80 ms |
| ANN search | 5-30 ms |
| Rerank (50 candidates) | 50-200 ms |
| LLM generation (streamed) | 500-2000 ms (TTFT 200-500 ms) |
| Network + overhead | 50-100 ms |

Stream the LLM response token-by-token — users perceive TTFT (time to first token), not total latency.

### Evaluation

- **Retrieval**: recall@K, precision@K. Build a small labeled set of (query, gold-chunk) pairs.
- **Answer quality**: LLM-as-judge with a clear rubric, cross-checked by humans on a sample.
- **Faithfulness**: does the answer only cite information from retrieved chunks? Measurable with citation overlap.

## Discussion prompts

1. Walk through one query end-to-end with concrete latency numbers. Where is your budget spent?
2. Why is the same embedding model required for ingestion and query? What does the bug look like if you accidentally mix?
3. Your retrieval top-5 contains the answer but the LLM confidently produces a different wrong answer. What is happening and how do you fix it?
4. How would you decide whether to add a cross-encoder reranker? What signals would trigger that investment?
5. A user asks a question whose answer is not in your corpus. Walk through the refusal path and how you avoid hallucination.

## Things to defend

- [ ] Draw the full pipeline (chunk → embed → index → retrieve → rerank → prompt → generate) and cite one concrete technology per stage.
- [ ] Defend your chunk size with a tradeoff (precision vs coherence).
- [ ] Explain hybrid BM25 + vector retrieval and when it beats pure vector.
- [ ] Cite lost-in-the-middle and propose a prompt layout that mitigates it.
- [ ] State the latency budget decomposition with numbers.
- [ ] Name at least two retrieval metrics (recall@K, precision@K, MRR) and how you'd collect labels.
- [ ] Describe the grounding and refusal path end-to-end.
- [ ] Know the failure mode where retrieval is correct but generation hallucinates anyway, and one mitigation.
