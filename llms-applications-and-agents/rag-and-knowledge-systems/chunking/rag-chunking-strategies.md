---
id: rag-chunking-strategies
title: RAG Chunking Strategies
minutes: 20
category: rag-and-knowledge-systems
---
# RAG chunking strategies

> Curated concept note, migrated from the in-app practice library into the canonical repo so the content lives once at the source. Original wording.

## Problem statement

You are building retrieval for a knowledge-base QA system. Embedding, indexing, and reranking are largely off-the-shelf — but **the unit you index is a choice you make, and it caps the quality of everything downstream.** A reranker cannot rescue a chunk that split a sentence in half; an LLM cannot ground an answer in context that was never retrieved because the relevant fact landed in a different chunk than the one that matched the query. The interviewer will push on how you decide chunk boundaries, chunk size, and overlap, what metadata you attach, how you handle tables and code, and the failure modes when chunking goes wrong. Treat this as **the highest-leverage decision in the whole RAG pipeline.**

## Core concepts

- **Chunking** = splitting source documents into the passages you embed and retrieve. The chunk is the atomic unit of retrieval: you retrieve whole chunks, never fragments of them.
- **The retrieval-precision vs. context-completeness tradeoff** is the spine of every chunking decision. Small chunks match queries precisely but may not contain the full answer; large chunks carry more context but dilute the embedding and drag in irrelevant text.
- **Chunking must respect the structure of meaning in the source**, not impose a uniform token grid. A Slack thread, a Confluence page, a Jira ticket, and a code file each carry meaning at a different granularity. One strategy produces garbage on at least half of heterogeneous sources.
- **Chunk size must match the embedding model's effective window**, not just its max-token limit. Embedding models degrade well before their stated maximum.
- **Metadata travels with the chunk** — source, section, timestamps, and access permissions — and is as load-bearing as the text itself for filtering, ranking, and citation.

## Deep dive

### Why chunking is the highest-leverage decision

Retrieval quality dominates end-to-end answer quality, and chunking sets the ceiling on retrieval quality. Three independent things have to go right and all are decided at chunk time:

1. **Matchability** — the chunk's embedding has to land near the query's embedding. A chunk that mixes three unrelated topics produces a muddy averaged vector that matches nothing well.
2. **Sufficiency** — once retrieved, the chunk has to actually contain the answer. If the answer spans a boundary, no single chunk is sufficient and the LLM either hallucinates the gap or refuses.
3. **Citability** — the chunk has to map back to a precise, linkable location in the source so the answer can be grounded and verified.

Every other stage (rerank, prompt assembly, generation) operates on chunks it cannot improve. This is why chunking, not model choice, is usually the first thing to tune when retrieval is weak.

### Fixed-size chunking

Split on a fixed token count (e.g. 512 tokens) with a fixed overlap, ignoring document structure. This is the **cheap baseline**: trivial to implement, uniform, predictable storage. Its weakness is that it is blind to meaning — it happily cuts mid-sentence, mid-paragraph, mid-table, and mid-function. Use it as a starting point or for genuinely unstructured text (raw logs, transcripts with no markup), and graduate to a structure-aware strategy the moment the source has structure worth preserving.

### Semantic (paragraph/section-aware) chunking

Split on the document's natural boundaries — paragraphs, headings, list items, sentence groups — so each chunk is a coherent unit of thought. **Heading-aware splitting** (break on H1/H2/H3) keeps a chunk scoped to a single section and lets you record the section heading as metadata. A more advanced variant detects topic shifts by measuring the embedding distance between adjacent sentences and cutting where similarity drops. Semantic chunking produces cleaner embeddings and far better citations because each chunk corresponds to a real, nameable location. The cost is a structure-aware parser per format and more variable chunk sizes.

### Hierarchical / parent-document chunking

Store the document at **multiple granularities at once** — section, paragraph, and sentence — and decouple what you *match on* from what you *return*. The canonical pattern is **parent-document retrieval**: index small child chunks (precise matching) but, on a hit, return the larger parent chunk (full context) to the LLM. This dissolves the precision-vs-context tradeoff: you get the matchability of a small chunk and the sufficiency of a large one. The price is extra storage (you keep both levels) and a small indirection at retrieval time to resolve child → parent. This is the strongest default when answers tend to need surrounding context.

### Sentence-window chunking

A lightweight cousin of hierarchical: embed and match on a **single sentence**, but at retrieval time expand to a window of N sentences before and after for context. You get the precision of sentence-level matching with enough surrounding text for the LLM to reason. Cheaper than full parent-document indexing (no separate parent store) and well suited to dense, fact-heavy prose where the answer is a sentence but its meaning depends on its neighbours.

### Chunk size and overlap tradeoffs

- **Size.** Typical chunks run ~300–800 tokens. **Smaller = higher retrieval precision** (tighter embedding, sharper match) **but misses cross-chunk context**; **larger = better coherence but a noisier, diluted embedding** and more irrelevant text in the prompt. There is no universal number — it depends on how meaning is packed in the source and on the embedding model (below).
- **Overlap.** Carry ~10–20% of tokens from the end of one chunk into the start of the next so a fact that straddles a boundary appears whole in at least one chunk. Overlap is the cheap insurance against splitting an answer across two chunks. Too little overlap loses cross-boundary facts; too much inflates storage and produces near-duplicate retrievals that crowd the top-K. Tune overlap to the source's sentence/paragraph length, not a blind percentage.

### Metadata to attach to every chunk

The chunk text is only half the chunk. Attach, at minimum:

- **Source / provenance** — `source_type`, `source_url` (a deep link), `document_id`, `title` — so every answer is citable and verifiable.
- **Structural position** — `section_heading`, `parent_document_id`, `thread_id` — for hierarchical retrieval and human-readable citations.
- **Temporal fields** — `created_at`, `updated_at`, `ingested_at` — the pair `updated_at` vs `ingested_at` is what lets you detect a stale chunk later.
- **Access control (`acl_groups`)** — in any enterprise corpus, the chunk's permissions are materialized onto the chunk so retrieval can filter on who-can-see-it. A permission error here is a security incident, not a relevance bug.
- **Ranking signals** — `author`, team/department, `content_type` — features a reranker can weigh beyond raw similarity.

Metadata is what turns a vector store into a *filterable, citable, permission-aware* retrieval tier.

### Tables and code

Generic splitters destroy structured content, so handle it explicitly:

- **Tables.** Never cut a table mid-row. Keep the table (or a logically complete sub-table) together, and prepend the column headers and a one-line caption to each table chunk so the rows stay interpretable out of context. For wide tables, consider serializing to a linearized "header: value" form that embeds more meaningfully than raw grid text.
- **Code.** Split on **syntactic boundaries** (function, class, method) using an AST or language-aware splitter, not arbitrary token counts that bisect a function. Attach the file path, language, and enclosing symbol as metadata. For long functions, keep the signature and docstring with each piece so a fragment is still identifiable.

The general rule: **identify the natural atomic unit of each content type and refuse to split below it.**

### Matching chunk size to the embedding model's effective window

A model's max-token limit is not its *effective* window. Most embedding models are trained on relatively short passages and their representation quality degrades well before the hard limit — pack 2,000 tokens into a model whose sweet spot is ~512 and the resulting vector blurs across too many topics to match anything precisely. Conversely, chunks far below the model's effective window waste capacity and fragment answers. **Choose chunk size for the model you are actually using**, and use the same model (and the same chunk-size regime) for ingestion and query — mismatched embedding spaces are the single most common, and most silent, RAG bug.

## Common pitfalls

- **Splitting mid-sentence / mid-idea.** Fixed-size chunking with no overlap guarantees this. The fix is overlap plus structure-aware boundaries.
- **Losing cross-chunk context.** The answer needs two adjacent paragraphs but they landed in different chunks and only one was retrieved. Fix with overlap, parent-document, or sentence-window chunking.
- **Chunk-size/embedding-window mismatch.** Chunks far larger than the model's effective window produce diluted vectors that match poorly; far smaller fragment answers. Size to the model.
- **Shredding tables and code.** A uniform token splitter cuts a table mid-row and a function mid-body, yielding chunks that are unparseable and unretrievable. Use content-type-aware splitting.
- **Forgetting metadata.** A chunk with no source link can't be cited; with no ACL can't be safely filtered; with no timestamps can't be detected as stale. Metadata is not optional.
- **One strategy for heterogeneous sources.** A generic splitter that works on prose destroys Slack threads, Jira tickets, and spreadsheets. Make chunking source-adaptive.

## Things to defend

- [ ] Name four chunking strategies (fixed-size, semantic/section-aware, hierarchical/parent-document, sentence-window) and when each wins.
- [ ] Defend a chunk size and an overlap with the precision-vs-context tradeoff.
- [ ] Explain parent-document retrieval and how it dissolves that tradeoff.
- [ ] List the metadata you attach to every chunk and why (source, section, timestamps, ACL, ranking signals).
- [ ] Describe how you chunk a table and a code file differently from prose.
- [ ] State why chunk size must match the embedding model's *effective* window, not its max.
- [ ] Identify the failure mode where an answer is split across chunks and the mitigation.
