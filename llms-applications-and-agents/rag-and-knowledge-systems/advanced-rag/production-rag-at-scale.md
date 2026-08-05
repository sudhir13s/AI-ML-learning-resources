---
id: production-rag-at-scale
title: Production RAG at Scale
minutes: 25
category: rag-and-knowledge-systems
---
# Production RAG at scale

> Curated concept note, migrated from the in-app practice library into the canonical repo so the content lives once at the source. Original wording.

## Problem statement

A demo RAG pipeline — embed, retrieve, generate — is a weekend project. **Production RAG at enterprise scale is a different system**, and this is a hard gate because the things that make it hard are precisely the things a demo ignores: permissions (never surface a document the user can't see), freshness (the corpus changes under you all day), grounding (the LLM will confidently exceed its evidence), evaluation (failures are silent — a wrong answer looks exactly like a right one), and the operational failure modes that emerge only at scale (stale index, retrieval drift, runaway cost). The interviewer will push on the end-to-end pipeline and then probe each of these as a place where a naive design breaks. The reference case here is enterprise document QA over fragmented SaaS sources, where **a permission error is not a relevance bug — it is a security incident.**

## Core concepts

- The production pipeline has two halves that meet at storage: an **ingestion path** (ingest → chunk → embed → index) that lands correct, permission-tagged, well-chunked content, and a **query path** (retrieve → rerank → generate → eval) that answers in seconds without ever violating a permission.
- **ACL-aware retrieval** is the defining enterprise constraint: every chunk carries a materialized access-control list, and the permission filter is pushed *into* the retrieval call so disallowed chunks are never even scored.
- **Hybrid retrieval** (dense + sparse, fused) plus a **cross-encoder reranker** is the standard quality stack; pure vector search silently fails on exact tokens.
- **Grounding is a defense, not a guarantee.** The LLM can hallucinate despite correct context; you defend with grounding prompts, citation, and a post-hoc faithfulness check.
- **RAG fails silently**, so evaluation is split by stage — retrieval recall vs. answer faithfulness — and ACL correctness is a security invariant measured at exactly 100%, not a quality target.
- The **scale failure modes** — stale index, retrieval drift, cost — are correctness- and economics-shaped, not availability-shaped, so no latency dashboard catches them.

## Deep dive

### The end-to-end production pipeline

Read it as a directed flow where each stage's output bounds the next:

```
ingest → chunk → embed → index → retrieve → rerank → generate → eval
```

- **Ingest.** A per-source connector authenticates, does incremental delta sync, and pulls content *plus permissions plus author identity*. Heterogeneous sources (a 20-token Slack message, a 5,000-token Confluence page, a structured Jira ticket) demand source-specific extraction.
- **Chunk.** Source-adaptive: thread-based for Slack, heading-based for Confluence, structured-field-fused for Jira, AST-based for code. One universal splitter produces garbage on half the sources.
- **Embed.** Same model for ingestion and query (mismatched spaces are the classic silent bug). Cost is one-time per document plus a delta on updates; the query embed is on the hot path.
- **Index.** Vectors into an ANN index (HNSW/IVF-PQ) **with the ACL groups stored as a server-side metadata filter**, and the same text into a sparse (BM25) index with the same ACL terms.
- **Retrieve.** ACL-filtered hybrid: dense top-K and sparse top-K, each permission-filtered, fused by Reciprocal Rank Fusion.
- **Rerank.** A cross-encoder reads (query, chunk) jointly and narrows top-50 → top-5, optionally weighting organizational signals (team, recency, source affinity).
- **Generate.** The LLM answers from the assembled, source-labeled context under a grounding prompt, streaming with citations.
- **Eval.** Online and offline measurement of every stage, because nothing upstream errors loudly when it degrades.

### ACL / permission-aware retrieval — the multi-tenant security problem

This is the section that separates enterprise RAG from generic RAG. The rule: **ACL is part of retrieval, not a post-processing step, and the bar is 100%.**

- **Materialize permissions onto every chunk.** Each source's idiosyncratic model — Slack channel membership, Confluence space restrictions, Drive file sharing, Jira project roles — is normalized into one filterable group namespace (`…:group:…`). At query time the user's identity resolves (via the corporate IdP / SCIM plus source memberships, cached for ~5ms lookups) to a set of group IDs, and retrieval demands `acl_groups ∩ user_groups ≠ ∅`.
- **Push the predicate into the database.** The ACL filter is applied *during* the vector and BM25 search (Pinecone `filter`, Weaviate `where`), so a chunk the user can't see is never scored and never enters application memory where a bug could leak it. Application-level post-filtering is a security bug waiting to happen.
- **Defense in depth.** Re-validate in application code too — two independent layers, no single bypass — and alarm on any query that reaches retrieval without an ACL filter attached.
- **Tenant isolation.** A shared index with ACL filtering is cost-efficient for most; carve regulated departments (HR, Legal, anything with PII or compensation data) into separate namespaces for hard isolation.
- **The operational gotchas.** Memberships go stale — refresh every ~15 minutes. Identity is sometimes ambiguous (`@john` = John Smith = jsmith@) — resolve it through the IdP as the single source of truth, and **on any ambiguity default to the most restrictive ACL, never the most permissive.**

### Incremental re-indexing on document updates

The corpus is not static, and freshness is per-source. Each connector does **delta sync** against the source's change cursor (Slack `oldest`, Confluence `lastModified`, Drive `changes.list` pageToken), so only changed items are re-processed. Match cadence to how fast a source actually changes: **real-time** webhooks for Slack and Drive, **hourly batch** for Confluence and Jira. On an update, re-chunk and re-embed only the affected document and upsert; deduplicate with a content hash so unchanged chunks aren't needlessly re-embedded. Because webhook events genuinely get dropped, run a **daily full-scan reconciliation** that compares each chunk's `ingested_at` against the source's `updated_at` and re-embeds anything that drifted. A full re-index of the whole corpus is reserved for an embedding-model upgrade.

### Latency budget decomposition

Stream the response — users perceive **time-to-first-token**, not total latency. A representative budget at enterprise scale (low QPS, complexity-per-query is the pressure, not throughput):

| Stage | Typical cost |
|---|---|
| Query embedding | ~10 ms |
| ACL resolution (cached) | ~5 ms |
| Vector search + ACL filter | ~15 ms |
| BM25 search + ACL filter | ~10 ms |
| RRF fusion | ~1 ms |
| Cross-encoder rerank (top-50) | ~20 ms |
| Context assembly | ~3 ms |
| LLM generation (streamed) | ~800 ms |
| **First token** | **~265 ms** |
| **Full response** | **~1,100 ms** |

The teaching point: **the LLM dominates both latency and cost** (often >80% of monthly spend). Retrieval is cheap. So the engineering budget goes into retrieval *quality* and ACL *correctness*, not shaving milliseconds off ANN search.

### Hallucination / grounding defenses and citation

Correct retrieval does not guarantee a correct answer — the LLM can confidently assert claims beyond its evidence, the most insidious RAG failure because the citations look legitimate. Defend in layers:

- **Grounding prompt.** Instruct the model to answer *only* from the provided context and to say "I don't know" otherwise. Mitigate lost-in-the-middle by placing the question both before and after long context, and keep the context tight (relevance degrades past ~5–10 chunks).
- **Citation + validation.** Require a chunk-id citation for every factual claim, then verify each citation actually exists in the retrieved set and — stronger — that the cited chunk **entails** the claim via an NLI check. A citation that doesn't support its claim is worse than none; it manufactures false confidence.
- **Refusal path.** A confidence threshold on the top retrieval score plus an explicit fallback — "I found relevant sources but can't confidently answer" — which users trust far more than a fabrication.

### RAG evaluation

Because RAG **fails silently**, evaluation is the only thing standing between a confident wrong answer and a user who trusts it. Split it by stage so a regression is diagnosable, not just visible:

- **Retrieval — recall, not faithfulness.** Recall@10 (target >0.85) and MRR (>0.70) against a **golden set** of ~500 hand-labeled (query → relevant-chunk) pairs. A retrieval regression doesn't error; it quietly feeds the LLM worse context that it dutifully answers wrong. Alarm if recall drops below ~0.80.
- **Generation — faithfulness, not recall.** NLI-based **faithfulness** (>0.90): is the answer entailed by the retrieved context? Plus **citation precision** (>0.92). These catch the hallucinate-despite-grounding case.
- **LLM-as-judge.** For answer quality at scale where golden labels don't exist, score answers with an LLM against a clear rubric, cross-checked by humans on a sample. Useful and cheap, but calibrate it against human judgments — don't trust the judge blind.
- **ACL correctness — a security invariant, not a target.** Measured at **exactly 100%**; `acl_violations_24h` must read 0, and any violation pages on-call. This is the one number that is not a quality dial.
- **Cross-source synthesis accuracy** (>0.75) on a labeled multi-source set, since single-source eval misses the system's hardest job.

### Scale failure modes

The dangerous failures at scale are silent and correctness-shaped, not availability-shaped — they never turn a latency dashboard red:

- **Stale index.** A dropped webhook or missed delta sync means a chunk reflects last week's decision, and the system answers confidently from it. Detect via `ingested_at` vs `updated_at` drift; mitigate with daily reconciliation, webhook retries, and a prominent "last updated" stamp on every citation.
- **Retrieval drift.** Quality erodes slowly — new jargon the embedding model never saw, a source whose longer chunks crowd out shorter relevant ones (source imbalance), distribution shift in queries. Detect by tracking recall@K on the golden set over time and the source distribution of top-5 vs. the corpus; mitigate by normalizing scores within source type before fusion, periodic embedding refresh, and golden-set expansion.
- **Cost.** The LLM is the dominant line item, so cost scales with traffic and context length. Control it with query/embedding caching for repetitive traffic, tight context budgets (more chunks ≠ better), a smaller model routed to easy queries, and self-hosting embeddings/LLM once volume or data-residency demands it.

## Common pitfalls

- **ACL applied after retrieval** instead of inside the query — a leak waiting to happen. Push the predicate into the DB filter and validate in app code.
- **Trusting grounding as a guarantee.** Correct context, wrong answer; you still need a faithfulness check and a refusal path.
- **Evaluating end-to-end only.** A blended score hides whether retrieval or generation regressed. Measure each stage.
- **Treating ACL correctness as a quality metric.** It's a 100% security invariant; 99.9% is a breach.
- **Full re-index on every update.** Re-embed only changed documents; reserve full re-index for a model upgrade.
- **Ignoring silent drift.** No alarm fires when recall slowly decays or the index goes stale — you must instrument for it.

## Tradeoffs

- **ACL at DB level vs. app level** — DB-level (filter never lets disallowed chunks leave the store) plus app-level re-validation for defense in depth. Never app-only.
- **Shared index + ACL filter vs. per-tenant index** — shared for most (cost), separate namespaces for regulated/PII departments (isolation).
- **Real-time vs. batch sync** — hybrid: real-time for fast-changing sources (Slack), batch for stable ones (Confluence). Match freshness cost to change rate.
- **Managed vs. self-hosted embeddings/LLM** — managed for speed-to-ship and quality; self-hosted once data residency or volume (cost) demands it, keeping the architecture identical.
- **Single-pass vs. decompose-and-synthesize retrieval** — single-pass for the ~80% simple majority; route detected complex/multi-source queries to the slower decompose path.

## Things to defend

- [ ] Draw the full pipeline (ingest → chunk → embed → index → retrieve → rerank → generate → eval) with one technology per stage.
- [ ] Explain ACL-aware retrieval end-to-end and why the filter must live inside the retrieval call.
- [ ] Describe incremental re-indexing and how you detect a stale chunk.
- [ ] State the latency budget with numbers and name which stage dominates cost.
- [ ] Give three layered defenses against hallucination and explain citation validation via NLI.
- [ ] Separate retrieval recall from answer faithfulness; describe golden sets and LLM-as-judge.
- [ ] Name the three scale failure modes (stale index, retrieval drift, cost) and a mitigation for each.
- [ ] Defend why ACL correctness is a 100% invariant, not a quality target.
