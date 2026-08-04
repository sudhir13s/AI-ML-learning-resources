---
id: "15-rag-and-llm-apps/citations-and-attribution/references"
topic: "Citations & Attribution (post-hoc attribution · citation precision/recall) — References"
parent: "15-rag-and-llm-apps/citations-and-attribution"
type: references
updated: 2026-07-02
---

# Citations & Attribution — references and further reading

> Companion link library for **[Citations & Attribution](citations-and-attribution.md)** (the
> concept page). External sources *and* internal cross-links, kept separate so it can be reused as a
> standalone list. Grouped by type, best-first. Every entry is free/open (no paywall) and chosen for
> depth on *this* topic — attaching each claim in a RAG answer to its source passage (attribution),
> measuring citation quality (precision/recall), and the entailment judge that separates a real
> citation from a topical one. Every formula/mechanism cited on the concept page (ALCE citation
> recall/precision, AIS attribution, the NLI entailment support function) appears here as a primary
> source.

**Start here — suggested path**:
1. **See the methods** — read [How to get a model to cite sources](https://python.langchain.com/docs/how_to/qa_citations/) (**LangChain**). *The menu of citation techniques (tool-calling IDs, snippets, prompting, post-processing), with code.*
2. **Make citations verifiable** — watch [RAG — but with Verified Citations!](https://www.youtube.com/watch?v=-wGzSnhQKPM) (**Trelis Research**). *Forcing citations to point at spans that genuinely support each claim — the entailment idea in practice.*
3. **Use model-native citations** — read [Anthropic: Introducing Citations](https://www.anthropic.com/news/introducing-citations-api) + the [Citations docs](https://platform.claude.com/docs/en/docs/build-with-claude/citations). *Generation-time, sentence-level grounding built into the API — `cited_text` + character spans.*
4. **Know the metrics precisely** — skim [ALCE (Gao et al. 2023)](https://arxiv.org/abs/2305.14627). *Citation recall & precision via an NLI entailment model — the metrics this chapter derives.*
5. **Read the foundation** — skim [AIS: Measuring Attribution in NLG (Rashkin et al. 2021)](https://arxiv.org/abs/2112.12870). *"Attributable to Identified Sources" — the definition of attribution the whole field builds on.*

**Videos**:
- [RAG — but with Verified Citations!](https://www.youtube.com/watch?v=-wGzSnhQKPM) — **Trelis Research** — making citations point to spans that genuinely support each claim (the entailment, not topical, idea).
- [How to Get LLM Answers With Sources — Advanced RAG](https://www.youtube.com/watch?v=69gUQ4XHg0o) — **M&M Tech** — returning source documents alongside answers via LangChain LCEL, the plumbing of post-hoc attribution.
- [Building a RAG System with In-line Citations Using Workflows](https://www.youtube.com/watch?v=P4xHWojIB-M) — **LlamaIndex** — inline `[1][2]` citations wired into a retrieval workflow (the `CitationQueryEngine` idea).
- [Unlocking Advanced RAG: Citations and Attributions](https://www.youtube.com/watch?v=RnCuOL-LBAw) — **Zilliz** — attribution patterns and why grounding-with-citations reduces hallucination.
- [How to stop LLM Hallucinations: Grounding via RAG (with RAGAS)](https://www.youtube.com/watch?v=4xcbXDjnjS4) — **Underfitted** — grounding and groundedness evaluation, the metric attribution must satisfy.

**Interactive & visual**:
- [Perplexity](https://www.perplexity.ai/) — **Perplexity AI** — a live consumer answer engine where every answer carries inline numbered citations to its web sources; the clearest working demo of the pattern.
- [Anthropic Citations — Claude docs](https://platform.claude.com/docs/en/docs/build-with-claude/citations) — **Anthropic** — the request/response format for generation-time citations (`citations.enabled`, `cited_text`, `document_index`, char spans), runnable.
- [princeton-nlp/ALCE (GitHub)](https://github.com/princeton-nlp/ALCE) — **Princeton NLP** — the open-source benchmark + evaluation code (NLI-based citation recall/precision) to read and run.

**Courses (free)**:
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × TruLens** — groundedness evaluation (claim-by-claim support in the retrieved context), the eval attribution must pass.
- [RAG from Scratch — Generation](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — free notebooks where returning sources/citations fits naturally into the generation step.

**Articles / blogs (free, no paywall)**:
- [How to get a model to cite sources](https://python.langchain.com/docs/how_to/qa_citations/) — **LangChain** — the canonical menu of citation techniques (tool IDs, snippets, prompting, retrieval/generation post-processing) with code.
- [Introducing Citations](https://www.anthropic.com/news/introducing-citations-api) — **Anthropic** — model-native, sentence-level source grounding; why it beats many custom prompt-based setups.
- [Attributing Sources in RAG Generated Output](https://apxml.com/courses/getting-started-rag/chapter-4-rag-generation-augmentation/attributing-sources) — **APX ML** — a clear, free walkthrough of prompting for and post-processing citations in a RAG pipeline.
- [The RAG Triad — Groundedness](https://www.trulens.org/getting_started/core_concepts/rag_triad/) — **TruLens** — groundedness = decompose the response into claims and find support for each in the context; the attribution-adjacent metric.
- [Vertex AI — Grounding & citation metadata](https://cloud.google.com/vertex-ai/generative-ai/docs/reference/rest/v1beta1/GroundingMetadata) — **Google Cloud** — the `groundingMetadata` / `groundingChunks` / `groundingSupports` response that links generated segments to their sources.
- [CitationQueryEngine](https://developers.llamaindex.ai/python/examples/query_engine/citation_query_engine/) — **LlamaIndex** — how the engine splits sources into citation chunks, injects `[1][2]`, and returns `source_nodes` for verification.

**Key papers**:
- [Enabling Large Language Models to Generate Text with Citations (ALCE)](https://arxiv.org/abs/2305.14627) — **Gao, Yen, Yu & Chen (2023, EMNLP)** — the primary source: the ALCE benchmark and **citation recall & precision** defined via an **NLI entailment** support function (recall = concatenated citations entail the statement; precision = no irrelevant citation) — the metrics this chapter derives.
- [Measuring Attribution in Natural Language Generation Models (AIS)](https://arxiv.org/abs/2112.12870) — **Rashkin et al. (2021, Computational Linguistics)** — the **Attributable to Identified Sources** framework: a statement is attributable iff a generic hearer, given the source, would affirm it — the definition of attribution the page's math rests on.
- [Attributed Question Answering: Evaluation and Modeling for Attributed LLMs](https://arxiv.org/abs/2212.08037) — **Bohnet et al. (2022)** — formalizes attributed QA and AIS-based evaluation (answer attributable iff *fully supported* by the cited source), the QA setting of this chapter.
- [RARR: Researching and Revising What Language Models Say, Using Language Models](https://arxiv.org/abs/2210.08726) — **Gao et al. (2022, ACL 2023)** — **post-hoc** attribution: find attribution for *any* model's output and edit unsupported claims; the research lineage of the post-hoc attributor built here.
- [Learning Fine-Grained Grounded Citations for Attributed LLMs](https://arxiv.org/abs/2408.04568) — **Huang et al. (2024, ACL)** — training models to cite at **sub-sentence** granularity, the fine-grained end of the coarse-vs-fine axis.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — the original RAG whose retrieve-then-generate pipeline is the thing this chapter attributes.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering & Information Retrieval"](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — the QA-grounding foundations (answer support, evidence) behind attribution, free PDF.

**In this platform**:
- Concept page (full explanation): [Citations & Attribution](citations-and-attribution.md)
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../../AI-ML-intuition/memory-retrieval-and-context/retrieval-augmented-generation/rag-intuition.md)
- Machinery reused here: [05 Hybrid Search (the DenseRetriever this reuses)](../hybrid-search/hybrid-search.md) · [08 Advanced RAG (the encoder-cosine support check reused as the attribution proxy)](../advanced-rag/advanced-rag.md) · [11 RAG Evaluation (faithfulness & the cosine≠entailment caveat)](../rag-evaluation/rag-evaluation.md)
- Foundations: [01 RAG Fundamentals](../rag-foundations/rag-foundations.md) · [03 Embedding Models](../embedding-models/embedding-models.md)
- Next / related: [14 Guardrails & Hallucination Mitigation](../../reasoning-evaluation-and-alignment/hallucination-and-grounding/hallucination-and-grounding.md)
- Related domain: [09. LLMs — Hallucination & Alignment Basics](../../../09.%20LLMs/20-Hallucination-and-Alignment-Basics/20-Hallucination-and-Alignment-Basics.md) · [06. NLP — NLP Evaluation Metrics (entailment-based evaluation)](../../../modalities-and-generative-models/natural-language-processing/nlp-evaluation-metrics/nlp-evaluation-metrics.md)
