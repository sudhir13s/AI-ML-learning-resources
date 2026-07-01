---
id: "11-rag-and-llm-apps/caching-and-cost-optimization/references"
topic: "Caching & Cost Optimization for LLM Apps — References"
parent: "11-rag-and-llm-apps/caching-and-cost-optimization"
type: references
updated: 2026-07-02
---

# Caching & Cost Optimization — references and further reading

> Companion link library for **[Caching & Cost Optimization](16-Caching-and-Cost-Optimization.md)** (the
> concept page). External sources *and* internal cross-links, kept separate so it can be reused as a
> standalone list. Grouped by type, best-first. Every entry is free/open (no paywall) and chosen for
> depth on *this* topic — semantic caching (embedding-similarity answer reuse), prompt/prefix caching
> (provider KV reuse), and the cost model. Every framework/mechanism cited on the concept page
> (GPTCache, Anthropic/OpenAI prompt caching, the false-hit tradeoff, the cost identity) appears here as
> a primary source.

**Start here — suggested path**:
1. **Cut input cost first** — watch [Prompt Caching Guide](https://www.youtube.com/watch?v=RDjaUJz-uWo) (**PromptHub**). *How OpenAI/Anthropic/Google prefix-cache the static parts of a prompt for 50–90% savings.*
2. **Read the mechanics** — read [Anthropic: Prompt Caching](https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching) + [OpenAI: Prompt Caching](https://developers.openai.com/api/docs/guides/prompt-caching). *Cache breakpoints, automatic vs explicit, cache-write vs cache-read pricing.*
3. **Add semantic caching** — watch [Semantic Caching Explained (Redis)](https://www.youtube.com/watch?v=NrqvtsnjIHU) (**Nariman Codes**). *Serve cached answers for *similar* (not identical) queries via embeddings.*
4. **Build a semantic cache** — watch [GPTCache — Save Cost on LLMs](https://www.youtube.com/watch?v=Yug3gObpX-g) (**Fahd Mirza**), and read the [GPTCache repo](https://github.com/zilliztech/GPTCache). *A working embedding-based response cache.*
5. **Read the source** — skim [GPTCache (Bang 2023, NLP-OSS)](https://aclanthology.org/2023.nlposs-1.24/). *The semantic-cache design — embedding similarity + a threshold — this chapter builds from scratch.*

**Videos**:
- [Prompt Caching Guide](https://www.youtube.com/watch?v=RDjaUJz-uWo) — **PromptHub** — how prefix caching works across OpenAI/Anthropic/Google, and how to structure prompts for hits.
- [Semantic Caching Explained: Reduce AI API Costs with Redis](https://www.youtube.com/watch?v=NrqvtsnjIHU) — **Nariman Codes** — embedding-similarity caching at the gateway, and when it pays off.
- [GPTCache — Save Cost on LLMs](https://www.youtube.com/watch?v=Yug3gObpX-g) — **Fahd Mirza** — installing and using a semantic response cache locally.
- [Caching Strategies to Slash Your LLM Bill](https://www.youtube.com/watch?v=j9wVKM89XFU) — **MadeForCloud** — prompt + semantic caching combined, with a cost demo.

**Interactive & visual**:
- [GPTCache — GitHub](https://github.com/zilliztech/GPTCache) — **Zilliz** — the reference semantic-cache library to read and run: embedding + similarity evaluation + cache store, wrapping the LLM client.
- [Anthropic Prompt Caching — docs](https://platform.claude.com/docs/en/docs/build-with-claude/prompt-caching) — **Anthropic** — the exact `cache_control` request format, TTLs, and cache-write/read pricing, runnable.
- [OpenAI Prompt Caching — guide](https://developers.openai.com/api/docs/guides/prompt-caching) — **OpenAI** — automatic prefix caching, the 1,024-token minimum, and the cached-token discount (up to ~90% input-cost / ~80% latency reduction).

**Courses (free)**:
- [LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) — **DeepLearning.AI × LangChain** — covers caching and efficient chains as part of building real apps.
- [HF: LLM Inference Optimization (guide)](https://huggingface.co/docs/transformers/en/llm_optims) — **Hugging Face** — free, hands-on guide to latency/cost optimizations (caching, batching, quantization pointers).

**Articles / blogs (free, no paywall)**:
- [What is Semantic Caching?](https://redis.io/blog/what-is-semantic-caching/) — **Redis** — how similarity-based caching works, the false-hit risk, and when it pays off.
- [zilliztech/GPTCache (GitHub README)](https://github.com/zilliztech/GPTCache) — **Zilliz** — the semantic-cache library's design (embedding, similarity eval, eviction) with runnable examples.
- [Prompt Caching in the API](https://openai.com/index/api-prompt-caching/) — **OpenAI** — the announcement + how automatic prompt caching cuts input cost and latency.
- [LangChain — LLM caching (how-to)](https://python.langchain.com/docs/how_to/llm_caching/) — **LangChain** — `set_llm_cache(InMemoryCache())` and semantic-cache backends, the API used on the page.

**Key papers**:
- [GPTCache: An Open-Source Semantic Cache for LLM Applications Enabling Faster Answers and Cost Savings](https://aclanthology.org/2023.nlposs-1.24/) — **Fu Bang (2023, NLP-OSS @ EMNLP)** — the reference semantic cache: embed the query, match by similarity + threshold, serve the stored answer; 2–10× faster on a hit. The design this chapter builds from scratch.
- [Prompt Cache: Modular Attention Reuse for Low-Latency Inference](https://arxiv.org/abs/2311.04934) — **Gim et al. (2023, MLSys 2024)** — reuse precomputed attention (KV) for recurring prompt segments (system messages, documents); the mechanism behind provider prompt/prefix caching (8–60× TTFT speedup).
- [GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching](https://arxiv.org/abs/2411.05276) — **Regmi & Pun (2024)** — embedding-based query caching; up to ~68% fewer API calls, with the similarity-threshold tradeoff.
- [MeanCache: User-Centric Semantic Caching for LLM Web Services](https://arxiv.org/abs/2403.02694) — **Gill et al. (2024)** — a privacy-aware, per-user semantic cache and the false-hit calibration problem.
- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — **Kwon et al. (2023)** — the KV-cache/serving optimization that makes the *inference* behind each miss cheap and high-throughput.

**Books (free chapters)**:
- [Hugging Face Transformers — "LLM inference optimization" guide](https://huggingface.co/docs/transformers/en/llm_optims) — **Hugging Face** — the closest free book-length reference: caching, batching, and the latency/cost levers, fully open.

**In this platform**:
- Concept page (full explanation): [Caching & Cost Optimization](16-Caching-and-Cost-Optimization.md)
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md) · [1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md) (the basis of semantic caching)
- Reused here: [05 Hybrid Search (the DenseRetriever/encoder)](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md) · [12 Long-Context vs RAG (the token cost model)](../12-Long-Context-vs-RAG/12-Long-Context-vs-RAG.md) · [14 Guardrails (the same threshold-tradeoff shape)](../14-Guardrails-and-Hallucination-Mitigation/14-Guardrails-and-Hallucination-Mitigation.md)
- Foundations: [01 RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [04 Vector Databases & ANN (fast cache lookup)](../04-Vector-Databases-and-ANN-Indexes/04-Vector-Databases-and-ANN-Indexes.md) · [15 LLM App Orchestration](../15-LLM-App-Orchestration/15-LLM-App-Orchestration.md)
- Related domain: [09. LLMs — KV Cache](../../09.%20LLMs/05-KV-Cache/05-KV-Cache.md) · [09. LLMs — Inference Optimization & Serving (vLLM)](../../09.%20LLMs/09-Inference-Optimization-and-Serving/09-Inference-Optimization-and-Serving.md)
