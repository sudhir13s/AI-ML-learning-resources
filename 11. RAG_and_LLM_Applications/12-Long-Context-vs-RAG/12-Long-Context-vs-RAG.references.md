---
id: "11-rag-and-llm-apps/long-context-vs-rag/references"
topic: "Long-Context vs RAG — References"
parent: "11-rag-and-llm-apps/long-context-vs-rag"
type: references
updated: 2026-07-02
---

# Long-Context vs RAG — references and further reading

> Companion link library for **[Long-Context vs RAG](12-Long-Context-vs-RAG.md)** (the concept page).
> External sources *and* internal cross-links, kept separate so it can be reused as a standalone list.
> Grouped by type, best-first. Every entry is free/open (no paywall) and chosen for depth on *this*
> topic — the cost/accuracy tradeoff between stuffing a long context and retrieving with RAG:
> lost-in-the-middle, effective-vs-advertised context, per-query cost, and the hybrid answer. Every
> formula/fact cited on the concept page (attention O(n²), Liu et al. U-curve, RULER, provider window
> sizes/prices) appears here as a primary source.

**Start here — suggested path**:
1. **Learn the core failure** — watch [Lost in the Middle, Explained](https://www.youtube.com/watch?v=Kf3LeaUGwlg) (**Weaviate**), then skim the [paper](https://arxiv.org/abs/2307.03172). *Why stuffing everything in context degrades mid-document recall.*
2. **See the empirical verdict** — read [Long-Context RAG Performance of LLMs](https://www.databricks.com/blog/long-context-rag-performance-llms) (**Databricks**). *Where long context helps, where it breaks, measured across models.*
3. **Get the decision framework** — read [Towards Long-Context RAG](https://www.llamaindex.ai/blog/towards-long-context-rag) (**LlamaIndex**). *When to retrieve, when to fill the window, and the hybrid middle path.*
4. **Watch the debate** — watch [Is RAG Really Dead? Testing GPT-4-128k](https://www.youtube.com/watch?v=UlmyyYQGhzc) and [RAG for long-context LLMs](https://www.youtube.com/watch?v=SsHUNfhF32s) (**LangChain, Lance Martin**). *Multi-fact retrieval over long windows, and how RAG adapts.*
5. **Read the comparison** — skim [Long Context vs RAG: An Evaluation and Revisits](https://arxiv.org/abs/2501.01880). *A careful head-to-head with the conditions under which each wins.*

**Videos**:
- [Lost in the Middle: How Language Models Use Long Context — Explained!](https://www.youtube.com/watch?v=Kf3LeaUGwlg) — **Weaviate** — the U-shaped recall curve that motivates retrieval even with big windows.
- [Is RAG Really Dead? Testing Multi-Fact Retrieval in GPT-4-128k](https://www.youtube.com/watch?v=UlmyyYQGhzc) — **LangChain (Lance Martin)** — needle/multi-needle tests probing long-context limits.
- [RAG for long context LLMs](https://www.youtube.com/watch?v=SsHUNfhF32s) — **LangChain (Lance Martin)** — how RAG architectures change (not vanish) as windows grow.
- [Are long context LLMs the death of RAG?](https://www.youtube.com/watch?v=Ng-EnWrwsAg) — **Aggregate Intellect** — a balanced discussion of the trade-offs.

**Interactive & visual**:
- [Long context (Gemini API docs)](https://ai.google.dev/gemini-api/docs/long-context) — **Google** — what large windows can/can't do, needle-in-haystack framing, and context caching.
- [Gemini 1.5 Pro 2M context window (Google Developers Blog)](https://developers.googleblog.com/en/new-features-for-the-gemini-api-and-google-ai-studio/) — **Google** — the official 2M-token announcement + context caching for cost reduction (the page's Gemini window/caching source).

**Courses (free)**:
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × LlamaIndex** — sentence-window / auto-merging retrieval that bridges chunk-level and long-context.
- [Long context (Gemini API docs)](https://ai.google.dev/gemini-api/docs/long-context) — **Google** — a free, practical guide to using and reasoning about very large context windows.

**Articles / blogs (free, no paywall)**:
- [Long-Context RAG Performance of LLMs](https://www.databricks.com/blog/long-context-rag-performance-llms) — **Databricks** — measured results across context lengths and models; where long context helps and breaks.
- [Towards Long-Context RAG](https://www.llamaindex.ai/blog/towards-long-context-rag) — **LlamaIndex** — new architectures and trade-offs when windows are large; the hybrid framing.
- [OpenAI GPT-4o model docs (128k context)](https://developers.openai.com/api/docs/models/gpt-4o) — **OpenAI** — the verified GPT-4o context-window and pricing source used on the page.
- [Claude models (200k context) + pricing](https://platform.claude.com/docs/en/docs/about-claude/models) — **Anthropic** — the verified Claude context-window and input-price source used on the page.

**Key papers**:
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — **Liu, Lin, Hewitt, Paranjape, Bevilacqua, Petroni & Liang (2023)** — the canonical accuracy-vs-position U-curve (best at the edges, worst in the middle); the page's cited lost-in-the-middle result.
- [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654) — **Hsieh et al. (2024)** — effective context is well below the advertised window; the page's cited advertised-vs-effective source.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — scaled dot-product attention, §3.2; the O(n²)-in-context-length scaling behind long-context compute/latency cost.
- [Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study](https://arxiv.org/abs/2407.16833) — **Li et al. (2024)** — when each wins and a "self-route" hybrid that picks per query.
- [Long Context vs RAG for LLMs: An Evaluation and Revisits](https://arxiv.org/abs/2501.01880) — **Yu et al. (2025)** — a careful re-evaluation across conditions; a nuanced head-to-head.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering & Information Retrieval"](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — retrieval foundations explaining why *selection* still beats dumping everything into context, free PDF.

**In this platform**:
- Concept page (full explanation): [Long-Context vs RAG](12-Long-Context-vs-RAG.md)
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
- Foundations: [01 RAG Fundamentals (retrieve-then-generate)](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [02 Document Chunking](../02-Document-Chunking-Strategies/02-Document-Chunking-Strategies.md) · [05 Hybrid Search (the DenseRetriever the dilution proxy reuses)](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md)
- Measure it: [11 RAG Evaluation (why a smaller focused context scores higher)](../11-RAG-Evaluation/11-RAG-Evaluation.md)
- Long-context mechanics: [09. LLMs — Long-Context Methods (RoPE scaling, ALiBi)](../../09.%20LLMs/08-Long-Context-Methods/08-Long-Context-Methods.md) · [09. LLMs — KV Cache (the memory/latency cost of a long prompt)](../../09.%20LLMs/05-KV-Cache/05-KV-Cache.md)
- Next: [13 Citations & Attribution](../13-Citations-and-Attribution/13-Citations-and-Attribution.md)
