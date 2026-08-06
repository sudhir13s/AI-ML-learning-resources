---
id: "15-rag-and-llm-apps"
topic: "RAG & LLM Applications"
level: advanced
built_from: ["llms", "nlp"]
updated: 2026-06-27
---

# RAG & LLM Applications
> Building real products on LLMs — retrieval-augmented generation, vector search, evaluation,
> prompting, and the engineering that makes them reliable.

**Start here:** [RAG from scratch](https://github.com/langchain-ai/rag-from-scratch) — **LangChain** — build retrieval-augmented generation step by step.

## Concept Index
Each topic below is a self-contained page with a curated companion resource card (free, open
courses, videos, papers, articles, books and cross-links). New here? Read the field framing above,
then work top to bottom.

### Foundations of retrieval-augmented generation
1. [RAG Fundamentals (retrieve-then-generate)](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/rag-foundations/notes-theory)
2. [Document Chunking Strategies](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/chunking/notes-theory)
3. [Embedding Models for Retrieval](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/embedding-models/notes-theory)

### Indexing & search
4. [Vector Databases & ANN Indexes (HNSW · IVF)](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/vector-search/notes-theory)
5. [Hybrid Search (BM25 + dense)](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/hybrid-search/notes-theory)
6. [Re-ranking (cross-encoders)](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/reranking/notes-theory)
7. [Query Transformation (HyDE · multi-query)](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/query-transformation/notes-theory)

### Advanced retrieval architectures
8. [Advanced RAG (parent-doc · fusion · self-RAG)](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/advanced-rag/notes-theory)
9. [GraphRAG](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/graph-rag/notes-theory)
10. [Agentic RAG](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/agentic-rag/notes-theory)

### Quality, reliability & evaluation
11. [RAG Evaluation (RAGAS · faithfulness · groundedness)](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/rag-evaluation/notes-theory)
12. [Long-Context vs RAG](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/long-context-vs-rag/notes-theory)
13. [Citations & Attribution](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/citations-and-attribution/notes-theory)
14. [Guardrails & Hallucination Mitigation](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/hallucination-and-grounding/notes-theory)

### Building & operating LLM apps
15. [LLM App Orchestration (chains · routing)](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/llm-app-orchestration/notes-theory)
16. [Caching & Cost Optimization for LLM Apps](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/caching-and-cost-optimization/notes-theory)

### Related concepts (canonical home is another section)
> These topics are foundations or neighbors of RAG, but their canonical home is another section —
> linked here to avoid repetition.
- **Word & sentence/document embeddings (the encoders RAG retrieves with)** → [NLP](/ai-ml/ai-ml-learning-resources/modalities-and-generative-models/overview/overview-6) ([Word Embeddings](/ai-ml/ai-ml-learning-resources/modalities-and-generative-models/word-embeddings-word2vec-glove-fasttext/notes-theory) · [Sentence & Document Embeddings](/ai-ml/ai-ml-learning-resources/modalities-and-generative-models/sentence-and-document-embeddings/notes-theory) · [Information Retrieval & Semantic Search](/ai-ml/ai-ml-learning-resources/modalities-and-generative-models/information-retrieval-and-semantic-search/notes-theory))
- **Transformer architecture · Attention** (the generator's engine) → [Deep Learning](/ai-ml/ai-ml-learning-resources/deep-learning/overview/overview)
- **LLM internals — prompting, fine-tuning, decoding, RLHF, KV-cache, long-context** → [LLMs](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/overview/overview-3) ([Prompting & In-Context Learning](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/prompting-and-in-context-learning/notes-theory) · [Long-Context Methods](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/long-context-architectures/notes-theory) · [Hallucination & Alignment](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/safety-and-alignment/notes-theory))
- **Agents & tool use** (the broader agent loop that Agentic RAG specializes) → [Agentic AI](/ai-ml/ai-ml-learning-resources/llms-applications-and-agents/overview/overview)
- **ANN / clustering math** (the geometry under vector indexes) → [Unsupervised Learning](/ai-ml/ai-ml-learning-resources/core-machine-learning/overview/overview-3)

## Courses (free)
- [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) — **DeepLearning.AI** — the canonical free RAG short course.
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × LlamaIndex** — retrieval quality + evaluation.

## Videos
- [RAG explained + production tips](https://www.youtube.com/watch?v=ahnGLM-RC1Y) — **OpenAI / community** — what breaks in real RAG systems.

## Key Papers / Articles
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — the original RAG.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — **Liu et al. (2023)** — why long context ≠ good retrieval.
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — **Anthropic** — a strong modern chunking recipe.

## Books
- [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) — **Chip Huyen (2025)** — the definitive text on building LLM products (RAG, agents, eval).

## In this platform
- Math/mechanism: [AI-ML-intuition 8.02 RAG](/ai-ml/ai-ml-intuitions/memory-retrieval-context/rag-intuition), [8.01 Prompting](/ai-ml/ai-ml-intuitions/reasoning-agency/in-context-learning-and-prompting-intuition)
