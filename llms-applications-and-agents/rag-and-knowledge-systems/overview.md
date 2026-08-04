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

**⭐ Start here:** [RAG from scratch](https://github.com/langchain-ai/rag-from-scratch) — **LangChain** — build retrieval-augmented generation step by step.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page and a curated
`.references.md` resource card (free, open courses · videos · papers · articles · books · cross-links).
> **✅ ready.** New here? Start with the field overview above, then work top to bottom.

### Foundations of retrieval-augmented generation
1. ✅ [RAG Fundamentals (retrieve-then-generate)](rag-foundations/rag-foundations.md)
2. ✅ [Document Chunking Strategies](chunking/chunking.md)
3. ✅ [Embedding Models for Retrieval](embedding-models/embedding-models.md)

### Indexing & search
4. ✅ [Vector Databases & ANN Indexes (HNSW · IVF)](vector-search/vector-search.md)
5. ✅ [Hybrid Search (BM25 + dense)](hybrid-search/hybrid-search.md)
6. ✅ [Re-ranking (cross-encoders)](reranking/reranking.md)
7. ✅ [Query Transformation (HyDE · multi-query)](query-transformation/query-transformation.md)

### Advanced retrieval architectures
8. ✅ [Advanced RAG (parent-doc · fusion · self-RAG)](advanced-rag/advanced-rag.md)
9. ✅ [GraphRAG](graph-rag/graph-rag.md)
10. ✅ [Agentic RAG](agentic-rag/agentic-rag.md)

### Quality, reliability & evaluation
11. ✅ [RAG Evaluation (RAGAS · faithfulness · groundedness)](rag-evaluation/rag-evaluation.md)
12. ✅ [Long-Context vs RAG](long-context-vs-rag/long-context-vs-rag.md)
13. ✅ [Citations & Attribution](citations-and-attribution/citations-and-attribution.md)
14. ✅ [Guardrails & Hallucination Mitigation](../reasoning-evaluation-and-alignment/hallucination-and-grounding/hallucination-and-grounding.md)

### Building & operating LLM apps
15. ✅ [LLM App Orchestration (chains · routing)](../agentic-ai/llm-app-orchestration/llm-app-orchestration.md)
16. ✅ [Caching & Cost Optimization for LLM Apps](../inference-and-runtime/caching-and-cost-optimization/caching-and-cost-optimization.md)

### Related concepts (canonical home is another section)
> These topics are foundations or neighbors of RAG, but their canonical home is another section —
> linked here to avoid repetition.
- **Word & sentence/document embeddings (the encoders RAG retrieves with)** → [NLP](../../modalities-and-generative-models/natural-language-processing/README.md) ([Word Embeddings](../../modalities-and-generative-models/natural-language-processing/word-embeddings-word2vec-glove-fasttext/word-embeddings-word2vec-glove-fasttext.md) · [Sentence & Document Embeddings](../../modalities-and-generative-models/natural-language-processing/sentence-and-document-embeddings/sentence-and-document-embeddings.md) · [Information Retrieval & Semantic Search](../../modalities-and-generative-models/natural-language-processing/information-retrieval-and-semantic-search/information-retrieval-and-semantic-search.md))
- **Transformer architecture · Attention** (the generator's engine) → [Deep Learning](../../deep-learning/README.md)
- **LLM internals — prompting, fine-tuning, decoding, RLHF, KV-cache, long-context** → [LLMs](../../09.%20LLMs/README.md) ([Prompting & In-Context Learning](../../09.%20LLMs/16-Prompting-and-In-Context-Learning/16-Prompting-and-In-Context-Learning.md) · [Long-Context Methods](../../09.%20LLMs/08-Long-Context-Methods/08-Long-Context-Methods.md) · [Hallucination & Alignment](../../09.%20LLMs/20-Hallucination-and-Alignment-Basics/20-Hallucination-and-Alignment-Basics.md))
- **Agents & tool use** (the broader agent loop that Agentic RAG specializes) → [Agentic AI](../agentic-ai/overview.md)
- **ANN / clustering math** (the geometry under vector indexes) → [Unsupervised Learning](../../core-machine-learning/unsupervised-learning/README.md)

## 🎓 Courses (free)
- [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) — **DeepLearning.AI** — the canonical free RAG short course.
- [Building & Evaluating Advanced RAG](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI × LlamaIndex** — retrieval quality + evaluation.

## 🎥 Videos
- [RAG explained + production tips](https://www.youtube.com/watch?v=ahnGLM-RC1Y) — **OpenAI / community** — what breaks in real RAG systems.

## 📄 Key Papers / Articles
- [Retrieval-Augmented Generation](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — the original RAG.
- [Lost in the Middle](https://arxiv.org/abs/2307.03172) — **Liu et al. (2023)** — why long context ≠ good retrieval.
- [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — **Anthropic** — a strong modern chunking recipe.

## 📚 Books
- [AI Engineering](https://www.oreilly.com/library/view/ai-engineering/9781098166298/) — **Chip Huyen (2025)** — the definitive text on building LLM products (RAG, agents, eval).

## 🔗 In this platform
- Math/mechanism: [AI-ML-intuition 8.02 RAG](../../../AI-ML-intuition/memory-retrieval-and-context/retrieval-augmented-generation/rag-intuition.md), [8.01 Prompting](../../../AI-ML-intuition/reasoning-and-agency/in-context-behavior/in-context-learning-and-prompting-intuition.md)
