---
id: "11-rag-and-llm-apps"
topic: "RAG & LLM Applications"
level: advanced
prereqs: ["llms", "nlp"]
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
1. ✅ [RAG Fundamentals (retrieve-then-generate)](01-RAG-Fundamentals/01-RAG-Fundamentals.md)
2. ✅ [Document Chunking Strategies](02-Document-Chunking-Strategies/02-Document-Chunking-Strategies.md)
3. ✅ [Embedding Models for Retrieval](03-Embedding-Models-for-Retrieval/03-Embedding-Models-for-Retrieval.md)

### Indexing & search
4. ✅ [Vector Databases & ANN Indexes (HNSW · IVF)](04-Vector-Databases-and-ANN-Indexes/04-Vector-Databases-and-ANN-Indexes.md)
5. ✅ [Hybrid Search (BM25 + dense)](05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md)
6. ✅ [Re-ranking (cross-encoders)](06-Re-ranking-Cross-Encoders/06-Re-ranking-Cross-Encoders.md)
7. ✅ [Query Transformation (HyDE · multi-query)](07-Query-Transformation-HyDE-Multi-Query/07-Query-Transformation-HyDE-Multi-Query.md)

### Advanced retrieval architectures
8. ✅ [Advanced RAG (parent-doc · fusion · self-RAG)](08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG/08-Advanced-RAG-Parent-Doc-Fusion-Self-RAG.md)
9. ✅ [GraphRAG](09-GraphRAG/09-GraphRAG.md)
10. ✅ [Agentic RAG](10-Agentic-RAG/10-Agentic-RAG.md)

### Quality, reliability & evaluation
11. ✅ [RAG Evaluation (RAGAS · faithfulness · groundedness)](11-RAG-Evaluation/11-RAG-Evaluation.md)
12. ✅ [Long-Context vs RAG](12-Long-Context-vs-RAG/12-Long-Context-vs-RAG.md)
13. ✅ [Citations & Attribution](13-Citations-and-Attribution/13-Citations-and-Attribution.md)
14. ✅ [Guardrails & Hallucination Mitigation](14-Guardrails-and-Hallucination-Mitigation/14-Guardrails-and-Hallucination-Mitigation.md)

### Building & operating LLM apps
15. ✅ [LLM App Orchestration (chains · routing)](15-LLM-App-Orchestration/15-LLM-App-Orchestration.md)
16. ✅ [Caching & Cost Optimization for LLM Apps](16-Caching-and-Cost-Optimization/16-Caching-and-Cost-Optimization.md)

### Related concepts (canonical home is another section)
> These topics are foundations or neighbors of RAG, but their canonical home is another section —
> linked here to avoid repetition.
- **Word & sentence/document embeddings (the encoders RAG retrieves with)** → [NLP](../06.%20NLP/README.md) ([Word Embeddings](../06.%20NLP/05-Word-Embeddings-Word2Vec-GloVe-FastText/05-Word-Embeddings-Word2Vec-GloVe-FastText.md) · [Sentence & Document Embeddings](../06.%20NLP/07-Sentence-and-Document-Embeddings/07-Sentence-and-Document-Embeddings.md) · [Information Retrieval & Semantic Search](../06.%20NLP/16-Information-Retrieval-and-Semantic-Search/16-Information-Retrieval-and-Semantic-Search.md))
- **Transformer architecture · Attention** (the generator's engine) → [Deep Learning](../05.%20Deep_Learning/README.md)
- **LLM internals — prompting, fine-tuning, decoding, RLHF, KV-cache, long-context** → [LLMs](../09.%20LLMs/README.md) ([Prompting & In-Context Learning](../09.%20LLMs/16-Prompting-and-In-Context-Learning/16-Prompting-and-In-Context-Learning.md) · [Long-Context Methods](../09.%20LLMs/08-Long-Context-Methods/08-Long-Context-Methods.md) · [Hallucination & Alignment](../09.%20LLMs/20-Hallucination-and-Alignment-Basics/20-Hallucination-and-Alignment-Basics.md))
- **Agents & tool use** (the broader agent loop that Agentic RAG specializes) → [Agentic AI](../12.%20Agentic_AI/README.md)
- **ANN / clustering math** (the geometry under vector indexes) → [Unsupervised Learning](../04.%20Unsupervised_Learning/README.md)

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
- Math/mechanism: [AI-ML-intuition 8.02 RAG](../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md), [8.01 Prompting](../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md)
