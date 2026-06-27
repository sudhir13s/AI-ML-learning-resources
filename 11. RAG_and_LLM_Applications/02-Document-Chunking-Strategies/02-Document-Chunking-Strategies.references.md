---
id: "11-rag-and-llm-apps/document-chunking/references"
topic: "Document Chunking Strategies — References"
parent: "11-rag-and-llm-apps/document-chunking"
type: references
updated: 2026-06-27
---

# Document Chunking Strategies — references and further reading

> Companion link library for **[Document Chunking Strategies](02-Document-Chunking-Strategies.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is a free, no-paywall link from a primary author or a recognized deep explainer — chosen for depth on *this* topic (how you split documents for retrieval), not popularity.

**Start here — suggested path**:
1. **See the landscape** — watch [The 5 Levels of Text Splitting](https://www.youtube.com/watch?v=8OJC21T2SL4) (**Greg Kamradt**). *The canonical tour: character → recursive → document → semantic → agentic, with intuition for each.*
2. **Get the practical rules** — read [Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/) (**Pinecone**). *Concrete guidance on chunk size, overlap, and matching strategy to content type.*
3. **Understand the key failure mode** — read [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) (**Anthropic**). *Why isolated chunks lose meaning, and how prepending context cuts retrieval failures 35–49%.*
4. **See it measured** — read [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking) (**Chroma Research**). *A rare empirical study — how to actually measure whether a chunking choice helped (the recall/precision tradeoff).*
5. **Go deeper on semantic/agentic** — watch [The BEST Way to Chunk Text for RAG](https://www.youtube.com/watch?v=Pk2BeaGbcTE) (**Adam Lucek**). *Hands-on comparison of recursive vs semantic vs cluster-based splitting.*

**Videos**:
- [The 5 Levels of Text Splitting for Retrieval](https://www.youtube.com/watch?v=8OJC21T2SL4) — **Greg Kamradt** — the definitive map of chunking strategies, with a live notebook and ChunkViz.
- [The BEST Way to Chunk Text for RAG](https://www.youtube.com/watch?v=Pk2BeaGbcTE) — **Adam Lucek** — practical, code-first comparison of recursive, semantic, and cluster-based chunking.
- [Chunking Best Practices for RAG Applications](https://www.youtube.com/watch?v=uhVMFZjUOJI) — **KX** — how chunk size/overlap interact with embeddings and retrieval, with live tuning.
- [Build Contextual Retrieval with Anthropic and Pinecone](https://www.youtube.com/watch?v=u-ocR-2P_YA) — **Pinecone** — implements the contextual-chunking recipe end-to-end.

**Interactive & visual**:
- [ChunkViz](https://chunkviz.up.railway.app/) — **Greg Kamradt** — paste text, pick a splitter and size/overlap, and *see* the chunk boundaries highlighted live; the fastest way to build intuition for where cuts land.

**Courses (free)**:
- [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) — **DeepLearning.AI** — its "Document Splitting" lesson walks character vs recursive vs token splitters hands-on.
- [RAG from Scratch — Indexing](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — the indexing notebooks cover splitting and multi-representation indexing with runnable code.

**Articles / blogs (free, no paywall)**:
- [Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/) — **Pinecone** — the standard practical reference on size, overlap, and method selection; the source for the overlap step/cost derivation on the concept page.
- [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — **Anthropic** — prepend chunk-specific context before embedding to recover lost meaning; reports the 35%/49%/67% retrieval-failure reductions cited on the page.
- [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking) — **Chroma Research** — a rare empirical study; how to measure whether a chunking choice helped, and the recall/precision tradeoff formalized on the page.
- [Chunking Strategies to Improve RAG Performance](https://weaviate.io/blog/chunking-strategies-for-rag) — **Weaviate** — survey of methods with code and when-to-use guidance.
- [Building and Evaluating Advanced RAG (chunking section)](https://www.deeplearning.ai/short-courses/building-evaluating-advanced-rag/) — **DeepLearning.AI / LlamaIndex** — sentence-window and parent-document retrieval, the retrieve-small-feed-large pattern.

**Key papers**:
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — **Reimers & Gurevych (2019)** — sentence embeddings whose cosine similarity tracks semantic relatedness; the basis for semantic chunking's adjacent-sentence similarity and the centroid-coherence intuition on the page.
- [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://arxiv.org/abs/2312.06648) — **Chen et al. (2023)** — proposes "propositions" as retrieval units; direct evidence that chunk granularity strongly affects retrieval.
- [LumberChunker: Long-Form Narrative Document Segmentation](https://arxiv.org/abs/2406.17526) — **Duarte et al. (2024)** — LLM-driven dynamic ("agentic") chunking that beats fixed-size splitting on long documents.

**Books (free, with chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering, Information Retrieval, and RAG" (indexing / passage units)](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — the IR foundations behind why passage granularity matters, free PDF.

**In this platform**:
- Concept page (full explanation): [Document Chunking Strategies](02-Document-Chunking-Strategies.md)
- Prereq (the pipeline this fits in): [01 RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
- Foundations (tokens & sentence embeddings): [06. NLP — Tokenization & Subword Algorithms](../../06.%20NLP/02-Tokenization-and-Subword-Algorithms/02-Tokenization-and-Subword-Algorithms.md) · [06. NLP — Sentence & Document Embeddings](../../06.%20NLP/07-Sentence-and-Document-Embeddings/07-Sentence-and-Document-Embeddings.md)
- Next in this domain (retrieve better): [03 Embedding Models for Retrieval](../03-Embedding-Models-for-Retrieval/03-Embedding-Models-for-Retrieval.md) · [04 Vector Databases & ANN Indexes](../04-Vector-Databases-and-ANN-Indexes/04-Vector-Databases-and-ANN-Indexes.md) · [06 Re-ranking with Cross-Encoders](../06-Re-ranking-Cross-Encoders/06-Re-ranking-Cross-Encoders.md)
