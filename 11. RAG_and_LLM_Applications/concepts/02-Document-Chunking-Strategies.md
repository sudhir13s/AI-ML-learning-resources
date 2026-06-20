---
id: "11-rag-and-llm-apps/document-chunking"
topic: "Document Chunking Strategies"
parent: "11-rag-and-llm-apps"
level: intermediate
prereqs: ["rag-fundamentals", "tokenization", "embeddings"]
interview_frequency: high
updated: 2026-06-20
---

# Document Chunking Strategies
> How you split documents into retrievable pieces — fixed-size, recursive, document-structure,
> semantic, or LLM-driven "agentic" — and how chunk size, overlap, and added context decide whether
> retrieval surfaces the right passage. Chunking is the cheapest, highest-leverage knob in a RAG
> pipeline: bad chunks cap quality no matter how good the embedder or LLM is.

**Why it matters:** the most practical RAG question — *how do you chunk, and why?* Interviewers probe
the trade-off (small chunks = precise but context-poor; large chunks = rich but noisy), when to use
recursive vs semantic splitting, why overlap matters, and modern tricks (contextual retrieval,
parent-document) that fix the "chunk lost its context" problem.

**⭐ Start here — suggested path:**

1. **See the landscape** — watch [The 5 Levels Of Text Splitting](https://www.youtube.com/watch?v=8OJC21T2SL4). *The canonical tour: character → recursive → document → semantic → agentic, with intuition for each.*
2. **Get the practical rules** — read [Pinecone: Chunking Strategies](https://www.pinecone.io/learn/chunking-strategies/). *Concrete guidance on chunk size, overlap, and matching strategy to content type.*
3. **Understand the failure mode** — read [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval). *Why isolated chunks lose meaning, and how prepending context fixes recall.*
4. **See it measured** — read [Weaviate: Chunking Strategies for RAG](https://weaviate.io/blog/chunking-strategies-for-rag). *How chunking choices change retrieval quality, with code.*
5. **Go deeper on semantic/agentic** — watch [The BEST Way to Chunk Text for RAG](https://www.youtube.com/watch?v=Pk2BeaGbcTE). *Hands-on comparison of recursive vs semantic vs cluster-based splitting.*

## 🎓 Courses (free)
- [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) — **DeepLearning.AI** — its "Document Splitting" lesson walks character vs recursive vs token splitters hands-on.
- [RAG from Scratch — Indexing](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — the indexing notebooks cover splitting and multi-representation indexing with runnable code.

## 🎥 Videos
- [The 5 Levels Of Text Splitting For Retrieval](https://www.youtube.com/watch?v=8OJC21T2SL4) — **Greg Kamradt** — the definitive map of chunking strategies, with a live notebook and ChunkViz.
- [The BEST Way to Chunk Text for RAG](https://www.youtube.com/watch?v=Pk2BeaGbcTE) — **Adam Lucek** — practical, code-first comparison of recursive, semantic, and cluster-based chunking.
- [Chunking Best Practices for RAG Applications](https://www.youtube.com/watch?v=uhVMFZjUOJI) — **KX** — a livestream covering how chunk size/overlap interact with embeddings and retrieval.
- [Build Contextual Retrieval with Anthropic and Pinecone](https://www.youtube.com/watch?v=u-ocR-2P_YA) — **Pinecone** — implements the contextual-chunking recipe end-to-end.

## 📄 Key Papers
- [Dense X Retrieval: What Retrieval Granularity Should We Use?](https://arxiv.org/abs/2312.06648) — **Chen et al. (2023)** — proposes "propositions" as retrieval units; evidence that granularity strongly affects retrieval.
- [LumberChunker: Long-Form Narrative Document Segmentation](https://arxiv.org/abs/2406.17526) — **Duarte et al. (2024)** — LLM-driven dynamic chunking that beats fixed-size splitting on long docs.

## 📰 Articles / Blogs (free, no paywall)
- [Chunking Strategies for LLM Applications](https://www.pinecone.io/learn/chunking-strategies/) — **Pinecone** — the standard practical reference on size, overlap, and method selection.
- [Introducing Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — **Anthropic** — a strong modern recipe: prepend chunk-specific context before embedding to recover lost meaning.
- [Chunking Strategies to Improve RAG Performance](https://weaviate.io/blog/chunking-strategies-for-rag) — **Weaviate** — survey of methods with code and when-to-use guidance.
- [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking) — **Chroma Research** — a rare empirical study; how to actually *measure* whether a chunking choice helped.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 14 (IR & RAG, indexing/passage units)**](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — the IR foundations behind why passage granularity matters, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
- Prereqs: [01 RAG Fundamentals](01-RAG-Fundamentals.md) · Next: [03 Embedding Models for Retrieval](03-Embedding-Models-for-Retrieval.md)
- Related domain: [06. NLP — Tokenization & Subword Algorithms](../../06.%20NLP/concepts/02-Tokenization-and-Subword-Algorithms.md) · [06. NLP — Sentence & Document Embeddings](../../06.%20NLP/concepts/07-Sentence-and-Document-Embeddings.md)
