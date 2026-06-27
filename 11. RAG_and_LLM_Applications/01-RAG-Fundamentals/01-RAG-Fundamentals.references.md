---
id: "11-rag-and-llm-apps/rag-fundamentals/references"
topic: "RAG Fundamentals — References"
parent: "11-rag-and-llm-apps/rag-fundamentals"
type: references
updated: 2026-06-27
---

# RAG Fundamentals — references and further reading

> Companion link library for **[RAG Fundamentals](01-RAG-Fundamentals.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is a free, no-paywall link from a primary author or a recognized deep explainer — chosen for depth on *this* topic (retrieve-then-generate fundamentals), not popularity.

**Start here — suggested path**:
1. **Get the one-paragraph mental model** — watch [What is Retrieval-Augmented Generation (RAG)?](https://www.youtube.com/watch?v=T-D1OfcDW1M) (**IBM Technology**). *Five minutes to the "retrieve facts, then answer with sources" picture before any code.*
2. **See the whole pipeline** — read [Retrieval-Augmented Generation (RAG)](https://www.pinecone.io/learn/retrieval-augmented-generation/) (**Pinecone**). *Walks indexing → retrieval → generation and the fine-tuning trade-off, vendor-neutral.*
3. **Read the source** — skim [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (**Lewis et al. 2020**). *Where "RAG" was coined; the retriever + generator framing and the marginalization equation.*
4. **Build one end-to-end** — follow [RAG from Scratch](https://github.com/langchain-ai/rag-from-scratch) (**LangChain**) or watch [Learn RAG From Scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8) (**freeCodeCamp**). *Wiring loader → splitter → embedder → vector store → retriever → LLM cements every term.*
5. **Ground it in theory** — read [SLP3 Ch. 14 (IR + RAG)](https://web.stanford.edu/~jurafsky/slp3/14.pdf) (**Jurafsky & Martin**). *Retrieve-then-read and evaluation, from the standard NLP textbook.*

**Videos**:
- [What is Retrieval-Augmented Generation (RAG)?](https://www.youtube.com/watch?v=T-D1OfcDW1M) — **IBM Technology (Marina Danilevsky)** — the clearest 6-minute conceptual overview; *why* retrieval beats memorization.
- [Learn RAG From Scratch — Python AI Tutorial](https://www.youtube.com/watch?v=sVcwVQRHIc8) — **freeCodeCamp (Lance Martin, LangChain)** — full-length end-to-end build covering indexing, retrieval, generation, and query translation.
- [RAG Explained in 12 Minutes](https://www.youtube.com/watch?v=v0ynfDPpe4E) — **Aishwarya Srinivasan** — a tight refresher on the moving parts and when RAG actually helps.
- [The 5 Levels of Text Splitting for Retrieval](https://www.youtube.com/watch?v=8OJC21T2SL4) — **Greg Kamradt** — a preview of the chunking step that makes or breaks retrieval quality (deep-dived in concept 2).

**Interactive & visual**:
- [The Illustrated Retrieval Transformer](https://jalammar.github.io/illustrated-retrieval-transformer/) — **Jay Alammar** — visual intuition for retrieval-augmented models, embeddings, and nearest-neighbour lookup.
- [Embedding Projector](https://projector.tensorflow.org/) — **TensorFlow** — interactively explore a real embedding space in 2D/3D; *see* semantically similar text cluster, the geometry retrieval relies on.

**Courses (free)**:
- [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) — **DeepLearning.AI** — the canonical free, hands-on RAG short course (load → split → embed → retrieve → chat).
- [RAG from Scratch (course + notebooks)](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — a dozen short lessons, each with open-source code, building RAG up from first principles.
- [Stanford CS224N — Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford** — the lectures on question answering and retrieval that underpin dense retrieval and RAG.

**Articles / blogs (free, no paywall)**:
- [Retrieval-Augmented Generation (RAG)](https://www.pinecone.io/learn/retrieval-augmented-generation/) — **Pinecone** — clean, vendor-neutral walkthrough of the pipeline and the fine-tuning trade-off.
- [Understanding RAG](https://docs.llamaindex.ai/en/stable/understanding/rag/) — **LlamaIndex** — the five stages (loading, indexing, storing, querying, evaluation) with crisp definitions.
- [Retrieval Augmented Generation (RAG)](https://www.promptingguide.ai/techniques/rag) — **DAIR.AI** — concise reference that ties RAG to prompting and lists the key variants.
- [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1) — **Anyscale** — a production-grade end-to-end RAG build with the engineering decisions spelled out.
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — **Gao et al. (2023)** — the map of the whole field (naive → advanced → modular RAG); the best single reference for where each later chapter fits.

**Key papers**:
- [Dense Passage Retrieval for Open-Domain Question Answering (DPR)](https://arxiv.org/abs/2004.04906) — **Karpukhin et al. (2020)** — the dual-encoder retriever that made dense retrieval beat BM25 for open-domain QA; the retriever half of the original RAG.
- [REALM: Retrieval-Augmented Language Model Pre-Training](https://arxiv.org/abs/2002.08909) — **Guu et al. (2020)** — the contemporaneous idea of training a retriever jointly with a masked LM; foundational context for RAG.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — the original RAG paper; **Eq. 1–2 give the marginalization $P(y\mid x)=\sum_z P(z\mid x)P(y\mid x,z)$** used on the concept page.
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) — **Liu et al. (2024, TACL)** — the U-shaped accuracy curve (evidence used best at the start/end of context, worst in the middle) behind the lost-in-the-middle figure.
- [Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — **Gao et al. (2023)** — comprehensive survey; the taxonomy that organizes the rest of this domain.

**Books (free, with chapters)**:
- [Introduction to Information Retrieval — §6.3 "The vector space model" / cosine similarity](https://nlp.stanford.edu/IR-book/html/htmledition/dot-products-1.html) — **Manning, Raghavan & Schütze** — **derives cosine similarity** as the angle between document vectors, the source for the similarity formula on the concept page; full book free online.
- [Speech and Language Processing, 3rd ed. — Ch. 14 "Question Answering, Information Retrieval, and RAG"](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — grounds RAG in IR theory: retrieve-then-read, dense retrieval, and evaluation (free PDF).

**In this platform**:
- Concept page (full explanation): [RAG Fundamentals](01-RAG-Fundamentals.md)
- Foundations (the *why* behind embeddings & retrieval): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md) · [8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md) · [06. NLP — Information Retrieval & Semantic Search](../../06.%20NLP/16-Information-Retrieval-and-Semantic-Search/16-Information-Retrieval-and-Semantic-Search.md)
- Next in this domain (retrieve better): [02 Document Chunking Strategies](../02-Document-Chunking-Strategies/02-Document-Chunking-Strategies.md) · [03 Embedding Models for Retrieval](../03-Embedding-Models-for-Retrieval/03-Embedding-Models-for-Retrieval.md) · [04 Vector Databases & ANN Indexes](../04-Vector-Databases-and-ANN-Indexes/04-Vector-Databases-and-ANN-Indexes.md) · [05 Hybrid Search (BM25 + Dense)](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md) · [06 Re-ranking with Cross-Encoders](../06-Re-ranking-Cross-Encoders/06-Re-ranking-Cross-Encoders.md)
- Weighing the alternative: [12 Long-Context vs RAG](../12-Long-Context-vs-RAG/12-Long-Context-vs-RAG.md) · [11 RAG Evaluation](../11-RAG-Evaluation/11-RAG-Evaluation.md)
