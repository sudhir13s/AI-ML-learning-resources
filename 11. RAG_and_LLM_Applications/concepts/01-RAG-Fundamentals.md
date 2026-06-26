---
id: "11-rag-and-llm-apps/rag-fundamentals"
topic: "RAG Fundamentals (retrieve-then-generate)"
parent: "11-rag-and-llm-apps"
level: intermediate
prereqs: ["embeddings", "transformers", "prompting"]
interview_frequency: very-high
updated: 2026-06-20
---

# RAG Fundamentals — Retrieve-then-Generate
> Give a frozen LLM fresh, private, or factual knowledge at query time by **retrieving** relevant
> documents and **stuffing them into the prompt** before the model generates — instead of baking that
> knowledge into the weights. The simplest fix for stale training data and hallucination, and the
> backbone of nearly every production LLM app.

**Why it matters:** the single most-asked LLM-application question — *why RAG over fine-tuning?*,
the indexing → retrieve → augment → generate pipeline, what each component (chunker, embedder, vector
store, retriever, generator) does, and where real systems break (bad chunks, low recall, the model
ignoring the context).

**⭐ Start here — suggested path:**

1. **Get the one-paragraph mental model** — watch [IBM: What is RAG?](https://www.youtube.com/watch?v=T-D1OfcDW1M). *Five minutes to the "retrieve facts, then answer with sources" picture before any code.*
2. **See the full pipeline** — read the [Pinecone RAG guide](https://www.pinecone.io/learn/retrieval-augmented-generation/). *Walks indexing → retrieval → generation and the trade-off vs fine-tuning.*
3. **Read the source** — skim the [original RAG paper](https://arxiv.org/abs/2005.11401). *Where "retrieval-augmented generation" was coined; understand the retriever + generator framing.*
4. **Build one end-to-end** — follow the [LangChain RAG tutorial](https://python.langchain.com/docs/tutorials/rag/) or watch [freeCodeCamp: RAG from Scratch](https://www.youtube.com/watch?v=sVcwVQRHIc8). *Wiring a loader → splitter → embedder → vector store → retriever → LLM cements every term.*
5. **Learn what breaks** — read [SLP3 Ch. 14 (RAG section)](https://web.stanford.edu/~jurafsky/slp3/14.pdf). *Grounds the pipeline in retrieval theory and failure modes (recall, faithfulness).*

## 🎓 Courses (free)
- [LangChain: Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/) — **DeepLearning.AI** — the canonical free, hands-on RAG short course (load → split → embed → retrieve → chat).
- [RAG from Scratch (course + notebooks)](https://github.com/langchain-ai/rag-from-scratch) — **LangChain (Lance Martin)** — a dozen short lessons, each with open-source code, building RAG up from first principles.

## 🎥 Videos
- [What is Retrieval-Augmented Generation (RAG)?](https://www.youtube.com/watch?v=T-D1OfcDW1M) — **IBM Technology (Marina Danilevsky)** — the clearest 6-minute conceptual overview; why retrieval beats memorization.
- [Learn RAG From Scratch – Python AI Tutorial](https://www.youtube.com/watch?v=sVcwVQRHIc8) — **freeCodeCamp (Lance Martin)** — full-length, end-to-end build covering indexing, retrieval, generation, and query translation.
- [RAG Explained in 12 Minutes](https://www.youtube.com/watch?v=v0ynfDPpe4E) — **Aishwarya Srinivasan** — a tight refresher on the moving parts and when RAG helps.
- [The 5 Levels Of Text Splitting For Retrieval](https://www.youtube.com/watch?v=8OJC21T2SL4) — **Greg Kamradt** — preview of the chunking step that makes or breaks retrieval quality (deep-dived in concept 2).

## 📄 Key Papers
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — the original RAG; retriever + generator trained together.
- [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906) — **Karpukhin et al. (2020)** — the dual-encoder retriever that made dense retrieval beat BM25 for open-domain QA.
- [Retrieval-Augmented Generation for LLMs: A Survey](https://arxiv.org/abs/2312.10997) — **Gao et al. (2023)** — the map of the whole field (naive → advanced → modular RAG); great reference.

## 📰 Articles / Blogs (free, no paywall)
- [Retrieval-Augmented Generation (RAG)](https://www.pinecone.io/learn/retrieval-augmented-generation/) — **Pinecone** — clean, vendor-neutral walkthrough of the pipeline and the fine-tuning trade-off.
- [RAG (prompting guide)](https://www.promptingguide.ai/techniques/rag) — **DAIR.AI** — concise reference that ties RAG to prompting and lists the key variants.
- [Understanding RAG](https://docs.llamaindex.ai/en/stable/understanding/rag/) — **LlamaIndex** — the five stages (loading, indexing, storing, querying, evaluation) with crisp definitions.
- [The Illustrated Retrieval Transformer](https://jalammar.github.io/illustrated-retrieval-transformer/) — **Jay Alammar** — visual intuition for retrieval-augmented models.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 14 "Question Answering, Information Retrieval, and RAG"**](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — grounds RAG in IR theory; retrieve-then-read and evaluation, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md) · [8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md)
- Next concepts: [02 Document Chunking Strategies](02-Document-Chunking-Strategies.md) · [03 Embedding Models for Retrieval](03-Embedding-Models-for-Retrieval.md)
- Related domain: [06. NLP — Information Retrieval & Semantic Search](../../06.%20NLP/concepts/16-Information-Retrieval-and-Semantic-Search.md) · [09. LLMs — Prompting & In-Context Learning](../../09.%20LLMs/16-Prompting-and-In-Context-Learning/16-Prompting-and-In-Context-Learning.md)
