---
id: "06-nlp/information-retrieval-semantic-search"
topic: "Information Retrieval & Semantic Search"
parent: "06-nlp"
level: intermediate
prereqs: ["bow-tfidf", "sentence-document-embeddings", "vector-similarity"]
interview_frequency: high
updated: 2026-06-19
---

# Information Retrieval & Semantic Search
> Finding the documents most relevant to a query. **Lexical IR** ranks by term overlap
> (TF-IDF / **BM25**); **semantic search** ranks by meaning, embedding query and documents into a
> shared vector space and retrieving nearest neighbors. The retrieval half of modern RAG systems.

**Why it matters:** IR is the backbone of search and the **retriever** in RAG, so it's a frequent
system-design interview topic. Be ready to explain **BM25** (and why it beats raw TF-IDF), the
**bi-encoder vs cross-encoder** trade-off, **dense retrieval (DPR)** with approximate nearest-neighbor
indexes, **hybrid** lexical+dense retrieval with reranking, and IR metrics like **MRR / nDCG /
Recall@k**.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [What is semantic search?](https://www.elastic.co/what-is/semantic-search) (**Elastic**). *Lexical vs semantic retrieval, plainly.*
2. **Get lexical IR** — read [Scoring, term weighting & the vector space model](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) (**IR Book**). *TF-IDF/BM25 ranking, the classical baseline.*
3. **See semantic search** — watch [Sentence Transformers: Semantic Search & Clustering](https://www.youtube.com/watch?v=OlhNZg4gOvA) (**Pradip Nichite**), and read the [SBERT semantic-search guide](https://www.sbert.net/examples/applications/semantic-search/README.html). *Bi-encoder retrieval + cross-encoder rerank.*
4. **Read the source** — [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906) → [RAG](https://arxiv.org/abs/2005.11401). *Dense retrieval, then grounding generation in it.*
5. **Reference** — [SLP3 Ch. 11](https://web.stanford.edu/~jurafsky/slp3/11.pdf) for IR + RAG in the standard text.

## 🎓 Courses (free)
- [Hugging Face LLM Course — semantic search with embeddings](https://huggingface.co/learn/llm-course/chapter5/6) — **Hugging Face** — build a semantic search index in code.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the IR + retrieval-for-QA lecture.

## 🎥 Videos
- [Sentence Transformers: Embedding, Similarity, Semantic Search & Clustering](https://www.youtube.com/watch?v=OlhNZg4gOvA) — **Pradip Nichite** — end-to-end semantic search in code.
- [Intro to Sentence Embeddings with Transformers](https://www.youtube.com/watch?v=jVPd7lEvjtg) — **James Briggs** — the embedding step retrieval depends on.
- [Cohere AI's LLM for Semantic Search in Python](https://www.youtube.com/watch?v=ejpc-nbKY2Y) — **James Briggs** — embeddings → nearest-neighbor retrieval.
- [Supercharging Semantic Search with Pinecone and Cohere](https://www.youtube.com/watch?v=e2g5ya4ZFro) — **Pinecone** — vector-DB retrieval at scale.

## 📄 Key Papers
- [Dense Passage Retrieval for Open-Domain QA (DPR)](https://arxiv.org/abs/2004.04906) — **Karpukhin et al. (2020)** — dense bi-encoder retrieval beats BM25.
- [Retrieval-Augmented Generation (RAG)](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — combines a retriever with a generator.
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — **Reimers & Gurevych (2019)** — the bi-encoder that makes semantic search fast.

## 📰 Articles / Blogs (free, no paywall)
- [What is semantic search?](https://www.elastic.co/what-is/semantic-search) — **Elastic** — clear lexical-vs-semantic framing.
- [Semantic Search with SentenceTransformers](https://www.sbert.net/examples/applications/semantic-search/README.html) — **SBERT docs** — bi-encoder retrieval + cross-encoder reranking patterns.
- [Scoring, term weighting & the vector space model](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) — **Stanford IR Book** — the classical ranking baseline, free.

## 📚 Books (free, with chapters)
- [Introduction to Information Retrieval — **full book (esp. Ch. 6 vector space, Ch. 8 evaluation)**](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) — **Manning, Raghavan & Schütze** — the definitive free IR text.
- [Speech and Language Processing, 3rd ed. — **Ch. 11 "Information Retrieval and Retrieval-Augmented Generation"**](https://web.stanford.edu/~jurafsky/slp3/11.pdf) — **Jurafsky & Martin** — IR + dense retrieval + RAG.

## 🔗 In this platform
- Prior steps: [03 Bag-of-Words & TF-IDF](03-Bag-of-Words-and-TF-IDF.md) (lexical) · [07 Sentence & Document Embeddings](07-Sentence-and-Document-Embeddings.md) (semantic).
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md) · [1.06 Scaled Dot-Product (similarity)](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md)
- Used by: [11 Question Answering](11-Question-Answering.md) — the retriever in open-domain QA. Canonical RAG home: [16. RAG & LLM Applications](../../11.%20RAG_and_LLM_Applications/concepts/README.md)
