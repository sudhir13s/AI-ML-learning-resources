---
id: "06-nlp/sentence-document-embeddings/references"
topic: "Sentence & Document Embeddings — References"
parent: "06-nlp/sentence-document-embeddings"
type: references
updated: 2026-06-27
---

# Sentence & Document Embeddings — references and further reading

> Companion link library for **[Sentence & Document Embeddings](07-Sentence-and-Document-Embeddings.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Intro to Sentence Embeddings with Transformers](https://www.youtube.com/watch?v=jVPd7lEvjtg) (**James Briggs**). *Why a sentence needs its own vector, not a bag of token vectors, and the cosine-similarity intuition.*
2. **See why SBERT works** — watch [SBERT (Sentence Transformers) is not BERT Sentence Embedding](https://www.youtube.com/watch?v=lVqwznaVi78) (**Discover AI**). *The siamese trick that makes pooled vectors comparable.*
3. **Read the source** — [Sentence-BERT](https://arxiv.org/abs/1908.10084) (**Reimers & Gurevych, 2019**). *Architecture, pooling, the three objectives, and the 65-hours → 5-seconds result.*
4. **Get hands-on** — read the [SentenceTransformers docs](https://www.sbert.net/) + [semantic-search guide](https://www.sbert.net/examples/applications/semantic-search/README.html). *Embed, index, query, and rerank in a few lines.*
5. **Go contrastive** — read [SimCSE](https://arxiv.org/abs/2104.08821) (**Gao et al., 2021**). *Dropout as a free positive pair; the modern training recipe.*

**Videos**:
- [Intro to Sentence Embeddings with Transformers](https://www.youtube.com/watch?v=jVPd7lEvjtg) — **James Briggs** — clear motivation + cosine-similarity intuition for sentence vectors.
- [SBERT (Sentence Transformers) is not BERT Sentence Embedding](https://www.youtube.com/watch?v=lVqwznaVi78) — **Discover AI** — the siamese architecture and why pooled BERT alone fails.
- [Sentence Transformers: Embedding, Similarity, Semantic Search & Clustering](https://www.youtube.com/watch?v=OlhNZg4gOvA) — **Pradip Nichite** — end-to-end code for the four core applications.
- [Unsupervised Sentence Transformers (how TSDAE works)](https://www.youtube.com/watch?v=pNvujJ1XyeQ) — **James Briggs** — training sentence encoders without labels (a sibling of SimCSE).

**Interactive & visual**:
- [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) — **Jay Alammar** — the clearest visual mental model of the encoder whose token vectors we pool; `[CLS]` and the contextual layers, drawn.

**Courses (free)**:
- [Hugging Face LLM Course — Ch. 1: Transformer Models](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — how encoders produce poolable representations, code-first.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the pretraining + representation lectures that ground sentence encoders.

**Articles / blogs (free, no paywall)**:
- [SentenceTransformers documentation](https://www.sbert.net/) — **Reimers et al.** — the canonical, free hands-on reference for training and using bi-encoders + cross-encoders.
- [Semantic Search with SentenceTransformers](https://www.sbert.net/examples/applications/semantic-search/README.html) — **SBERT docs** — bi-encoder retrieval + cross-encoder reranking patterns, in code.
- [Sentence Embeddings (NLP series)](https://www.pinecone.io/learn/series/nlp/sentence-embeddings/) — **Pinecone (James Briggs)** — from mean-pooling's failure to SBERT, with the anisotropy story spelled out.
- [Getting Started with Embeddings](https://huggingface.co/blog/getting-started-with-embeddings) — **Hugging Face** — embed sentences and build semantic search end to end, free.
- [MTEB: Massive Text Embedding Benchmark](https://huggingface.co/blog/mteb) — **Hugging Face** — how to read the leaderboard and pick an embedding model for *your* task.

**Key papers**:
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — **Reimers & Gurevych (2019)** — the siamese fine-tune (NLI classification / STS regression / triplet) that made BERT embeddings cosine-comparable.
- [A Simple but Tough-to-Beat Baseline for Sentence Embeddings (SIF)](https://openreview.net/forum?id=SyK00v5xx) — **Arora, Liang & Ma (2017)** — the smooth-inverse-frequency weighting + common-component removal.
- [Distributed Representations of Sentences and Documents (Doc2Vec / Paragraph Vectors)](https://arxiv.org/abs/1405.4053) — **Le & Mikolov (2014)** — PV-DM and PV-DBOW: the first widely-used *learned* document vector, the bridge from averaging word vectors to running an encoder.
- [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) — **Gao, Yao & Chen (2021)** — dropout as the positive-pair augmentation; supervised NLI hard negatives; fixes anisotropy.
- [On the Sentence Embeddings from Pre-trained Language Models (BERT-flow)](https://arxiv.org/abs/2011.05864) — **Li et al. (2020)** — the anisotropy analysis and a flow-based whitening fix for raw BERT vectors.
- [Universal Sentence Encoder](https://arxiv.org/abs/1803.11175) — **Cer et al. (2018)** — transformer and DAN sentence encoders for transfer, predating SBERT.
- [Text Embeddings by Weakly-Supervised Contrastive Pre-training (E5)](https://arxiv.org/abs/2212.03533) — **Wang et al. (2022)** — the contrastive recipe + `query:`/`passage:` instruction prefixes behind modern embedding models.
- [C-Pack / BGE: Packed Resources for General Chinese & English Embeddings](https://arxiv.org/abs/2309.07597) — **Xiao et al. (2023)** — the BGE family, a strong open multi-stage contrastive embedding model.
- [MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — **Muennighoff et al. (2022)** — the 8-task, 50+-dataset benchmark that is the field's scoreboard.
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832) — **Khattab & Zaharia (2020)** — per-token vectors + MaxSim late interaction, between bi- and cross-encoders.
- [Dense Passage Retrieval for Open-Domain QA (DPR)](https://arxiv.org/abs/2004.04906) — **Karpukhin et al. (2020)** — dense bi-encoder retrieval that beats BM25; the retriever pattern behind RAG.

**Reference & API docs**:
- [E5 model card (intfloat/e5-base-v2)](https://huggingface.co/intfloat/e5-base-v2) — **Microsoft** — the exact prefixes and pooling a deployed E5 model expects (read before you use it).
- [OpenAI Embeddings guide](https://platform.openai.com/docs/guides/embeddings) — **OpenAI** — `text-embedding-3` usage, dimensions, and Matryoshka truncation.
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) — **MTEB** — the live ranking; filter by the **retrieval** column for search/RAG.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 6 "Vector Semantics and Embeddings"](https://web.stanford.edu/~jurafsky/slp3/6.pdf) — **Jurafsky & Martin** — vector semantics that generalize from words to sentences (cosine, similarity, the geometry).

**In this platform**:
- Concept page (full explanation): [Sentence & Document Embeddings](07-Sentence-and-Document-Embeddings.md)
- Builds on: [Word Embeddings (Word2Vec · GloVe · FastText)](../05-Word-Embeddings-Word2Vec-GloVe-FastText/05-Word-Embeddings-Word2Vec-GloVe-FastText.md) — the static vectors we average · [Contextual Embeddings (ELMo · BERT)](../06-Contextual-Embeddings-ELMo-BERT/06-Contextual-Embeddings-ELMo-BERT.md) — the token-level vectors we pool, and the anisotropy source.
- The training engine: [Contrastive / Self-Supervised Learning](../../04.%20Unsupervised_Learning/concepts/12-Contrastive-Self-Supervised-Learning.md) — InfoNCE, positives/negatives, alignment & uniformity behind SimCSE.
- Puts it to work: [Information Retrieval & Semantic Search](../16-Information-Retrieval-and-Semantic-Search/16-Information-Retrieval-and-Semantic-Search.md) — BM25, dense retrieval, ANN indexing, and retrieve-then-rerank.
- Related building blocks: [Bag-of-Words & TF-IDF](../03-Bag-of-Words-and-TF-IDF/03-Bag-of-Words-and-TF-IDF.md) — the lexical baseline (and the IDF instinct SIF echoes).
- Concept depth (the *why*): [AI-ML-intuition 1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md) · [1.13 Contrastive Learning (SimCLR · InfoNCE)](../../../AI-ML-intuition/Module_1_Representation/1.13_Representation_Contrastive_Learning_SimCLR_InfoNCE.md)
