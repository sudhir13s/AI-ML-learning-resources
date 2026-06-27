---
id: "06-nlp/information-retrieval-semantic-search/references"
topic: "Information Retrieval & Semantic Search — References"
parent: "06-nlp/information-retrieval-semantic-search"
type: references
updated: 2026-06-22
---

# Information Retrieval & Semantic Search — references and further reading

> Companion link library for **[Information Retrieval & Semantic Search](16-Information-Retrieval-and-Semantic-Search.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author (the paper's authors) or a recognized deep explainer, chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Frame lexical vs semantic** — read [What is semantic search?](https://www.elastic.co/what-is/semantic-search) (**Elastic**). *The clearest plain-English split between term-overlap and meaning-based retrieval.*
2. **Get the lexical baseline** — read [Scoring, term weighting & the vector space model](https://nlp.stanford.edu/IR-book/html/htmledition/scoring-term-weighting-and-the-vector-space-model-1.html) (**Manning IR Book**). *TF-IDF/BM25 ranking, the classical baseline you must beat.*
3. **See semantic search in code** — watch [Sentence Transformers: Semantic Search & Clustering](https://www.youtube.com/watch?v=OlhNZg4gOvA) (**Pradip Nichite**) and read the [SBERT semantic-search guide](https://www.sbert.net/examples/applications/semantic-search/README.html). *Bi-encoder retrieval + cross-encoder rerank, end to end.*
4. **Read the dense-retrieval source** — [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906) (**Karpukhin et al., 2020**). *The contrastive dual-encoder + in-batch negatives, derived on the page.*
5. **Connect to RAG** — read [SLP3 Ch. 14: Question Answering, Information Retrieval, and RAG](https://web.stanford.edu/~jurafsky/slp3/14.pdf) (**Jurafsky & Martin**). *IR + dense retrieval + RAG in the standard text.*

**Videos**:
- [Sentence Transformers: Embedding, Similarity, Semantic Search & Clustering](https://www.youtube.com/watch?v=OlhNZg4gOvA) — **Pradip Nichite** — end-to-end semantic search in code, the practical companion to this page.
- [Intro to Sentence Embeddings with Transformers](https://www.youtube.com/watch?v=jVPd7lEvjtg) — **James Briggs** — the embedding step dense retrieval depends on, built up carefully.
- [Hierarchical Navigable Small Worlds (HNSW) for Vector Search](https://www.youtube.com/watch?v=QvKMwLjdK-s) — **James Briggs (Pinecone)** — the layered greedy-search graph behind most vector databases, visualized.
- [Product Quantization for Vector Search](https://www.youtube.com/watch?v=t9mRf2S5vDI) — **James Briggs (Pinecone)** — how PQ compresses vectors to codes, with the memory math.
- [BM25 — The Best Search Algorithm You've Never Heard Of](https://www.youtube.com/watch?v=ruBm9WywevM) — **ML & DS** — the BM25 saturation/length-normalization terms, clearly.
- [What is Retrieval-Augmented Generation (RAG)?](https://www.youtube.com/watch?v=T-D1OfcDW1M) — **IBM Technology** — where the retriever you build plugs into the LLM, in five clear minutes.
- [FAISS — Facebook AI Similarity Search](https://www.youtube.com/watch?v=sKyvsdEv6rk) — **James Briggs (Pinecone)** — building Flat / IVF / IVFPQ indexes in code, the library every ANN benchmark uses.

**Interactive & visual**:
- [Nearest Neighbor Indexes for Similarity Search](https://www.pinecone.io/learn/series/faiss/vector-indexes/) — **Pinecone (James Briggs)** — Flat vs IVF vs HNSW vs PQ with recall/latency/memory plots, the recall–latency–memory surface made tangible.
- [The Building Blocks of LLMs: Vectors, Tokens and Embeddings](https://txt.cohere.com/what-is-similarity-between-sentences/) — **Cohere** — what "semantic similarity" geometrically means before you search on it.

**Courses (free)**:
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the IR + retrieval-for-QA lectures from the people who wrote the textbook.
- [Hugging Face LLM Course — semantic search with embeddings](https://huggingface.co/learn/llm-course/chapter5/6) — **Hugging Face** — build a working semantic-search index in code, FAISS included.
- [Pinecone — Faiss: The Missing Manual](https://www.pinecone.io/learn/series/faiss/) — **Pinecone** — a free, deep, hands-on course on FAISS index types (Flat/IVF/HNSW/PQ) and how to choose them.

**Articles / blogs (free, no paywall)**:
- [What is semantic search?](https://www.elastic.co/what-is/semantic-search) — **Elastic** — clear lexical-vs-semantic framing from a search-engine vendor.
- [Semantic Search with Sentence Transformers](https://www.sbert.net/examples/applications/semantic-search/README.html) — **SBERT docs (Reimers)** — the canonical bi-encoder retrieval + cross-encoder reranking patterns, in code.
- [Retrieve & Re-Rank](https://www.sbert.net/examples/applications/retrieve_rerank/README.html) — **SBERT docs (Reimers)** — the exact two-stage retrieve-then-rerank pipeline this page builds.
- [Reciprocal Rank Fusion (RRF)](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion) — **Elastic** — the $\frac{1}{k+r}$ fusion for combining BM25 + dense rankings, documented by a system that ships it.
- [tf–idf weighting](https://nlp.stanford.edu/IR-book/html/htmledition/tf-idf-weighting-1.html) — **Stanford IR Book** — the lexical-weighting math BM25 builds on, free.
- [HyDE: Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) — **Gao et al. (2022)** — generate a hypothetical answer, embed *that*, and search with it.

**Key papers**:
- [The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — **Robertson & Zaragoza (2009)** — the definitive derivation and justification of BM25.
- [Dense Passage Retrieval for Open-Domain QA (DPR)](https://arxiv.org/abs/2004.04906) — **Karpukhin et al. (2020)** — the contrastive dual-encoder with in-batch negatives that beats BM25; the dual-encoder dot-product score and the contrastive loss derived on the page.
- [Representation Learning with Contrastive Predictive Coding (InfoNCE)](https://arxiv.org/abs/1807.03748) — **van den Oord, Li & Vinyals (2018)** — the InfoNCE objective DPR's loss is an instance of.
- [Cumulated Gain-based Evaluation of IR Techniques (nDCG)](https://doi.org/10.1145/582415.582418) — **Järvelin & Kekäläinen (2002), ACM TOIS** — the original (discounted) cumulated gain and per-query ideal normalization the nDCG derivation follows.
- [Learning to Rank using Gradient Descent (RankNet)](https://www.microsoft.com/en-us/research/publication/learning-to-rank-using-gradient-descent/) — **Burges et al. (2005)** — the learning-to-rank line that popularized the exponential gain $2^{\mathrm{rel}}-1$ in DCG.
- [Sentence-BERT](https://arxiv.org/abs/1908.10084) — **Reimers & Gurevych (2019)** — the bi-encoder fine-tune that made semantic search fast and the cross-encoder rerank precise.
- [Efficient and Robust ANN Search using HNSW Graphs](https://arxiv.org/abs/1603.09320) — **Malkov & Yashunin (2018)** — the layered small-world graph behind most vector databases.
- [Product Quantization for Nearest Neighbor Search](https://inria.hal.science/inria-00514462v2/document) — **Jégou, Douze & Schmid (2011)** — splitting vectors into subspaces and quantizing each: the memory win for billion-scale search.
- [Billion-scale similarity search with GPUs (FAISS)](https://arxiv.org/abs/1702.08734) — **Johnson, Douze & Jégou (2019)** — the library every ANN benchmark is measured against.
- [Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — **Cormack, Clarke & Buettcher (2009)** — the original RRF paper; the $\frac{1}{k+r}$ fusion derived on this page.
- [ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction](https://arxiv.org/abs/2004.12832) — **Khattab & Zaharia (2020)** — the MaxSim late-interaction middle ground between bi- and cross-encoders.
- [Retrieval-Augmented Generation for Knowledge-Intensive NLP (RAG)](https://arxiv.org/abs/2005.11401) — **Lewis et al. (2020)** — where the retriever you built grounds the generator.
- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of IR Models](https://arxiv.org/abs/2104.08663) — **Thakur et al. (2021)** — the standard zero-shot retrieval benchmark; reports nDCG@10.

**Books (free chapters)**:
- [Introduction to Information Retrieval — **full book** (esp. Ch. 1 inverted index, Ch. 6 vector space, Ch. 8 evaluation)](https://nlp.stanford.edu/IR-book/html/htmledition/irbook.html) — **Manning, Raghavan & Schütze** — the definitive free IR text; the inverted index, BM25, and nDCG all live here.
- [Speech and Language Processing, 3rd ed. — **Ch. 14 "Question Answering, Information Retrieval, and RAG"**](https://web.stanford.edu/~jurafsky/slp3/14.pdf) — **Jurafsky & Martin** — lexical + dense retrieval + RAG in one chapter.

**In this platform**:
- Concept page (full explanation): [Information Retrieval & Semantic Search](16-Information-Retrieval-and-Semantic-Search.md)
- Prior steps (the inputs): [03 Bag-of-Words & TF-IDF](../03-Bag-of-Words-and-TF-IDF/03-Bag-of-Words-and-TF-IDF.md) (lexical / BM25 lineage) · [07 Sentence & Document Embeddings](../07-Sentence-and-Document-Embeddings/07-Sentence-and-Document-Embeddings.md) (dense vectors)
- Foundations: [k-Nearest-Neighbors](../../03.%20Supervised_Learning/concepts/04-k-Nearest-Neighbors.md) (the search ANN approximates) · [1.06 Vector Similarities — the Scaled Dot-Product](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md) (the scoring function)
- Puts it to work: [11 Question Answering](../11-Question-Answering/11-Question-Answering.md) (the retriever in open-domain QA) · canonical RAG home [11. RAG & LLM Applications](../../11.%20RAG_and_LLM_Applications/README.md) · the *why* [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md)
