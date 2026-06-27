---
id: "06-nlp/topic-modeling/references"
topic: "Topic Modeling — References"
parent: "06-nlp/topic-modeling"
type: references
updated: 2026-06-27
---

# Topic Modeling — references and further reading

> Companion link library for **[Topic Modeling (LDA · NMF)](15-Topic-Modeling-LDA-NMF.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Latent Dirichlet Allocation (Part 1)](https://www.youtube.com/watch?v=T05t-SqKArY) (**Luis Serrano**). *The generative story, visually, before any math — the clearest LDA intro anywhere.*
2. **See the machine** — watch [Training LDA: Gibbs Sampling (Part 2)](https://www.youtube.com/watch?v=BaM1uiCpj_E) (**Luis Serrano**). *How Gibbs sampling actually recovers the topics from the generated documents.*
3. **Get the NMF view** — read the [Lee & Seung NMF paper](https://www.nature.com/articles/44565) and the [scikit-learn NMF/LDA example](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html). *Topics as non-negative, parts-based matrix factors, side by side with LDA.*
4. **Read the source** — [the LDA paper](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) (**Blei, Ng & Jordan, 2003**). *The generative model and variational inference, from the authors.*
5. **Make it concrete** — code it with the [gensim LDA tutorial](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html). *Fit LDA on a real corpus, evaluate coherence, read the topics.*

**Videos**:
- [Latent Dirichlet Allocation (Part 1 of 2)](https://www.youtube.com/watch?v=T05t-SqKArY) — **Luis Serrano** — the clearest visual intro to the generative model; start here.
- [Training LDA: Gibbs Sampling (Part 2 of 2)](https://www.youtube.com/watch?v=BaM1uiCpj_E) — **Luis Serrano** — how Gibbs sampling recovers topics from the generated documents.
- [Topic Models: Gibbs Sampling](https://www.youtube.com/watch?v=u7l5hhmdc0M) — **Jordan Boyd-Graber** — the collapsed-Gibbs update derived and explained by a topic-modeling researcher (the update used on this page).
- [Topic Models — Introduction](https://www.youtube.com/watch?v=IUAHUEpFucA) — **Jordan Boyd-Graber** — the latent-variable framing and the generative story, by a domain researcher.
- [Intuition behind LDA for Topic Modeling](https://www.youtube.com/watch?v=Cpt97BpI-t4) — **Bhavesh Bhatt** — an applied walkthrough with code, documents → topics → words.
- [BERTopic explained](https://www.youtube.com/watch?v=uZxQz87lb84) — **Maarten Grootendorst (author)** — the embed → cluster → c-TF-IDF pipeline of the modern embedding-based approach, from its creator.

**Courses (free)**:
- [scikit-learn — Topic extraction with NMF & LDA (example)](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html) — **scikit-learn** — a runnable side-by-side of both methods on real text.
- [gensim — Topic Modelling tutorials](https://radimrehurek.com/gensim/auto_examples/index.html) — **gensim (Řehůřek)** — end-to-end LDA pipelines, coherence evaluation, and visualization, free.

**Articles / blogs (free, no paywall)**:
- [scikit-learn User Guide — Decomposition: LDA & NMF](https://scikit-learn.org/stable/modules/decomposition.html#latent-dirichlet-allocation-lda) — **scikit-learn** — the math of LDA's variational inference and NMF's objectives/updates, with the exact API.
- [Topic Modeling with Gensim (LDA, end-to-end)](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/) — **Machine Learning Plus** — preprocessing → LDA → coherence → pyLDAvis, the full practical pipeline.
- [BERTopic — Algorithm documentation](https://maartengr.github.io/BERTopic/algorithm/algorithm.html) — **Maarten Grootendorst** — the embed → UMAP → HDBSCAN → c-TF-IDF pipeline, from the author.
- [pyLDAvis — interactive topic-model visualization](https://github.com/bmabey/pyLDAvis) — **Ben Mabey** — the standard tool for inspecting topic separation and per-topic relevant terms.

**Key papers**:
- [Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — **Blei, Ng & Jordan (2003)** — the foundational topic model, the generative joint, and its variational inference (the source of the plate diagram and joint on this page).
- [Dynamic Topic Models](https://mimno.infosci.cornell.edu/info6150/readings/dynamic_topic_models.pdf) — **Blei & Lafferty (2006)** — topics that evolve over time, the basis of trend analysis.
- [BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — **Grootendorst (2022)** — the leading embedding-based topic model (embed → UMAP → HDBSCAN → c-TF-IDF).
- [Reading Tea Leaves: How Humans Interpret Topic Models](https://proceedings.neurips.cc/paper/2009/hash/f92586a25bb3145facd64ab20fd554ff-Abstract.html) — **Chang et al. (2009)** — the result that lower perplexity can mean *less* interpretable topics; why we trust coherence over perplexity.
- [Finding Scientific Topics](https://www.pnas.org/doi/10.1073/pnas.0307752101) — **Griffiths & Steyvers (2004)** — collapsed Gibbs sampling for LDA; Eq. (5) is the conditional derived and coded on this page.
- [Online Learning for Latent Dirichlet Allocation](https://papers.nips.cc/paper/2010/hash/71f6278d140af599e06ad9bf1ba03cb0-Abstract.html) — **Hoffman, Blei & Bach (2010)** — the online/stochastic mini-batch variational inference that scales LDA to huge corpora.
- [Probabilistic Latent Semantic Indexing](https://arxiv.org/abs/1301.6705) — **Hofmann (1999)** — pLSA, LDA's predecessor; the latent-topic-variable idea, the source of the pLSA mixture on this page.
- [Learning the Parts of Objects by Non-negative Matrix Factorization](https://www.nature.com/articles/44565) — **Lee & Seung (1999)** — NMF and the parts-based "faces" demonstration; the source of the $V \approx WH$ interpretation.
- [Algorithms for Non-negative Matrix Factorization](https://proceedings.neurips.cc/paper/2000/hash/f9d1152547c0bde01830b7e8bd60024c-Abstract.html) — **Lee & Seung (2001)** — the multiplicative update rules and the auxiliary-function convergence proof coded on this page.
- [Optimizing Semantic Coherence in Topic Models](https://aclanthology.org/D11-1024/) — **Mimno, Wallach, Talley, Leenders & McCallum (2011)** — the UMass coherence measure (intrinsic, single-corpus); the page computes a stabilized, pair-averaged variant of it.
- [Automatic Evaluation of Topic Coherence](https://aclanthology.org/N10-1012/) — **Newman, Lau, Grieser & Baldwin (2010)** — the PMI/NPMI coherence family that correlates with human judgments.
- [Exploring the Space of Topic Coherence Measures](http://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — **Röder, Both & Hinneburg (2015)** — the $C_v$ coherence measure and the systematic comparison behind it (open PDF).
- [Probabilistic Topic Models (chapter)](https://cocosci.princeton.edu/tom/papers/SteyversGriffiths.pdf) — **Steyvers & Griffiths (2007)** — the authors' open tutorial deriving collapsed Gibbs sampling for LDA, the update used on this page.

**Books (free chapters)**:
- [Introduction to Information Retrieval — Ch. 18 "Matrix decompositions and Latent Semantic Indexing"](https://nlp.stanford.edu/IR-book/html/htmledition/matrix-decompositions-and-latent-semantic-indexing-1.html) — **Manning, Raghavan & Schütze** — LSI/SVD, the matrix-factorization roots of NMF topic models.
- [Speech and Language Processing, 3rd ed. — Ch. 6 "Vector Semantics and Embeddings"](https://web.stanford.edu/~jurafsky/slp3/6.pdf) — **Jurafsky & Martin** — the term–document matrix and its low-rank structure that topic models exploit.

**In this platform**:
- Concept page (full explanation): [Topic Modeling (LDA · NMF)](15-Topic-Modeling-LDA-NMF.md)
- Runnable code: [teaching notebook](code/15-Topic-Modeling-LDA-NMF.ipynb) (executed, assert-before-print) · [source module](code/topic_modeling.py) (from-scratch Gibbs LDA, NMF, coherence) · [figure generator](code/make_figures_15.py)
- Prior step (the input matrix): [Bag-of-Words & TF-IDF](../03-Bag-of-Words-and-TF-IDF/03-Bag-of-Words-and-TF-IDF.md) — the document–term matrix topic models factorize.
- The latent-variable / EM foundation: [Gaussian Mixture Models & EM](../../04.%20Unsupervised_Learning/04-Gaussian-Mixture-Models-and-EM/04-Gaussian-Mixture-Models-and-EM.md) — same soft-assignment fitting, on continuous data.
- The dimensionality-reduction framing: [Dimensionality Reduction — Overview (PCA · SVD)](../../04.%20Unsupervised_Learning/06-Dimensionality-Reduction-Overview/06-Dimensionality-Reduction-Overview.md) — NMF/LSA as reductions of the term matrix.
- The modern embedding route: [Sentence & Document Embeddings](../07-Sentence-and-Document-Embeddings/07-Sentence-and-Document-Embeddings.md) — what BERTopic clusters instead of counts.
- Where LSA connects: [Information Retrieval & Semantic Search](../16-Information-Retrieval-and-Semantic-Search/16-Information-Retrieval-and-Semantic-Search.md) — latent semantic indexing links the two.
