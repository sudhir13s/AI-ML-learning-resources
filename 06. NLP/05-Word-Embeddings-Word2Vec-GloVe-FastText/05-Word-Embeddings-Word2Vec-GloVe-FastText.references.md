---
id: "06-nlp/word-embeddings/references"
topic: "Word Embeddings — References"
parent: "06-nlp/word-embeddings"
type: references
updated: 2026-06-27
---

# Word Embeddings — references and further reading

> Companion link library for **[Word Embeddings](05-Word-Embeddings-Word2Vec-GloVe-FastText.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first within each group. Every entry is free/open and from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity, and every link verified.

**Start here — suggested path**:
1. **Build intuition** — watch [Word2Vec, Clearly Explained](https://www.youtube.com/watch?v=viZrOnJclY0) (**StatQuest**), then read [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) (**Jay Alammar**). *Fastest route to a correct mental model.*
2. **Feel the geometry** — explore the [TensorFlow Embedding Projector](https://projector.tensorflow.org/) and watch [Vectoring Words](https://www.youtube.com/watch?v=gQddtTdmG_8) (**Computerphile**). *Makes king − man + woman ≈ queen tangible.*
3. **Get the math** — [CS224N Lecture 2](https://www.youtube.com/watch?v=ERibwqs9p38) + [SLP3 Ch. 6](https://web.stanford.edu/~jurafsky/slp3/6.pdf). *The objective, gradients, and negative sampling you'll be asked to derive.*
4. **Read the sources** — [word2vec](https://arxiv.org/abs/1301.3781) → [negative sampling](https://arxiv.org/abs/1310.4546) → [GloVe](https://nlp.stanford.edu/pubs/glove.pdf) → [FastText](https://arxiv.org/abs/1607.04606) → [Levy & Goldberg](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization). *Predictive → global co-occurrence → subword → the unification.*
5. **Make it concrete** — implement skip-gram + negative sampling with [d2l Ch. 15](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html).

**In this platform**:
- Concept page (full explanation): [Word Embeddings](05-Word-Embeddings-Word2Vec-GloVe-FastText.md)
- Runnable code: [teaching notebook](code/05-Word-Embeddings-Word2Vec-GloVe-FastText.ipynb) · [source-of-truth script](code/word_embeddings.py) · [figure generator](code/make_figures_05.py)
- Concept depth (the *why*): [AI-ML-intuition 1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md) · [1.02a Word2Vec / Skip-Gram](../../../AI-ML-intuition/Module_1_Representation/1.02a_Word_Embeddings_Word2Vec_Skip-Gram.md)
- Previous concept: [Bag-of-Words & TF-IDF](../03-Bag-of-Words-and-TF-IDF/03-Bag-of-Words-and-TF-IDF.md) — the sparse, count-based representations embeddings replace
- Related: [Tokenization & Subword Algorithms](../02-Tokenization-and-Subword-Algorithms/02-Tokenization-and-Subword-Algorithms.md) — FastText's subword idea in modern LLMs
- Next concept: [Contextual Embeddings (ELMo/BERT)](../06-Contextual-Embeddings-ELMo-BERT/06-Contextual-Embeddings-ELMo-BERT.md) — a vector per word *per sentence*
- Onward: [Sentence & Document Embeddings](../07-Sentence-and-Document-Embeddings/07-Sentence-and-Document-Embeddings.md) · [09. LLMs](../../09.%20LLMs/README.md)

**Videos** (best-first):
- [Word Embedding and Word2Vec, Clearly Explained](https://www.youtube.com/watch?v=viZrOnJclY0) — **StatQuest (Josh Starmer)** — gentle, from-scratch intuition for skip-gram + training; the fastest route to a correct mental model.
- [CS224N Lecture 2 — Word Vectors and word2vec](https://www.youtube.com/watch?v=ERibwqs9p38) — **Stanford (Christopher Manning)** — the rigorous derivation: objective, gradients, negative sampling; the lecture interviewers' questions come from.
- [Vectoring Words (Word Embeddings)](https://www.youtube.com/watch?v=gQddtTdmG_8) — **Computerphile (Rob Miles)** — vivid intuition for *why* vector arithmetic captures meaning.
- [The Illustrated Word2vec — A Gentle Intro to Word Embeddings](https://www.youtube.com/watch?v=ISPId9Lhc1g) — **Jay Alammar** — the video companion to the Start-here article; richly visual.

**Interactive & visual**:
- [TensorFlow Embedding Projector](https://projector.tensorflow.org/) — **Google** — load real pretrained word2vec/GloVe vectors and fly through the embedding space in 3-D; search a word, watch its nearest neighbours light up. The single best way to *feel* that this geometry is real.
- [GloVe project page](https://nlp.stanford.edu/projects/glove/) — **Stanford NLP** — the paper, the pretrained vectors, and an interactive nearest-neighbour / analogy explorer in one place.

**Courses (free)**:
- [Google ML Crash Course — Embeddings](https://developers.google.com/machine-learning/crash-course/embeddings) — **Google** — short, applied intro to why/how embeddings work.
- [Stanford CS224N — NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the definitive treatment (word2vec → GloVe), lectures + notes free.

**Articles / blogs (free, no paywall)** (best-first):
- [The Illustrated Word2Vec](https://jalammar.github.io/illustrated-word2vec/) — **Jay Alammar** — the most-loved visual walkthrough of embeddings and skip-gram; the clearest first read.
- [On Word Embeddings — Part 1](https://www.ruder.io/word-embeddings-1/) — **Sebastian Ruder** — the definitive survey series (history, models, math).
- [Word2Vec Tutorial — The Skip-Gram Model](https://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) — **Chris McCormick** — step-by-step skip-gram with negative sampling, from the input pairs to the gradients.
- [Word2Vec (TensorFlow tutorial)](https://www.tensorflow.org/text/tutorials/word2vec) — **TensorFlow** — implement skip-gram + negative sampling end to end in code.

**Papers**:
- [Distributional Structure](https://doi.org/10.1080/00437956.1954.11659520) — **Harris (1954)** — the linguistics primary source for the distributional hypothesis ("words in similar contexts have similar meanings") that every embedding method operationalizes.
- [Distributed Representations of Words and Phrases (negative sampling)](https://arxiv.org/abs/1310.4546) — **Mikolov et al. (2013)** — negative sampling, hierarchical softmax, freq^0.75, subsampling.
- [Efficient Estimation of Word Representations (word2vec)](https://arxiv.org/abs/1301.3781) — **Mikolov et al. (2013)** — introduces CBOW + skip-gram.
- [Enriching Word Vectors with Subword Information (FastText)](https://arxiv.org/abs/1607.04606) — **Bojanowski et al. (2017)** — character n-grams → handles OOV & morphology.
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — **Pennington, Socher & Manning (2014)** — co-occurrence-matrix factorization view.
- [Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://aclanthology.org/Q15-1016/) — **Levy, Goldberg & Dagan (2015)** — shows SVD-PPMI is competitive; all methods factorize co-occurrence.
- [Man is to Computer Programmer as Woman is to Homemaker? (debiasing)](https://arxiv.org/abs/1607.06520) — **Bolukbasi et al. (2016)** — measures and addresses gender bias baked into embeddings.
- [Neural Word Embedding as Implicit Matrix Factorization](https://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization) — **Levy & Goldberg (2014)** — proves SGNS implicitly factorizes a shifted-PMI matrix.
- [Word Association Norms, Mutual Information, and Lexicography](https://aclanthology.org/J90-1003/) — **Church & Hanks (1990)** — the origin of PMI as a word-association measure.

**Books (free chapters)**:
- [A Primer on Neural Network Models for NLP — §5 (word embeddings)](https://arxiv.org/abs/1510.00726) — **Yoav Goldberg** — concise, rigorous, free on arXiv.
- [Dive into Deep Learning — Ch. 15 (word2vec, approximate training, GloVe, subword)](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — free, with runnable code.
- [Speech and Language Processing, 3rd ed. — Ch. 6 "Vector Semantics and Embeddings"](https://web.stanford.edu/~jurafsky/slp3/6.pdf) — **Jurafsky & Martin** — the standard reference (TF-IDF → word2vec).
