---
id: "06-nlp/topic-modeling"
topic: "Topic Modeling (LDA · NMF)"
parent: "06-nlp"
level: intermediate
prereqs: ["bow-tfidf", "probability", "linear-algebra"]
interview_frequency: medium
updated: 2026-06-19
---

# Topic Modeling — LDA · NMF
> Unsupervised discovery of the **latent themes** in a collection of documents. Each document is a
> mixture of topics; each topic is a distribution over words. **LDA** is the generative-probabilistic
> approach; **NMF** is the linear-algebra (matrix-factorization) approach on the TF-IDF matrix.

**Why it matters:** topic modeling is the classic unsupervised-NLP interview topic and still the
go-to for exploring large unlabeled corpora. Be ready to describe **LDA**'s generative story
(Dirichlet priors over topic and word distributions), how inference works (Gibbs sampling /
variational), how **NMF** factorizes the term–document matrix into non-negative topic and weight
matrices, how the two differ, and how you choose the number of topics and evaluate with **coherence**.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Latent Dirichlet Allocation (Part 1)](https://www.youtube.com/watch?v=T05t-SqKArY) (**Luis Serrano**). *The generative story, visually, before any math.*
2. **See it applied** — watch [Intuition behind LDA for Topic Modeling](https://www.youtube.com/watch?v=Cpt97BpI-t4) (**Bhavesh Bhatt**). *Documents → topics → words in practice.*
3. **Read the source** — [LDA paper](https://jmlr.org/papers/volume3/blei03a/blei03a.pdf) (**Blei, Ng & Jordan, 2003**). *The generative model and variational inference.*
4. **Get the NMF view** — read the [Lee & Seung NMF paper](https://www.nature.com/articles/44565) and the [scikit-learn NMF/LDA example](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html). *Topics as non-negative matrix factors.*
5. **Make it concrete** — code it with [gensim LdaModel](https://radimrehurek.com/gensim/models/ldamodel.html). *Fit LDA on a real corpus and read the topics.*

## 🎓 Courses (free)
- [scikit-learn — Topic extraction with NMF & LDA (example)](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html) — **scikit-learn** — runnable side-by-side of both methods.
- [gensim — Topic Modelling tutorials](https://radimrehurek.com/gensim/auto_examples/index.html#documentation) — **gensim** — end-to-end LDA pipelines, free.

## 🎥 Videos
- [Latent Dirichlet Allocation (Part 1 of 2)](https://www.youtube.com/watch?v=T05t-SqKArY) — **Luis Serrano** — the clearest visual intro to the generative model.
- [Intuition behind LDA for Topic Modeling](https://www.youtube.com/watch?v=Cpt97BpI-t4) — **Bhavesh Bhatt** — applied walkthrough with code.
- [What is Latent Dirichlet Allocation (LDA)?](https://www.youtube.com/watch?v=b1L5wQQQXao) — **Dr Bo Han Class** — lecture-style explanation of LDA for text mining.
- [Latent Dirichlet Allocation for Topic Modeling](https://www.youtube.com/watch?v=5PUpLa0-g4w) — **upGrad** — concise NLP-tutorial framing.

## 📄 Key Papers
- [Latent Dirichlet Allocation](https://jmlr.org/papers/volume3/blei03a/blei03a.pdf) — **Blei, Ng & Jordan (2003)** — the foundational topic model.
- [Learning the Parts of Objects by Non-negative Matrix Factorization](https://www.nature.com/articles/44565) — **Lee & Seung (1999)** — the NMF method underlying NMF topic modeling.

## 📰 Articles / Blogs (free, no paywall)
- [Topic extraction with NMF & LDA (scikit-learn)](https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html) — **scikit-learn** — both methods, runnable.
- [gensim LdaModel documentation](https://radimrehurek.com/gensim/models/ldamodel.html) — **gensim** — the canonical free LDA implementation + usage.
- [LDA Model Tutorial (gensim)](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html) — **gensim** — an accessible end-to-end LDA walkthrough with code (free).

## 📚 Books (free, with chapters)
- [Introduction to Information Retrieval — **Ch. 18 "Matrix decompositions and Latent Semantic Indexing"**](https://nlp.stanford.edu/IR-book/html/htmledition/matrix-decompositions-and-latent-semantic-indexing-1.html) — **Manning, Raghavan & Schütze** — LSI/SVD, the matrix-factorization roots of NMF topic models.
- [Dive into Deep Learning — **Ch. 15 (NLP pretraining; representation context)**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — situates topic vectors within text representation.

## 🔗 In this platform
- Prior step: [03 Bag-of-Words & TF-IDF](03-Bag-of-Words-and-TF-IDF.md) — the term–document matrix topic models factorize.
- Concept depth (the *why*): [AI-ML-intuition 1.05 Spectral Methods (PCA · SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md) · [5.06 GMMs & EM](../../../AI-ML-intuition/Module_5_Generation/5.06_GMMs_and_EM.md)
- Related: [16 Information Retrieval & Semantic Search](16-Information-Retrieval-and-Semantic-Search.md) — latent semantic indexing connects the two.
