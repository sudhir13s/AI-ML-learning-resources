---
id: "03-supervised-learning/k-nearest-neighbors/references"
topic: "k-Nearest Neighbors — References"
parent: "03-supervised-learning/k-nearest-neighbors"
type: references
updated: 2026-06-22
---

# k-Nearest Neighbors — references and further reading

> Companion link library for **[k-Nearest Neighbors](04-k-Nearest-Neighbors.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity. All links verified (HTTP 200).

**Start here — suggested path**:
1. **Build the intuition** — watch [StatQuest: K-nearest neighbors, Clearly Explained](https://www.youtube.com/watch?v=HVXime0nQeI) (**Josh Starmer**). *See the neighbor vote decide the label — the whole algorithm in one short video.*
2. **See why k is a knob** — watch [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) (**Josh Starmer**). *Why small k is high-variance and large k is high-bias — the core of choosing k.*
3. **Get the math** — read [ISLR Ch. 2.2.3 & 3.5 (KNN)](https://www.statlearning.com/) (**James, Witten, Hastie & Tibshirani**). *k-NN as the running example for the Bayes boundary and bias–variance.*
4. **Understand the failure mode** — read [ESL Ch. 13.3 & 2.5 (curse of dimensionality)](https://hastie.su.domains/ElemStatLearn/) (**Hastie, Tibshirani & Friedman**). *Why distance-based methods degrade as dimensions grow.*
5. **Make it concrete** — work through the [scikit-learn Nearest Neighbors guide](https://scikit-learn.org/stable/modules/neighbors.html) (**scikit-learn**). *Distance metrics, KD-tree/ball-tree, weighting, and tuning k in code.*

**Videos**:
- [StatQuest: K-nearest neighbors, Clearly Explained](https://www.youtube.com/watch?v=HVXime0nQeI) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for the neighbor vote.
- [StatQuest: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA) — **StatQuest (Josh Starmer)** — why small k is high-variance and large k is high-bias; the lens for choosing k.
- [StatQuest: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — **StatQuest (Josh Starmer)** — how you actually pick k without fooling yourself on training error.
- [Introduction to Learning, Nearest Neighbors (6.034)](https://www.youtube.com/watch?v=09mb78oiPkA) — **Patrick Winston (MIT)** — a full lecture on nearest-neighbor learning, scaling, and the "sleeping at the wheel" intuition for the metric.
- [Nearest-neighbor regression example](https://www.youtube.com/watch?v=3lp5CmSwrHI) — **Victor Lavrenko (Edinburgh)** — the regression side of k-NN, worked visually.

**Interactive & visual**:
- [Nearest Neighbors — classification plots](https://scikit-learn.org/stable/auto_examples/neighbors/plot_classification.html) — **scikit-learn** — runnable example whose figures show the boundary going from jagged (small k) to smooth (large k).
- [HNSW, explained visually](https://www.pinecone.io/learn/series/faiss/hnsw/) — **Pinecone** — an animated walkthrough of the navigable-small-world graph that powers approximate k-NN in vector databases.
- [Nearest Neighbors and the Curse of Dimensionality](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html) — **Kilian Weinberger (Cornell CS4780)** — the cleanest lecture note deriving distance concentration, with the geometry.

**Courses (free)**:
- [Machine Learning Specialization — Course 1](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — k-NN among the foundational classifiers; free to audit.
- [CS229: Machine Learning — Lecture notes](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — non-parametric methods and the bias–variance view, rigorously.
- [Kaggle Learn — Intro to Machine Learning](https://www.kaggle.com/learn/intro-to-machine-learning) — **Kaggle** — hands-on first classifiers and over/underfitting, in code, free.

**Articles / blogs (free, no paywall)**:
- [Nearest Neighbors (scikit-learn user guide)](https://scikit-learn.org/stable/modules/neighbors.html) — **scikit-learn** — the practical reference: distance metrics, KD-tree/ball-tree, distance weighting, and `algorithm='auto'`.
- [Nearest Neighbors & the Curse of Dimensionality (lecture note)](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote02_kNN.html) — **Kilian Weinberger (Cornell)** — derives why distances concentrate and what it means for k-NN.
- [Vector indexes for similarity search](https://www.pinecone.io/learn/series/faiss/vector-indexes/) — **Pinecone** — how brute-force k-NN becomes KD-tree → IVF → HNSW at scale; the bridge to vector search and RAG.
- [MLU-Explain (visual ML explainers)](https://mlu-explain.github.io/) — **Amazon** — interactive explainers for the bias–variance and evaluation concepts a k-NN model lives or dies by.

**Key papers**:
- [Nearest Neighbor Pattern Classification](https://isl.stanford.edu/~cover/papers/transIT/0021cove.pdf) — **Cover & Hart (1967)** — the foundational paper proving 1-NN error is at most twice the Bayes error; author-hosted PDF.
- [An Algorithm for Finding Best Matches in Logarithmic Expected Time (KD-trees)](http://i.stanford.edu/pub/cstr/reports/cs/tr/75/482/CS-TR-75-482.pdf) — **Friedman, Bentley & Finkel (1977)** — how to make nearest-neighbor search fast; Stanford tech-report PDF.
- [Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality (LSH)](https://www.cs.princeton.edu/courses/archive/spr04/cos598B/bib/IndykM-curse.pdf) — **Indyk & Motwani (1998)** — locality-sensitive hashing for sub-linear high-dimensional search.
- [Efficient and robust approximate nearest neighbor search using HNSW graphs](https://arxiv.org/abs/1603.09320) — **Malkov & Yashunin (2018)** — the hierarchical navigable-small-world index behind modern vector databases.
- [When Is "Nearest Neighbor" Meaningful?](https://members.loria.fr/MOBerger/Enseignement/Master2/Exposes/beyer.pdf) — **Beyer, Goldstein, Ramakrishnan & Shaft (1999)** — the distance-concentration result formalized: nearest/farthest → 1 as d grows.

**Books (free, with chapters)**:
- [An Introduction to Statistical Learning (ISLR) — Ch. 2.2.3 & 3.5 (KNN)](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — k-NN as the running example for bias–variance and the Bayes boundary; free PDF.
- [The Elements of Statistical Learning — Ch. 13.3 "k-Nearest-Neighbor Classifiers" & 2.5](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment, including effective complexity ≈ n/k and the curse of dimensionality.
- [Introduction to Information Retrieval — Ch. 14 (vector space & k-NN classification)](https://nlp.stanford.edu/IR-book/) — **Manning, Raghavan & Schütze** — k-NN over TF-IDF/embeddings and why cosine is the metric for text; free online.
- [Dive into Deep Learning — k-NN background & softmax classification](https://d2l.ai/) — **Zhang, Lipton, Li & Smola** — instance-based learning framed alongside modern methods, with runnable code.

**In this platform**:
- Concept page (full explanation): [k-Nearest Neighbors](04-k-Nearest-Neighbors.md)
- Math prerequisites (the *why*): [01. Foundations](../../01.%20Foundations/concepts/README.md) — distance metrics, norms, the curse of dimensionality.
- The knob k turns: [Bias–Variance Tradeoff](12-Bias-Variance-Tradeoff.md) — small k = high variance, large k = high bias, made precise.
- How you actually pick k: [Cross-Validation](13-Cross-Validation.md) — never tune k on training error.
- Why scaling is mandatory: [2. Data Preprocessing](../../02.%20Data_Preprocessing/concepts/README.md) — k-NN is acutely sensitive to feature scaling and encoding.
- Concept depth (the *why*): [AI-ML-intuition 1.07–1.08 Euclidean vs Cosine Distance](../../../AI-ML-intuition/Module_1_Representation/1.07-1.08_Similarities_Distances_Euclidean_vs_Cosine.md) · [1.09 Manhattan (L1) Distance](../../../AI-ML-intuition/Module_1_Representation/1.09_Manhattan_L1_Distance.md) · [1.10 Mahalanobis Distance](../../../AI-ML-intuition/Module_1_Representation/1.10_Mahalanobis_Distance.md)
- A contrasting classifier: [Naive Bayes](05-Naive-Bayes.md) — a generative, parametric alternative to k-NN's lazy, geometric approach.
