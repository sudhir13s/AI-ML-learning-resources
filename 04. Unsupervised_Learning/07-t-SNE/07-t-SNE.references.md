---
id: "04-unsupervised-learning/t-sne/references"
topic: "t-SNE — References"
parent: "04-unsupervised-learning/t-sne"
type: references
updated: 2026-06-22
---

# t-SNE — references and further reading

> Companion link library for **[t-SNE](07-t-SNE.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [StatQuest: t-SNE, Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM) (**Josh Starmer**). *Why similar points attract and dissimilar ones repel in the low-D map — the gentlest first watch.*
2. **See the caveats** — read [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) (**Wattenberg, Viégas & Johnson, Distill**). *Interactive: how perplexity, iterations, and randomness change the picture — the single most important read here.*
3. **Get the math** — watch [t-SNE: Clearly Explained](https://www.youtube.com/watch?v=43ySR7_Yb4E) (**Deepia**), then read the [original paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf). *Conditional probabilities, the Student-t tail, the KL objective, the gradient.*
4. **Read the source** — [van der Maaten & Hinton (2008)](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) then [Barnes-Hut t-SNE (2014)](https://arxiv.org/abs/1301.3342). *The method, then the O(n log n) approximation behind every modern implementation.*
5. **Make it concrete** — code it with [scikit-learn TSNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne) and sweep perplexity. *Watch the embedding change to internalize "don't over-read it."*

**Videos**:
- [StatQuest: t-SNE, Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM) — **StatQuest (Josh Starmer)** — the gentle from-scratch intuition for attraction/repulsion; the best first watch.
- [t-SNE: Clearly Explained](https://www.youtube.com/watch?v=43ySR7_Yb4E) — **Deepia** — the full probabilistic derivation and the KL-divergence objective, visually.
- [t-SNE explained](https://www.youtube.com/watch?v=RJVL80Gg3lA) — **ritvikmath** — perplexity and the Gaussian/Student-t pairing, concisely and clearly.
- [PCA, t-SNE, and UMAP — Modern Approaches to Dimension Reduction](https://www.youtube.com/watch?v=YPJQydzTLwQ) — **Leland McInnes (PyData)** — situates t-SNE between PCA and UMAP, with its trade-offs, from the UMAP author.

**Interactive & visual**:
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) — **Wattenberg, Viégas & Johnson (Distill)** — interactive; *the* definitive guide to not over-interpreting t-SNE plots (cluster sizes, distances, perplexity, seeds).
- [Comparison of Manifold Learning methods (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html) — **scikit-learn** — PCA vs t-SNE vs Isomap vs LLE on the same data; the clearest "which method" picture.
- [t-SNE on the perplexity parameter (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html) — **scikit-learn** — runnable demo of exactly how perplexity reshapes the embedding.

**Courses (free)**:
- [scikit-learn — t-SNE user guide](https://scikit-learn.org/stable/modules/manifold.html#t-sne) — **scikit-learn** — perplexity, early exaggeration, the optimization, and the explicit warnings, with code.
- [scikit-learn — Manifold learning user guide](https://scikit-learn.org/stable/modules/manifold.html) — **scikit-learn** — t-SNE in context next to Isomap, LLE, and spectral embedding.

**Articles / blogs (free, no paywall)**:
- [t-SNE author page & FAQ](https://lvdmaaten.github.io/tsne/) — **Laurens van der Maaten** — the author's own implementations, parameter advice, and common pitfalls, from the source.
- [openTSNE documentation](https://opentsne.readthedocs.io/en/latest/) — **Pavlin Poličar** — a fast, well-documented modern t-SNE (FFT-accelerated, with a `transform`) and an excellent practical guide.
- [In Depth: Manifold Learning](https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html) — **Jake VanderPlas (Python Data Science Handbook)** — t-SNE and friends with worked code and honest caveats.

**Key papers**:
- [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) — **van der Maaten & Hinton (2008)** — the original; the crowding problem, the Student-t fix, the symmetric gradient.
- [Stochastic Neighbor Embedding](https://www.cs.toronto.edu/~hinton/absps/sne.pdf) — **Hinton & Roweis (2002/03)** — the SNE predecessor t-SNE improves on; where the probability formulation came from.
- [Accelerating t-SNE using Tree-Based Algorithms (Barnes-Hut)](https://arxiv.org/abs/1301.3342) — **van der Maaten (2014)** — the O(n log n) approximation behind every modern implementation.
- [Dimensionality Reduction: A Comparative Review](https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf) — **van der Maaten, Postma & van den Herik (2009)** — t-SNE benchmarked against PCA and the other non-linear methods; the landscape in one paper.
- [UMAP: Uniform Manifold Approximation and Projection](https://arxiv.org/abs/1802.03426) — **McInnes, Healy & Melville (2018)** — the modern alternative; read for the t-SNE-vs-UMAP comparison and the global-structure argument.

**Books (free chapters)**:
- [The Elements of Statistical Learning — §14.9 "Nonlinear Dimension Reduction and Local MDS"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the MDS/manifold family t-SNE belongs to.
- [Mathematics for Machine Learning — Ch. 10 "Dimensionality Reduction with PCA"](https://mml-book.github.io/) — **Deisenroth, Faisal & Ong** — free; the linear baseline (PCA) derived from first principles, the contrast t-SNE is defined against.
- [Dive into Deep Learning](https://d2l.ai/) — **Zhang, Lipton, Li & Smola** — free; how the learned embeddings that are the usual *input* to t-SNE are produced and inspected.

**In this platform**:
- Concept page (full explanation): [t-SNE](07-t-SNE.md)
- The linear baseline t-SNE is defined against: [06 Dimensionality Reduction — Overview (PCA / SVD)](../06-Dimensionality-Reduction-Overview/06-Dimensionality-Reduction-Overview.md)
- The modern alternative (faster, has a transform, more global): [08 UMAP](../08-UMAP/08-UMAP.md)
- The KL-divergence objective (the *why* of the cost): [AI-ML-intuition 5.01 Entropy & KL Divergence](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Concept depth (the *why*): [AI-ML-intuition 1.11–1.12 Dimensionality Reduction (t-SNE / UMAP)](../../../AI-ML-intuition/Module_1_Representation/1.11-1.12_Dimensionality_Reduction_for_Representation_t-SNE_UMAP.md)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../01.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)
