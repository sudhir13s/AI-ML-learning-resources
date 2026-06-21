---
id: "04-unsupervised-learning/umap/references"
topic: "UMAP — References"
parent: "04-unsupervised-learning/umap"
type: references
updated: 2026-06-22
---

# UMAP — references and further reading

> Companion link library for **[UMAP](08-UMAP.md)** (the concept page). Curated links — external sources *and* internal cross-links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is a primary author or a recognized deep explainer, chosen for depth on *this* topic, and every link verified.

**Start here — suggested path:**
1. **Build intuition** — watch [StatQuest: UMAP, Clearly Explained](https://www.youtube.com/watch?v=eN0wFzBA4Sc). *The fuzzy neighbor graph and how the low-D layout is pulled to match it — the best first watch.*
2. **See the trade-offs** — read [Understanding UMAP](https://pair-code.github.io/understanding-umap/) (Google PAIR). *Interactive: how `n_neighbors` / `min_dist` reshape the map, and how UMAP differs from t-SNE — the key read for reading plots responsibly.*
3. **Get the math** — watch [PyData: PCA, t-SNE, and UMAP](https://www.youtube.com/watch?v=YPJQydzTLwQ) by the author, then read [How UMAP Works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html). *The fuzzy-simplicial-set construction and the cross-entropy objective, explained plainly.*
4. **Read the source** — [McInnes, Healy & Melville (2018)](https://arxiv.org/abs/1802.03426). *The manifold/topology foundations; skim the theory, focus on the algorithm (§2–4) and parameters.*
5. **Make it concrete** — install [`umap-learn`](https://umap-learn.readthedocs.io/en/latest/) and run the [basic-usage tutorial](https://umap-learn.readthedocs.io/en/latest/basic_usage.html); sweep `n_neighbors` / `min_dist` and compare against your t-SNE plot. *The sweep cements the local↔global and tight↔spread trade-offs.*

**Videos:**
- [StatQuest: UMAP, Clearly Explained](https://www.youtube.com/watch?v=eN0wFzBA4Sc) — **StatQuest (Josh Starmer)** — gentle from-scratch intuition for the neighbor graph and layout; the best first watch.
- [UMAP Dimension Reduction, Main Ideas](https://www.youtube.com/watch?v=nq6iPZVUxZU) — **StatQuest (Josh Starmer)** — the companion "main ideas" video for a fast mental model before the math.
- [PCA, t-SNE, and UMAP — Modern Approaches to Dimension Reduction](https://www.youtube.com/watch?v=YPJQydzTLwQ) — **Leland McInnes (PyData, UMAP author)** — the definitive talk from the person who built it; situates UMAP between PCA and t-SNE.
- [UMAP explained — The best dimensionality reduction?](https://www.youtube.com/watch?v=6BPl81wGGP8) — **DeepFindr** — concise tour of the graph construction and the t-SNE comparison.

**Interactive & visual:**
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/) — **Coenen & Pearce (Google PAIR)** — interactive; the clearest treatment of `n_neighbors` / `min_dist` and the t-SNE contrast, and *why the geometry isn't literal*.
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) — **Wattenberg, Viégas & Johnson (Distill)** — the sibling caveats for t-SNE; the discipline transfers directly to reading UMAP plots.

**Courses (free):**
- [UMAP documentation & tutorials](https://umap-learn.readthedocs.io/en/latest/) — **McInnes, Healy & Melville** — the canonical guide: parameters, transforming new data, clustering, supervised UMAP, AlignedUMAP, with code.
- [How UMAP Works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) — **McInnes, Healy & Astels** — the algorithm walkthrough doubling as a structured mini-course on the fuzzy graph and the cross-entropy layout.
- [UMAP basic usage (tutorial)](https://umap-learn.readthedocs.io/en/latest/basic_usage.html) — **umap-learn docs** — the copy-paste recipe to embed and inspect your own data, plus the parameter pages.

**Articles / blogs (free, no paywall):**
- [`lmcinnes/umap` — source, README & theory notes](https://github.com/lmcinnes/umap) — **Leland McInnes** — the reference implementation; the README and `docs/` are an authoritative, equation-level account of the membership, normalization, and cross-entropy steps, straight from the author.
- [Comparison of Manifold Learning methods (scikit-learn)](https://scikit-learn.org/stable/auto_examples/manifold/plot_compare_methods.html) — **scikit-learn** — PCA vs t-SNE vs Isomap vs LLE on one dataset; the clearest "which method does what" picture, runnable.
- [Understanding UMAP (PAIR write-up)](https://pair-code.github.io/understanding-umap/) — **Coenen & Pearce (Google)** — the parameter and global-structure intuition in prose alongside the interactive demos.
- [The Specious Art of Single-Cell Genomics](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011288) — **Chari & Pachter (2023)** — a pointed, important critique of how literally UMAP/t-SNE plots are read in genomics; the honest counterweight every practitioner should see.

**Key papers:**
- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426) — **McInnes, Healy & Melville (2018)** — the original; the manifold/topology foundations, the algorithm, and the parameters.
- [Dimensionality reduction for visualizing single-cell data using UMAP](https://www.nature.com/articles/nbt.4314) — **Becht, McInnes, Healy et al. (2019, Nature Biotechnology)** — the paper that made UMAP the standard in single-cell genomics; the canonical real-world application.
- [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) — **van der Maaten & Hinton (2008)** — the method UMAP is most compared against; the crowding problem and the KL objective UMAP's cross-entropy contrasts with.
- [On UMAP's True Loss Function](https://arxiv.org/abs/2103.14608) — **Damrich & Hamprecht (2021)** — shows UMAP's *effective* loss differs from the stated cross-entropy; essential for an honest, senior-level account of why UMAP behaves as it does.
- [Attraction-Repulsion Spectrum in Neighbor Embeddings](https://arxiv.org/abs/2007.08902) — **Böhm, Berens & Kobak (2022)** — places t-SNE and UMAP on one continuum controlled by the attraction/repulsion balance; the clearest unifying view.
- [Dimensionality Reduction: A Comparative Review](https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf) — **van der Maaten, Postma & van den Herik (2009)** — free survey placing manifold methods against PCA and t-SNE; the landscape in one paper.

**Books (free, with chapters):**
- [The Elements of Statistical Learning — §14.9 "Nonlinear Dimension Reduction and Local MDS"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the manifold-learning family UMAP belongs to (Isomap, LLE, Laplacian eigenmaps).
- [Mathematics for Machine Learning — Ch. 10 "Dimensionality Reduction with PCA"](https://mml-book.github.io/) — **Deisenroth, Faisal & Ong** — free; the linear baseline UMAP departs from, derived from first principles.
- [Dive into Deep Learning — embeddings & visualization context](https://d2l.ai/) — **Zhang, Lipton, Li & Smola** — free; how the high-D embeddings you typically feed UMAP are produced and inspected.

**In this platform:**
- Concept page (full explanation): [UMAP](08-UMAP.md)
- The closest comparison (slower, weaker global structure, no reusable transform): [t-SNE](07-t-SNE.md)
- The linear baseline (fast, faithful-but-limited): [Dimensionality Reduction — Overview (PCA)](06-Dimensionality-Reduction-Overview.md)
- Concept depth (the *why*): [AI-ML-intuition 1.11–1.12 Dimensionality Reduction (t-SNE / UMAP)](../../../AI-ML-intuition/Module_1_Representation/1.11-1.12_Dimensionality_Reduction_for_Representation_t-SNE_UMAP.md)
- Often the next step after UMAP (its author also wrote HDBSCAN): [DBSCAN](03-DBSCAN.md) · [Spectral Clustering](05-Spectral-Clustering.md)
- The failure mode UMAP helps fix in retrieval: [k-Nearest Neighbors (curse of dimensionality)](../../03.%20Supervised_Learning/concepts/04-k-Nearest-Neighbors.md)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../01.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)
