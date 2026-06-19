---
id: "04-unsupervised-learning/umap"
topic: "UMAP (Uniform Manifold Approximation and Projection)"
parent: "04-unsupervised-learning"
level: intermediate
prereqs: ["dimensionality-reduction", "t-sne", "graphs", "topology-intuition"]
interview_frequency: high
updated: 2026-06-19
---

# UMAP — Uniform Manifold Approximation and Projection
> A non-linear dimensionality-reduction method built on manifold theory: construct a fuzzy
> nearest-neighbor graph that approximates the data's manifold, then optimize a low-D layout whose graph
> matches it (via cross-entropy). The modern default — faster than t-SNE, scales to millions of points,
> preserves more **global** structure, and can transform new data.

**Why it matters:** the natural "t-SNE vs UMAP" follow-up. You should explain UMAP's advantages — speed
and scalability, better global structure, a reusable `transform` for unseen points, and use beyond
visualization (as a preprocessing step for clustering) — while keeping the same humility: like t-SNE,
**distances and densities in the embedding are not fully faithful**, and `n_neighbors` / `min_dist`
trade local vs global structure. Knowing the two key knobs and the honest caveats is the bar.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: UMAP, Clearly Explained](https://www.youtube.com/watch?v=eN0wFzBA4Sc). *The fuzzy neighbor graph and how the low-D layout is pulled to match it.*
2. **See the trade-offs** — read [Understanding UMAP](https://pair-code.github.io/understanding-umap/) (Google PAIR). *Interactive: how `n_neighbors`/`min_dist` reshape the map, and how UMAP differs from t-SNE — the key read.*
3. **Get the math** — watch [PyData: PCA, t-SNE, and UMAP](https://www.youtube.com/watch?v=YPJQydzTLwQ) by the author, then read [How UMAP Works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html). *The fuzzy-simplicial-set construction and the cross-entropy objective, explained plainly.*
4. **Read the source** — [McInnes, Healy & Melville (2018)](https://arxiv.org/abs/1802.03426). *The manifold/topology foundations; skim the theory, focus on the algorithm and parameters.*
5. **Make it concrete** — install [`umap-learn`](https://umap-learn.readthedocs.io/en/latest/) and run the [basic-usage tutorial](https://umap-learn.readthedocs.io/en/latest/basic_usage.html); compare against your t-SNE plot. *Sweeping `n_neighbors`/`min_dist` cements the local↔global trade-off.*

## 🎓 Courses (free)
- [UMAP documentation & tutorials](https://umap-learn.readthedocs.io/en/latest/) — **McInnes, Healy & Melville** — the canonical guide: parameters, transforming new data, clustering, and supervised UMAP, with code.
- [How UMAP Works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) — **McInnes, Healy & Astels** — the algorithm walkthrough doubling as a structured mini-course.

## 🎥 Videos
- [StatQuest: UMAP, Clearly Explained](https://www.youtube.com/watch?v=eN0wFzBA4Sc) — **StatQuest (Josh Starmer)** — gentle from-scratch intuition for the neighbor graph and layout; the best first watch.
- [PCA, t-SNE, and UMAP — Modern Approaches to Dimension Reduction](https://www.youtube.com/watch?v=YPJQydzTLwQ) — **Leland McInnes (PyData, author)** — the definitive explanation from the person who built it.
- [UMAP explained — The best dimensionality reduction?](https://www.youtube.com/watch?v=6BPl81wGGP8) — **DeepFindr** — concise tour of the graph construction and the t-SNE comparison.
- [UMAP Dimension Reduction, Main Ideas](https://www.youtube.com/watch?v=nq6iPZVUxZU) — **StatQuest (Josh Starmer)** — the companion "main ideas" video for a quick mental model.

## 📄 Key Papers
- [UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction](https://arxiv.org/abs/1802.03426) — **McInnes, Healy & Melville (2018)** — the original; manifold foundations and the algorithm.
- [Dimensionality Reduction: A Comparative Review](https://lvdmaaten.github.io/publications/papers/TR_Dimensionality_Reduction_Review_2009.pdf) — **van der Maaten, Postma & van den Herik (2009)** — free survey placing UMAP-style manifold methods against PCA and t-SNE.

## 📰 Articles / Blogs (free, no paywall)
- [Understanding UMAP](https://pair-code.github.io/understanding-umap/) — **Coenen & Pearce (Google PAIR)** — interactive; the clearest treatment of `n_neighbors`/`min_dist` and the t-SNE contrast.
- [How UMAP Works](https://umap-learn.readthedocs.io/en/latest/how_umap_works.html) — **Leland McInnes** — the author's own plain-language explanation of the fuzzy graph and optimization.
- [UMAP basic usage (tutorial)](https://umap-learn.readthedocs.io/en/latest/basic_usage.html) — **umap-learn docs** — copy-paste recipe to embed and inspect your own data.

## 📚 Books (free, with chapters)
- [The Elements of Statistical Learning — **§14.9 "Nonlinear Dimension Reduction and Local MDS"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the manifold-learning family UMAP belongs to.
- [Dive into Deep Learning — embeddings & visualization context](https://d2l.ai/) — **Zhang, Lipton, Li & Smola** — free; how the high-D embeddings you feed UMAP are produced.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.11–1.12 Dimensionality Reduction (t-SNE / UMAP)](../../../AI-ML-intuition/Module_1_Representation/1.11-1.12_Dimensionality_Reduction_for_Representation_t-SNE_UMAP.md)
- Compare with: [07 t-SNE](07-t-SNE.md) (slower, weaker global structure, no reusable transform) · [06 Dimensionality Reduction overview](06-Dimensionality-Reduction-Overview.md) (PCA, the linear baseline)
- Related: [03 DBSCAN](03-DBSCAN.md) / [05 Spectral Clustering](05-Spectral-Clustering.md) — UMAP is often a preprocessing step before clustering (its author also wrote HDBSCAN)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../1.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)
