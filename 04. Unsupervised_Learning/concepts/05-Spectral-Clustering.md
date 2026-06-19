---
id: "04-unsupervised-learning/spectral-clustering"
topic: "Spectral Clustering"
parent: "04-unsupervised-learning"
level: intermediate
prereqs: ["k-means", "linear-algebra", "eigendecomposition", "graphs"]
interview_frequency: medium
updated: 2026-06-19
---

# Spectral Clustering
> Treat the data as a *graph* of similarities, then cluster using the eigenvectors of its Laplacian:
> embed points into a low-dimensional space where well-separated groups become linearly separable,
> and run k-means there. Finds clusters that k-means can't — non-convex shapes, intertwined rings.

**Why it matters:** the canonical "when does k-means fail and what do you reach for?" answer. Interviews
probe why you build a similarity graph (k-NN or Gaussian/RBF affinity), what the **graph Laplacian**
(`L = D − W`) is and why its smallest eigenvectors encode cluster structure, the link to **graph cuts**
(normalized cut ≈ a relaxed eigenvalue problem), and the practical recipe: affinity → Laplacian →
top-`k` eigenvectors → k-means on the rows. It's also your bridge from clustering to manifold learning.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Spectral Clustering — Explained](https://www.youtube.com/watch?v=KanQwD8h89w). *See why mapping to an eigenvector space turns tangled clusters into easy ones.*
2. **See the three steps** — [Lecture 34: Spectral Clustering Three Steps](https://www.youtube.com/watch?v=uxsDKhZHDcc) (Stanford / MMDS). *Affinity matrix → Laplacian eigenvectors → k-means, made concrete.*
3. **Get the math** — read [von Luxburg: A Tutorial on Spectral Clustering](https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf). *The Laplacian, unnormalized vs normalized cuts, and why the eigenvectors solve a relaxed graph cut.*
4. **Read the source** — [Ng, Jordan & Weiss (2002)](https://proceedings.neurips.cc/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf). *The normalized-Laplacian algorithm everyone cites and scikit-learn implements.*
5. **Make it concrete** — code it with [scikit-learn SpectralClustering](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering) and try the [image-segmentation demo](https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html). *Tuning the affinity and `n_clusters` cements it.*

## 🎓 Courses (free)
- [scikit-learn — Spectral clustering user guide](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering) — **scikit-learn** — affinity choices, the eigen-step, and when to prefer it over k-means, with code.
- [Mining of Massive Datasets — clustering/graphs](http://www.mmds.org/) — **Leskovec, Rajaraman & Ullman (Stanford)** — free course + book; the graph-cut framing behind spectral methods.

## 🎥 Videos
- [Spectral Clustering — Explained](https://www.youtube.com/watch?v=KanQwD8h89w) — **DataMListic** — the cleanest intuition for why the eigenvector embedding untangles non-convex clusters.
- [Lecture 34 — Spectral Clustering: Three Steps](https://www.youtube.com/watch?v=uxsDKhZHDcc) — **Stanford (MMDS, Leskovec)** — affinity → Laplacian → k-means, the canonical recipe.
- [3 Easy Steps to Understand and Implement Spectral Clustering](https://www.youtube.com/watch?v=YHz0PHcuJnk) — **Normalized Nerd** — adjacency matrix, eigenvalues, and a from-scratch implementation.
- [Spectral Clustering example in Python](https://www.youtube.com/watch?v=yTRG1NCIf1c) — **Data Science with Robert** — applying scikit-learn's `SpectralClustering` to data k-means gets wrong.

## 📄 Key Papers
- [On Spectral Clustering: Analysis and an Algorithm](https://proceedings.neurips.cc/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf) — **Ng, Jordan & Weiss (2002)** — the normalized-Laplacian algorithm that became the standard.
- [Normalized Cuts and Image Segmentation](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf) — **Shi & Malik (2000)** — the graph-cut view that motivates spectral clustering.
- [A Tutorial on Spectral Clustering](https://arxiv.org/abs/0711.0189) — **von Luxburg (2007)** — the definitive free survey tying Laplacians, cuts, and algorithms together.

## 📰 Articles / Blogs (free, no paywall)
- [A Tutorial on Spectral Clustering (PDF)](https://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf) — **Ulrike von Luxburg** — the most-readable derivation of the Laplacian and the algorithm.
- [Spectral clustering (scikit-learn user guide)](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering) — **scikit-learn** — practical affinity and parameter guidance with runnable code.
- [Spectral clustering for image segmentation (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html) — **scikit-learn** — copy-paste recipe showing the method on a toy image.

## 📚 Books (free, with chapters)
- [Mining of Massive Datasets — **Ch. 10 "Analysis of Social Networks"** (spectral/graph clustering)](http://infolab.stanford.edu/~ullman/mmds/ch10.pdf) — **Leskovec, Rajaraman & Ullman (Stanford)** — free PDF; graph partitioning and the Laplacian at scale.
- [The Elements of Statistical Learning — **§14.5.3 "Spectral Clustering"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; spectral clustering placed against k-means and PCA.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.05 Spectral Methods (PCA / SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md) — the eigendecomposition machinery spectral clustering relies on
- Compare with: [01 K-Means Clustering](01-K-Means-Clustering.md) (run *inside* the spectral embedding) · [03 DBSCAN](03-DBSCAN.md) · [02 Hierarchical Clustering](02-Hierarchical-Clustering.md)
- Related: [07 t-SNE](07-t-SNE.md) · [08 UMAP](08-UMAP.md) — manifold methods that also embed before clustering
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../01.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)
