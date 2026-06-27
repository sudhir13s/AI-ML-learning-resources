---
id: "04-unsupervised-learning/spectral-clustering/references"
topic: "Spectral Clustering — References"
parent: "04-unsupervised-learning/spectral-clustering"
type: references
updated: 2026-06-22
---

# Spectral Clustering — references and further reading

> Companion link library for **[Spectral Clustering](05-Spectral-Clustering.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Spectral Clustering — Explained](https://www.youtube.com/watch?v=KanQwD8h89w) (**DataMListic**). *See why mapping to an eigenvector space turns tangled clusters into easy ones.*
2. **See the three steps** — watch [Lecture 34: Spectral Clustering — Three Steps](https://www.youtube.com/watch?v=uxsDKhZHDcc) (**Stanford MMDS, Leskovec**). *Affinity matrix → Laplacian eigenvectors → k-means, made concrete.*
3. **Get the math** — read [A Tutorial on Spectral Clustering](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf) (**von Luxburg, 2007**). *The Laplacian, unnormalized vs normalized cuts, and why the eigenvectors solve a relaxed graph cut — the definitive treatment.*
4. **Read the source** — read [On Spectral Clustering: Analysis and an Algorithm](https://proceedings.neurips.cc/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf) (**Ng, Jordan & Weiss, 2002**). *The normalized-Laplacian algorithm everyone cites and scikit-learn implements.*
5. **Make it concrete** — code it with the [scikit-learn SpectralClustering guide](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering) and the [segmentation demo](https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html). *Tuning the affinity and `n_clusters` cements it.*

**Videos**:
- [Spectral Clustering — Explained](https://www.youtube.com/watch?v=KanQwD8h89w) — **DataMListic** — the cleanest intuition for why the eigenvector embedding untangles non-convex clusters; the best first watch.
- [Lecture 34 — Spectral Clustering: Three Steps](https://www.youtube.com/watch?v=uxsDKhZHDcc) — **Stanford (MMDS, Leskovec)** — affinity → Laplacian → k-means, the canonical recipe from the Mining of Massive Datasets course.
- [3 Easy Steps to Understand and Implement Spectral Clustering](https://www.youtube.com/watch?v=YHz0PHcuJnk) — **Normalized Nerd** — adjacency matrix, eigenvalues, and a from-scratch implementation in code.
- [The Graph Laplacian](https://www.youtube.com/watch?v=oNVB_KKDxRc) — **Visual Kernel** — a beautifully visual build-up of L = D − W and what its eigenvectors mean; grounds the algebra in pictures.

**Courses (free)**:
- [scikit-learn — Spectral clustering user guide](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering) — **scikit-learn** — affinity choices, the eigen-step, and when to prefer it over k-means, with runnable code; the practical reference.
- [Mining of Massive Datasets — clustering & graphs](http://www.mmds.org/) — **Leskovec, Rajaraman & Ullman (Stanford)** — free course + book; the graph-cut framing behind spectral methods at scale.
- [CS168: The Modern Algorithmic Toolbox — Spectral Graph Theory](https://web.stanford.edu/class/cs168/l/l11.pdf) — **Stanford (Valiant)** — lecture notes deriving the Laplacian, the Fiedler vector, and spectral partitioning from first principles.

**Articles / blogs (free, no paywall)**:
- [A Tutorial on Spectral Clustering (PDF)](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf) — **Ulrike von Luxburg** — the most-readable derivation of the Laplacian, the three cut objectives, and the algorithm; the single best reference on this topic.
- [Spectral clustering (scikit-learn user guide)](https://scikit-learn.org/stable/modules/clustering.html#spectral-clustering) — **scikit-learn** — practical affinity and parameter guidance with code, plus the comparison table against k-means/DBSCAN.
- [Spectral clustering for image segmentation](https://scikit-learn.org/stable/auto_examples/cluster/plot_segmentation_toy.html) — **scikit-learn** — copy-paste recipe showing the Shi–Malik Normalized-Cuts idea on a toy image.
- [Lecture Notes on Expansion, Sparsest Cut, and Spectral Graph Theory](https://lucatrevisan.github.io/books/expanders-2016.pdf) — **Luca Trevisan (Stanford/Bocconi)** — graduate notes tying λ₂ to the best cut (the Cheeger inequality); the rigorous "why a small eigengap means a good cut."

**Key papers**:
- [On Spectral Clustering: Analysis and an Algorithm](https://proceedings.neurips.cc/paper/2001/file/801272ee79cfde7fa5960571fee36b9b-Paper.pdf) — **Ng, Jordan & Weiss (2002)** — the normalized-Laplacian + row-normalization algorithm that became the standard (the one on the concept page).
- [Normalized Cuts and Image Segmentation](https://people.eecs.berkeley.edu/~malik/papers/SM-ncut.pdf) — **Shi & Malik (2000)** — the graph-cut view (NCut) that motivates spectral clustering; author-hosted PDF.
- [A Tutorial on Spectral Clustering](https://arxiv.org/abs/0711.0189) — **von Luxburg (2007)** — the definitive free survey tying Laplacians, cuts, random walks, and algorithms together (arXiv).
- [Algebraic Connectivity of Graphs](https://dml.cz/handle/10338.dmlcz/101168) — **Fiedler (1973)** — the original paper introducing λ₂ (algebraic connectivity) and the Fiedler vector; the root of spectral partitioning.
- [Kernel k-means, Spectral Clustering and Normalized Cuts](https://people.bu.edu/bkulis/pubs/spectral_techreport.pdf) — **Dhillon, Guan & Kulis (2004)** — proves spectral clustering and weighted kernel k-means optimize the same objective.
- [Learning Segmentation by Random Walks](https://www.cs.huji.ac.il/~werman/Papers/meila_shi_2000.pdf) — **Meilă & Shi (2000)** — the random-walk interpretation (NCut = probability of crossing between clusters); the L_rw view on the concept page.
- [Power Iteration Clustering](https://icml.cc/Conferences/2010/papers/387.pdf) — **Lin & Cohen (2010)** — a fast approximation that skips the full eigendecomposition; one of the scalability levers.
- [Self-Tuning Spectral Clustering](https://proceedings.neurips.cc/paper/2004/file/40173ea48d9567f1f393b20c855bb40b-Paper.pdf) — **Zelnik-Manor & Perona (2004)** — per-point local bandwidth σᵢ and automatic `k` selection; the practical fix for clusters of different densities/scales.

**Books (free chapters)**:
- [The Elements of Statistical Learning — §14.5.3 "Spectral Clustering"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; spectral clustering placed against k-means and PCA.
- [Mining of Massive Datasets — Ch. 10 "Mining Social-Network Graphs"](http://infolab.stanford.edu/~ullman/mmds/ch10.pdf) — **Leskovec, Rajaraman & Ullman (Stanford)** — free PDF; graph partitioning and the Laplacian at scale.
- [Networks, Crowds, and Markets — graph structure](https://www.cs.cornell.edu/home/kleinber/networks-book/) — **Easley & Kleinberg (Cornell)** — free book; the network/community-structure context spectral clustering operates in.

**In this platform**:
- Concept page (full explanation): [Spectral Clustering](05-Spectral-Clustering.md)
- Compare with the siblings: [K-Means Clustering](../01-K-Means-Clustering/01-K-Means-Clustering.md) (runs *inside* the spectral embedding) · [Hierarchical Clustering](../02-Hierarchical-Clustering/02-Hierarchical-Clustering.md) · [DBSCAN](../03-DBSCAN/03-DBSCAN.md) (the other arbitrary-shape method) · [Gaussian Mixture Models & EM](../04-Gaussian-Mixture-Models-and-EM/04-Gaussian-Mixture-Models-and-EM.md)
- Manifold cousins (same Laplacian-eigenvector machinery): [t-SNE](../07-t-SNE/07-t-SNE.md) · [UMAP](../08-UMAP/08-UMAP.md)
- Foundations (the eigendecomposition it relies on): [AI-ML-intuition 1.05 Spectral Methods (PCA / SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../01.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)
