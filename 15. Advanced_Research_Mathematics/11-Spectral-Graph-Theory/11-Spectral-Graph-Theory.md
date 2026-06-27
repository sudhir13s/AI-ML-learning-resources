---
id: "15-advanced-math/spectral-graph-theory"
topic: "Spectral Graph Theory"
parent: "15-advanced-research-mathematics"
level: advanced
prereqs: ["linear-algebra", "eigenvalues", "graph-theory"]
interview_frequency: low
updated: 2026-06-20
---

# Spectral Graph Theory
> Read a graph through the eigenvalues and eigenvectors of its matrices — the adjacency matrix and,
> above all, the **graph Laplacian** `L = D − A`. The spectrum encodes connectivity (the Fiedler
> value λ₂), bottlenecks (Cheeger's inequality), and a natural Fourier basis on graphs. It's the
> mathematics under spectral clustering, graph signal processing, and the spectral view of GNNs.

**Why it matters:** spectral clustering ("partition by the Fiedler vector") and graph convolution
("filter in the Laplacian eigenbasis") both come straight from here. The Laplacian is the graph
analogue of the continuous Laplacian operator, which is why "graph Fourier transform" and
"diffusion on graphs" make sense — tying this card to Fourier analysis (card 12) and geometric deep
learning (card 10). Cheeger's inequality is a classic theory talking point.

**⭐ Start here — suggested path:**

1. **Meet the Laplacian** — watch [Spectral Partitioning, Part 1: The Graph Laplacian](https://www.youtube.com/watch?v=rVnOANM0oJE). *L = D − A and what its eigenvectors mean — the entry point.*
2. **See spectral clustering** — read [von Luxburg's Tutorial on Spectral Clustering](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf). *The single best explanation of why low Laplacian eigenvectors cluster.*
3. **Get the theory** — work [Spielman's Spectral Graph Theory notes, Lec 1–3](https://www.cs.yale.edu/homes/spielman/561/). *Fundamental graphs, Courant–Fischer, and the Fiedler value, rigorously.*
4. **Reach Cheeger** — study the Cheeger-inequality lectures in those notes. *The eigenvalue ↔ conductance bound that justifies spectral partitioning.*
5. **Connect to GNNs** — watch [Geometric Deep Learning Lecture 1](https://www.youtube.com/watch?v=PtA0lg_e5nA) and a [GNN lecture](https://www.youtube.com/watch?v=pL5Nc8Axv5A). *The Laplacian eigenbasis is exactly the spectral view of graph convolution.*

## 🎓 Courses (free)
- [Spectral Graph Theory (AMath 561 / CS 662)](https://www.cs.yale.edu/homes/spielman/561/) — **Daniel Spielman (Yale)** — the definitive course: Laplacians, Cheeger, expanders, full lecture notes, free.
- [Geometric Deep Learning — proto-book & lectures](https://geometricdeeplearning.com/) — **Bronstein, Bruna, Cohen & Veličković** — spectral & spatial views of graph learning, fully free.
- [A Tutorial on Spectral Clustering](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf) — **Ulrike von Luxburg** — the standard reference write-up, free PDF.

## 🎥 Videos
- [Spectral Partitioning, Part 1: The Graph Laplacian](https://www.youtube.com/watch?v=rVnOANM0oJE) — **Udacity (HPC)** — the Laplacian and its eigenvectors as a partitioning tool.
- [Geometric Deep Learning — Lecture 1 (Introduction)](https://www.youtube.com/watch?v=PtA0lg_e5nA) — **Michael Bronstein** — graphs, symmetry, and the spectral perspective on graph learning.
- [Graph Neural Networks: Geometric, Structural and Algorithmic Perspectives, Part 1](https://www.youtube.com/watch?v=pL5Nc8Axv5A) — **Petar Veličković (DeepMind)** — message passing and the spectral↔spatial connection.
- [A Brief History of Geometric Deep Learning](https://www.youtube.com/watch?v=yuw_LwqHsgM) — **Michael Bronstein** — where spectral graph theory sits in the GDL story.

## 📄 Key Papers
- [Spectral Networks and Deep Locally Connected Networks on Graphs](https://arxiv.org/abs/1312.6203) — **Bruna et al. (2014)** — the first spectral GNN: convolution in the Laplacian eigenbasis, free on arXiv.
- [Semi-Supervised Classification with Graph Convolutional Networks (GCN)](https://arxiv.org/abs/1609.02907) — **Kipf & Welling (2017)** — the spectral filter simplified to the workhorse GCN.
- [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478) — **Bronstein et al. (2021)** — the unifying framework that situates spectral graph methods.

## 📰 Articles / Blogs (free, no paywall)
- [Spectral Graph Theory — chapter from Spielman](http://www.cs.yale.edu/homes/spielman/PAPERS/SGTChapter.pdf) — **Daniel Spielman** — a self-contained survey chapter (Laplacians, Cheeger), free PDF.
- [A Tutorial on Spectral Clustering](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf) — **Ulrike von Luxburg** — the clearest derivation of the spectral-clustering algorithm.

## 📚 Books (free, with chapters)
- [Spectral Graph Theory — **Lectures 1–6 (Laplacian, Courant–Fischer, Cheeger)**](https://www.cs.yale.edu/homes/spielman/561/) — **Daniel Spielman (Yale)** — the course as a free book of lecture notes.
- [Mathematics for Machine Learning — **Ch. 4 (Matrix Decompositions: eigen/spectral)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — the eigen-decomposition background, free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.04 Graph Representations](../../../AI-ML-intuition/Module_1_Representation/1.04_Graph_Representations.md) · [1.05 Spectral Methods (PCA/SVD)](../../../AI-ML-intuition/Module_1_Representation/1.05_Spectral_Methods_PCA_SVD.md)
- Foundations (the basics this builds on): [Eigenvalues & Eigenvectors](../../01.%20Foundations/04-Eigenvalues-and-Eigenvectors/04-Eigenvalues-and-Eigenvectors.md) · [Matrix Decompositions](../../01.%20Foundations/05-Matrix-Decompositions/05-Matrix-Decompositions.md)
- Prerequisite & next: [10 Differential Geometry & Manifolds](../10-Differential-Geometry-and-Manifolds/10-Differential-Geometry-and-Manifolds.md) · [12 Fourier Analysis & Signal Processing](../12-Fourier-Analysis-and-Signal-Processing/12-Fourier-Analysis-and-Signal-Processing.md) · [13 Random Matrix Theory](../13-Random-Matrix-Theory/13-Random-Matrix-Theory.md)
- Related domain (clustering): [04. Unsupervised Learning](../../04.%20Unsupervised_Learning/README.md)
</content>
