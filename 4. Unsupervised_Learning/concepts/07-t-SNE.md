---
id: "04-unsupervised-learning/t-sne"
topic: "t-SNE (t-Distributed Stochastic Neighbor Embedding)"
parent: "04-unsupervised-learning"
level: intermediate
prereqs: ["dimensionality-reduction", "probability", "kl-divergence", "gradient-descent"]
interview_frequency: high
updated: 2026-06-19
---

# t-SNE — t-Distributed Stochastic Neighbor Embedding
> A non-linear visualization method: model pairwise *similarities* as probabilities in high-D (Gaussian)
> and low-D (heavy-tailed Student-t), then move the 2-D points to minimize the KL divergence between the
> two. Produces strikingly clean clusters — the standard way to "see" embeddings.

**Why it matters:** the most-misused tool in the kit, so interviews test whether you understand its
caveats. You should explain why the Student-t in low-D fixes the **crowding problem**, what
**perplexity** controls (effective neighborhood size), and — critically — why t-SNE is for
*visualization only*: cluster **sizes and inter-cluster distances are not meaningful**, runs are
stochastic, and it doesn't define a reusable transform for new points (unlike PCA/UMAP). Knowing *when
not to trust the picture* is the whole point.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: t-SNE, Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM). *Why similar points attract and dissimilar ones repel in the low-D map.*
2. **See the caveats** — read [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) (Distill). *Interactive: how perplexity, iterations, and randomness change the picture — the single most important read here.*
3. **Get the math** — [t-SNE: Clearly Explained](https://www.youtube.com/watch?v=43ySR7_Yb4E) (Deepia) + the [original paper](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf). *Conditional probabilities, the Student-t tail, the KL objective, and gradient descent on point positions.*
4. **Read the source** — [van der Maaten & Hinton (2008)](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) and [Barnes-Hut t-SNE (2014)](https://arxiv.org/abs/1301.3342). *The method, then the `O(n log n)` approximation that made it practical.*
5. **Make it concrete** — code it with [scikit-learn TSNE](https://scikit-learn.org/stable/modules/manifold.html#t-sne) and sweep perplexity. *Watch the embedding change to internalize "don't over-read it."*

## 🎓 Courses (free)
- [scikit-learn — t-SNE user guide](https://scikit-learn.org/stable/modules/manifold.html#t-sne) — **scikit-learn** — perplexity, early exaggeration, the optimization, and the explicit warnings, with code.
- [scikit-learn — Manifold learning user guide](https://scikit-learn.org/stable/modules/manifold.html) — **scikit-learn** — t-SNE in context next to Isomap, LLE, and spectral embedding.

## 🎥 Videos
- [StatQuest: t-SNE, Clearly Explained](https://www.youtube.com/watch?v=NEaUSP4YerM) — **StatQuest (Josh Starmer)** — the gentle from-scratch intuition; the best first watch.
- [t-SNE: Clearly Explained](https://www.youtube.com/watch?v=43ySR7_Yb4E) — **Deepia** — the full probabilistic derivation and the KL-divergence objective.
- [PCA, t-SNE, and UMAP — Modern Approaches to Dimension Reduction](https://www.youtube.com/watch?v=YPJQydzTLwQ) — **Leland McInnes (PyData)** — situates t-SNE between PCA and UMAP, with its trade-offs.
- [t-SNE explained](https://www.youtube.com/watch?v=RJVL80Gg3lA) — **ritvikmath** — perplexity and the Gaussian/Student-t pairing, concisely.

## 📄 Key Papers
- [Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) — **van der Maaten & Hinton (2008)** — the original; the crowding problem and the Student-t fix.
- [Accelerating t-SNE using Tree-Based Algorithms (Barnes-Hut)](https://arxiv.org/abs/1301.3342) — **van der Maaten (2014)** — the `O(n log n)` approximation behind every modern implementation.
- [Stochastic Neighbor Embedding](https://www.cs.toronto.edu/~hinton/absps/sne.pdf) — **Hinton & Roweis (2003)** — the SNE predecessor that t-SNE improves on; good for "where did this come from?"

## 📰 Articles / Blogs (free, no paywall)
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) — **Wattenberg, Viégas & Johnson (Distill)** — interactive; the definitive guide to *not* over-interpreting t-SNE plots.
- [t-SNE author page & FAQ](https://lvdmaaten.github.io/tsne/) — **Laurens van der Maaten** — the author's own implementations, parameter advice, and common pitfalls.
- [Visualizing with t-SNE (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/manifold/plot_t_sne_perplexity.html) — **scikit-learn** — see exactly how perplexity reshapes the embedding.

## 📚 Books (free, with chapters)
- [The Elements of Statistical Learning — **§14.9 "Nonlinear Dimension Reduction and Local Multidimensional Scaling"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the MDS/manifold family t-SNE belongs to.
- [Dive into Deep Learning — visualization & embeddings context](https://d2l.ai/) — **Zhang, Lipton, Li & Smola** — free; how embeddings (the usual input to t-SNE) are learned and inspected.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.11–1.12 Dimensionality Reduction (t-SNE / UMAP)](../../../AI-ML-intuition/Module_1_Representation/1.11-1.12_Dimensionality_Reduction_for_Representation_t-SNE_UMAP.md)
- Information-theory prereq (the KL objective): [AI-ML-intuition 5.01 Entropy & KL Divergence](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Compare with: [06 Dimensionality Reduction overview](06-Dimensionality-Reduction-Overview.md) (PCA, the linear baseline) · [08 UMAP](08-UMAP.md) (faster, preserves more global structure)
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../1.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)
