---
id: "04-unsupervised-learning"
topic: "Unsupervised Learning"
level: intermediate
prereqs: ["foundations", "linear-algebra"]
updated: 2026-06-27
---

# Unsupervised Learning
> Finding structure without labels — clustering (k-means, DBSCAN), dimensionality reduction
> (PCA, t-SNE, UMAP), and density estimation.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page and a curated
`.references.md` resource card (free, open courses · videos · papers · articles · books · cross-links).
> **✅ ready · ⬜ coming soon.** New to the area? Start with the field overview below, then work top to bottom.

### Clustering
1. ✅ [K-Means Clustering](01-K-Means-Clustering/01-K-Means-Clustering.md)
2. ✅ [Hierarchical Clustering (agglomerative & divisive)](02-Hierarchical-Clustering/02-Hierarchical-Clustering.md)
3. ✅ [DBSCAN (density-based clustering)](03-DBSCAN/03-DBSCAN.md)
4. ✅ [Gaussian Mixture Models & EM](04-Gaussian-Mixture-Models-and-EM/04-Gaussian-Mixture-Models-and-EM.md)
5. ✅ [Spectral Clustering](05-Spectral-Clustering/05-Spectral-Clustering.md)

### Dimensionality reduction & manifold learning
6. ✅ [Dimensionality Reduction — overview (PCA · SVD framing, cross-link to math)](06-Dimensionality-Reduction-Overview/06-Dimensionality-Reduction-Overview.md)
7. ✅ [t-SNE](07-t-SNE/07-t-SNE.md)
8. ✅ [UMAP](08-UMAP/08-UMAP.md)

### Density & anomaly
9. ✅ [Anomaly / Outlier Detection (Isolation Forest · LOF · One-Class SVM)](09-Anomaly-Outlier-Detection/09-Anomaly-Outlier-Detection.md)
10. ✅ [Kernel Density Estimation](10-Kernel-Density-Estimation/10-Kernel-Density-Estimation.md)

### Patterns & structure
11. ✅ [Association Rule Learning (Apriori · FP-Growth)](11-Association-Rule-Learning/11-Association-Rule-Learning.md)

### Representation (self-supervised)
12. ✅ [Contrastive / Self-Supervised Learning](12-Contrastive-Self-Supervised-Learning/12-Contrastive-Self-Supervised-Learning.md)

### Related concepts (canonical home is another section)
> These topics are used across many areas, so they're kept in one place to avoid repetition.
- **PCA / SVD (the math)** — eigendecomposition, variance maximization, the SVD view → [Foundations — Maths for AI-ML](../01.%20Foundations/Maths%20for%20AI-ML/README.md)
- **Autoencoders** — non-linear, learned dimensionality reduction → [Deep Learning](../05.%20Deep_Learning/README.md)
- **Word / sentence embeddings** — representation learning over text → [NLP](../06.%20NLP/README.md)

## 🎓 Courses (free)
- [Machine Learning Specialization (Course 3)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng** — clustering & anomaly detection.
- [Kaggle Learn: Clustering & PCA notebooks](https://www.kaggle.com/learn) — **Kaggle** — applied, runnable.

## 🎥 Videos
- [StatQuest: PCA, t-SNE, k-means, Hierarchical clustering](https://www.youtube.com/playlist?list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF) — **Josh Starmer** — the canonical visual explanations.

## 📄 Key Papers / Articles
- [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/) — **Distill** — interactive; the pitfalls everyone hits.
- [UMAP](https://arxiv.org/abs/1802.03426) — **McInnes et al. (2018)** — the modern manifold method + [great docs](https://umap-learn.readthedocs.io/).

## 📚 Books (free)
- [An Introduction to Statistical Learning (ISLP), Ch. 12](https://www.statlearning.com/) — **James et al.** — free; clustering & PCA, applied.

## 🔗 In this platform
- The geometry behind it: [AI-ML-intuition 1.05 PCA/SVD, 1.11–1.12 t-SNE/UMAP, 1.18 k-Means](../../AI-ML-intuition/Module_1_Representation/)
