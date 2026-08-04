---
id: "01-foundations/singular-value-decomposition/references"
topic: "Singular Value Decomposition — References"
parent: "01-foundations/singular-value-decomposition"
type: references
updated: 2026-07-02
---

# Singular Value Decomposition — references and further reading

> Companion link library for **[Singular Value Decomposition](singular-value-decomposition.md)**
> (the concept page). Kept separate so it can be reused as a standalone reference list. Grouped by
> type, best-first. Everything here is **free / open** — no paywall. Every source cited under a
> "Source / derivation" line on the concept page appears here, so each formula is traceable to a
> primary source. Chosen for depth on *this* topic, not popularity.

**⭐ Start here — suggested path**:
1. **Get the picture** — watch [Singular Value Decomposition (SVD): Overview](https://www.youtube.com/watch?v=gXbThCXjZFM) (**Steve Brunton**). *The rotate–scale–rotate intuition and why SVD is everywhere in data science.*
2. **See the geometry** — watch [The determinant](https://www.youtube.com/watch?v=Ip3X9LOh2dk) and the change-of-basis videos in **3Blue1Brown**'s [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab). *Builds the "matrix = linear transformation" mental model SVD rests on.*
3. **Do the math** — watch [SVD: Mathematical Overview](https://www.youtube.com/watch?v=nbBvuuNVfco) (**Steve Brunton**), then read [MML Ch. 4.5](https://mml-book.github.io/book/mml-book.pdf). *Connects `A = UΣVᵀ` to the eigenvectors of `AᵀA` and `AAᵀ`.*
4. **The authoritative lecture** — watch [Gilbert Strang: SVD (MIT 18.06, Lec 29)](https://www.youtube.com/watch?v=TX_vooSnhm8). *Strang's geometric derivation and the four-subspaces tie-in.*
5. **Low-rank & applications** — read [Data-Driven Science & Engineering, Ch. 1 (SVD)](https://databookuw.com/) (**Brunton & Kutz**, free PDF). *Eckart–Young, truncation, and real applications with code.*

**🎥 Videos** (trusted creators only):
- [Singular Value Decomposition (SVD): Overview](https://www.youtube.com/watch?v=gXbThCXjZFM) — **Steve Brunton (UW)** — the clearest high-level intuition; first stop.
- [SVD: Mathematical Overview](https://www.youtube.com/watch?v=nbBvuuNVfco) — **Steve Brunton** — the derivation via `AᵀA` eigenvectors, step by step.
- [SVD: Image Compression](https://www.youtube.com/watch?v=DG7YTlGnCEo) — **Steve Brunton** — the exact demo this chapter builds, worked on a real image.
- [Singular Value Decomposition (MIT 18.06, Lec 29)](https://www.youtube.com/watch?v=TX_vooSnhm8) — **Gilbert Strang (MIT OCW)** — the definitive full lecture with the geometry and the four subspaces.
- [Computing the SVD (MIT 18.065)](https://www.youtube.com/watch?v=mBcLRGuAFUk) — **Gilbert Strang (MIT OCW)** — the data-methods framing, `AᵀA` and `AAᵀ`.
- [Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab) — **3Blue1Brown** — the geometric "matrix = transformation" series that makes rotate→scale→rotate obvious.
- [Principal Component Analysis (PCA), Step-by-Step](https://www.youtube.com/watch?v=FgakZw6K1QQ) — **StatQuest (Josh Starmer)** — PCA explained plainly; the sister view of truncated SVD.

**🎓 Courses (free)**:
- [MIT 18.06 — Linear Algebra (SVD lecture)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/) — **Gilbert Strang (MIT OCW)** — the classic course; SVD is Lecture 29.
- [MIT 18.065 — Matrix Methods in Data Analysis, Signal Processing, and ML](https://ocw.mit.edu/courses/18-065-matrix-methods-in-data-analysis-signal-processing-and-machine-learning-spring-2018/) — **Gilbert Strang (MIT OCW)** — an entire data-focused course built around the SVD.
- [Stanford CS229 — Linear Algebra Review (SVD & PSD matrices)](https://cs229.stanford.edu/section/cs229-linalg.pdf) — **Stanford** — the ML-oriented summary, SVD in context of PCA.

**📰 Articles / blogs (free, no paywall)**:
- [We Recommend a Singular Value Decomposition](https://www.ams.org/publicoutreach/feature-column/fcarc-svd) — **David Austin (AMS Feature Column)** — a beautifully illustrated, free geometric walkthrough (the circle→ellipse picture).
- [Understanding the SVD, geometrically](https://gregorygundersen.com/blog/2018/12/10/svd/) — **Gregory Gundersen** — a careful, honest derivation of the geometry and the eigen-connection.
- [Singular Value Decomposition (SVD) tutorial](https://web.mit.edu/be.400/www/SVD/Singular_Value_Decomposition.htm) — **Kirk Baker (MIT)** — a compact worked example, `AᵀA` and `AAᵀ` by hand.
- [Eigenfaces / faces recognition with an SVD-based approach](https://scikit-learn.org/stable/auto_examples/decomposition/plot_faces_decomposition.html) — **scikit-learn** — the eigen-digits idea of this chapter, on faces, runnable.

**📄 Key papers**:
- [The approximation of one matrix by another of lower rank](https://doi.org/10.1007/BF02288367) — **Eckart & Young (1936), *Psychometrika***. — the original Eckart–Young theorem: truncated SVD is the best low-rank approximation in the Frobenius norm. (DOI link; the theorem statement is reproduced in every reference below.)
- [Symmetric gauge functions and unitarily invariant norms](https://doi.org/10.1093/qmath/11.1.50) — **Mirsky (1960), *Quarterly J. Math.***. — the Mirsky extension: truncated SVD is optimal in *every* unitarily invariant norm (Frobenius and spectral both).
- [Finding Structure with Randomness: randomized algorithms for constructing approximate matrix decompositions](https://arxiv.org/abs/0909.4061) — **Halko, Martinsson & Tropp (2011), *SIAM Review***. — the standard method for computing an approximate SVD at scale.
- [On the Early History of the Singular Value Decomposition](https://doi.org/10.1137/1035134) — **G. W. Stewart (1993), *SIAM Review***. — where the SVD comes from (Beltrami, Jordan, Sylvester); good historical grounding.

**📚 Books (free chapters / full PDFs)**:
- [Mathematics for Machine Learning — Ch. 4.5 "Singular Value Decomposition"](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — `A = UΣVᵀ`, the derivation from `AᵀA`, low-rank approximation, and the link to PCA. Free full PDF.
- [Data-Driven Science & Engineering — Ch. 1 "Singular Value Decomposition"](https://databookuw.com/) — **Brunton & Kutz** — an SVD-first treatment with code; the source of the image-compression and Eckart–Young framing. Chapter PDFs free.
- [MIT 18.06 — Lecture 29: Singular Value Decomposition (video + downloadable transcript)](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-29-singular-value-decomposition/) — **Gilbert Strang (MIT OCW)** — the specific free SVD lecture ("the final and best factorization of a matrix"), with the four-subspaces picture; the closest thing to a free Strang SVD chapter, one click.
- [Numerical Linear Algebra — Lecture 4 "The Singular Value Decomposition"](https://people.maths.ox.ac.uk/trefethen/lec4.ps) — **Trefethen & Bau** — the exact SVD lecture (with [Lecture 5, "More on the SVD"](https://people.maths.ox.ac.uk/trefethen/lec5.ps)) deriving the SVD and its geometry rigorously; the standard numerical reference, posted free by the author.

**🔗 In this platform**:
- Concept page (full explanation): [Singular Value Decomposition](singular-value-decomposition.md)
- Prerequisite (the *why* behind Σ² and eigenvectors of `AᵀA`): [04 Eigenvalues & Eigenvectors](../eigenvalues-and-eigenvectors/eigenvalues-and-eigenvectors.md) · [05 Matrix Decompositions](../matrix-decompositions/matrix-decompositions.md)
- Foundations it rests on: [02 Matrices & Matrix Operations](../matrices-and-matrix-operations/matrices-and-matrix-operations.md) · [03 Norms, Inner Products & Orthogonality](../norms-inner-products-and-orthogonality/norms-inner-products-and-orthogonality.md)
- Where it goes next: [07 PCA — the math](../principal-component-analysis-math/principal-component-analysis-math.md)
- Applied downstream: dimensionality reduction and latent factors → [Unsupervised Learning](../../../core-machine-learning/unsupervised-learning/README.md); low-rank adaptation of LLM weights → [LoRA & PEFT](../../../llms-applications-and-agents/training-and-adaptation/lora-and-parameter-efficient-fine-tuning/lora-and-parameter-efficient-fine-tuning.md)
