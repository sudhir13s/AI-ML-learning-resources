---
id: "15-advanced-math/rkhs"
topic: "Hilbert Spaces & RKHS (Kernels)"
parent: "15-advanced-research-mathematics"
level: advanced
prereqs: ["functional-analysis", "linear-algebra", "kernel-trick"]
interview_frequency: medium
updated: 2026-06-20
---

# Hilbert Spaces & RKHS — the Math of Kernels
> A Reproducing Kernel Hilbert Space is a Hilbert space of functions where evaluation `f ↦ f(x)` is a
> bounded linear functional — so by Riesz it equals an inner product `⟨f, k(·,x)⟩`. That one fact
> (the *reproducing property*) is the engine behind the kernel trick, SVMs, Gaussian processes,
> kernel ridge regression, MMD, and the neural tangent kernel.

**Why it matters:** the kernel trick you learned as "replace dot products with k(x,y)" is *justified*
here — Mercer's theorem, the representer theorem (your solution is a finite combo of kernels on the
data), and the feature-map view all live in this card. It's also the lens for infinite-width networks
(NTK) and two-sample testing (MMD). A favorite theory-interview thread: "why does the representer
theorem hold?"

**⭐ Start here — suggested path:**

1. **Recall the trick** — skim the platform's [Kernel Trick intuition](../../../AI-ML-intuition/Module_1_Representation/1.16_The_Kernel_Trick.md). *Know the "what" before formalizing the "why".*
2. **Build the RKHS** — watch [Lecture 2 on kernel methods: RKHS](https://www.youtube.com/watch?v=2uvpOKoiYoI) (Mairal). *Constructs the space from a kernel and derives the reproducing property.*
3. **Read the primer** — work [A Primer on RKHS](https://arxiv.org/abs/1408.0952) or [Twelve Pages or Less](http://users.umiacs.umd.edu/~hal3//docs/daume04rkhs.pdf). *Mercer → feature map → reproducing property, tightly.*
4. **Connect to GPs** — watch [Understanding Kernels and Gaussian Processes](https://www.youtube.com/watch?v=1U5sIaTD6xA) (Hennig). *The function-space view: a GP prior is a distribution over an RKHS-flavored space.*
5. **Get the theorems** — read Gretton's [Introduction to RKHS notes](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf) for the representer theorem and MMD. *The results you'll actually be asked to state.*

## 🎓 Courses (free)
- [Kernel Methods — course slides & notes](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf) — **Arthur Gretton (Gatsby, UCL)** — the standard ML-flavored RKHS course (kernels → RKHS → MMD), free.
- [MIT 18.102 — Hilbert space theory (the foundation)](https://ocw.mit.edu/courses/18-102-introduction-to-functional-analysis-spring-2021/) — **MIT OCW** — the rigorous Hilbert-space backbone RKHS sits on.
- [Gaussian Processes for Machine Learning — full free book](https://gaussianprocess.org/gpml/chapters/RW.pdf) — **Rasmussen & Williams** — the function-space companion to kernels, free PDF.

## 🎥 Videos
- [Lecture 2 on kernel methods: RKHS](https://www.youtube.com/watch?v=2uvpOKoiYoI) — **Julien Mairal (Inria)** — constructs the RKHS and proves the reproducing property.
- [Understanding Kernels and Gaussian Processes](https://www.youtube.com/watch?v=1U5sIaTD6xA) — **Philipp Hennig (Tübingen)** — kernels as covariance functions; the GP/RKHS connection.
- [MIT 18.102 — Lecture 14: Basic Hilbert Space Theory](https://www.youtube.com/watch?v=EBdgFFf54U0) — **Casey Rodriguez (MIT OCW)** — Riesz representation, the theorem that makes RKHS work.
- [MIT 18.102 — Lecture 1: Basic Banach Space Theory](https://www.youtube.com/watch?v=uoL4lQxfgwg) — **Casey Rodriguez (MIT OCW)** — completeness and bounded functionals, the prerequisites for "reproducing".

## 📄 Key Papers
- [Kernel methods in machine learning](https://projecteuclid.org/journals/annals-of-statistics/volume-36/issue-3/Kernel-methods-in-machine-learning/10.1214/009053607000000677.full) — **Hofmann, Schölkopf & Smola (2008)** — the canonical RKHS-in-ML survey: representer theorem, kernels, SVMs, GPs.
- [A Primer on Reproducing Kernel Hilbert Spaces](https://arxiv.org/abs/1408.0952) — **Manton & Amblard (2015)** — self-contained construction of RKHS from first principles.
- [Reproducing Kernel Hilbert Space, Mercer's Theorem, Eigenfunctions, Nyström — Tutorial & Survey](https://arxiv.org/abs/2106.08443) — **Ghojogh et al. (2021)** — Mercer + practical kernel methods in one free reference.
- [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572) — **Jacot, Gabriel & Hongler (2018)** — infinite-width nets become an RKHS, the modern payoff.

## 📰 Articles / Blogs (free, no paywall)
- [From Zero to Reproducing Kernel Hilbert Spaces in Twelve Pages or Less](http://users.umiacs.umd.edu/~hal3//docs/daume04rkhs.pdf) — **Hal Daumé III** — the fastest correct path to the reproducing property.
- [Introduction to RKHS and simple kernel algorithms](https://www.gatsby.ucl.ac.uk/~gretton/coursefiles/lecture4_introToRKHS.pdf) — **Arthur Gretton** — the representer theorem and kernel mean embeddings, clearly.

## 📚 Books (free, with chapters)
- [Gaussian Processes for Machine Learning — **Ch. 4 (covariance functions) & App. B (Gaussian Markov / RKHS)**](https://gaussianprocess.org/gpml/chapters/RW.pdf) — **Rasmussen & Williams** — kernels as the heart of GP modeling, free.
- [Understanding Machine Learning — **Ch. 16 (Kernel Methods)**](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) — **Shalev-Shwartz & Ben-David** — kernels with the representer theorem and margin bounds, free PDF.
- [Mathematics for Machine Learning — **Ch. 3 (inner-product geometry)**](https://mml-book.github.io/book/mml-book.pdf) — **Deisenroth, Faisal & Ong** — the finite-dimensional intuition that generalizes to RKHS.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.16 The Kernel Trick](../../../AI-ML-intuition/Module_1_Representation/1.16_The_Kernel_Trick.md) · [1.06 Scaled Dot-Product](../../../AI-ML-intuition/Module_1_Representation/1.06_Vector_Similarities_The_Scaled_Dot-Product.md)
- Foundations (the basics this builds on): [Norms, Inner Products & Orthogonality](../../01.%20Foundations/03-Norms-Inner-Products-and-Orthogonality/03-Norms-Inner-Products-and-Orthogonality.md)
- Prerequisite & next: [02 Functional Analysis](../02-Functional-Analysis/02-Functional-Analysis.md) · [08 Information Geometry](../08-Information-Geometry/08-Information-Geometry.md)
- Related domain: [04. Unsupervised Learning](../../04.%20Unsupervised_Learning/README.md)
</content>
