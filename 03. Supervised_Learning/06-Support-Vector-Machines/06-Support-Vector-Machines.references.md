---
id: "03-supervised-learning/support-vector-machines/references"
topic: "Support Vector Machines — References"
parent: "03-supervised-learning/support-vector-machines"
type: references
updated: 2026-06-22
---

# Support Vector Machines — references and further reading

> Companion link library for **[Support Vector Machines](06-Support-Vector-Machines.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [SVM Main Ideas](https://www.youtube.com/watch?v=efR1C6CvhmE) (**StatQuest**). *See the maximum-margin "street" and the support vectors that define it.*
2. **See the kernel trick** — watch [Polynomial Kernel](https://www.youtube.com/watch?v=Toet3EiSFcM) then [RBF Kernel](https://www.youtube.com/watch?v=Qc5IyLW_hns) (**StatQuest**). *How a kernel computes high-dim dot products without leaving the original space.*
3. **Get the math** — read [ISLR Ch. 9](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + the [SVM tutorial](https://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf). *Margin objective, soft-margin slack, and the dual that exposes the kernel.*
4. **Read the source** — skim [Support-Vector Networks](https://link.springer.com/content/pdf/10.1007/BF00994018.pdf) (**Cortes & Vapnik 1995**). *The paper that introduced soft-margin SVMs.*
5. **Make it concrete** — fit `SVC` with [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html); sweep `C` and RBF `gamma`.

**Videos**:
- [Support Vector Machines, Clearly Explained!!!](https://www.youtube.com/watch?v=efR1C6CvhmE) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for margins and support vectors.
- [SVMs Part 2: The Polynomial Kernel](https://www.youtube.com/watch?v=Toet3EiSFcM) — **StatQuest (Josh Starmer)** — how the polynomial kernel builds non-linear boundaries.
- [SVMs Part 3: The Radial (RBF) Kernel](https://www.youtube.com/watch?v=Qc5IyLW_hns) — **StatQuest (Josh Starmer)** — the most-used kernel, and what `gamma` controls.
- [Support Vector Machines — The Math You Should Know](https://www.youtube.com/watch?v=05VABNfa1ds) — **CodeEmporium** — the optimization view: margin objective, Lagrangian, and the dual.

**Interactive & visual**:
- [RBF SVM parameters — visual sweep](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html) — **scikit-learn** — a grid of `C` × `gamma` showing the boundary go from underfit (too smooth) to overfit (too wiggly).

**Courses (free)**:
- [CS229: Machine Learning — Lecture notes (SVMs)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — the rigorous derivation: margins, the dual, KKT conditions, and kernels.
- [Learning From Data — caltech (online course)](https://home.work.caltech.edu/telecourse.html) — **Yaser Abu-Mostafa (Caltech)** — the VC-dimension / generalization theory behind *why* a large margin works; the formal backbone of the "wide margin generalizes" claim.
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — classification context and the kernel idea; free to audit.
- [A Tutorial on Support Vector Machines (course PDF)](https://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf) — **Jason Weston (Columbia)** — a compact, self-contained derivation used in teaching.

**Articles / blogs (free, no paywall)**:
- [A Tutorial on Support Vector Machines for Pattern Recognition](https://web.mit.edu/6.034/wwwbob/svm.pdf) — **Christopher Burges (1998)** — the classic, much-cited tutorial deriving margins, the dual, KKT, and kernels in full; the standard reference write-up.
- [Support Vector Machines (scikit-learn user guide)](https://scikit-learn.org/stable/modules/svm.html) — **scikit-learn** — the practical reference: kernels, `C`, `gamma`, multiclass (one-vs-one / one-vs-rest), and `probability=True`.
- [A Tutorial on Support Vector Machines (Weston)](https://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf) — **Jason Weston** — a clear derivation of the margin objective and the kernel trick.

**Key papers**:
- [Support-Vector Networks](https://link.springer.com/article/10.1007/BF00994018) — **Cortes & Vapnik (1995)** — the paper that introduced the soft-margin SVM; publisher page.
- [Support-Vector Networks (PDF)](https://link.springer.com/content/pdf/10.1007/BF00994018.pdf) — **Cortes & Vapnik (1995)** — the same landmark paper as a free PDF.
- [Probabilistic Outputs for SVMs and Comparisons to Regularized Likelihood Methods](https://www.cs.colorado.edu/~mozer/Teaching/syllabi/6622/papers/Platt1999.pdf) — **John Platt (1999)** — the original **Platt scaling** method for turning SVM scores into calibrated probabilities.

**Books (free chapters)**:
- [An Introduction to Statistical Learning (ISLR) — Ch. 9 "Support Vector Machines"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the best applied chapter: maximal-margin → support-vector classifier → kernels, with labs.
- [The Elements of Statistical Learning — Ch. 12 "Support Vector Machines and Flexible Discriminants"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment (the dual, the loss + penalty view).
- [The Nature of Statistical Learning Theory](https://link.springer.com/book/10.1007/978-1-4757-3264-1) — **Vladimir Vapnik (1995)** — the source of the margin / VC-dimension bounds that justify maximizing the margin (publisher page).
- [Information Theory, Inference, and Learning Algorithms — Ch. 40 (kernel methods)](https://www.inference.org.uk/itprnn/book.pdf) — **David MacKay** — kernels and large-margin classifiers in a probabilistic frame.

**In this platform**:
- Concept page (full explanation): [Support Vector Machines](06-Support-Vector-Machines.md)
- Concept depth (the *why*): [AI-ML-intuition 1.16 The Kernel Trick](../../../AI-ML-intuition/Module_1_Representation/1.16_The_Kernel_Trick.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Related: [Logistic Regression](02-Logistic-Regression.md) (another linear classifier; SVM maximizes margin, not likelihood) · [Regularization (Linear Models)](03-Regularization-Linear-Models.md) (C is inverse regularization)
- Math prerequisites: [01. Foundations](../../01.%20Foundations/concepts/README.md) — linear algebra, convex optimization, Lagrangian duality
