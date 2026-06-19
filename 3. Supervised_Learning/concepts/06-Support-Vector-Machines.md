---
id: "03-supervised-learning/support-vector-machines"
topic: "Support Vector Machines (SVM)"
parent: "03-supervised-learning"
level: intermediate
prereqs: ["linear-algebra", "convex-optimization", "kernel-trick"]
interview_frequency: high
updated: 2026-06-19
---

# Support Vector Machines (SVM)
> Find the hyperplane that separates the classes with the **largest margin** — the widest "street"
> between them — depending only on the boundary points (the *support vectors*). The **kernel trick**
> then lets the same linear machinery carve non-linear boundaries in an implicit high-dimensional space.

**Why it matters:** the classic optimization-flavored interview topic. Expect: define the margin and
why maximizing it helps generalization, the role of support vectors, the hard- vs soft-margin
formulation and the `C` hyperparameter, the primal–dual view, and — the crown jewel — the **kernel
trick** (RBF, polynomial) that gives non-linear boundaries without ever computing the high-dimensional features.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: SVM Main Ideas](https://www.youtube.com/watch?v=efR1C6CvhmE). *See the maximum-margin "street" and the support vectors that define it.*
2. **See the kernel trick** — watch [StatQuest: Polynomial Kernel](https://www.youtube.com/watch?v=Toet3EiSFcM) then [RBF Kernel](https://www.youtube.com/watch?v=Qc5IyLW_hns). *How a kernel computes high-dimensional dot products without leaving the original space.*
3. **Get the math** — read [ISLR Ch. 9 "Support Vector Machines"](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) + [SVM tutorial](https://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf). *The margin objective, soft-margin slack variables, and the dual that exposes the kernel.*
4. **Read the source** — skim [Cortes & Vapnik: Support-Vector Networks](https://link.springer.com/article/10.1007/BF00994018). *The paper that introduced soft-margin SVMs (publisher page; PDF mirror linked below).*
5. **Make it concrete** — implement with [scikit-learn SVM](https://scikit-learn.org/stable/modules/svm.html). *Fit `SVC`, sweep `C` and the RBF `gamma`, and watch the boundary tighten and the support vectors change.*

## 🎓 Courses (free)
- [CS229: Machine Learning — Lecture notes (SVMs)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — the rigorous derivation: margins, the dual, KKT conditions, and kernels.
- [Machine Learning Specialization](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — classification context and the kernel idea; free to audit.
- [A Tutorial on Support Vector Machines (course PDF)](https://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf) — **Jason Weston (Columbia)** — a compact, self-contained derivation used in teaching, free.

## 🎥 Videos
- [Support Vector Machines, Clearly Explained!!!](https://www.youtube.com/watch?v=efR1C6CvhmE) — **StatQuest (Josh Starmer)** — the gentle, from-scratch intuition for margins and support vectors.
- [SVMs Part 2: The Polynomial Kernel](https://www.youtube.com/watch?v=Toet3EiSFcM) — **StatQuest (Josh Starmer)** — how the polynomial kernel builds non-linear boundaries.
- [SVMs Part 3: The Radial (RBF) Kernel](https://www.youtube.com/watch?v=Qc5IyLW_hns) — **StatQuest (Josh Starmer)** — the most-used kernel, and what `gamma` controls.
- [Support Vector Machines — The Math You Should Know](https://www.youtube.com/watch?v=05VABNfa1ds) — **CodeEmporium** — the optimization view: margin objective, Lagrangian, and the dual.

## 📄 Key Papers
- [Support-Vector Networks](https://link.springer.com/article/10.1007/BF00994018) — **Cortes & Vapnik (1995)** — the paper that introduced the soft-margin SVM; publisher page.
- [Support-Vector Networks (PDF)](https://link.springer.com/content/pdf/10.1007/BF00994018.pdf) — **Cortes & Vapnik (1995)** — the same landmark paper as a free PDF.

## 📰 Articles / Blogs (free, no paywall)
- [Support Vector Machines (scikit-learn user guide)](https://scikit-learn.org/stable/modules/svm.html) — **scikit-learn** — the practical reference: kernels, `C`, `gamma`, and multiclass strategies.
- [RBF SVM parameters — visual example](https://scikit-learn.org/stable/auto_examples/svm/plot_rbf_parameters.html) — **scikit-learn** — a visual sweep of `C` and `gamma` showing under/overfitting.
- [A Tutorial on Support Vector Machines (Weston)](https://www.cs.columbia.edu/~kathy/cs4701/documents/jason_svm_tutorial.pdf) — **Jason Weston** — a clear, free derivation of the margin objective and the kernel trick.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLR) — **Ch. 9 "Support Vector Machines"**](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the best applied chapter: maximal-margin → support-vector classifier → kernels, with labs.
- [The Elements of Statistical Learning — **Ch. 12 "Support Vector Machines and Flexible Discriminants"**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous treatment (the dual, the loss + penalty view).
- [Information Theory, Inference, and Learning Algorithms — **Ch. 40 (kernel methods context)**](https://www.inference.org.uk/itprnn/book.pdf) — **David MacKay** — kernels and large-margin classifiers in a probabilistic frame, free PDF.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 1.16 The Kernel Trick](../../../AI-ML-intuition/Module_1_Representation/1.16_The_Kernel_Trick.md) · [3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Math prerequisites (the *why*): [1. Foundations](../../1.%20Foundations/README.md) — linear algebra, convex optimization, Lagrangian duality.
- Prior / related concepts: [02 Logistic Regression](02-Logistic-Regression.md) — another linear classifier; SVM maximizes margin, not likelihood · [03 Regularization](03-Regularization-Linear-Models.md) — `C` is an inverse-regularization knob.
- Related domain: [2. Data Preprocessing](../../2.%20Data_Preprocessing/README.md) — SVMs need feature scaling for distance-based kernels to behave.
