---
id: "04-unsupervised-learning/kernel-density-estimation/references"
topic: "Kernel Density Estimation — References"
parent: "04-unsupervised-learning/kernel-density-estimation"
type: references
updated: 2026-06-22
---

# Kernel Density Estimation — references and further reading

> Companion link library for **[Kernel Density Estimation](10-Kernel-Density-Estimation.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer, chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [The Histogram and Kernel Density Estimation](https://www.youtube.com/watch?v=SUvPJ4URYGA) (**Justin Esarey**). *See KDE as a smoothed, bin-free histogram — bumps summed over points, the clearest first picture.*
2. **See the bandwidth** — play with [the interactive KDE explainer](https://mathisonian.github.io/kde/) (**Matthew Conlen**). *Drag the bandwidth and kernel and watch over- vs under-smoothing live — the single most important intuition.*
3. **Get the math** — read [In Depth: Kernel Density Estimation](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html) (**Jake VanderPlas**). *The estimator, kernels, bandwidth selection, and KDE as a Bayesian classifier, with runnable code.*
4. **Go deeper on selection** — watch [Kernel Density Estimation](https://www.youtube.com/watch?v=qc9elACH8LA) (**Kapil Sachdeva**). *Bandwidth selection (Silverman, cross-validation) and where KDE breaks down.*
5. **Read the source** — [On Estimation of a Probability Density Function and Mode](https://projecteuclid.org/euclid.aoms/1177704472) (**Parzen 1962**). *The foundational "Parzen window" paper that defines KDE.*

**Videos**:
- [The Histogram and Kernel Density Estimation](https://www.youtube.com/watch?v=SUvPJ4URYGA) — **Justin Esarey** — KDE as a smoothed histogram; the clearest first intuition.
- [Kernel Density Estimation](https://www.youtube.com/watch?v=qc9elACH8LA) — **Kapil Sachdeva** — bandwidth selection and the bias–variance trade-off in depth.
- [Kernel Density Estimation — Explained](https://www.youtube.com/watch?v=6sGOMbC5xdE) — **DataMListic** — the estimator formula and the role of the kernel, concisely.
- [Kernel Density Estimation Explained | Statistics for Data Science](https://www.youtube.com/watch?v=k0uyEzcNj4U) — **DataMites** — a worked example end to end, good for a second pass.

**Interactive & visual**:
- [Kernel Density Estimation — interactive explainer](https://mathisonian.github.io/kde/) — **Matthew Conlen** — drag the bandwidth and kernel; the best visual intuition, fully free.
- [Simple 1D Kernel Density Estimation example](https://scikit-learn.org/stable/auto_examples/neighbors/plot_kde_1d.html) — **scikit-learn** — a runnable walkthrough of kernels and bandwidth as a focused lesson.

**Courses (free)**:
- [Density Estimation — scikit-learn user guide](https://scikit-learn.org/stable/modules/density.html) — **scikit-learn** — KDE kernels, bandwidth, and using density scores for novelty detection, with code.
- [All of Nonparametric Statistics — Ch. 6 "Density Estimation"](https://www.stat.cmu.edu/~larry/=sml/densityestimation.pdf) — **Larry Wasserman (CMU)** — the rigorous treatment: bias/variance, AMISE, cross-validation, all derived.

**Articles / blogs (free, no paywall)**:
- [In Depth: Kernel Density Estimation](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html) — **Jake VanderPlas (Python Data Science Handbook)** — KDE for density, visualization, and a Bayesian classifier, with code.
- [Kernel density estimation — Wikipedia](https://en.wikipedia.org/wiki/Kernel_density_estimation) — **Wikipedia** — a careful reference for the estimator, kernel table, AMISE, and Silverman's rule, with citations.
- [Kernel Density Estimation (scikit-learn density guide)](https://scikit-learn.org/stable/modules/density.html) — **scikit-learn** — the practical reference on kernels, bandwidth, and `KernelDensity`.

**Key papers**:
- [On Estimation of a Probability Density Function and Mode](https://projecteuclid.org/euclid.aoms/1177704472) — **Parzen (1962)** — the foundational paper (the "Parzen window") that defines KDE in its modern form.
- [Remarks on Some Nonparametric Estimates of a Density Function](https://projecteuclid.org/euclid.aoms/1177728190) — **Rosenblatt (1956)** — the earliest non-parametric density estimator, the idea KDE generalizes.
- [Density Estimation](https://projecteuclid.org/journals/statistical-science/volume-19/issue-4/Density-Estimation/10.1214/088342304000000297.full) — **Sheather (2004)** — an open-access review by the co-author of the Sheather–Jones plug-in selector; the modern reference on bandwidth selection.

**Books (free, with chapters)**:
- [Density Estimation for Statistics and Data Analysis](https://ned.ipac.caltech.edu/level5/March02/Silverman/paper.pdf) — **Silverman (1986)** — the classic monograph; the source of Silverman's rule and the standard reference on KDE.
- [The Elements of Statistical Learning — §6.6 "Kernel Density Estimation and Classification"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; KDE, bandwidth, and the kernel-classifier connection.
- [Mathematics for Machine Learning — Ch. 6 "Probability and Distributions"](https://mml-book.github.io/) — **Deisenroth, Faisal & Ong** — free; the density/probability foundations KDE estimates.
- [Multivariate Density Estimation: Theory, Practice, and Visualization](https://www.wiley.com/en-us/Multivariate+Density+Estimation%3A+Theory%2C+Practice%2C+and+Visualization%2C+2nd+Edition-p-9780471697558) — **David W. Scott (1992)** — the standard reference on multivariate KDE, the bandwidth matrix, and Scott's rule.

**In this platform**:
- Concept page (full explanation): [Kernel Density Estimation](10-Kernel-Density-Estimation.md)
- Parametric counterpart (the contrast): [Gaussian Mixture Models & EM](../04-Gaussian-Mixture-Models-and-EM/04-Gaussian-Mixture-Models-and-EM.md) — KDE is a GMM with one fixed bump per point
- Puts it to work: [Anomaly / Outlier Detection](../09-Anomaly-Outlier-Detection/09-Anomaly-Outlier-Detection.md) — low estimated density = outlier
- Method tie: [The Kernel Trick (intuition)](../../../AI-ML-intuition/Module_1_Representation/1.16_The_Kernel_Trick.md) — the same Gaussian bump, used for inner products rather than density
- Foundations (the *why*): [Distributions & the Gaussian (intuition)](../../../AI-ML-intuition/Module_0_Foundations/0.02_Distributions_and_the_Gaussian.md) — the kernel is usually a Gaussian bump
- Prereq math: [Foundations — Linear Algebra (vectors & matrices)](../../01.%20Foundations/Maths%20for%20AI-ML/1.%20Linear%20Algebra/VectorsAndMatrices.md)
- Field overview: [4. Unsupervised Learning](../README.md)
