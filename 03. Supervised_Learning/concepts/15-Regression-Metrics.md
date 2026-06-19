---
id: "03-supervised-learning/regression-metrics"
topic: "Regression Metrics (RMSE · MAE · R²)"
parent: "03-supervised-learning"
level: beginner
prereqs: ["linear-regression", "mean-squared-error", "variance"]
interview_frequency: high
updated: 2026-06-19
---

# Regression Metrics — RMSE · MAE · R²
> Score a continuous-target model by how far its predictions miss: **MAE** averages the absolute
> errors (robust, same units), **RMSE** averages the squared errors then square-roots (penalizes large
> misses), and **R²** reports the fraction of variance the model explains (1 = perfect, 0 = no better
> than the mean). Knowing *which* to report — and why — is the whole point.

**Why it matters:** the regression counterpart to classification metrics, and a common interview
follow-up. Expect: derive each metric, explain **RMSE vs MAE** (RMSE punishes outliers via squaring;
MAE is the median-like, robust choice), why RMSE shares units with the target while MSE doesn't, what
**R²** means and its traps (it can go negative, and it inflates with more features → use **adjusted
R²**), and how the metric you optimize (squared vs absolute loss) changes the model you get.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: Linear Regression](https://www.youtube.com/watch?v=nk2CQITm_eo). *See residuals — the raw material every regression metric is built from.*
2. **See why we square** — watch [StatQuest: Fitting a Line (Least Squares)](https://www.youtube.com/watch?v=PaFPbb66DxQ). *Why squared error (→ MSE/RMSE) is the default, and how it differs from absolute error.*
3. **Master R²** — watch [StatQuest: R-squared, Clearly Explained!](https://www.youtube.com/watch?v=2AQKmw14mHM). *The variance-explained interpretation and how to read it honestly.*
4. **Get the math** — read [scikit-learn: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) + [ISLR Ch. 3.1.3 (assessing fit: RSE, R²)](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf). *Exact formulas for MAE, MSE, RMSE, R², and adjusted R².*
5. **Make it concrete** — compute them with [`mean_absolute_error`, `mean_squared_error`, `r2_score`](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) and add one big outlier. *Watch RMSE jump while MAE barely moves — the difference made concrete.*

## 🎓 Courses (free)
- [Machine Learning Specialization — Course 1 (Regression)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — cost functions and how regression error is measured; free to audit.
- [Google ML Crash Course — Linear Regression (Loss)](https://developers.google.com/machine-learning/crash-course/linear-regression) — **Google** — short, applied intro to squared error and model fit.
- [CS229: Machine Learning — Lecture notes §1](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — squared-error loss and its MLE-under-Gaussian-noise meaning.

## 🎥 Videos
- [Linear Regression, Clearly Explained!!!](https://www.youtube.com/watch?v=nk2CQITm_eo) — **StatQuest (Josh Starmer)** — residuals, fit, and R² introduced from scratch.
- [The Main Ideas of Fitting a Line (Least Squares)](https://www.youtube.com/watch?v=PaFPbb66DxQ) — **StatQuest (Josh Starmer)** — why we square residuals (→ MSE/RMSE) rather than take absolutes.
- [R-squared, Clearly Explained!!!](https://www.youtube.com/watch?v=2AQKmw14mHM) — **StatQuest (Josh Starmer)** — the variance-explained interpretation, step by step.
- [Root-Mean-Square Error (RMSD)](https://www.youtube.com/watch?v=zMFdb__sUpw) — **Khan Academy** — RMSE/RMSD as the standard deviation of the residuals.

## 📄 Key Papers
- [Root Mean Square Error (RMSE) or Mean Absolute Error (MAE)?](https://gmd.copernicus.org/articles/7/1247/2014/gmd-7-1247-2014.pdf) — **Chai & Draxler (2014)** — the careful argument for when each metric is appropriate; open-access PDF.
- [Another Look at Measures of Forecast Accuracy](https://robjhyndman.com/papers/mase.pdf) — **Hyndman & Koehler (2006)** — the careful comparison of error measures (and why scale matters); author-hosted PDF, free.

## 📰 Articles / Blogs (free, no paywall)
- [Metrics and scoring — Regression metrics (scikit-learn)](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) — **scikit-learn** — the practical reference: MAE, MSE, RMSE, R², explained variance, MAPE.
- [MLU-Explain: Linear Regression](https://mlu-explain.github.io/linear-regression/) — **Amazon** — interactive: drag points and watch residuals, R², and the fit update.
- [Ordinary Least Squares Regression (visual)](https://setosa.io/ev/ordinary-least-squares-regression/) — **Victor Powell / Setosa** — a visual feel for residuals and variance explained.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLR) — **Ch. 3.1.3 "Assessing the Accuracy of the Model"**](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — RSE and R² explained in the regression chapter.
- [The Elements of Statistical Learning — **Ch. 2.4 & 7 (loss & assessment)**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — squared vs absolute loss and model assessment, rigorously.
- [Dive into Deep Learning — **Ch. 3 (loss functions)**](https://d2l.ai/chapter_linear-regression/index.html) — **Zhang et al.** — squared-error loss as the metric you optimize, with runnable code.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.01 Mean Squared Error (MSE / L2)](../../../AI-ML-intuition/Module_3_Evaluation/3.01_Mean_Squared_Error_MSE_L2_Loss.md) · [3.02 Mean Absolute Error (MAE / L1)](../../../AI-ML-intuition/Module_3_Evaluation/3.02_Mean_Absolute_Error_MAE_L1_Loss.md)
- Math prerequisites (the *why*): [01. Foundations](../../01.%20Foundations/README.md) — variance, residuals, L1 vs L2 norms.
- Related concepts: [01 Linear Regression](01-Linear-Regression.md) — the model these score · [14 Classification Metrics](14-Classification-Metrics.md) — the discrete-target counterpart · [13 Cross-Validation](13-Cross-Validation.md) — how you estimate them reliably.
- Related domain: [5. Deep Learning](../../05.%20Deep_Learning/README.md) — MSE/MAE are the standard regression losses for neural nets too.
