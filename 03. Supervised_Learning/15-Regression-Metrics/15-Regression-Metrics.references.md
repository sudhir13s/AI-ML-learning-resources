---
id: "03-supervised-learning/regression-metrics/references"
topic: "Regression Metrics — References"
parent: "03-supervised-learning/regression-metrics"
type: references
updated: 2026-06-22
---

# Regression Metrics — references and further reading

> Companion link library for **[Regression Metrics](15-Regression-Metrics.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [Linear Regression, Clearly Explained](https://www.youtube.com/watch?v=nk2CQITm_eo) (**StatQuest**). *See residuals — the raw material every regression metric is built from.*
2. **See why we square** — watch [The Main Ideas of Fitting a Line (Least Squares)](https://www.youtube.com/watch?v=PaFPbb66DxQ) (**StatQuest**). *Why squared error (→ MSE/RMSE) is the default, and how it differs from absolute error.*
3. **Master R²** — watch [R-squared, Clearly Explained](https://www.youtube.com/watch?v=2AQKmw14mHM) (**StatQuest**). *The variance-explained interpretation and how to read it honestly.*
4. **Get the math** — read [scikit-learn: Regression metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) + [ISLR Ch. 3.1.3](https://www.statlearning.com/). *Exact formulas for MAE, MSE, RMSE, R², adjusted R².*
5. **Make it concrete** — compute `mean_absolute_error`, `mean_squared_error`, `r2_score` on a dataset, then add one big outlier. *Watch RMSE jump while MAE barely moves.*

**Videos**:
- [Linear Regression, Clearly Explained](https://www.youtube.com/watch?v=nk2CQITm_eo) — **StatQuest (Josh Starmer)** — residuals, fit, and R² introduced from scratch.
- [The Main Ideas of Fitting a Line (Least Squares)](https://www.youtube.com/watch?v=PaFPbb66DxQ) — **StatQuest (Josh Starmer)** — why we square residuals (→ MSE/RMSE) rather than take absolutes.
- [R-squared, Clearly Explained](https://www.youtube.com/watch?v=2AQKmw14mHM) — **StatQuest (Josh Starmer)** — the variance-explained interpretation, step by step.
- [Quantile Regression](https://www.youtube.com/watch?v=s203ScSy8RM) — **ritvikmath** — pinball loss and predicting quantiles, drawn out clearly.

**Interactive & visual**:
- [MLU-Explain: Linear Regression](https://mlu-explain.github.io/linear-regression/) — **Amazon** — drag points and watch residuals, R², and the fit update live.
- [Ordinary Least Squares Regression (visual)](https://setosa.io/ev/ordinary-least-squares-regression/) — **Victor Powell / Setosa** — a visual feel for residuals and variance explained.

**Courses (free)**:
- [Machine Learning Specialization — Course 1 (Regression)](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — cost functions and how regression error is measured; free to audit.
- [Google ML Crash Course — Linear Regression (Loss)](https://developers.google.com/machine-learning/crash-course/linear-regression) — **Google** — short, applied intro to squared error and model fit.
- [CS229: Machine Learning — Lecture notes](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — squared-error loss and its MLE-under-Gaussian-noise meaning.

**Articles / blogs (free, no paywall)**:
- [Metrics and scoring — Regression metrics (scikit-learn)](https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics) — **scikit-learn** — the practical reference: MAE, MSE, RMSE, R², MAPE, MSLE, pinball.
- [Forecasting: Principles and Practice — Evaluating accuracy](https://otexts.com/fpp3/accuracy.html) — **Hyndman & Athanasopoulos** — MAE, RMSE, MAPE, and scaled errors (MASE) for forecasting, by the authority on the topic.
- [Quantile regression (scikit-learn example)](https://scikit-learn.org/stable/auto_examples/linear_model/plot_quantile_regression.html) — **scikit-learn** — fitting models at several τ with the pinball loss to build prediction intervals, with runnable code.

**Key papers**:
- [Root Mean Square Error (RMSE) or Mean Absolute Error (MAE)?](https://gmd.copernicus.org/articles/7/1247/2014/gmd-7-1247-2014.pdf) — **Chai & Draxler (2014)** — the careful argument for when each metric is appropriate; open-access PDF.
- [Another Look at Measures of Forecast Accuracy](https://robjhyndman.com/papers/mase.pdf) — **Hyndman & Koehler (2006)** — why scale matters and the case for scaled errors (MASE); author-hosted PDF, free.
- [Quantile Regression (survey)](http://www.econ.uiuc.edu/~roger/research/rq/rq.pdf) — **Roger Koenker** — author-hosted overview of quantile regression and the pinball (check) loss, building on the founding **Koenker & Bassett (1978)** "Regression Quantiles."
- [Robust Estimation of a Location Parameter](https://projecteuclid.org/journals/annals-of-mathematical-statistics/volume-35/issue-1/Robust-Estimation-of-a-Location-Parameter/10.1214/aoms/1177703732.full) — **Huber (1964)** — the founding paper of robust statistics and the Huber loss.

**Books (free chapters)**:
- [An Introduction to Statistical Learning (ISLR) — Ch. 3.1.3 "Assessing the Accuracy of the Model"](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — RSE, R², and adjusted R² in the regression chapter; free PDF.
- [The Elements of Statistical Learning — Ch. 2.4 & 7 (loss & assessment)](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — squared vs absolute loss, Gaussian-MLE, and model assessment, rigorously; free PDF.
- [Dive into Deep Learning — Ch. 3 (loss functions)](https://d2l.ai/chapter_linear-regression/index.html) — **Zhang et al.** — squared-error loss as the metric you optimize, with runnable code.

**In this platform**:
- Concept page (full explanation): [Regression Metrics](15-Regression-Metrics.md)
- Concept depth (the *why*): [AI-ML-intuition 3.01 Mean Squared Error (MSE / L2)](../../../AI-ML-intuition/Module_3_Evaluation/3.01_Mean_Squared_Error_MSE_L2_Loss.md) · [3.02 Mean Absolute Error (MAE / L1)](../../../AI-ML-intuition/Module_3_Evaluation/3.02_Mean_Absolute_Error_MAE_L1_Loss.md)
- Related: [Linear Regression](../01-Linear-Regression/01-Linear-Regression.md) (the model these score) · [Classification Metrics](../14-Classification-Metrics/14-Classification-Metrics.md) (the discrete-target counterpart) · [Cross-Validation](../13-Cross-Validation/13-Cross-Validation.md) (how you estimate these reliably) · [Bias–Variance Tradeoff](../12-Bias-Variance-Tradeoff/12-Bias-Variance-Tradeoff.md)
- Math prerequisites: [01. Foundations](../../01.%20Foundations/README.md) — variance, residuals, L1 vs L2 norms
- Related domain: [5. Deep Learning](../../05.%20Deep_Learning/README.md) — MSE/MAE are the standard regression losses for neural nets too
