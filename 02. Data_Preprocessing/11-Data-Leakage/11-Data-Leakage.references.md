---
id: "02-data-preprocessing/data-leakage/references"
topic: "Data Leakage — References"
parent: "02-data-preprocessing/data-leakage"
type: references
updated: 2026-07-03
---

# Data Leakage — references and further reading

> Companion link library for **[Data Leakage](11-Data-Leakage.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Everything here is free / open, and every "Source / derivation" citation on the concept page appears below.

**Start here — suggested path**:
1. **Get the definition** — watch [What is Data Leakage in Machine Learning?](https://www.youtube.com/watch?v=n9jz7G68pVg) (**Krish Naik**). *The clearest short definition, with practical examples of both contamination and target leakage.*
2. **See concrete leaky features** — watch [Examples of Data or Target Leakage](https://www.youtube.com/watch?v=NaySLPTCgDM) (**Rajistics**). *Real features that secretly encode the answer, and how they fool a model.*
3. **Do the hands-on lesson** — do [Kaggle Learn: Data Leakage](https://www.kaggle.com/code/alexisbcook/data-leakage) (**Kaggle**). *Target leakage vs train-test contamination, worked on real data with code.*
4. **Learn the structural fix** — read [Common pitfalls and recommended practices — Data leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage) (**scikit-learn**). *Why `Pipeline` + fit-on-train-only makes leakage structurally impossible, with runnable code.*
5. **Read the formal treatment** — read [Leakage in Data Mining](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf) (**Kaufman, Rosset, Perlich & Stitelman, 2012**). *The definitive taxonomy, detection, and avoidance — the paper everyone cites.*
6. **Run it yourself** — open [this chapter's notebook](code/11-Data-Leakage.ipynb). *Watch a leaky protocol report 0.86 on pure noise and the Pipeline fix collapse it to 0.48, then measure a real target leak and a temporal leak.*

**Videos**:
- [What is Data Leakage in Machine Learning?](https://www.youtube.com/watch?v=n9jz7G68pVg) — **Krish Naik** — concise, practical definition with examples of both leakage classes.
- [Examples of Data or Target Leakage](https://www.youtube.com/watch?v=NaySLPTCgDM) — **Rajistics** — concrete leaky features (proxies for the label) and how they inflate a score.
- [Machine Learning Fundamentals: Cross Validation](https://www.youtube.com/watch?v=fSytzGwwBVw) — **StatQuest (Josh Starmer)** — the canonical, intuition-first explanation of the resampling protocol leakage corrupts.
- [Selecting the best model with cross-validation](https://www.youtube.com/watch?v=6dbrR-WymjI) — **Data School (Kevin Markham)** — how to cross-validate *correctly* (including preprocessing) so you don't leak, with scikit-learn.
- [Cross Validation — the right way (and how leakage sneaks in)](https://www.youtube.com/watch?v=wjILv3-UGM8) — **Abhishek Thakur (Kaggle Grandmaster)** — a practitioner's tour of CV done right and the leakage traps competitions are full of.
- [What is data leakage?](https://www.youtube.com/watch?v=cApPa55X2JU) — **CodeEmporium** — intuition and how to reason about train/test contamination.

**Courses (free)**:
- [Kaggle Learn — Data Leakage](https://www.kaggle.com/code/alexisbcook/data-leakage) — **Kaggle** — the single best free hands-on lesson: target leakage vs train-test contamination, on real data.
- [Google ML Crash Course — Overfitting: dividing datasets](https://developers.google.com/machine-learning/crash-course/overfitting/dividing-datasets) — **Google** — the splitting discipline that prevents contamination, with interactive widgets.
- [scikit-learn — Cross-validation: evaluating estimator performance](https://scikit-learn.org/stable/modules/cross_validation.html) — **scikit-learn** — the authoritative guide to CV done right, including time-series and grouped data.

**Interactive & visual**:
- [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html) — **scikit-learn docs** — the canonical, runnable "how to avoid data leakage" reference, with correct-vs-leaky code side by side.
- [Visualizing cross-validation behavior in scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html) — **scikit-learn docs** — plots of `KFold`, `GroupKFold`, and `TimeSeriesSplit` fold assignments — exactly the train/validation geometry leakage exploits.
- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) — **scikit-learn docs** — the forward-chronological splitter that prevents temporal leakage, with a worked example.

**Articles / blogs (free, no paywall)**:
- [Common pitfalls: how to avoid data leakage](https://scikit-learn.org/stable/common_pitfalls.html#how-to-avoid-data-leakage) — **scikit-learn** — the `Pipeline` pattern that fits every transform on training folds only.
- [Data Leakage in Machine Learning](https://machinelearningmastery.com/data-leakage-machine-learning/) — **Jason Brownlee** — an accessible overview of the failure modes and how to guard against them.
- [Time series cross-validation](https://otexts.com/fpp3/tscv.html) — **Hyndman & Athanasopoulos (Forecasting: Principles and Practice)** — why random splits leak on time-ordered data, and the forward-evaluation fix (free online).

**Key sources / derivations** (cited on the concept page):
- [Leakage in Data Mining: Formulation, Detection, and Avoidance](https://dl.acm.org/doi/10.1145/2382577.2382579) — **Kaufman, Rosset, Perlich & Stitelman, *ACM TKDD* 6(4), 2012 (KDD 2011)** — the definitive taxonomy of leakage, why it produces optimistically biased estimates, and how to detect/avoid it (author copy [free PDF](https://www.cs.umb.edu/~ding/history/470_670_fall_2011/papers/cs670_Tran_PreferredPaper_LeakingInDataMining.pdf)).
- [An Introduction to Statistical Learning — Ch. 5 "Resampling Methods"](https://www.statlearning.com/) — **James, Witten, Hastie & Tibshirani** — free PDF; why every data-dependent step must be fit inside the resampling loop, and the right way to cross-validate.
- [The Elements of Statistical Learning — §7.10.2 "The Wrong and Right Way to Do Cross-validation"](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — free PDF; the exact feature-selection-before-CV leak this chapter reproduces on noise, worked through formally.
- [Leakage and the Reproducibility Crisis in ML-based Science](https://arxiv.org/abs/2207.07048) — **Kapoor & Narayanan, 2022** — free on arXiv; documents hundreds of published results across fields invalidated by leakage, and a checklist to prevent it.
- [Pipelines and composite estimators](https://scikit-learn.org/stable/modules/compose.html) & [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html) — **scikit-learn docs** — the mechanism that fits all preprocessing inside the training folds, structurally preventing preprocessing leakage.
- [TimeSeriesSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html) & [Forecasting: Principles and Practice §5.10](https://otexts.com/fpp3/tscv.html) — **scikit-learn / Hyndman & Athanasopoulos** — the correct evaluation of time-ordered data (train on the past, predict the future).

**Books (free, with chapters)**:
- [An Introduction to Statistical Learning (Python) — Ch. 5 "Resampling Methods"](https://www.statlearning.com/) — **James et al.** — resampling that respects the train/test boundary; free PDF.
- [Approaching (Almost) Any Machine Learning Problem](https://github.com/abhishekkrthakur/approachingalmost) — **Abhishek Thakur** — a Kaggle Grandmaster's practical treatment of cross-validation and the leakage traps that decide competitions (free PDF on GitHub).
- [Feature Engineering and Selection — Ch. 3 "A Review of the Predictive Modeling Process"](http://www.feat.engineering/) — **Kuhn & Johnson** — disciplined data spending and resampling that structurally prevents leakage; free online.

**In this platform**:
- Concept page (full explanation): [Data Leakage](11-Data-Leakage.md)
- The classic mild leak, measured — fitting a scaler on all data: [02 Feature Scaling & Normalization](../02-Feature-Scaling-and-Normalization/02-Feature-Scaling-and-Normalization.md)
- The split leakage is defined against — train/validation/test discipline: [10 Train/Validation/Test Splits](../10-Train-Validation-Test-Splits/10-Train-Validation-Test-Splits.md)
- The structural cure in a real pipeline (`Pipeline` + `ColumnTransformer`): [13 Data Pipelines](../13-Data-Pipelines/13-Data-Pipelines.md)
- Building past-only features so time series don't leak: [08 Date-Time & Cyclical Features](../08-Date-Time-and-Cyclical-Features/08-Date-Time-and-Cyclical-Features.md)
- Why an inflated score is really an overfitting/generalization story: [00 Basics — 05 Overfitting & Underfitting](../../00.%20Basics/05-Overfitting-and-Underfitting/05-Overfitting-and-Underfitting.md)
- The concept depth (the *why* of generalization): [AI-ML-intuition 3.07 Bias–Variance & Generalization](../../../AI-ML-intuition/Module_3_Evaluation/3.07_Bias_Variance_and_Generalization.md)
- Where leakage meets production — train/serve skew: [14 Deployment & MLOps](../../14.%20Deployment_and_MLOps/README.md) · [12 Data & Concept Drift](../../14.%20Deployment_and_MLOps/12-Data-and-Concept-Drift-Detection/README.md)
