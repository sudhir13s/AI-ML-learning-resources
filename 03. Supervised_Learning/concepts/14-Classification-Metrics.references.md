---
id: "03-supervised-learning/classification-metrics/references"
topic: "Classification Metrics — References"
parent: "03-supervised-learning/classification-metrics"
type: references
updated: 2026-06-22
---

# Classification Metrics — references and further reading

> Companion link library for **[Classification Metrics](14-Classification-Metrics.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic.

**Start here — suggested path**:
1. **Build intuition** — watch [The Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o) (**StatQuest**), then play with [MLU-Explain: Precision & Recall](https://mlu-explain.github.io/precision-recall/). *Everything else is derived from these four cells.*
2. **See the tradeoff** — watch [Sensitivity and Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8) (**StatQuest**). *Why moving the threshold trades recall for specificity.*
3. **Master ROC/AUC** — watch [ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM) (**StatQuest**), then explore [MLU-Explain: ROC & AUC](https://mlu-explain.github.io/roc-auc/). *The threshold-free ranking view; when PR-AUC beats ROC-AUC.*
4. **Get the math** — read [scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) + [ISLR Ch. 4.4.2](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf). *Exact formulas, multiclass averaging, and the curves.*
5. **Go one level deeper** — read [Davis & Goadrich (2006)](https://www.biostat.wisc.edu/~page/rocpr.pdf) on PR-vs-ROC under imbalance, then skim [Guo et al. (2017)](https://arxiv.org/abs/1706.04599) for why discrimination ≠ calibration. *The two ideas that separate a strong answer from a textbook one.*
6. **Make it concrete** — compute `classification_report`, `average_precision_score`, and `calibration_curve` on an imbalanced dataset; watch accuracy stay high while recall and precision collapse.

**Videos**:
- [Machine Learning Fundamentals: The Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o) — **StatQuest (Josh Starmer)** — the four cells everything else is built from.
- [Machine Learning Fundamentals: Sensitivity and Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8) — **StatQuest (Josh Starmer)** — recall vs specificity and the threshold tradeoff.
- [ROC and AUC, Clearly Explained!](https://www.youtube.com/watch?v=4jRBRDbJemM) — **StatQuest (Josh Starmer)** — the threshold-free ranking metric, drawn out step by step.
- [Precision, Recall and F1-score Explained Clearly](https://www.youtube.com/watch?v=Rm-6TagS71U) — **Neuro Splash** — a clean visual walkthrough of precision, recall, and the harmonic-mean F1.
- [How to evaluate a classifier in scikit-learn](https://www.youtube.com/watch?v=85dtiMz9tSo) — **Data School (Kevin Markham)** — confusion matrix, precision/recall, ROC/AUC, and threshold tuning, end to end in code.

**Interactive & visual**:
- [Explaining the ROC curve (interactive)](https://www.evidentlyai.com/classification-metrics/explain-roc-curve) — **Evidently AI** — step-by-step build of the ROC curve from the confusion matrix, with the threshold sweep animated.
- [MLU-Explain: ROC & AUC](https://mlu-explain.github.io/roc-auc/) — **Amazon** — fully interactive: move the threshold and watch the ROC curve and AUC respond.
- [MLU-Explain: Precision & Recall](https://mlu-explain.github.io/precision-recall/) — **Amazon** — interactive confusion matrix; see precision and recall trade off live.

**Courses (free)**:
- [Machine Learning Specialization — Course 2](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — precision/recall, F1, and metrics for skewed data; free to audit.
- [Google ML Crash Course — Classification (ROC, AUC, precision/recall)](https://developers.google.com/machine-learning/crash-course/classification) — **Google** — short, applied, with threshold and ROC interactives.
- [CS229: Machine Learning — Lecture notes (Evaluation)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — the evaluation framing alongside the models.

**Articles / blogs (free, no paywall)**:
- [Metrics and scoring (scikit-learn user guide)](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) — **scikit-learn** — the practical reference: every metric, multiclass averaging, and curves.
- [Probability calibration of classifiers (worked example)](https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html) — **scikit-learn** — reliability diagrams, Brier score, and Platt/isotonic recalibration in code.

**Key papers**:
- [The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf) — **Davis & Goadrich (2006)** — *why* PR curves are more informative than ROC under class imbalance (and the non-linear PR interpolation).
- [An Introduction to ROC Analysis](https://people.inf.elte.hu/kiss/13dwhdm/roc.pdf) — **Fawcett (2006)** — the definitive tutorial on ROC curves and AUC.
- [The Precision-Recall Plot Is More Informative than the ROC Plot on Imbalanced Datasets](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) — **Saito & Rehmsmeier (2015)** — extends Davis & Goadrich with extensive imbalanced-data experiments; the PRROC reference.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — **Guo et al. (2017)** — modern nets are over-confident; ECE, reliability diagrams, and temperature scaling — why discrimination ≠ calibration.
- [The advantages of the Matthews correlation coefficient (MCC) over F1 and accuracy](https://bmcgenomics.biomedcentral.com/articles/10.1186/s12864-019-6413-7) — **Chicco & Jurman (2020)** — the case for MCC as the most informative single binary metric on imbalanced data.

**Books (free chapters)**:
- [An Introduction to Statistical Learning (ISLR) — Ch. 4.4.2 (Confusion matrix, ROC)](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the applied treatment of classification evaluation.
- [The Elements of Statistical Learning — Ch. 9.2.5 & 7 (ROC / assessment)](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous view of ROC and model assessment.
- [Speech and Language Processing, 3rd ed. — Ch. 4 (Evaluation: precision/recall/F1)](https://web.stanford.edu/~jurafsky/slp3/4.pdf) — **Jurafsky & Martin** — the cleanest derivation of precision/recall/F1.

**In this platform**:
- Concept page (full explanation): [Classification Metrics](14-Classification-Metrics.md)
- Concept depth (the *why*): [AI-ML-intuition 3.05 Classification Metrics](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md) · [3.06 ROC, AUC & PR Curves](../../../AI-ML-intuition/Module_3_Evaluation/3.06_ROC_AUC_PR_Curves.md)
- Related: [Logistic Regression](02-Logistic-Regression.md) (produces the probabilities you threshold) · [Regression Metrics](15-Regression-Metrics.md) (continuous-target counterpart) · [Cross-Validation](13-Cross-Validation.md) (how you estimate these reliably) · [Bias–Variance Tradeoff](12-Bias-Variance-Tradeoff.md)
- Math prerequisites: [01. Foundations](../../01.%20Foundations/concepts/README.md) — conditional probability, base rates, thresholds
