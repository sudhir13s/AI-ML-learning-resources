---
id: "03-supervised-learning/classification-metrics"
topic: "Classification Metrics (precision · recall · F1 · ROC-AUC · PR-AUC)"
parent: "03-supervised-learning"
level: beginner
prereqs: ["logistic-regression", "confusion-matrix", "probability"]
interview_frequency: very-high
updated: 2026-06-19
---

# Classification Metrics — precision · recall · F1 · ROC-AUC · PR-AUC
> Accuracy lies on imbalanced data, so you measure classifiers with the **confusion matrix** and what
> it yields: **precision** (of predicted positives, how many are right), **recall** (of actual
> positives, how many you caught), their harmonic mean **F1**, and threshold-free summaries
> **ROC-AUC** and **PR-AUC**. Choosing the right metric for the problem is the real skill.

**Why it matters:** the most practically important evaluation topic and a guaranteed interview area.
Expect: define TP/FP/TN/FN and derive precision/recall, explain the precision–recall **tradeoff** and
when you'd favor each (cancer screening vs spam), why **accuracy is misleading** under class
imbalance, what ROC-AUC means (probability a random positive ranks above a random negative) and why
**PR-AUC** is better for rare positives, and how the decision **threshold** moves all of these.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: The Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o), then play with [MLU-Explain: Precision & Recall](https://mlu-explain.github.io/precision-recall/). *Everything else is derived from these four cells.*
2. **See the tradeoff** — watch [StatQuest: Sensitivity and Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8). *Recall (sensitivity) vs specificity, and why moving the threshold trades one for the other.*
3. **Master ROC/AUC** — watch [StatQuest: ROC and AUC](https://www.youtube.com/watch?v=4jRBRDbJemM), then explore [MLU-Explain: ROC & AUC](https://mlu-explain.github.io/roc-auc/). *The threshold-free ranking view, and when PR-AUC beats ROC-AUC.*
4. **Get the math** — read [scikit-learn: Classification metrics](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) + [ISLR Ch. 4.4.2 (confusion matrix / ROC)](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf). *Exact formulas, averaging for multiclass, and the ROC/PR curves.*
5. **Make it concrete** — compute them with [`classification_report` and `roc_auc_score`](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) on an **imbalanced** dataset. *Watch accuracy stay high while recall collapses — the lesson that sticks.*

## 🎓 Courses (free)
- [Machine Learning Specialization — Course 2](https://www.coursera.org/specializations/machine-learning-introduction) — **Andrew Ng (DeepLearning.AI)** — precision/recall, F1, and metrics for skewed data; free to audit.
- [Google ML Crash Course — Classification (ROC, AUC, precision/recall)](https://developers.google.com/machine-learning/crash-course/classification) — **Google** — short, applied, with threshold and ROC interactives.
- [CS229: Machine Learning — Lecture notes (Evaluation)](https://cs229.stanford.edu/main_notes.pdf) — **Stanford (Ng)** — the evaluation framing alongside the models.

## 🎥 Videos
- [Machine Learning Fundamentals: The Confusion Matrix](https://www.youtube.com/watch?v=Kdsp6soqA7o) — **StatQuest (Josh Starmer)** — the four cells everything else is built from.
- [Machine Learning Fundamentals: Sensitivity and Specificity](https://www.youtube.com/watch?v=vP06aMoz4v8) — **StatQuest (Josh Starmer)** — recall vs specificity and the threshold tradeoff.
- [ROC and AUC, Clearly Explained!](https://www.youtube.com/watch?v=4jRBRDbJemM) — **StatQuest (Josh Starmer)** — the threshold-free ranking metric, drawn out step by step.
- [Precision, Recall and F1-score Explained Clearly](https://www.youtube.com/watch?v=Rm-6TagS71U) — **Neuro Splash** — a clean visual walkthrough of precision, recall, and the harmonic-mean F1.

## 📄 Key Papers
- [The Relationship Between Precision-Recall and ROC Curves](https://www.biostat.wisc.edu/~page/rocpr.pdf) — **Davis & Goadrich (2006)** — *why* PR curves are more informative than ROC under class imbalance; author-hosted PDF, free.
- [An Introduction to ROC Analysis](https://people.inf.elte.hu/kiss/13dwhdm/roc.pdf) — **Fawcett (2006)** — the definitive tutorial on ROC curves and AUC; open PDF.

## 📰 Articles / Blogs (free, no paywall)
- [MLU-Explain: ROC & AUC](https://mlu-explain.github.io/roc-auc/) — **Amazon** — fully interactive: move the threshold and watch the ROC curve and AUC respond.
- [MLU-Explain: Precision & Recall](https://mlu-explain.github.io/precision-recall/) — **Amazon** — interactive confusion matrix; see precision and recall trade off live.
- [Metrics and scoring (scikit-learn user guide)](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) — **scikit-learn** — the practical reference: every metric, multiclass averaging, and curves.

## 📚 Books (free, with chapters)
- [An Introduction to Statistical Learning (ISLR) — **Ch. 4.4.2 (Confusion matrix, ROC)**](https://www.statlearning.com/s/ISLR-Seventh-Printing.pdf) — **James, Witten, Hastie & Tibshirani** — the applied treatment of classification evaluation.
- [The Elements of Statistical Learning — **Ch. 9.2.5 & 7 (ROC / assessment)**](https://hastie.su.domains/ElemStatLearn/) — **Hastie, Tibshirani & Friedman** — the rigorous view of ROC and model assessment.
- [Speech and Language Processing, 3rd ed. — **Ch. 4 (Evaluation: precision/recall/F1)**](https://web.stanford.edu/~jurafsky/slp3/4.pdf) — **Jurafsky & Martin** — the cleanest derivation of precision/recall/F1 from a text-classification angle.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.05 Classification Metrics (Precision/Recall/F1)](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md) · [3.06 ROC, AUC & PR Curves](../../../AI-ML-intuition/Module_3_Evaluation/3.06_ROC_AUC_PR_Curves.md)
- Math prerequisites (the *why*): [1. Foundations](../../1.%20Foundations/README.md) — conditional probability, base rates, thresholds.
- Related concepts: [02 Logistic Regression](02-Logistic-Regression.md) — produces the probabilities you threshold · [15 Regression Metrics](15-Regression-Metrics.md) — the continuous-target counterpart · [13 Cross-Validation](13-Cross-Validation.md) — how you estimate these reliably.
- Related domain: [2. Data Preprocessing](../../2.%20Data_Preprocessing/README.md) — class imbalance and resampling shape which metric you trust.
