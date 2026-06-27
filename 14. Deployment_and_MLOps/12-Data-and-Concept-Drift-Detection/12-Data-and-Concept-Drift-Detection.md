---
id: "14-mlops/data-and-concept-drift-detection"
topic: "Data & Concept Drift Detection"
parent: "14-mlops-and-deployment"
level: advanced
prereqs: ["model-monitoring-and-observability", "statistics"]
interview_frequency: high
updated: 2026-06-20
---

# Data & Concept Drift Detection
> Detecting when the world has moved away from the training distribution. **Data drift**: P(X) changes
> (inputs shift). **Concept drift**: P(Y|X) changes (the input→output relationship shifts). Both quietly
> erode accuracy — drift detection tells you *when to retrain* before users notice.

**Why it matters:** the deep-dive that follows monitoring, and a strong differentiator. Interviewers
want a crisp **data drift vs concept drift** distinction, detection methods (KS test, PSI, Chi-square,
Jensen–Shannon / Wasserstein distance), the **delayed-/no-label problem** (use prediction drift + input
drift as proxies), and the response (alert → root-cause → retrain). The trigger for continuous training.

**⭐ Start here — suggested path:**

1. **Nail the definitions** — read [Evidently: What is Data Drift](https://www.evidentlyai.com/ml-in-production/data-drift) and [What is Concept Drift](https://www.evidentlyai.com/ml-in-production/concept-drift). *P(X) vs P(Y|X) — get this distinction exactly right.*
2. **Learn the tests** — read [Evidently: Data Drift Detection Deep Dive](https://learn.evidentlyai.com/ml-observability-course/module-2-ml-monitoring-metrics/data-drift-deep-dive). *KS, PSI, Chi-square, JS/Wasserstein — which test for which data type.*
3. **Detect it in code** — watch [Detect Data Drift Using Evidently](https://www.youtube.com/watch?v=WXu4NRG5WQU) and [Evidently in Jupyter: data & prediction drift](https://www.youtube.com/watch?v=g0Z2e-IqmmU). *Running the tests and reading the reports.*
4. **Read the theory** — skim [A Survey on Concept Drift Adaptation](https://eprints.bournemouth.ac.uk/22491/). *Drift types (sudden/gradual/recurring) and adaptation strategies — the rigorous backbone.*
5. **Close the loop** — connect drift to retraining via [07 CI/CD for ML & CT](07-CICD-for-ML-and-Continuous-Training.md). *Drift detected → continuous-training pipeline fires.*

## 🎓 Courses (free)
- [Evidently — ML Observability Course (Module 2: Drift)](https://learn.evidentlyai.com/ml-observability-course/module-2-ml-monitoring-metrics/data-drift-deep-dive) — **Evidently AI** — the statistical drift tests, hands-on.
- [Made With ML — Monitoring](https://madewithml.com/courses/mlops/monitoring/) — **Goku Mohandas** — drift detection as part of production monitoring.

## 🎥 Videos
- [Easiest Way to Detect Data Drift Using Evidently](https://www.youtube.com/watch?v=WXu4NRG5WQU) — **community** — minimal end-to-end drift detection.
- [Use Evidently in Jupyter to Evaluate Data & Prediction Drift](https://www.youtube.com/watch?v=g0Z2e-IqmmU) — **community** — running the drift report and interpreting it.
- [Data Drift & Early Monitoring for ML Models](https://www.youtube.com/watch?v=N12uMO-fj40) — **Emeli (Evidently AI)** — drift detection methods and pitfalls.
- [Drift Monitoring & Evaluation for LLM Apps](https://www.youtube.com/watch?v=eQ6cGzDUtMU) — **Evidently AI** — extending drift detection to LLM/embedding outputs.

## 📄 Key Papers
- [A Survey on Concept Drift Adaptation](https://eprints.bournemouth.ac.uk/22491/) — **Gama et al. (2014)** — the canonical taxonomy of drift types and adaptation methods.
- [Learning under Concept Drift: A Review](https://arxiv.org/abs/1010.4784) — **Žliobaitė (2010)** — foundational survey of drift detection and handling.
- [Challenges in Deploying ML: A Survey of Case Studies](https://arxiv.org/abs/2011.09926) — **Paleyes et al. (2020)** — drift-driven degradation in real deployments.

## 📰 Articles / Blogs (free, no paywall)
- [What is Data Drift in ML](https://www.evidentlyai.com/ml-in-production/data-drift) — **Evidently AI** — P(X) shift, detection tests, and handling.
- [What is Concept Drift in ML](https://www.evidentlyai.com/ml-in-production/concept-drift) — **Evidently AI** — P(Y|X) shift and how to detect it without fresh labels.
- [Data Drift Detection Deep Dive](https://learn.evidentlyai.com/ml-observability-course/module-2-ml-monitoring-metrics/data-drift-deep-dive) — **Evidently AI** — the statistical tests, compared.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 8 "Data Distribution Shifts & Monitoring"**](https://huyenchip.com/mlops/) — **Chip Huyen** — covariate/label/concept shift and detection (author notes free).
- [Machine Learning Engineering — **Ch. 9 "Monitoring & Maintenance"** (distribution shift)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [11 Model Monitoring & Observability](11-Model-Monitoring-and-Observability.md)
- Next concepts: [07 CI/CD for ML & CT](07-CICD-for-ML-and-Continuous-Training.md) · [14 A/B Testing · Shadow & Canary](14-AB-Testing-Shadow-and-Canary-Deployment.md)
- Related concept (covered elsewhere): distribution-shift theory & generalization → [01. Foundations](../../01.%20Foundations/concepts/README.md)
