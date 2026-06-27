---
id: "14-mlops/model-monitoring-and-observability"
topic: "Model Monitoring & Observability"
parent: "14-mlops-and-deployment"
level: advanced
prereqs: ["model-serving", "ml-lifecycle"]
interview_frequency: very-high
updated: 2026-06-20
---

# Model Monitoring & Observability
> Watching a deployed model so silent failures don't go unnoticed: operational health (latency, errors,
> throughput), data quality (schema, missing/out-of-range inputs), and model quality (accuracy when labels
> arrive, proxy metrics when they don't). Observability is being able to ask *why* it failed, not just *that* it did.

**Why it matters:** the "your model is live — how do you know it's still working?" question, and a near-universal
one. Models fail silently: no exception, just degrading predictions. Interviewers want the three layers
(operational / data / model quality), the **delayed-label problem** (and proxy metrics when ground truth is
late or absent), and the tool stack (Evidently, Prometheus, Grafana). Drift detection is the deep dive that follows.

**⭐ Start here — suggested path:**

1. **Get the layers** — read [How to Monitor Machine Learning Models](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/). *Operational vs data vs model-quality monitoring — the mental model.*
2. **Monitoring vs observability** — read [Evidently: What is ML Monitoring & Observability](https://learn.evidentlyai.com/ml-observability-course/module-1-introduction/ml-monitoring-observability). *Why "that it broke" (monitoring) and "why it broke" (observability) are different.*
3. **Build a dashboard** — work [Made With ML: Monitoring](https://madewithml.com/courses/mlops/monitoring/), then watch [Evidently AI: ML Monitoring & Observability Tutorial](https://www.youtube.com/watch?v=cgc3dSEAel0). *Turns the theory into a real metrics dashboard.*
4. **Batch vs online monitoring** — watch [Evidently: Batch architecture dashboard](https://www.youtube.com/watch?v=u4Mcu0hXfMA) and [Online architecture dashboard](https://www.youtube.com/watch?v=2hTRXEOJF8k). *Two deployment shapes, two monitoring designs.*
5. **Go to drift** — move to [12 Data & Concept Drift Detection](../12-Data-and-Concept-Drift-Detection/12-Data-and-Concept-Drift-Detection.md). *The statistical heart of why model quality degrades.*

## 🎓 Courses (free)
- [Made With ML — Monitoring](https://madewithml.com/courses/mlops/monitoring/) — **Goku Mohandas** — monitor performance, data, and drift in a real system.
- [Evidently — ML Observability Course](https://learn.evidentlyai.com/ml-observability-course/module-1-introduction/ml-monitoring-observability) — **Evidently AI** — free, end-to-end open course on monitoring/observability.

## 🎥 Videos
- [Evidently AI Tutorial — Open-Source ML Monitoring & Observability](https://www.youtube.com/watch?v=cgc3dSEAel0) — **community** — the metrics, reports, and dashboards in practice.
- [ML Model Monitoring Dashboard with Evidently — Batch Architecture](https://www.youtube.com/watch?v=u4Mcu0hXfMA) — **Evidently AI** — code practice for batch monitoring.
- [ML Model Monitoring Dashboard with Evidently — Online Architecture](https://www.youtube.com/watch?v=2hTRXEOJF8k) — **Evidently AI** — code practice for live-service monitoring.
- [Data Drift & Early Monitoring for ML Models](https://www.youtube.com/watch?v=N12uMO-fj40) — **Emeli (Evidently AI)** — meetup talk on what to monitor and why.

## 📄 Key Papers
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (2015)** — "monitoring & testing" debt; why ML systems erode silently.
- [Challenges in Deploying ML: A Survey of Case Studies](https://arxiv.org/abs/2011.09926) — **Paleyes et al. (2020)** — post-deployment monitoring failures across industry.

## 📰 Articles / Blogs (free, no paywall)
- [How to Monitor Machine Learning Models](https://christophergs.com/machine%20learning/2020/03/14/how-to-monitor-machine-learning-models/) — **Christopher Samiullah** — the canonical free three-layer monitoring guide.
- [Evidently — Model Monitoring](https://www.evidentlyai.com/ml-in-production/model-monitoring) — **Evidently AI** — what to track and how, with the delayed-label problem.
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — **Google** — monitoring and "test in production" heuristics.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 8 "Data Distribution Shifts & Monitoring"**](https://huyenchip.com/mlops/) — **Chip Huyen** — the canonical chapter on monitoring + shift (author notes free).
- [Machine Learning Engineering — **Ch. 9 "Model Serving, Monitoring & Maintenance"**](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [01 ML Lifecycle & MLOps Maturity](../01-ML-Lifecycle-and-MLOps-Maturity/01-ML-Lifecycle-and-MLOps-Maturity.md) · [09 Model Serving](../09-Model-Serving/09-Model-Serving.md)
- Next concepts: [12 Data & Concept Drift Detection](../12-Data-and-Concept-Drift-Detection/12-Data-and-Concept-Drift-Detection.md) · [13 Model Registry & Governance](../13-Model-Registry-and-Governance/13-Model-Registry-and-Governance.md)
