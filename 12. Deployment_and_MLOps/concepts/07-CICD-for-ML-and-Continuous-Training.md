---
id: "12-mlops/cicd-for-ml-and-continuous-training"
topic: "CI/CD for ML & Continuous Training (CT)"
parent: "12-mlops-and-deployment"
level: advanced
prereqs: ["ml-pipelines-and-orchestration", "ml-lifecycle"]
interview_frequency: high
updated: 2026-06-20
---

# CI/CD for ML & Continuous Training (CT)
> Extending DevOps CI/CD to ML, plus a third loop unique to ML — **Continuous Training**: pipelines that
> automatically retrain, validate, and redeploy models as new data arrives or performance drops. The jump
> from MLOps maturity level 1 to level 2.

**Why it matters:** the "what's different about CI/CD for ML vs software?" question. Interviewers want
the extra validations (data + model, not just code), what gets versioned and tested (data schemas,
feature logic, model quality gates), and the CT loop: trigger → retrain → evaluate against a baseline →
canary/deploy. Tools: GitHub Actions, CML, and pipeline orchestrators.

**⭐ Start here — suggested path:**

1. **See the levels** — re-read [Google: MLOps level 2 (CI/CD/CT)](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning). *Where CT plugs into the maturity model.*
2. **Get the principles** — read [Martin Fowler: Continuous Delivery for ML (CD4ML)](https://martinfowler.com/articles/cd4ml.html). *The canonical CD4ML article — the vocabulary interviewers use.*
3. **See ML-specific CI** — watch [MLOps Tutorial #1: Intro to CI for ML](https://www.youtube.com/watch?v=9BgIDqAzfuA). *Why ML CI needs data + model checks, not just unit tests.*
4. **Automate model tracking in CI** — watch [MLOps Tutorial #3: Track ML models with Git & GitHub Actions](https://www.youtube.com/watch?v=xPncjKH6SPk). *Comparing model metrics across branches in a PR — the model quality gate.*
5. **Build the pipeline** — work [Made With ML: CI/CD](https://madewithml.com/courses/mlops/cicd/). *A complete GitHub Actions CI/CD/CT flow for an ML project.*

## 🎓 Courses (free)
- [Made With ML — CI/CD](https://madewithml.com/courses/mlops/cicd/) — **Goku Mohandas** — full GitHub Actions pipeline for testing, training, and deploying.
- [CML — Documentation](https://cml.dev/doc) — **Iterative** — continuous ML in CI runners (reports, metrics, cloud GPUs).

## 🎥 Videos
- [MLOps Tutorial #1: Intro to Continuous Integration for ML](https://www.youtube.com/watch?v=9BgIDqAzfuA) — **DVCorg** — CI for ML projects, the data/model angle.
- [MLOps Tutorial #3: Track ML Models with Git & GitHub Actions](https://www.youtube.com/watch?v=xPncjKH6SPk) — **DVCorg** — automated model comparison and quality gates in PRs.
- [MLOps Tutorial #4: GitHub Actions with Your Own GPUs](https://www.youtube.com/watch?v=rVq-SCNyxVc) — **DVCorg** — self-hosted GPU runners for continuous training.
- [Learn MLOps with MLflow and Databricks — Full Course](https://www.youtube.com/watch?v=tVskbekONlw) — **freeCodeCamp** — the train → register → deploy CD path end to end.

## 📄 Key Papers
- [MLOps: Continuous delivery and automation pipelines in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — **Google Cloud** — defines CI/CD/CT and the level-2 architecture.
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (2015)** — why untested glue code and configs are the dominant production risk.

## 📰 Articles / Blogs (free, no paywall)
- [Continuous Delivery for Machine Learning (CD4ML)](https://martinfowler.com/articles/cd4ml.html) — **Sato, Wider & Windheuser (martinfowler.com)** — the definitive CD4ML reference.
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — **Google** — testing and launch discipline for production ML.
- [CML — Continuous Machine Learning](https://cml.dev/) — **Iterative** — CI/CD patterns purpose-built for ML.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 9 "Continual Learning & Test in Production"**](https://huyenchip.com/mlops/) — **Chip Huyen** — continuous training, retraining cadence, and testing in prod (author notes free).
- [Machine Learning Engineering — **Ch. 8 "Model Deployment"** (CI/CD & retraining)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [01 ML Lifecycle & MLOps Maturity](01-ML-Lifecycle-and-MLOps-Maturity.md) · [06 ML Pipelines & Orchestration](06-ML-Pipelines-and-Orchestration.md)
- Next concepts: [12 Data & Concept Drift Detection](12-Data-and-Concept-Drift-Detection.md) · [14 A/B Testing · Shadow & Canary](14-AB-Testing-Shadow-and-Canary-Deployment.md)
