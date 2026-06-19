---
id: "12-mlops/experiment-tracking"
topic: "Experiment Tracking (MLflow · Weights & Biases)"
parent: "12-mlops-and-deployment"
level: intermediate
prereqs: ["ml-lifecycle", "reproducibility"]
interview_frequency: high
updated: 2026-06-20
---

# Experiment Tracking — MLflow · Weights & Biases
> Logging every run's params, metrics, code version, and artifacts to a central store so experiments
> are comparable, searchable, and reproducible. The lab notebook of ML — turns "I think run 47 was
> best" into a queryable record.

**Why it matters:** a near-universal MLOps interview topic. Expect "how do you track experiments and
pick the best model," the anatomy of a run (params vs metrics vs tags vs artifacts), why this beats
spreadsheets/filenames, MLflow's four components (Tracking, Projects, Models, Registry), and how
tracking feeds the model registry and CI/CD downstream.

**⭐ Start here — suggested path:**

1. **Get the mental model** — read [MLflow Tracking docs](https://mlflow.org/docs/latest/tracking/). *Runs / experiments / params / metrics / artifacts — the vocabulary everything else uses.*
2. **Do it end-to-end** — follow [Made With ML: Experiment Tracking](https://madewithml.com/courses/mlops/experiment-tracking/). *Logs a real training loop and compares runs — the fastest path to fluency.*
3. **See a second tool** — watch [MLOps Zoomcamp: W&B Experiment Tracking](https://www.youtube.com/watch?v=yNyqFMwEyL4). *W&B shows the hosted/collaborative flavor; comparing tools sharpens the "why."*
4. **Go deep on MLflow** — watch the [MLflow + Databricks full course](https://www.youtube.com/watch?v=tVskbekONlw). *Tracking → registry → deployment, the full lifecycle MLflow spans.*
5. **Wire into your workflow** — add tracking to a PyTorch loop with [W&B + PyTorch](https://www.youtube.com/watch?v=KESSYZExK44). *Making logging a one-liner is what makes the habit stick.*

## 🎓 Courses (free)
- [Made With ML — Experiment Tracking](https://madewithml.com/courses/mlops/experiment-tracking/) — **Goku Mohandas** — log, organize, and compare runs in a real project.
- [Made With ML — MLOps Course](https://madewithml.com/courses/mlops/) — **Goku Mohandas** — situates tracking inside the full develop→deploy loop.

## 🎥 Videos
- [MLOps Zoomcamp — Experiment Tracking with Weights & Biases](https://www.youtube.com/watch?v=yNyqFMwEyL4) — **DataTalks.Club** — hands-on tracking, sweeps, and artifacts with W&B.
- [Learn MLOps with MLflow and Databricks — Full Course](https://www.youtube.com/watch?v=tVskbekONlw) — **freeCodeCamp** — deep, end-to-end MLflow (tracking → registry → serving).
- [Track Your PyTorch Experiments with Weights & Biases](https://www.youtube.com/watch?v=KESSYZExK44) — **Weights & Biases** — instrument a training loop in minutes.
- [Learn Weights and Biases Now! — Beginner Tutorial](https://www.youtube.com/watch?v=_3z0r2k8Nh4) — **Weights & Biases** — gentle intro to runs, metrics, and dashboards.
- [Using W&B beyond experiment tracking](https://www.youtube.com/watch?v=_2DNYGv3jiM) — **Weights & Biases** — sweeps, artifacts, and reports built on the tracking core.

## 📄 Key Papers
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (2015)** — why ad-hoc experimentation accrues debt; motivates systematic tracking.
- [Improving Reproducibility in ML Research](https://arxiv.org/abs/2003.12206) — **Pineau et al. (2020)** — what to log so a result is reproducible (params, seeds, env).

## 📰 Articles / Blogs (free, no paywall)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking/) — **MLflow** — the canonical reference: runs, experiments, autologging, backends.
- [W&B Experiment Tracking](https://wandb.ai/site/experiment-tracking/) — **Weights & Biases** — the hosted-tracking model with collaboration and sweeps.
- [MLflow Quickstart / Getting Started](https://mlflow.org/docs/latest/index.html) — **MLflow** — the four components (Tracking, Projects, Models, Registry) in one page.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 6 "Model Development & Offline Evaluation"**](https://huyenchip.com/mlops/) — **Chip Huyen** — experiment tracking and versioning in context (author notes free).
- [Machine Learning Engineering — **Ch. 5 "Supervised Model Training"** (experiment management)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [01 ML Lifecycle & MLOps Maturity](01-ML-Lifecycle-and-MLOps-Maturity.md) · [02 Reproducibility](02-Reproducibility.md)
- Next concepts: [04 Data & Model Versioning](04-Data-and-Model-Versioning.md) · [13 Model Registry & Governance](13-Model-Registry-and-Governance.md)
