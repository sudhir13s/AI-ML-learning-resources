---
id: "14-mlops/ml-pipelines-and-orchestration"
topic: "ML Pipelines & Orchestration (Airflow · Kubeflow)"
parent: "14-mlops-and-deployment"
level: advanced
prereqs: ["ml-lifecycle", "reproducibility"]
interview_frequency: high
updated: 2026-06-20
---

# ML Pipelines & Orchestration — Airflow · Kubeflow
> Turning an ad-hoc notebook into a reliable DAG of steps (ingest → validate → transform → train →
> evaluate → deploy) that an orchestrator schedules, retries, and monitors. The automation backbone of
> MLOps maturity level 1 — pipelines, not people, run the workflow.

**Why it matters:** the "how do you automate retraining / move from a notebook to a production pipeline?"
question. Expect the DAG model, idempotency and retries, why data pipelines differ from generic workflows
(data validation, lineage, caching), and the tool landscape: **Airflow** (general workflow scheduler) vs
**Kubeflow Pipelines** (Kubernetes-native, ML-specific) vs lighter options (Metaflow, Prefect).

**⭐ Start here — suggested path:**

1. **Get the levels** — read [ml-ops.org: Three Levels of ML Software](https://ml-ops.org/content/three-levels-of-ml-software). *Why pipelines (not scripts) are what "automation" means in MLOps.*
2. **Learn the DAG model** — watch [Learn Apache Airflow in 10 Minutes](https://www.youtube.com/watch?v=5peQThvQmQk). *DAGs, tasks, scheduling, retries — the orchestration vocabulary.*
3. **See the ML-native option** — read [Kubeflow Pipelines Overview](https://www.kubeflow.org/docs/components/pipelines/overview/) and watch [Intro to Kubeflow Pipelines](https://www.youtube.com/watch?v=_AY8mmbR1o4). *Containerized, reusable ML components on Kubernetes.*
4. **Build a real pipeline** — work [Made With ML: Orchestration](https://madewithml.com/courses/mlops/orchestration/). *Wires data → train → evaluate into a runnable, scheduled DAG.*
5. **Compare tools** — watch [Orchestrating ML with Airflow, Kubeflow & Prefect](https://www.youtube.com/watch?v=d2kFomZJJAc). *The decision criteria interviewers expect you to weigh.*

## 🎓 Courses (free)
- [Made With ML — Orchestration](https://madewithml.com/courses/mlops/orchestration/) — **Goku Mohandas** — build and schedule a real ML DAG.
- [Apache Airflow — Fundamentals Tutorial](https://airflow.apache.org/docs/apache-airflow/stable/index.html) — **Apache Airflow** — official docs + tutorial for authoring DAGs.

## 🎥 Videos
- [Learn Apache Airflow in 10 Minutes](https://www.youtube.com/watch?v=5peQThvQmQk) — **community** — crisp intro to DAGs, tasks, and scheduling.
- [Intro to Kubeflow Pipelines](https://www.youtube.com/watch?v=_AY8mmbR1o4) — **Google Cloud Tech** — ML-native pipelines, reusable components, metadata tracking.
- [Orchestrating ML with Airflow, Kubeflow & Prefect](https://www.youtube.com/watch?v=d2kFomZJJAc) — **Uplatz** — side-by-side of the major orchestrators for ML.
- [Machine Learning Pipelines with DVC (Hands-On)](https://www.youtube.com/watch?v=71IGzyH95UY) — **DVCorg** — lightweight, reproducible pipelines with DAG stages and caching.

## 📄 Key Papers
- [MLOps: Continuous delivery and automation pipelines in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — **Google Cloud** — the whitepaper defining ML pipeline automation (level 1).
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (2015)** — "pipeline jungles" and glue code that orchestration is meant to tame.

## 📰 Articles / Blogs (free, no paywall)
- [Three Levels of ML Software](https://ml-ops.org/content/three-levels-of-ml-software) — **ml-ops.org** — data/model/code pipelines and how they automate the lifecycle.
- [Kubeflow Pipelines — Overview](https://www.kubeflow.org/docs/components/pipelines/overview/) — **Kubeflow** — the ML-pipeline component model on Kubernetes.
- [Apache Airflow — Documentation](https://airflow.apache.org/docs/apache-airflow/stable/index.html) — **Apache** — DAGs, operators, scheduling, and executors.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 6 "Model Development"** & **Ch. 10 "Infrastructure & Tooling"** (orchestration)](https://huyenchip.com/mlops/) — **Chip Huyen** — author notes/talks free.
- [Machine Learning Engineering — **Ch. 7–8** (model deployment & pipelines)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [01 ML Lifecycle & MLOps Maturity](../01-ML-Lifecycle-and-MLOps-Maturity/01-ML-Lifecycle-and-MLOps-Maturity.md) · [05 Feature Stores](../05-Feature-Stores/05-Feature-Stores.md)
- Next concepts: [07 CI/CD for ML & CT](../07-CICD-for-ML-and-Continuous-Training/07-CICD-for-ML-and-Continuous-Training.md) · [10 Scaling Inference](../10-Scaling-Inference/10-Scaling-Inference.md)
