---
id: "14-mlops/feature-stores"
topic: "Feature Stores (Feast)"
parent: "14-mlops-and-deployment"
level: advanced
prereqs: ["data-and-model-versioning", "feature-engineering"]
interview_frequency: medium
updated: 2026-06-20
---

# Feature Stores — Feast
> A central system that computes, stores, and serves ML features for both training (offline, historical,
> point-in-time correct) and inference (online, low-latency) — guaranteeing the **same** feature logic
> in both. The fix for train/serve skew and duplicated feature pipelines across teams.

**Why it matters:** the "how do you stop training and serving from computing features differently?"
question. Interviewers want the offline vs online split, **point-in-time correctness** (no future
leakage in training labels), feature reuse/discovery across teams, and where a feature store sits in
the architecture. Feast is the canonical open-source example.

**⭐ Start here — suggested path:**

1. **Understand the problem** — read [Feature Stores for ML](https://eugeneyan.com/writing/feature-stores/). *Train/serve skew and the offline/online duality — the "why" before any tool.*
2. **See the canonical design** — watch [Feature Store: The Beginner's Guide (+Feast)](https://www.youtube.com/watch?v=0InBN_a4cQQ). *Architecture: offline store, online store, registry, materialization.*
3. **Build one** — follow [Feast Quickstart](https://docs.feast.dev/getting-started/quickstart). *Define features, materialize to the online store, and fetch for inference — point-in-time joins click here.*
4. **Put it in a pipeline** — work [Made With ML: Feature Store](https://madewithml.com/courses/mlops/feature-store/). *Connects the store to training + serving in a real MLOps flow.*
5. **Survey the landscape** — browse [featurestore.org](https://www.featurestore.org/). *Where managed/open options differ, and when a feature store is overkill.*

## 🎓 Courses (free)
- [Made With ML — Feature Store](https://madewithml.com/courses/mlops/feature-store/) — **Goku Mohandas** — feature stores in a full production ML workflow.
- [Feast — Getting Started](https://docs.feast.dev/getting-started/quickstart) — **Feast** — guided, hands-on intro to defining and serving features.

## 🎥 Videos
- [Feature Store for Machine Learning: The Beginner's Guide (+Feast)](https://www.youtube.com/watch?v=0InBN_a4cQQ) — **community** — clear architecture + Feast walkthrough.
- [Feast: feature store for Machine Learning](https://www.youtube.com/watch?v=DaNv-Wf1MBA) — **community** — concepts and live demo of offline/online serving.
- [An AI Engineer Technical Guide to Feature Store with FEAST](https://www.youtube.com/watch?v=p2cuq4eJ2BY) — **Prodramp** — deeper, code-led tour of Feast internals.
- [Feast as Feature Store in Machine Learning — Live Demo](https://www.youtube.com/watch?v=iZ8R_EUf_pM) — **Ashutosh Tripathi** — end-to-end Python demo with point-in-time joins.

## 📄 Key Papers
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (2015)** — "data dependencies" and pipeline-jungle debt that feature stores exist to cut.
- [MLOps: Continuous delivery and automation pipelines in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — **Google Cloud** — where feature engineering/serving fits the automated lifecycle.

## 📰 Articles / Blogs (free, no paywall)
- [Feature Stores — A Hierarchy of Needs](https://eugeneyan.com/writing/feature-stores/) — **Eugene Yan** — the clearest free explainer of the problem and the design.
- [Feast Documentation](https://docs.feast.dev/) — **Feast** — concepts, architecture, and the offline/online split in depth.
- [Feature Store for ML (featurestore.org)](https://www.featurestore.org/) — **community** — vendor-neutral landscape, definitions, and patterns.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 5 "Feature Engineering"** & feature-store discussion](https://huyenchip.com/mlops/) — **Chip Huyen** — feature engineering, train/serve skew, and stores (author notes free).
- [Machine Learning Engineering — **Ch. 4 "Feature Engineering"**](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — feature lifecycle and serving; read-first chapters free.

## 🔗 In this platform
- Builds on: [04 Data & Model Versioning](../04-Data-and-Model-Versioning/04-Data-and-Model-Versioning.md)
- Next concepts: [06 ML Pipelines & Orchestration](../06-ML-Pipelines-and-Orchestration/06-ML-Pipelines-and-Orchestration.md) · [11 Model Monitoring & Observability](../11-Model-Monitoring-and-Observability/11-Model-Monitoring-and-Observability.md)
- Related concept (covered elsewhere): feature engineering theory → [02. Data_Preprocessing](../../02.%20Data_Preprocessing/README.md)
