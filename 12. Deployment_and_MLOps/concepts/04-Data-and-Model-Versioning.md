---
id: "12-mlops/data-and-model-versioning"
topic: "Data & Model Versioning (DVC · lakeFS)"
parent: "12-mlops-and-deployment"
level: intermediate
prereqs: ["reproducibility", "git"]
interview_frequency: high
updated: 2026-06-20
---

# Data & Model Versioning — DVC · lakeFS
> Git for data and models: track large datasets and model artifacts by content hash, store the bytes in
> object storage, and keep only lightweight pointers in Git — so any commit reconstructs the exact
> data + model that produced a result. Reproducibility's missing half (Git versions code, not 10 GB of data).

**Why it matters:** the "Git can't hold your training set — so how do you version data?" question.
Expect the DVC model (pointer files + remote cache + content-addressable storage), why this differs
from versioning code, data lineage (which dataset version → which model), and the difference between
file-level tools (DVC) and data-lake/branching tools (lakeFS). Pairs with experiment tracking for full reproducibility.

**⭐ Start here — suggested path:**

1. **Get the idea** — read [DVC: Get Started](https://dvc.org/doc/start) and watch [Versioning Data with DVC](https://www.youtube.com/watch?v=kLKBcPonMYw). *The pointer-file + remote-cache model is the whole concept.*
2. **Version a real dataset** — follow [DVC: Data Management user guide](https://dvc.org/doc/user-guide/data-management). *`dvc add` / `dvc push` on a real dataset makes content-addressing concrete.*
3. **Tie data to models** — read [DVC: Versioning Data & Models scenario](https://dvc.org/doc/use-cases/versioning-data-and-models). *Switching dataset + model versions together is the lineage payoff.*
4. **Put it in a project** — follow [Made With ML: Versioning](https://madewithml.com/courses/mlops/versioning/). *Integrates DVC with code + experiment tracking for an end-to-end reproducible run.*
5. **See the data-lake approach** — read [lakeFS: Data Version Control](https://lakefs.io/data-version-control/). *Branch/commit/merge over object storage — the scale-out alternative to file-level DVC.*

## 🎓 Courses (free)
- [Made With ML — Versioning](https://madewithml.com/courses/mlops/versioning/) — **Goku Mohandas** — version code, data, and models together in a project.
- [DVC — Get Started (interactive docs)](https://dvc.org/doc/start) — **Iterative** — guided, hands-on intro to data + model versioning.

## 🎥 Videos
- [Versioning Data with DVC (Hands-On Tutorial!)](https://www.youtube.com/watch?v=kLKBcPonMYw) — **DVCorg** — official walkthrough of `dvc add/push/checkout`.
- [Data Version Control using DVC | DVC in MLOps](https://www.youtube.com/watch?v=XMnuZF53LAY) — **community** — clean from-scratch DVC + Git workflow.
- [Hands-on with DVC | Data Versioning in MLOps](https://www.youtube.com/watch?v=efnw2QvlhZM) — **community** — DVC inside a real project with remotes.
- [MLOps Zoomcamp 1.1 — Introduction](https://www.youtube.com/watch?v=s0uaFZSzwfI) — **DataTalks.Club** — frames versioning within the MLOps foundation.

## 📄 Key Papers
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (2015)** — "data dependencies cost more than code dependencies" — the case for data versioning.
- [Improving Reproducibility in ML Research](https://arxiv.org/abs/2003.12206) — **Pineau et al. (2020)** — pinning data versions as a reproducibility requirement.

## 📰 Articles / Blogs (free, no paywall)
- [DVC — Data Management](https://dvc.org/doc/user-guide/data-management) — **Iterative** — how content-addressed storage + remotes actually work.
- [DVC — Versioning Data & Models](https://dvc.org/doc/use-cases/versioning-data-and-models) — **Iterative** — the data-and-model lineage use case end to end.
- [lakeFS — Data Version Control](https://lakefs.io/data-version-control/) — **lakeFS** — git-like branch/commit/merge for data lakes at scale.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 6 "Model Development & Offline Evaluation"** (versioning & lineage)](https://huyenchip.com/mlops/) — **Chip Huyen** — author notes/talks free.
- [Machine Learning Engineering — **Ch. 3 "Data Collection & Preparation"** (data versioning)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [02 Reproducibility](02-Reproducibility.md) · [03 Experiment Tracking](03-Experiment-Tracking.md)
- Next concepts: [05 Feature Stores](05-Feature-Stores.md) · [13 Model Registry & Governance](13-Model-Registry-and-Governance.md)
- Related concept (covered elsewhere): data preprocessing & feature engineering → [02. Data_Preprocessing](../../02.%20Data_Preprocessing/concepts/README.md)
