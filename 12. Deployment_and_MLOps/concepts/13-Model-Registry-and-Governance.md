---
id: "12-mlops/model-registry-and-governance"
topic: "Model Registry & Governance"
parent: "12-mlops-and-deployment"
level: advanced
prereqs: ["experiment-tracking", "ml-lifecycle"]
interview_frequency: medium
updated: 2026-06-20
---

# Model Registry & Governance
> A central, versioned store for trained models with stage transitions (staging → production → archived),
> lineage, approvals, and metadata. Governance adds the controls around it: who can promote a model, what
> documentation it carries (model cards), audit trails, and compliance. The control plane between "trained"
> and "deployed."

**Why it matters:** "how do you manage which model version is in production, and roll back?" Interviewers
want the registry's role (single source of truth, stage transitions, lineage back to the run + data),
how it gates deployment (approvals, quality checks), and governance/responsible-AI concerns (model cards,
auditability, access control). Sits between experiment tracking and serving.

**⭐ Start here — suggested path:**

1. **Get the registry concept** — read [MLflow Model Registry](https://mlflow.org/docs/latest/ml/model-registry/). *Registered models, versions, stages, lineage — the core data model.*
2. **See the workflow** — watch [MLflow Model Registry: Staging vs Production vs Archived](https://www.youtube.com/watch?v=-aWL-c8ybm0). *Stage transitions and promotion — the deployment gate.*
3. **Do it hands-on** — watch [MLFlow Tutorial: Model Versioning & Registry](https://www.youtube.com/watch?v=iIiPo4qv97o). *Registering, versioning, and promoting a model in code.*
4. **Add governance** — read [Model Cards (paper)](https://arxiv.org/abs/1810.03993) and [Model Card Toolkit](https://modelcards.withgoogle.com/about). *Documentation, intended use, and limitations — the governance layer.*
5. **Production-readiness rubric** — read [The ML Test Score](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/). *The checklist a governance process should enforce before promotion.*

## 🎓 Courses (free)
- [MLflow — Model Registry docs](https://mlflow.org/docs/latest/ml/model-registry/) — **MLflow** — registry concepts, stages, and APIs.
- [Made With ML — MLOps Course](https://madewithml.com/courses/mlops/) — **Goku Mohandas** — model management within the production lifecycle.

## 🎥 Videos
- [MLflow Model Registry Explained — Staging vs Production vs Archived](https://www.youtube.com/watch?v=-aWL-c8ybm0) — **community** — the stage lifecycle and promotion flow.
- [MLFlow Tutorial — Model Versioning & Model Registry](https://www.youtube.com/watch?v=iIiPo4qv97o) — **Karndeep Singh** — hands-on registry usage.
- [MLflow Crash Course — Model Registry & Deployment](https://www.youtube.com/watch?v=bDflB17YUNc) — **community** — registry → deployment end to end.
- [MLflow Model Registry: A Complete Guide to Managing ML Models](https://www.youtube.com/watch?v=cUxRj3Jc0lQ) — **CodeKamikaze** — versioning, tags, and lifecycle management.

## 📄 Key Papers
- [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) — **Mitchell et al. (2019)** — the standard for documenting a model's intended use, performance, and limitations.
- [The ML Test Score: A Rubric for Production Readiness](https://research.google/pubs/the-ml-test-score-a-rubric-for-ml-production-readiness-and-technical-debt-reduction/) — **Breck et al. (Google, 2017)** — what to verify before promoting a model.
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (2015)** — undeclared consumers and versioning debt that governance addresses.

## 📰 Articles / Blogs (free, no paywall)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html) — **MLflow** — the canonical reference for registry workflows.
- [Model Card Toolkit](https://modelcards.withgoogle.com/about) — **Google** — practical model-card templates for governance.
- [Best Practices for ML on Google Cloud](https://cloud.google.com/architecture/ml-on-gcp-best-practices) — **Google Cloud** — model management, lineage, and governance practices.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 6 "Model Development"** & **Ch. 11 "The Human Side of ML"** (governance/responsible AI)](https://huyenchip.com/mlops/) — **Chip Huyen** — author notes/talks free.
- [Machine Learning Engineering — **Ch. 8–9** (deployment, versioning & maintenance)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [03 Experiment Tracking](03-Experiment-Tracking.md) · [04 Data & Model Versioning](04-Data-and-Model-Versioning.md)
- Next concepts: [14 A/B Testing · Shadow & Canary](14-AB-Testing-Shadow-and-Canary-Deployment.md) · [11 Model Monitoring & Observability](11-Model-Monitoring-and-Observability.md)
