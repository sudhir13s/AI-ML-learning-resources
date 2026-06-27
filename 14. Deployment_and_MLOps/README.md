---
id: "14-mlops-and-deployment"
topic: "MLOps & Deployment"
level: advanced
prereqs: ["tools-and-frameworks", "software-engineering"]
updated: 2026-06-27
---

# MLOps & Deployment
> Getting models into production and keeping them healthy — serving, pipelines, CI/CD,
> monitoring, and the systems discipline around ML.

**⭐ Start here:** [Made With ML](https://madewithml.com/) — **Goku Mohandas** — the best free, end-to-end MLOps course (design → develop → deploy → iterate).

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) — a short guided learning
path plus the best **free, open** courses, videos, papers, articles, and books for that topic.
> **✅ ready · ⬜ coming soon.** New here? Start with the field overview above, then work top to bottom.

### Foundations & lifecycle
1. ✅ [ML Lifecycle & MLOps Maturity](01-ML-Lifecycle-and-MLOps-Maturity/01-ML-Lifecycle-and-MLOps-Maturity.md)
2. ✅ [Reproducibility (seeds, environments, lineage)](02-Reproducibility/02-Reproducibility.md)
3. ✅ [Experiment Tracking (MLflow · Weights & Biases)](03-Experiment-Tracking/03-Experiment-Tracking.md)
4. ✅ [Data & Model Versioning (DVC · lakeFS)](04-Data-and-Model-Versioning/04-Data-and-Model-Versioning.md)

### Pipelines & automation
5. ✅ [Feature Stores (Feast)](05-Feature-Stores/05-Feature-Stores.md)
6. ✅ [ML Pipelines & Orchestration (Airflow · Kubeflow)](06-ML-Pipelines-and-Orchestration/06-ML-Pipelines-and-Orchestration.md)
7. ✅ [CI/CD for ML & Continuous Training (CT)](07-CICD-for-ML-and-Continuous-Training/07-CICD-for-ML-and-Continuous-Training.md)

### Packaging & serving
8. ✅ [Model Packaging & Containerization (Docker)](08-Model-Packaging-and-Containerization/08-Model-Packaging-and-Containerization.md)
9. ✅ [Model Serving (REST/gRPC · batch vs online · BentoML/Triton/TF-Serving)](09-Model-Serving/09-Model-Serving.md)
10. ✅ [Scaling Inference (autoscaling · GPU · Ray Serve)](10-Scaling-Inference/10-Scaling-Inference.md)

### Operations, monitoring & governance
11. ✅ [Model Monitoring & Observability](11-Model-Monitoring-and-Observability/11-Model-Monitoring-and-Observability.md)
12. ✅ [Data & Concept Drift Detection](12-Data-and-Concept-Drift-Detection/12-Data-and-Concept-Drift-Detection.md)
13. ✅ [Model Registry & Governance](13-Model-Registry-and-Governance/13-Model-Registry-and-Governance.md)
14. ✅ [A/B Testing · Shadow & Canary Deployment](14-AB-Testing-Shadow-and-Canary-Deployment/14-AB-Testing-Shadow-and-Canary-Deployment.md)

### LLMs & cost
15. ✅ [LLMOps (eval · guardrails · prompt versioning · cost/latency)](15-LLMOps/15-LLMOps.md)
16. ✅ [Cost Optimization for ML Systems](16-Cost-Optimization/16-Cost-Optimization.md)

### Related concepts (canonical home is another section)
> These topics have a canonical home elsewhere in the platform — linked here, not duplicated.
- **Online experimentation & A/B statistics theory** (hypothesis tests, power, CUPED) → [01. Foundations](../01.%20Foundations/README.md)
- **LLM inference internals** (KV-cache, quantization, paged attention, serving stacks) → [09. LLMs](../09.%20LLMs/README.md)
- **Data preprocessing & feature engineering** (cleaning, encoding, scaling, splits) → [02. Data_Preprocessing](../02.%20Data_Preprocessing/README.md)

## 🎓 Courses (free)
- [Made With ML](https://madewithml.com/) — **Goku Mohandas** — production ML + MLOps, code and reasoning.
- [Full Stack Deep Learning](https://fullstackdeeplearning.com/) — **FSDL** — free lectures on shipping ML products.

## 🎥 Videos
- [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) — **DataTalks.Club** — free, hands-on, project-based (experiment tracking → orchestration → monitoring → deployment).

## 📰 Articles / Reference
- [Rules of Machine Learning](https://developers.google.com/machine-learning/guides/rules-of-ml) — **Google** — 43 hard-won best practices for production ML.
- [Designing ML Systems — notes & talks](https://huyenchip.com/mlops/) — **Chip Huyen** — the practitioner's map of the MLOps landscape.

## 📚 Books
- [Designing Machine Learning Systems](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/) — **Chip Huyen** — the definitive modern text (paid, but chapters/notes are free online).

## 🔗 In this platform
- Inference economics: [AI-ML-intuition Module 7](../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/) · [LLM Systems curriculum](../llm_systems_curriculum.md)
