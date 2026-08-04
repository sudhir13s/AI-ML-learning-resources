---
id: "18-mlops-and-deployment"
topic: "MLOps & Deployment"
level: advanced
built_from: ["tools-and-frameworks", "software-engineering"]
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
1. ✅ [ML Lifecycle & MLOps Maturity](lifecycle-and-reproducibility/ml-lifecycle-and-mlops-maturity/ml-lifecycle-and-mlops-maturity.md)
2. ✅ [Reproducibility (seeds, environments, lineage)](lifecycle-and-reproducibility/reproducibility/reproducibility.md)
3. ✅ [Experiment Tracking (MLflow · Weights & Biases)](lifecycle-and-reproducibility/experiment-tracking/experiment-tracking.md)
4. ✅ [Data & Model Versioning (DVC · lakeFS)](lifecycle-and-reproducibility/data-and-model-versioning/data-and-model-versioning.md)

### Pipelines & automation
5. ✅ [Feature Stores (Feast)](data-and-training-platforms/feature-stores/feature-stores.md)
6. ✅ [ML Pipelines & Orchestration (Airflow · Kubeflow)](data-and-training-platforms/ml-pipelines-and-orchestration/ml-pipelines-and-orchestration.md)
7. ✅ [CI/CD for ML & Continuous Training (CT)](release-and-deployment/cicd-for-ml-and-continuous-training/cicd-for-ml-and-continuous-training.md)

### Packaging & serving
8. ✅ [Model Packaging & Containerization (Docker)](packaging-and-serving/model-packaging-and-containerization/model-packaging-and-containerization.md)
9. ✅ [Model Serving (REST/gRPC · batch vs online · BentoML/Triton/TF-Serving)](packaging-and-serving/model-serving/model-serving.md)
10. ✅ [Scaling Inference (autoscaling · GPU · Ray Serve)](packaging-and-serving/scaling-inference/scaling-inference.md)

### Operations, monitoring & governance
11. ✅ [Model Monitoring & Observability](monitoring-and-reliability/model-monitoring-and-observability/model-monitoring-and-observability.md)
12. ✅ [Data & Concept Drift Detection](monitoring-and-reliability/data-and-concept-drift-detection/data-and-concept-drift-detection.md)
13. ✅ [Model Registry & Governance](governance-and-economics/model-registry-and-governance/model-registry-and-governance.md)
14. ✅ [A/B Testing · Shadow & Canary Deployment](release-and-deployment/ab-testing-shadow-and-canary-deployment/ab-testing-shadow-and-canary-deployment.md)

### LLMs & cost
15. ✅ [LLMOps (eval · guardrails · prompt versioning · cost/latency)](governance-and-economics/llmops/llmops.md)
16. ✅ [Cost Optimization for ML Systems](governance-and-economics/cost-optimization/cost-optimization.md)

### Related concepts (canonical home is another section)
> These topics have a canonical home elsewhere in the platform — linked here, not duplicated.
- **Online experimentation & A/B statistics theory** (hypothesis tests, power, CUPED) → [01. Foundations](../foundations/mathematical-foundations/README.md)
- **LLM inference internals** (KV-cache, quantization, paged attention, serving stacks) → [09. LLMs](../09.%20LLMs/README.md)
- **Data preprocessing & feature engineering** (cleaning, encoding, scaling, splits) → [02. Data_Preprocessing](../foundations/data-preparation/README.md)

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
