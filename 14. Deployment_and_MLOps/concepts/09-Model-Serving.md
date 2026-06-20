---
id: "14-mlops/model-serving"
topic: "Model Serving (REST/gRPC · batch vs online · BentoML/Triton/TF-Serving)"
parent: "14-mlops-and-deployment"
level: advanced
prereqs: ["model-packaging-and-containerization", "apis"]
interview_frequency: very-high
updated: 2026-06-20
---

# Model Serving — REST/gRPC · Batch vs Online · BentoML · Triton · TF-Serving
> Exposing a model so applications can get predictions: wrap it behind an API (REST or gRPC), or run it
> over a dataset in bulk. The choice — **online** (low-latency, per-request) vs **batch/offline**
> (high-throughput, scheduled) — and the runtime (BentoML, Triton, TF-Serving) shape your whole
> production architecture.

**Why it matters:** the single most common deployment question. Interviewers want the online vs batch
trade-off (latency vs throughput vs cost), REST vs gRPC, request batching, why purpose-built servers
(Triton, TF-Serving, BentoML) beat a hand-rolled Flask app, and the prediction-service contract
(versioning, warmup, health checks). The core of "how is your model actually used."

**⭐ Start here — suggested path:**

1. **Get the split** — read [Google: Static vs Dynamic (batch vs online) Inference](https://developers.google.com/machine-learning/crash-course/production-ml-systems/static-vs-dynamic-inference). *The first decision; everything else follows from it.*
2. **See the patterns** — read [Serving ML Models in Production: Common Patterns](https://www.anyscale.com/blog/serving-ml-models-in-production-common-patterns). *Pipeline, ensemble, online/offline — the vocabulary of serving architectures.*
3. **Build a REST service** — work [Made With ML: API](https://madewithml.com/courses/mlops/api/), then watch [Getting Started with BentoML](https://www.youtube.com/watch?v=dnceIPcWJyA). *Model → versioned REST endpoint in minutes.*
4. **See the deployment modes** — watch [MLOps Zoomcamp: Three Ways of Deploying a Model](https://www.youtube.com/watch?v=JMGe4yIoBRA). *Web service vs streaming vs batch, side by side.*
5. **Meet the high-perf servers** — skim [Triton Inference Server](https://github.com/triton-inference-server/server) and [TF-Serving architecture](https://www.tensorflow.org/tfx/serving/architecture). *Dynamic batching, multi-framework, GPU — why these exist.*

## 🎓 Courses (free)
- [Made With ML — Serving / API](https://madewithml.com/courses/mlops/api/) — **Goku Mohandas** — design and ship a production prediction API.
- [BentoML — Documentation](https://docs.bentoml.com/en/latest/) — **BentoML** — turn any model into a REST/gRPC service with batching and packaging.

## 🎥 Videos
- [Getting Started with BentoML — Deploy Your First ML Model in 10 Minutes](https://www.youtube.com/watch?v=dnceIPcWJyA) — **community** — model → REST API with BentoML.
- [MLOps Zoomcamp 4.1 — Three Ways of Deploying a Model](https://www.youtube.com/watch?v=JMGe4yIoBRA) — **DataTalks.Club** — web service vs streaming vs batch.
- [Introduction to Model Deployment with Ray Serve](https://www.youtube.com/watch?v=TdjJpAHLuxQ) — **Anyscale** — programmable online serving and composition.
- [Ray Serve: Patterns of ML Models in Production](https://www.youtube.com/watch?v=mM4hJLelzSw) — **Simon Mo (Anyscale)** — serving patterns: pipelines, ensembles, business logic.

## 📄 Key Papers
- [Challenges in Deploying ML: A Survey of Case Studies](https://arxiv.org/abs/2011.09926) — **Paleyes et al. (2020)** — real serving/latency/scaling failures from industry.
- [MLOps: Continuous delivery and automation pipelines in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — **Google Cloud** — where the prediction service sits in the architecture.

## 📰 Articles / Blogs (free, no paywall)
- [Static vs Dynamic Inference (batch vs online)](https://developers.google.com/machine-learning/crash-course/production-ml-systems/static-vs-dynamic-inference) — **Google** — the foundational serving trade-off.
- [Serving ML Models in Production: Common Patterns](https://www.anyscale.com/blog/serving-ml-models-in-production-common-patterns) — **Anyscale** — the serving-architecture pattern catalog.
- [TF-Serving Architecture](https://www.tensorflow.org/tfx/serving/architecture) — **TensorFlow** — a production server's internals: servables, loaders, batching.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 7 "Model Deployment & Prediction Service"** (batch vs online, REST/gRPC)](https://huyenchip.com/mlops/) — **Chip Huyen** — author notes/talks free.
- [Machine Learning Engineering — **Ch. 8 "Model Deployment"** (serving patterns)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [08 Model Packaging & Containerization](08-Model-Packaging-and-Containerization.md)
- Next concepts: [10 Scaling Inference](10-Scaling-Inference.md) · [11 Model Monitoring & Observability](11-Model-Monitoring-and-Observability.md) · [14 A/B Testing · Shadow & Canary](14-AB-Testing-Shadow-and-Canary-Deployment.md)
- Related concept (covered elsewhere): LLM inference internals (KV-cache, paged attention) → [09. LLMs — Inference Optimization & Serving](../../09.%20LLMs/concepts/09-Inference-Optimization-and-Serving.md)
