---
id: "12-mlops/scaling-inference"
topic: "Scaling Inference (autoscaling · GPU · Ray Serve)"
parent: "12-mlops-and-deployment"
level: advanced
prereqs: ["model-serving", "distributed-systems"]
interview_frequency: high
updated: 2026-06-20
---

# Scaling Inference — Autoscaling · GPU · Ray Serve
> Meeting variable demand without over-paying: horizontally scale replicas, autoscale on the right
> signal (queue depth / latency, not just CPU), pack and share GPUs, and batch requests to raise
> throughput. The systems layer that turns a single prediction service into one that survives traffic spikes.

**Why it matters:** the "your model gets 10× traffic at peak — how do you scale it cost-effectively?"
question. Interviewers want horizontal vs vertical scaling, the right autoscaling metric (queue depth /
concurrency beats CPU% for inference), GPU specifics (cold starts, sharing/MIG, batching for utilization),
and tools like Ray Serve + Kubernetes HPA. The bridge between serving and cost.

**⭐ Start here — suggested path:**

1. **Frame the goal** — read [Considerations for Deploying ML Models in Production](https://www.anyscale.com/blog/considerations-for-deploying-machine-learning-models-in-production). *Throughput, latency, and cost as the scaling objective function.*
2. **Learn the autoscaling signal** — read [Ray Serve Autoscaling Guide](https://docs.ray.io/en/latest/serve/autoscaling-guide.html) and [Kubernetes Autoscaling](https://kubernetes.io/docs/concepts/workloads/autoscaling/). *Why queue depth/concurrency is the right metric for inference, not CPU%.*
3. **See it in a framework** — read [Ray Serve docs](https://docs.ray.io/en/latest/serve/index.html) and watch [Introducing Ray Serve](https://www.youtube.com/watch?v=gV4YS4e1CXg). *Replicas, autoscaling, and batching in one programmable layer.*
4. **Scale on GPUs** — watch [Productionizing ML at Scale with Ray Serve](https://www.youtube.com/watch?v=UtH-CMpmxvI). *GPU packing, batching, and multi-model serving under load.*
5. **Connect to cost** — move to [16 Cost Optimization](16-Cost-Optimization.md). *Autoscaling, batching, and GPU sharing are the main levers on the inference bill.*

## 🎓 Courses (free)
- [Ray Serve — Documentation](https://docs.ray.io/en/latest/serve/index.html) — **Anyscale** — scalable serving with built-in autoscaling and batching.
- [Made With ML — MLOps Course (serving & scaling)](https://madewithml.com/courses/mlops/) — **Goku Mohandas** — situates scaling in the full production workflow.

## 🎥 Videos
- [Introducing Ray Serve: Scalable & Programmable ML Serving](https://www.youtube.com/watch?v=gV4YS4e1CXg) — **Simon Mo (Anyscale)** — the scaling model: replicas, autoscaling, batching.
- [Introduction to Model Deployment with Ray Serve](https://www.youtube.com/watch?v=TdjJpAHLuxQ) — **Anyscale** — deploy and scale online inference.
- [Ray Serve: Patterns of ML Models in Production](https://www.youtube.com/watch?v=mM4hJLelzSw) — **Simon Mo (Anyscale)** — scaling composite/multi-model pipelines.
- [Productionizing ML at Scale with Ray Serve](https://www.youtube.com/watch?v=UtH-CMpmxvI) — **Anyscale** — GPU serving and scaling under real load.

## 📄 Key Papers
- [Challenges in Deploying ML: A Survey of Case Studies](https://arxiv.org/abs/2011.09926) — **Paleyes et al. (2020)** — scalability and resource failures in production ML.
- [MLOps: Continuous delivery and automation pipelines in ML](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning) — **Google Cloud** — infrastructure and scaling in the level-2 architecture.

## 📰 Articles / Blogs (free, no paywall)
- [Considerations for Deploying ML Models in Production](https://www.anyscale.com/blog/considerations-for-deploying-machine-learning-models-in-production) — **Anyscale** — throughput/latency/cost trade-offs that drive scaling.
- [Ray Serve — Autoscaling Guide](https://docs.ray.io/en/latest/serve/autoscaling-guide.html) — **Anyscale** — the right autoscaling signals for inference.
- [Kubernetes — Autoscaling Workloads](https://kubernetes.io/docs/concepts/workloads/autoscaling/) — **Kubernetes** — HPA/VPA fundamentals underneath ML autoscaling.

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 7 "Model Deployment"** & **Ch. 10 "Infrastructure & Tooling"** (scaling, GPUs)](https://huyenchip.com/mlops/) — **Chip Huyen** — author notes/talks free.
- [Machine Learning Engineering — **Ch. 8 "Model Deployment"** (scaling & throughput)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [06 ML Pipelines & Orchestration](06-ML-Pipelines-and-Orchestration.md) · [09 Model Serving](09-Model-Serving.md)
- Next concepts: [16 Cost Optimization](16-Cost-Optimization.md)
- Related concept (covered elsewhere): LLM inference optimization (quantization, paged attention, batching) → [08. LLMs — Inference Optimization & Serving](../../08.%20LLMs/concepts/09-Inference-Optimization-and-Serving.md)
