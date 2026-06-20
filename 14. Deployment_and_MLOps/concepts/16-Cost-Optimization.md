---
id: "14-mlops/cost-optimization"
topic: "Cost Optimization for ML Systems"
parent: "14-mlops-and-deployment"
level: advanced
prereqs: ["scaling-inference", "model-serving"]
interview_frequency: medium
updated: 2026-06-20
---

# Cost Optimization for ML Systems
> Delivering the required quality and latency at the lowest sustainable cost. The levers: right-size and
> share GPUs, batch requests, autoscale to demand (and scale to zero), use spot/preemptible capacity for
> training, cache and quantize for inference, and pick batch over online when latency allows. Compute is
> often the dominant line item — and the one engineers control most directly.

**Why it matters:** the "your inference bill is exploding — how do you cut it without hurting users?"
question, increasingly central for LLM systems. Interviewers want the cost drivers (GPU-hours, idle
capacity, token volume), the trade-offs (latency vs throughput vs cost vs quality), and concrete levers:
batching, autoscaling/scale-to-zero, spot instances, quantization, caching, and batch vs online. Ties
together serving, scaling, and LLMOps.

**⭐ Start here — suggested path:**

1. **Find the cost drivers** — read [Cost-Effective Machine Learning with Ray](https://www.anyscale.com/blog/cost-effective-machine-learning-with-ray). *Idle GPUs, over-provisioning, and where the money actually goes.*
2. **Lever 1 — batch vs online** — re-read [Static vs Dynamic Inference](https://developers.google.com/machine-learning/crash-course/production-ml-systems/static-vs-dynamic-inference). *Batch is dramatically cheaper when latency permits.*
3. **Lever 2 — autoscaling & sharing** — read [Ray Serve Autoscaling](https://docs.ray.io/en/latest/serve/autoscaling-guide.html). *Scale to demand (and to zero); pack/share GPUs to raise utilization.*
4. **Lever 3 — cheaper inference** — watch [Enabling Cost-Efficient LLM Serving with Ray Serve](https://www.youtube.com/watch?v=TJ5K1CO9Wbs). *Batching + quantization + right-sized hardware on real LLM workloads.*
5. **LLM-specific economics** — read [Building LLM Applications for Production](https://huyenchip.com/2023/04/11/llm-engineering.html) (cost/latency section). *Token cost, caching, model-size trade-offs for LLM systems.*

## 🎓 Courses (free)
- [Ray Serve — Documentation (autoscaling & batching)](https://docs.ray.io/en/latest/serve/index.html) — **Anyscale** — the main cost levers for serving, hands-on.
- [Made With ML — MLOps Course](https://madewithml.com/courses/mlops/) — **Goku Mohandas** — efficiency considerations across the production stack.

## 🎥 Videos
- [Enabling Cost-Efficient LLM Serving with Ray Serve](https://www.youtube.com/watch?v=TJ5K1CO9Wbs) — **Anyscale** — batching, autoscaling, and hardware choices to cut serving cost.
- [Optimizing LLM Inference with AWS Trainium, Ray, vLLM, and Anyscale](https://www.youtube.com/watch?v=tzzduslVzos) — **Anyscale** — cheaper accelerators + efficient serving stack.
- [Accelerated LLM Inference with Anyscale (Ray Summit 2024)](https://www.youtube.com/watch?v=_sDMsg0STqs) — **Anyscale** — throughput-per-dollar improvements for inference.
- [Productionizing ML at Scale with Ray Serve](https://www.youtube.com/watch?v=UtH-CMpmxvI) — **Anyscale** — GPU utilization and scaling efficiency under load.

## 📄 Key Papers
- [Challenges in Deploying ML: A Survey of Case Studies](https://arxiv.org/abs/2011.09926) — **Paleyes et al. (2020)** — resource/cost constraints as a recurring deployment challenge.
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (2015)** — infrastructure debt that quietly inflates cost.

## 📰 Articles / Blogs (free, no paywall)
- [Cost-Effective Machine Learning with Ray](https://www.anyscale.com/blog/cost-effective-machine-learning-with-ray) — **Anyscale** — practical cost levers: utilization, spot, autoscaling.
- [Static vs Dynamic Inference](https://developers.google.com/machine-learning/crash-course/production-ml-systems/static-vs-dynamic-inference) — **Google** — batch vs online as a cost decision.
- [Building LLM Applications for Production](https://huyenchip.com/2023/04/11/llm-engineering.html) — **Chip Huyen** — LLM cost/latency economics (caching, model size, tokens).

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 7 "Model Deployment"** & **Ch. 10 "Infrastructure & Tooling"** (cost & efficiency)](https://huyenchip.com/mlops/) — **Chip Huyen** — author notes/talks free.
- [Machine Learning Engineering — **Ch. 8 "Model Deployment"** (efficiency & resource trade-offs)](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — read-first chapters free.

## 🔗 In this platform
- Builds on: [09 Model Serving](09-Model-Serving.md) · [10 Scaling Inference](10-Scaling-Inference.md) · [15 LLMOps](15-LLMOps.md)
- Related concept (covered elsewhere): LLM inference cost techniques (quantization, KV-cache, batching) → [09. LLMs — Inference Optimization & Serving](../../09.%20LLMs/concepts/09-Inference-Optimization-and-Serving.md) · [Quantization](../../09.%20LLMs/concepts/10-Quantization.md)
