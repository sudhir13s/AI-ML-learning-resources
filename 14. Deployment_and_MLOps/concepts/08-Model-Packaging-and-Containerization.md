---
id: "14-mlops/model-packaging-and-containerization"
topic: "Model Packaging & Containerization (Docker)"
parent: "14-mlops-and-deployment"
level: intermediate
prereqs: ["reproducibility", "software-engineering"]
interview_frequency: high
updated: 2026-06-20
---

# Model Packaging & Containerization — Docker
> Bundling a model, its serving code, and every dependency into a portable, reproducible artifact —
> usually a Docker image — that runs identically on a laptop, in CI, and in production. The unit of
> deployment that makes "works on my machine" a non-problem.

**Why it matters:** "how do you ship a trained model to production?" Interviewers probe the artifact
(weights + code + env + interface), why containers solve dependency/parity problems, Dockerfile
fundamentals (layers, caching, slim/multi-stage builds, pinned base images), and ML-specific packaging
(model formats like ONNX/SavedModel, GPU images, image size). The bridge from "trained" to "served."

**⭐ Start here — suggested path:**

1. **Learn containers** — watch [Learn Docker in 7 Easy Steps](https://www.youtube.com/watch?v=gAkwW2tuIqE) and read [Docker: Get Started](https://docs.docker.com/get-started/). *Images, containers, Dockerfiles — the foundation.*
2. **Containerize a model** — watch [Convert your ML Model into a Docker Image](https://www.youtube.com/watch?v=JigSpm6KORI). *Wrap a model + API into an image you can run anywhere.*
3. **Do it the MLOps way** — work [Made With ML: Docker](https://madewithml.com/courses/mlops/docker/). *Packaging within a real serving workflow, with a clean interface.*
4. **Build production images** — read [Docker: Dockerfile best practices](https://docs.docker.com/build/building/best-practices/). *Layer caching, multi-stage builds, slim images — what separates a toy image from a deployable one.*
5. **Connect to serving** — move to [09 Model Serving](09-Model-Serving.md). *The image is the artifact a serving runtime runs; packaging and serving are two halves of deployment.*

## 🎓 Courses (free)
- [Made With ML — Docker](https://madewithml.com/courses/mlops/docker/) — **Goku Mohandas** — package an ML app into a reproducible image.
- [Docker — Get Started Guide](https://docs.docker.com/get-started/) — **Docker** — official, hands-on intro to images and containers.

## 🎥 Videos
- [Learn Docker in 7 Easy Steps — Full Beginner's Tutorial](https://www.youtube.com/watch?v=gAkwW2tuIqE) — **Fireship** — fast, complete mental model of Docker.
- [Docker Tutorial for Beginners — Full DevOps Course](https://www.youtube.com/watch?v=fqMOX6JJhGo) — **freeCodeCamp** — thorough run through images, volumes, compose.
- [How to Convert Your ML Model into a Docker Image](https://www.youtube.com/watch?v=JigSpm6KORI) — **community** — Python model → image, step by step.
- [Deploy Your ML Model Anywhere: Production-Ready AI Containers](https://www.youtube.com/watch?v=l1Qfy4zj21k) — **community** — containerizing an ML model + prediction API.

## 📄 Key Papers
- [Challenges in Deploying ML: A Survey of Case Studies](https://arxiv.org/abs/2011.09926) — **Paleyes et al. (2020)** — environment/dependency failures that containerization prevents.
- [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/file/86df7dcfd896fcaf2674f757a2463eba-Paper.pdf) — **Sculley et al. (2015)** — config and environment debt; packaging is part of the cure.

## 📰 Articles / Blogs (free, no paywall)
- [Dockerfile Best Practices](https://docs.docker.com/build/building/best-practices/) — **Docker** — layers, caching, multi-stage, slim images.
- [Docker — Get Started](https://docs.docker.com/get-started/) — **Docker** — the canonical reference for images/containers.
- [Cog — Containers for ML](https://cog.run/) — **Replicate** — opinionated, ML-specific packaging on top of Docker (GPU, weights, predict interface).

## 📚 Books (free, with chapters)
- [Designing Machine Learning Systems — **Ch. 7 "Model Deployment & Prediction Service"** (packaging & serving)](https://huyenchip.com/mlops/) — **Chip Huyen** — author notes/talks free.
- [Machine Learning Engineering — **Ch. 8 "Model Deployment"**](http://www.mlebook.com/wiki/doku.php) — **Andriy Burkov** — packaging and deployment patterns; read-first chapters free.

## 🔗 In this platform
- Builds on: [02 Reproducibility](02-Reproducibility.md)
- Next concepts: [09 Model Serving](09-Model-Serving.md) · [10 Scaling Inference](10-Scaling-Inference.md)
- Related concept (covered elsewhere): LLM-specific serving stacks → [09. LLMs](../../09.%20LLMs/concepts/README.md)
