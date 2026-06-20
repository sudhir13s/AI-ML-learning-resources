---
id: "11-tools-and-frameworks/onnx"
topic: "ONNX & Model Interchange (export, runtime, portability)"
parent: "11-tools-and-frameworks"
level: intermediate
prereqs: ["python", "pytorch", "neural-networks"]
interview_frequency: medium
updated: 2026-06-20
---

# ONNX & Model Interchange — Export · Runtime · Portability
> The **Open Neural Network Exchange**: a standard, framework-agnostic format for representing models
> as a computation graph, plus **ONNX Runtime**, a high-performance cross-platform inference engine.
> Train in PyTorch or TensorFlow, export to ONNX, and run anywhere — server, browser, mobile, edge —
> with graph optimizations and hardware acceleration.

**Why it matters:** model portability and inference performance are core deployment concerns, and
ONNX is the lingua franca that decouples training framework from serving target. Interviews and
real work touch on why you'd export to ONNX, how the graph + opsets work, ONNX Runtime's graph
optimizations (fusion, constant folding), and how it enables hardware-specific execution providers.

**⭐ Start here — suggested path:**

1. **Grasp the concept** — read [ONNX Concepts](https://onnx.ai/onnx/intro/concepts.html). *Understand the model-as-graph + operators/opsets idea — that's the whole format.*
2. **See an export end to end** — [Export a PyTorch model to ONNX](https://pytorch.org/tutorials/beginner/onnx/intro_onnx.html). *The canonical workflow: train → `torch.onnx.export` → run with ONNX Runtime.*
3. **Watch it explained** — [ONNX Explained with Example](https://www.youtube.com/watch?v=cZtXdMao7Ic). *A quick, concrete walkthrough of what ONNX is and does.*
4. **Learn the runtime** — read the [ONNX Runtime docs](https://onnxruntime.ai/docs/) and watch [Introduction to ONNX Runtime](https://www.youtube.com/watch?v=Wp5PaRpudlk). *The optimizations + execution providers are what make ONNX worth it in production.*
5. **Adapt a tutorial** — browse [onnx/tutorials](https://github.com/onnx/tutorials) for your framework. *Find the closest conversion example and adapt it.*

## 🎓 Courses (free)
- [ONNX documentation](https://onnx.ai/onnx/intro/concepts.html) — **ONNX team** — the official intro to the format, graphs, and operators.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — **ONNX Runtime team (Microsoft)** — the guide to optimizing and deploying ONNX models.
- [onnx/tutorials](https://github.com/onnx/tutorials) — **ONNX team** — a free collection of conversion and inference tutorials across frameworks.

## 🎥 Videos
- [ONNX Explained with Example](https://www.youtube.com/watch?v=cZtXdMao7Ic) — **Daniel Krei** — a quick, concrete intro to what ONNX is.
- [Introduction to ONNX Runtime](https://www.youtube.com/watch?v=Wp5PaRpudlk) — **NVIDIA Developer** — the runtime, optimizations, and accelerated inference.
- [TensorFlow 2.0 Complete Course](https://www.youtube.com/watch?v=tPYj3fFJGjk) — **freeCodeCamp** — context on a source framework you'd commonly export from.
- [PyTorch for Deep Learning — Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA) — **freeCodeCamp** — the other primary source framework for ONNX export.

## 📄 Key Papers
- [ONNX Concepts (format specification)](https://onnx.ai/onnx/intro/concepts.html) — **ONNX team** — the authoritative description of the interchange format.
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/) — **Microsoft** — the canonical reference for the inference engine and its optimizations.

## 📰 Articles / Blogs (free, no paywall)
- [ONNX home](https://onnx.ai/) — **ONNX team** — what ONNX is and the ecosystem around it.
- [Export a PyTorch model to ONNX](https://pytorch.org/tutorials/beginner/onnx/intro_onnx.html) — **PyTorch team** — the end-to-end export + run workflow.
- [ONNX Runtime home](https://onnxruntime.ai/) — **Microsoft** — the cross-platform, accelerated inference engine.

## 📚 Books (free, with chapters)
- [onnx/tutorials (collection)](https://github.com/onnx/tutorials) — **ONNX team** — a free, chapter-like set of conversion/inference guides.
- [ONNX Runtime documentation (full)](https://onnxruntime.ai/docs/) — **Microsoft** — a book-length, free guide to deployment and optimization.

## 🔗 In this platform
- Related domain: [12. Deployment & MLOps](../../12.%20Deployment_and_MLOps/concepts/README.md) · [05. Deep Learning](../../05.%20Deep_Learning/concepts/README.md)
- Pairs with: [05 PyTorch](05-PyTorch.md) · [06 TensorFlow & Keras](06-TensorFlow-and-Keras.md)
- Deeper concept (the *why*): model serving & optimization → [Deployment & MLOps](../../12.%20Deployment_and_MLOps/concepts/README.md)
