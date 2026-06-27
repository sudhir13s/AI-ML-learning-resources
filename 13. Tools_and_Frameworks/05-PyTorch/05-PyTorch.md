---
id: "13-tools-and-frameworks/pytorch"
topic: "PyTorch (tensors, autograd, nn, training loops)"
parent: "13-tools-and-frameworks"
level: intermediate
prereqs: ["python", "numpy", "neural-networks"]
interview_frequency: very-high
updated: 2026-06-20
---

# PyTorch — Tensors · Autograd · nn · Training Loops
> The dominant research and production deep-learning framework: GPU-accelerated tensors with a
> NumPy-like API, reverse-mode autodiff (`autograd`), the `nn.Module` building blocks, and the
> explicit Python training loop that makes it easy to read, debug, and extend.

**Why it matters:** PyTorch is the default for modern DL and the framework most interviews assume —
explaining autograd and the computation graph, writing a correct training loop (`zero_grad` →
`forward` → `loss` → `backward` → `step`), the train/eval mode distinction, and moving tensors
between CPU/GPU. It is also the substrate for Hugging Face, Lightning, and most LLM code.

**⭐ Start here — suggested path:**

1. **Do the 60-minute blitz** — [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html). *The official fast intro to tensors, autograd, and a first network.*
2. **Learn the basics properly** — work the [Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) series. *A clean, modern path through datasets, models, autograd, and the training loop.*
3. **Build it end to end on video** — [PyTorch for Deep Learning — Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA) (Daniel Bourke / freeCodeCamp). *Watching the workflow built from scratch cements the training-loop pattern.*
4. **Understand autograd deeply** — focus on how `requires_grad`, `.backward()`, and the dynamic graph work (covered in the basics + [docs](https://pytorch.org/docs/stable/index.html)). *Autograd is the most-asked PyTorch interview topic.*
5. **Go to mastery** — follow the free [Zero to Mastery PyTorch book/course](https://www.learnpytorch.io/). *A structured, project-driven path to real competence.*

## 🎓 Courses (free)
- [PyTorch official tutorials](https://pytorch.org/tutorials/) — **PyTorch team** — the authoritative library of tutorials, from the blitz to advanced topics.
- [Learn PyTorch for Deep Learning (Zero to Mastery)](https://www.learnpytorch.io/) — **Daniel Bourke** — a free, comprehensive online book + course with runnable notebooks.

## 🎥 Videos
- [PyTorch for Deep Learning & Machine Learning — Full Course](https://www.youtube.com/watch?v=V_xro1bcAuA) — **Daniel Bourke / freeCodeCamp** — the 25h beginner→capable course.
- [Learn PyTorch for deep learning in a day. Literally.](https://www.youtube.com/watch?v=Z_ikDlimN6A) — **Daniel Bourke** — a single-sitting, hands-on fundamentals course.
- [Deep Learning With PyTorch — Full Course](https://www.youtube.com/watch?v=c36lUUr864M) — **Patrick Loeber** — a concise, well-structured tour of the core API.
- [PyTorch 101 Crash Course for Beginners](https://www.youtube.com/watch?v=LyJtbe__2i0) — **Daniel Bourke / Zero To Mastery** — an up-to-date fast-start crash course.

## 📄 Key Papers
- [PyTorch: An Imperative Style, High-Performance Deep Learning Library](https://arxiv.org/abs/1912.01703) — **Paszke et al. (2019), NeurIPS** — the foundational paper on PyTorch's design.
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html) — **PyTorch team** — the canonical reference for tensors, autograd, and `nn`.

## 📰 Articles / Blogs (free, no paywall)
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) — **PyTorch team** — the fastest correct on-ramp.
- [PyTorch — Learn the Basics](https://pytorch.org/tutorials/beginner/basics/intro.html) — **PyTorch team** — the modern step-by-step fundamentals series.
- [Get Started (install + first run)](https://pytorch.org/get-started/locally/) — **PyTorch team** — the official install/setup guide.

## 📚 Books (free, with chapters)
- [Learn PyTorch for Deep Learning (online book)](https://www.learnpytorch.io/) — **Daniel Bourke** — a free, chapter-structured book with notebooks.
- [pytorch-deep-learning (course materials)](https://github.com/mrdbourke/pytorch-deep-learning) — **Daniel Bourke** — all notebooks and exercises, free on GitHub.

## 🔗 In this platform
- Related domain: [05. Deep Learning](../../05.%20Deep_Learning/README.md) · [09. LLMs](../../09.%20LLMs/README.md)
- Pairs with: [08 Hugging Face](../08-Hugging-Face/08-Hugging-Face.md) · [12 Weights & Biases](../12-Weights-and-Biases/12-Weights-and-Biases.md) · [09 ONNX](../09-ONNX-and-Model-Interchange/09-ONNX-and-Model-Interchange.md)
