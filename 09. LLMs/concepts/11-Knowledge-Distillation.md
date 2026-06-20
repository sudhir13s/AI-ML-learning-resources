---
id: "09-llms/knowledge-distillation"
topic: "Knowledge Distillation"
parent: "09-llms"
level: intermediate
prereqs: ["softmax", "cross-entropy", "language-modeling-objectives"]
interview_frequency: high
updated: 2026-06-20
---

# Knowledge Distillation
> Train a small **student** to mimic a large **teacher**. The trick: match the teacher's *soft*
> probability distribution (the "dark knowledge" in the logits), softened by a temperature, not just
> the hard labels. For LLMs this powers smaller chat models (DistilBERT, the *-Instruct* distillations)
> and increasingly **sequence-level** distillation where the student learns from teacher generations.

**Why it matters:** a classic ML question that's now central to LLM compression. Be ready to write the
KD loss (KL on softened logits + task loss), explain temperature and why soft targets carry more
information than hard labels, and contrast logit/response/feature distillation and online vs offline.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Knowledge Distillation in Neural Networks — Explained!](https://www.youtube.com/watch?v=BUCSTKQOzcM). *Teacher/student, soft targets, and temperature in plain terms.*
2. **See it in code** — [Knowledge Distillation: Full Tutorial with Code](https://www.youtube.com/watch?v=l44uC7jfnvY). *The KD loss implemented end to end.*
3. **Read the source** — [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531). *The original soft-target + temperature formulation.*
4. **Follow the PyTorch tutorial** — [PyTorch KD tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html). *Implement teacher→student distillation yourself.*
5. **Connect to LLM compression** — [Inference Optimization](09-Inference-Optimization-and-Serving.md) + [Quantization](10-Quantization.md). *Distillation, quantization, and pruning are the three compression levers.*

## 🎓 Courses (free)
- [Hugging Face — Knowledge distillation guide](https://huggingface.co/docs/transformers/en/llm_optims) — **Hugging Face** — distilling transformers, with code references.
- [Stanford CS336 — Model compression](https://stanford-cs336.github.io/spring2025/) — **Stanford** — distillation among efficiency techniques.

## 🎥 Videos
- [Knowledge Distillation in Neural Networks — Explained!](https://www.youtube.com/watch?v=BUCSTKQOzcM) — **CodeEmporium** — the cleanest conceptual intro.
- [Knowledge Distillation in ML: Full Tutorial with Code](https://www.youtube.com/watch?v=l44uC7jfnvY) — **MLWorks** — implementation walkthrough.
- [LoRA explained (precision and compression context)](https://www.youtube.com/watch?v=t509sv5MT0w) — **DeepFindr** — places distillation among compression methods.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — where distilled small models fit in the LLM landscape.

## 📄 Key Papers
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) — **Hinton, Vinyals & Dean (2015)** — soft targets + temperature; the founding paper.
- [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108) — **Sanh et al. (2019)** — 40% smaller, 60% faster, ~97% of BERT.
- [Sequence-Level Knowledge Distillation](https://arxiv.org/abs/1606.07947) — **Kim & Rush (2016)** — distilling on teacher *generations*, the LLM-relevant form.

## 📰 Articles / Blogs (free, no paywall)
- [Knowledge Distillation — PyTorch tutorial](https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html) — **PyTorch** — hands-on teacher→student with the KD loss.
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/) — **Lilian Weng** — distillation in the compression toolbox.
- [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) — **Sebastian Raschka** — where distillation fits in modern LLM training pipelines.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 12 "Computational Performance"**](https://d2l.ai/chapter_natural-language-processing-pretraining/index.html) — **Zhang et al.** — efficiency and compression foundations.
- [Deep Learning — **Ch. 7 "Regularization"**](https://www.deeplearningbook.org/) — **Goodfellow, Bengio & Courville** — soft targets as a form of teacher-guided regularization.

## 🔗 In this platform
- Concept depth (the *why*): [Module 7.04 Knowledge Distillation](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.04_Knowledge_Distillation.md)
- Related concepts: [Quantization](10-Quantization.md) · [Inference Optimization & Serving](09-Inference-Optimization-and-Serving.md) · [Supervised Fine-Tuning](13-Supervised-Fine-Tuning.md)
