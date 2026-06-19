---
id: "08-llms"
topic: "Large Language Models"
level: advanced
prereqs: ["nlp", "deep-learning"]
updated: 2026-06-20
---

# Large Language Models (LLMs)
> Transformer language models at scale — pretraining, scaling laws, fine-tuning, alignment, and the
> systems that serve them. For *building* one from scratch, see the platform links below.

**⭐ Start here:** [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — the best 1-hour mental model of how LLMs work.

## 📑 Per-concept resources
Looking for the best **free, open** resources on a *specific* LLM topic — scaling laws, KV cache,
LoRA, RLHF, decoding, serving? See the **[Concept Index →](concepts/README.md)**, where every owned
concept has its own guided resource card (courses, videos, papers, articles, books, cross-links).

## 🎓 Courses (free)
- [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — **Andrej Karpathy** — culminates in building & training a GPT from scratch.
- [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — fine-tuning, RLHF, deployment, free.
- [Stanford CS324 — Large Language Models](https://stanford-cs324.github.io/winter2022/) — **Stanford** — capabilities, harms, scaling, alignment; lecture notes free.
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) — **Stanford** — build an LLM end to end (data → train → align → serve).

## 🎥 Videos
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — pretraining → SFT → RLHF, end to end.
- [Let's build the GPT Tokenizer](https://www.youtube.com/watch?v=zduSFxRajkE) — **Karpathy** — BPE and why LLMs are weird at characters/math.
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — **3Blue1Brown** — the visual canon for the underlying architecture.
- [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU) — **Karpathy** — a full pretraining run, spelled out.

## 📄 Key Papers
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — **Brown et al. (2020)** — in-context learning emerges.
- [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — **Hoffmann et al. (2022)** — the scaling-law correction.
- [InstructGPT / RLHF](https://arxiv.org/abs/2203.02155) — **Ouyang et al. (2022)** — how chat models are aligned.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — the field map (pretraining → adaptation → use → eval).

## 📰 Articles / Blogs (free, no paywall)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — **Jay Alammar** — the visual canon for decoder-only LMs.
- [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — **Sebastian Raschka** — a curated path through the key papers.
- [Transformer Circuits / mechanistic interpretability](https://transformer-circuits.pub/2021/framework/index.html) — **Anthropic** — how transformers actually compute.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 10 "Large Language Models"**](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — the standard reference chapter, free PDF.

## 🔗 In this platform
- Build one: [project_06 ChatGPT-from-scratch](../../AI-ML-problemsets/projects/project_06_chatgpt_from_scratch/) · Intuition: [Module 8 — LLMs & Agentic Systems](../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/) · Systems: [LLM Systems curriculum](../llm_systems_curriculum.md)
- Foundations: [Transformer Architecture](../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) · [Attention](../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Contextual Embeddings (BERT)](../06.%20NLP/concepts/06-Contextual-Embeddings-ELMo-BERT.md)
