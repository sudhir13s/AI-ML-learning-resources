---
id: "08-llms/instruction-tuning"
topic: "Instruction Tuning"
parent: "08-llms"
level: advanced
prereqs: ["supervised-fine-tuning", "prompting-and-in-context-learning"]
interview_frequency: high
updated: 2026-06-20
---

# Instruction Tuning
> A specific, powerful kind of SFT: fine-tune on a **diverse mix of tasks phrased as instructions**
> so the model learns to follow instructions in general — including ones it never saw. This is what
> unlocks zero-shot generalization (FLAN, T0) and, with self-generated data (Self-Instruct, Alpaca),
> made open instruction-following models cheap to build.

**Why it matters:** the "why can an LLM do tasks it was never trained on?" question. Be ready to
distinguish instruction tuning from plain SFT (task *diversity* and instruction *framing* are the
point), explain FLAN-style multi-task tuning, and describe Self-Instruct / Alpaca's synthetic-data loop.

**⭐ Start here — suggested path:**

1. **See where it fits** — watch [Karpathy: Deep Dive into LLMs](https://www.youtube.com/watch?v=7xTGNNLPyMI). *Instruction data is what makes the SFT model usable.*
2. **Read the Alpaca story** — [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html). *Self-Instruct + 52K examples → a capable instruction-follower cheaply.*
3. **Read the source** — [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652). *Multi-task instruction tuning → emergent zero-shot ability.*
4. **Go deeper** — [Self-Instruct](https://arxiv.org/abs/2212.10560) and [Scaling Instruction-Finetuned LMs (FLAN-T5/PaLM)](https://arxiv.org/abs/2210.11416). *Synthetic data + scaling the task mix.*
5. **Do it hands-on** — [Fine-tuning LLMs with example code](https://www.youtube.com/watch?v=eC6Hd1hFvos). *Build an instruction-tuned model yourself.*

## 🎓 Courses (free)
- [Hugging Face LLM Course — instruction tuning & alignment](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — instruction datasets and the `trl` workflow.
- [Stanford CS336 — Post-training & alignment](https://stanford-cs336.github.io/spring2025/) — **Stanford** — instruction tuning in the post-training pipeline.

## 🎥 Videos
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — instruction/conversation data in the alignment pipeline.
- [Fine-tuning Large Language Models (with example code)](https://www.youtube.com/watch?v=eC6Hd1hFvos) — **Shaw Talebi** — instruction-style SFT in code.
- [LLM Fine-Tuning Crash Course: 1-Hour End-to-End](https://www.youtube.com/watch?v=mrKuDK9dGlg) — **AI Anytime** — instruction dataset prep → training.
- [Chain-of-thought prompting — Explained!](https://www.youtube.com/watch?v=AFE6x81AP4k) — **CodeEmporium** — how instruction-tuned models follow reasoning instructions.

## 📄 Key Papers
- [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) — **Wei et al. (2021)** — instruction tuning → emergent zero-shot generalization.
- [Self-Instruct: Aligning LMs with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — **Wang et al. (2022)** — bootstrap instruction data from the model itself.
- [Scaling Instruction-Finetuned Language Models (FLAN-T5/PaLM)](https://arxiv.org/abs/2210.11416) — **Chung et al. (2022)** — scaling tasks, CoT data, and model size.

## 📰 Articles / Blogs (free, no paywall)
- [Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) — **Stanford CRFM** — the canonical cheap-instruction-tuning recipe.
- [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — **Sebastian Raschka** — instruction tuning vs plain SFT.
- [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — **Sebastian Raschka** — where instruction tuning fits across the key papers.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 12 "Model Alignment, Prompting & In-Context Learning"**](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — instruction tuning in the alignment chapter.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — §5 instruction tuning and alignment (free book-length reference).

## 🔗 In this platform
- Concept depth (the *why*): [Module 8.01 In-Context Learning & Prompting](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.01_In-Context_Learning_and_Prompting.md) · [Module 7.03 Transfer Learning & Fine-Tuning](../../../AI-ML-intuition/Module_7_Scaling_and_Adaptation/7.03_Transfer_Learning_and_Fine_Tuning.md)
- Related concepts: [Supervised Fine-Tuning](13-Supervised-Fine-Tuning.md) · [RLHF & DPO](15-RLHF-and-DPO.md) · [Prompting & In-Context Learning](16-Prompting-and-In-Context-Learning.md)
