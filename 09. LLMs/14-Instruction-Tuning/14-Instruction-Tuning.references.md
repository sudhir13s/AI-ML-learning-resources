---
id: "09-llms/instruction-tuning/references"
topic: "Instruction Tuning — References"
parent: "09-llms/instruction-tuning"
type: references
updated: 2026-06-27
---

# Instruction Tuning — references and further reading

> Companion link library for **[Instruction Tuning](14-Instruction-Tuning.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity. All links are free / no-paywall.

**Start here — suggested path**:
1. **See where it fits** — watch [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) (**Andrej Karpathy**). *Instruction/SFT data is what turns a base model into something you can ask things of — the whole post-training pipeline in one talk.*
2. **Read the source** — [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) (**Wei et al. 2021**). *The result: multitask instruction tuning → zero-shot generalization to held-out task types.*
3. **Get the scaling story** — [Scaling Instruction-Finetuned Language Models (FLAN-T5)](https://arxiv.org/abs/2210.11416) (**Chung et al. 2022**). *More tasks, more diversity, and CoT data — what each buys.*
4. **See the cheap recipe** — [Stanford Alpaca](https://crfm.stanford.edu/2023/03/13/alpaca.html) + [Self-Instruct](https://arxiv.org/abs/2212.10560). *Bootstrap instruction data from the model itself → an open instruction-follower cheaply.*
5. **Do it hands-on** — [Fine-tuning LLMs with example code](https://www.youtube.com/watch?v=eC6Hd1hFvos) (**Shaw Talebi**) + this chapter's [notebook](code/14-Instruction-Tuning.ipynb). *Build and watch the generalization gap yourself.*

**Videos**:
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — the clearest tour of pretraining → instruction/SFT → RLHF, with instruction data's exact role.
- [Fine-tuning Large Language Models (with example code)](https://www.youtube.com/watch?v=eC6Hd1hFvos) — **Shaw Talebi** — instruction-style SFT end to end, in code.
- [LLM Fine-Tuning Crash Course: 1-Hour End-to-End](https://www.youtube.com/watch?v=mrKuDK9dGlg) — **AI Anytime** — instruction-dataset prep → training → eval, hands-on.
- [Stanford CS25 — Building Llama from scratch / instruction tuning](https://www.youtube.com/watch?v=rE7lTk7tlFY) — **Stanford CS25** — instruction tuning within a modern post-training stack.

**Interactive & visual**:
- [LLM Visualizer (3D)](https://bbycroft.net/llm) — **Brendan Bycroft** — walk a token through a small GPT's forward pass; grounds *what* the instruction-conditioned model is actually computing.
- [PromptSource / P3 templates](https://github.com/bigscience-workshop/promptsource) — **BigScience** — browse the public template/verbalizer library behind T0; *see* multiple phrasings per task.

**Courses (free)**:
- [Hugging Face LLM Course — instruction tuning & alignment](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — instruction datasets and the `trl` SFT workflow.
- [Stanford CS336 — Post-training & alignment](https://stanford-cs336.github.io/spring2025/) — **Stanford** — instruction tuning's place in the post-training pipeline.

**Articles / blogs (free, no paywall)**:
- [Stanford Alpaca: A Strong, Replicable Instruction-Following Model](https://crfm.stanford.edu/2023/03/13/alpaca.html) — **Stanford CRFM** — the canonical cheap instruction-tuning recipe (Self-Instruct + 52K examples).
- [Finetuning Large Language Models](https://magazine.sebastianraschka.com/p/finetuning-large-language-models) — **Sebastian Raschka** — instruction tuning vs plain SFT, data quality, gotchas.
- [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — **Sebastian Raschka** — where instruction tuning sits across the key papers.
- [The FLAN Collection: Designing Data and Methods for Effective Instruction Tuning](https://arxiv.org/abs/2301.13688) — **Longpre et al. (2023)** — the modern open instruction mix, with mix-design ablations (diversity, balancing, templates).

**Key papers**:
- [Finetuned Language Models Are Zero-Shot Learners (FLAN)](https://arxiv.org/abs/2109.01652) — **Wei et al. (2021)** — introduces instruction tuning; the task-count scaling curve (Fig. 5), the scale interaction (Fig. 6), and the cluster-level held-out protocol.
- [Multitask Prompted Training Enables Zero-Shot Task Generalization (T0)](https://arxiv.org/abs/2110.08207) — **Sanh et al. (2021)** — concurrent multitask-prompt result; the P3/PromptSource template library.
- [Super-NaturalInstructions: Generalization via Declarative Instructions on 1600+ Tasks](https://arxiv.org/abs/2204.07705) — **Wang et al. (2022)** — pushing the diversity axis hard (1,616 tasks, 76 task types).
- [Scaling Instruction-Finetuned Language Models (FLAN-T5/PaLM)](https://arxiv.org/abs/2210.11416) — **Chung et al. (2022)** — scaling task count to 1,836, the diversity-beats-count finding, and CoT data lifting reasoning.
- [Training Language Models to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — **Ouyang et al. (2022)** — §3 is the SFT/instruction-tuning stage; the staging that distinguishes it from RLHF.
- [Self-Instruct: Aligning LMs with Self-Generated Instructions](https://arxiv.org/abs/2212.10560) — **Wang et al. (2022)** — bootstrap instruction data from the model itself; the synthetic-data recipe behind Alpaca.
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206) — **Zhou et al. (2023)** — ~1,000 high-quality instruction examples can rival far larger noisy mixes; quality > quantity.
- [The FLAN Collection](https://arxiv.org/abs/2301.13688) — **Longpre et al. (2023)** — systematic study of *what makes an instruction mix effective* (mixing strategy, balancing, input inversion).

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 12 "Model Alignment, Prompting & In-Context Learning"](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — instruction tuning within the alignment chapter.
- [A Survey of Large Language Models — §5 Instruction Tuning & Alignment](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — book-length reference; the instruction-tuning section maps the dataset/method landscape.

**In this platform**:
- Concept page (full explanation): [Instruction Tuning](14-Instruction-Tuning.md)
- Reuses its loss from: [Supervised Fine-Tuning (SFT)](../13-Supervised-Fine-Tuning/13-Supervised-Fine-Tuning.md) — the masked next-token cross-entropy this chapter applies over a diverse mix.
- The next stage (different objective): [RLHF & DPO](../15-RLHF-and-DPO/15-RLHF-and-DPO.md) — preference optimization for helpfulness/harmlessness.
- Related concepts: [Prompting & In-Context Learning](../16-Prompting-and-In-Context-Learning/16-Prompting-and-In-Context-Learning.md) · [Chain-of-Thought Reasoning](../17-Chain-of-Thought-Reasoning/17-Chain-of-Thought-Reasoning.md) · [LoRA / PEFT](../12-LoRA-and-PEFT/12-LoRA-and-PEFT.md) (how instruction tuning is run cheaply)
- Language-modeling foundation: [Language Modeling Objectives](../01-Language-Modeling-Objectives/01-Language-Modeling-Objectives.md) — the next-token objective instruction tuning specializes
