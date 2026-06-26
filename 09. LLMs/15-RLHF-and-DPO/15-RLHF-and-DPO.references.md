---
id: "09-llms/rlhf-and-dpo/references"
topic: "RLHF & DPO — References"
parent: "09-llms/rlhf-and-dpo"
type: references
updated: 2026-06-22
---

# RLHF & DPO — references and further reading

> Companion link library for **[RLHF & DPO](15-RLHF-and-DPO.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [RLHF, Clearly Explained](https://www.youtube.com/watch?v=qPN_XZcJf_s) (**StatQuest**). *The 3-stage pipeline without the heavy math.*
2. **Read the illustrated guide** — [Illustrating RLHF](https://huggingface.co/blog/rlhf) (**Hugging Face**). *Reward model + PPO + KL penalty, visually.*
3. **Get the systems view** — read [RLHF and its alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) (**Sebastian Raschka**). *RLHF, DPO, and the preference-tuning landscape.*
4. **Read the sources** — [InstructGPT](https://arxiv.org/abs/2203.02155) then [DPO](https://arxiv.org/abs/2305.18290). *The full RLHF recipe, then the reward-model-free alternative.*
5. **Follow DPO's derivation** — watch [DPO math insight](https://www.youtube.com/watch?v=PZ6k5T5s5lY) (**Ricardo Calix**). *Why DPO and RLHF optimize the same objective.*

**Videos**:
- [Reinforcement Learning with Human Feedback (RLHF), Clearly Explained](https://www.youtube.com/watch?v=qPN_XZcJf_s) — **StatQuest (Josh Starmer)** — the gentlest correct overview of the pipeline.
- [RLHF explained with math derivations and PyTorch code](https://www.youtube.com/watch?v=qGyFrqc34yc) — **Umar Jamil** — reward model + PPO + KL, derived and coded line by line.
- [Direct Preference Optimization (DPO) — math insight explained](https://www.youtube.com/watch?v=PZ6k5T5s5lY) — **Ricardo Calix** — the DPO objective and its equivalence to RLHF.
- [DPO — fine-tune LLMs without reinforcement learning](https://www.youtube.com/watch?v=k2pD3k1485A) — **Luis Serrano** — DPO intuition, clearly.

**Courses (free)**:
- [Hugging Face — RLHF & DPO with `trl`](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — preference alignment in code (PPO and DPO trainers).
- [Stanford CS336 — Alignment](https://stanford-cs336.github.io/spring2025/) — **Stanford** — reward modeling, PPO, and DPO in the post-training stack.

**Articles / blogs (free, no paywall)**:
- [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf) — **Hugging Face** — the canonical illustrated explainer.
- [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html) — **Chip Huyen** — a clear, systems-minded walkthrough.
- [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) — **Sebastian Raschka** — RLHF, DPO, and the preference-tuning landscape.

**Key papers**:
- [Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons](https://www.jstor.org/stable/2334029) — **Bradley & Terry (1952)** — the 70-year-old pairwise-comparison model the reward-model loss is built on; $P(i \succ j) = \sigma(\text{strength gap})$.
- [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — **Christiano et al. (2017)** — the origin of learning a reward from *pairwise comparisons* of trajectories; the seed RLHF grew from.
- [Learning to Summarize from Human Feedback](https://arxiv.org/abs/2009.01325) — **Stiennon et al. (2020)** — the reward-model + PPO recipe applied to summarization; the bridge from Christiano to InstructGPT.
- [Training LMs to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — **Ouyang et al. (2022)** — the canonical 3-stage RLHF pipeline.
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — **Rafailov et al. (2023)** — preference alignment without a reward model or RL loop; the derivation in full.
- [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) — **Schulman et al. (2017)** — the clipped-surrogate RL algorithm RLHF uses to optimize the policy.
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760) — **Gao, Schulman & Hilton (2022)** — the Goodhart/over-optimization curve, measured.
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — **Bai et al. (2022)** — RLAIF: replace human labels with AI feedback against written principles.
- [A General Theoretical Paradigm to Understand Learning from Preferences (IPO)](https://arxiv.org/abs/2310.12036) — **Azar et al. (2023)** — fixes DPO's tendency to overfit the preference margin.
- [KTO: Model Alignment as Prospect Theoretic Optimization](https://arxiv.org/abs/2402.01306) — **Ethayarajh et al. (2024)** — alignment from unpaired good/bad labels.
- [ORPO: Monolithic Preference Optimization without a Reference Model](https://arxiv.org/abs/2403.07691) — **Hong, Lee & Thorne (2024)** — fold preference tuning into SFT, no reference model.
- [DeepSeekMath (GRPO)](https://arxiv.org/abs/2402.03300) — **Shao et al. (2024)** — critic-free group-relative policy optimization.
- [Llama 2](https://arxiv.org/abs/2307.09288) — **Touvron et al. (2023)** — a detailed, reproducible RLHF chat recipe (rejection sampling + PPO).

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 12 "Model Alignment, Prompting & In-Context Learning"](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — RLHF in the alignment chapter.
- [A Survey of Large Language Models — §5 Alignment Tuning](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — RLHF and its alternatives, book-length reference.

**In this platform**:
- Concept page (full explanation): [RLHF & DPO](15-RLHF-and-DPO.md)
- Hands-on project: [RLHF & Alignment workflow](../../../Practitioner-Workflows/RLHF-and-Alignment/RLHF-and-Alignment.md) (collect pairs → reward model → PPO → DPO, step by step)
- Concept depth (the *why*): [AI-ML-intuition 6.03 PPO and RLHF](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.03_PPO_and_RLHF.md) · [6.02 Policy Gradients / REINFORCE](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.02_Policy_Gradients_REINFORCE.md)
- Prerequisites: [Supervised Fine-Tuning](../13-Supervised-Fine-Tuning/13-Supervised-Fine-Tuning.md) · [Instruction Tuning](../14-Instruction-Tuning/14-Instruction-Tuning.md)
- RL foundations: [Reinforcement Learning](../../08.%20Reinforcement_Learning/concepts/README.md) (PPO & policy gradients)
- Related: [Hallucination & Alignment Basics](../20-Hallucination-and-Alignment-Basics/20-Hallucination-and-Alignment-Basics.md)
