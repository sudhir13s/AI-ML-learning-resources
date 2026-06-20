---
id: "08-llms/rlhf-and-dpo"
topic: "RLHF & DPO (preference alignment)"
parent: "08-llms"
level: advanced
prereqs: ["supervised-fine-tuning", "policy-gradients", "kl-divergence"]
interview_frequency: very-high
updated: 2026-06-20
---

# RLHF & DPO — Preference Alignment
> After SFT, align the model to *human preferences*. **RLHF**: train a reward model on pairwise
> preferences, then optimize the policy with **PPO** against that reward (plus a KL penalty to stay
> near the SFT model). **DPO**: skip the reward model and RL loop entirely — a clever reparameterization
> turns preference data into a simple classification-style loss. DPO is simpler and now widely used.

**Why it matters:** the alignment interview centerpiece. Be ready to draw the 3-stage RLHF pipeline
(SFT → reward model → PPO), explain the Bradley-Terry reward loss and the KL penalty, then derive
*why* DPO's closed-form objective is equivalent — and discuss reward hacking and over-optimization.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [StatQuest: RLHF Clearly Explained](https://www.youtube.com/watch?v=qPN_XZcJf_s). *The 3-stage pipeline without the heavy math.*
2. **Read the illustrated guide** — [HF: Illustrating RLHF](https://huggingface.co/blog/rlhf). *Reward model + PPO + KL penalty, visually.*
3. **Get the RL mechanics** — review [PPO & policy gradients](../../10.%20Reinforcement_Learning/concepts/README.md) (the RL engine under RLHF). *PPO is the optimizer doing the work.*
4. **Read the sources** — [InstructGPT/RLHF](https://arxiv.org/abs/2203.02155) then [DPO](https://arxiv.org/abs/2305.18290). *The full RLHF recipe, then the reward-model-free alternative.*
5. **See DPO's derivation** — [DPO: math insight explained](https://www.youtube.com/watch?v=PZ6k5T5s5lY). *Why DPO and RLHF optimize the same objective.*

## 🎓 Courses (free)
- [Hugging Face — RLHF & DPO with `trl`](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — preference alignment in code (PPO, DPO trainers).
- [Stanford CS336 — Alignment](https://stanford-cs336.github.io/spring2025/) — **Stanford** — reward modeling, PPO, and DPO in the post-training stack.

## 🎥 Videos
- [Reinforcement Learning with Human Feedback (RLHF), Clearly Explained](https://www.youtube.com/watch?v=qPN_XZcJf_s) — **StatQuest** — the gentlest correct overview.
- [RLHF explained with math derivations and PyTorch code](https://www.youtube.com/watch?v=qGyFrqc34yc) — **Umar Jamil** — reward model + PPO + KL, derived and coded.
- [Direct Preference Optimization (DPO) — math insight explained](https://www.youtube.com/watch?v=PZ6k5T5s5lY) — **Ricardo Calix** — the DPO objective and its equivalence to RLHF.
- [DPO — fine-tune LLMs without reinforcement learning](https://www.youtube.com/watch?v=k2pD3k1485A) — **Luis Serrano** — DPO intuition, clearly.

## 📄 Key Papers
- [Training LMs to Follow Instructions with Human Feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — **Ouyang et al. (2022)** — the canonical 3-stage RLHF pipeline.
- [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) — **Rafailov et al. (2023)** — preference alignment without a reward model or RL loop.
- [LLaMA-2](https://arxiv.org/abs/2307.09288) — **Touvron et al. (2023)** — a detailed, reproducible RLHF chat recipe (incl. rejection sampling + PPO).

## 📰 Articles / Blogs (free, no paywall)
- [Illustrating Reinforcement Learning from Human Feedback (RLHF)](https://huggingface.co/blog/rlhf) — **Hugging Face** — the canonical illustrated explainer.
- [RLHF: Reinforcement Learning from Human Feedback](https://huyenchip.com/2023/05/02/rlhf.html) — **Chip Huyen** — a clear, systems-minded walkthrough.
- [LLM Training: RLHF and Its Alternatives](https://magazine.sebastianraschka.com/p/llm-training-rlhf-and-its-alternatives) — **Sebastian Raschka** — RLHF, DPO, and the preference-tuning landscape.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 12 "Model Alignment, Prompting & In-Context Learning"**](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — RLHF in the alignment chapter.
- [A Survey of Large Language Models](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — §5 alignment tuning (RLHF + alternatives), free book-length reference.

## 🔗 In this platform
- Concept depth (the *why*): [Module 6.03 PPO and RLHF](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.03_PPO_and_RLHF.md) · [Module 6.02 Policy Gradients / REINFORCE](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.02_Policy_Gradients_REINFORCE.md)
- Foundations (covered elsewhere): [PPO & policy-gradient mechanics → Reinforcement Learning](../../10.%20Reinforcement_Learning/concepts/README.md)
- Related concepts: [Supervised Fine-Tuning](13-Supervised-Fine-Tuning.md) · [Instruction Tuning](14-Instruction-Tuning.md) · [Hallucination & Alignment Basics](20-Hallucination-and-Alignment-Basics.md)
