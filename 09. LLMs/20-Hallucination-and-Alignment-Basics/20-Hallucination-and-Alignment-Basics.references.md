---
id: "09-llms/hallucination-and-alignment-basics/references"
topic: "Hallucination & Alignment Basics — References"
parent: "09-llms/hallucination-and-alignment-basics"
type: references
updated: 2026-06-27
---

# Hallucination & Alignment Basics — references and further reading

> Companion link library for **[Hallucination & Alignment Basics](20-Hallucination-and-Alignment-Basics.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer, free / open-access, chosen for depth on *this* topic. Every formula on the concept page cites a source here.

**Start here — suggested path**:
1. **Build the intuition** — watch [Why Large Language Models Hallucinate](https://www.youtube.com/watch?v=cfqtFvWOfg0) (**IBM Technology**). *The clearest short intro to types, causes, and mitigations.*
2. **Get the incentive argument** — read [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) (**Kalai et al. 2025**). *Hallucination is the optimal response to a train/eval rule that punishes "I don't know."*
3. **Get the taxonomy & survey** — read [Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/) (**Lilian Weng**). *The definitive free survey: taxonomy, detection, mitigation.*
4. **See the alignment mechanism** — watch [RLHF, Clearly Explained](https://www.youtube.com/watch?v=qPN_XZcJf_s) (**StatQuest**) then read [InstructGPT](https://arxiv.org/abs/2203.02155) (**Ouyang et al. 2022**). *What preference tuning actually optimizes.*
5. **Connect the fixes** — [RLHF & DPO](../15-RLHF-and-DPO/15-RLHF-and-DPO.md) (preference tuning) + [RAG](../../11.%20RAG_and_LLM_Applications/concepts/README.md) (grounding) + [Decoding & Sampling](../18-Decoding-and-Sampling/18-Decoding-and-Sampling.md) (temperature). *The main levers, each its own chapter.*

**Videos**:
- [Why Large Language Models Hallucinate](https://www.youtube.com/watch?v=cfqtFvWOfg0) — **IBM Technology** — the clearest concise explainer of causes and mitigations; the best first watch.
- [Reinforcement Learning with Human Feedback (RLHF), Clearly Explained](https://www.youtube.com/watch?v=qPN_XZcJf_s) — **StatQuest (Josh Starmer)** — the alignment mechanism that reshapes honesty, built up from scratch.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — where hallucination and alignment sit in the full pretraining → SFT → RLHF pipeline (the "why models hallucinate" segment is excellent).
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — the "LLM OS" view plus the safety, jailbreak, and alignment discussion.
- [Direct Preference Optimization (DPO), explained](https://www.youtube.com/watch?v=hvGa5Mba4c8) — **Umar Jamil** — DPO's objective derived and coded line-by-line, the no-reward-model alignment route.
- [Aligning LLMs with Direct Preference Optimization](https://www.youtube.com/watch?v=QXVCqtAZAn4) — **DeepLearning.AI** — RLHF vs DPO in the alignment pipeline, with the preference-data framing.

**Interactive & visual**:
- [Has it Hallucinated? — TruthfulQA explorer](https://github.com/sylinrl/TruthfulQA) — **Lin et al.** — the benchmark's questions and answers; browse the misconception-triggering prompts that define factuality testing.
- [BBQ / safety-and-refusal demos in the HF Evaluate hub](https://huggingface.co/docs/evaluate/index) — **Hugging Face** — runnable faithfulness/factuality metrics (entailment, QA-based) you can apply to your own outputs.

**Courses (free)**:
- [Stanford CS324 — Large Language Models: Harms, safety & alignment](https://stanford-cs324.github.io/winter2022/) — **Stanford** — capabilities vs harms, alignment, and evaluation, in lecture form.
- [Stanford CS336 — Language Modeling from Scratch (alignment & safety)](https://stanford-cs336.github.io/spring2025/) — **Stanford** — alignment and post-training within the full LLM build.
- [Hugging Face — RLHF and alignment](https://huggingface.co/learn/nlp-course/chapter12/1) — **Hugging Face** — preference tuning, RLHF, and DPO with runnable examples.

**Articles / blogs (free, no paywall)**:
- [Extrinsic Hallucinations in LLMs](https://lilianweng.github.io/posts/2024-07-07-hallucination/) — **Lilian Weng (OpenAI)** — the definitive free survey: taxonomy (intrinsic/extrinsic, factuality/faithfulness), detection, and mitigation in one place.
- [Core Views on AI Safety: When, Why, What, and How](https://www.anthropic.com/research/core-views-on-ai-safety) — **Anthropic** — what "alignment" means, why it's hard, and how HHH frames the goal.
- [Detecting hallucinations in large language models using semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) — **Farquhar et al. (Nature 2024)** — open-access: cluster sampled answers by meaning and measure entropy *over meanings* — the strongest sampling-based hallucination detector.
- [Reducing hallucination in structured outputs via RAG](https://www.anthropic.com/research) — **Anthropic** — grounding and retrieval as the primary factuality lever (research index).
- [Measuring faithfulness: SummaC, AlignScore and the NLI approach](https://github.com/yuh-zha/AlignScore) — **Zha et al.** — code + explanation of entailment-based faithfulness scoring.

**Key papers** (every formula on the page cites one of these):
- [Why Language Models Hallucinate](https://arxiv.org/abs/2509.04664) — **Kalai et al. (2025)** — hallucination as the *statistically optimal* response to a binary scoring rule that punishes abstention; the incentive-misalignment argument.
- [Survey of Hallucination in Natural Language Generation](https://arxiv.org/abs/2202.03629) — **Ji et al. (2022)** — the taxonomy (intrinsic/extrinsic, factuality/faithfulness) + causes + metrics reference.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — §3.4 defines the softmax LM head ($p_i = e^{z_i}/\sum_j e^{z_j}$) the "softmax floor" argument rests on.
- [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) — **Holtzman et al. (2019)** — temperature/nucleus sampling; the basis for the $R(T)$ unsupported-claim-rate-vs-temperature analysis.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — **Guo et al. (2017)** — defines Expected Calibration Error (ECE) and documents systematic over-confidence (bars below the diagonal) — the reliability-diagram source.
- [Selective Classification for Deep Neural Networks](https://arxiv.org/abs/1705.08500) — **Geifman & El-Yaniv (2017)** — the risk–coverage framework behind abstention: trade coverage for accuracy by thresholding confidence.
- [How Language Model Hallucinations Can Snowball](https://arxiv.org/abs/2305.13534) — **Zhang et al. (2023)** — an early wrong token forces consistent (wrong) continuations; models assert claims they can separately flag as false.
- [TruthfulQA: Measuring How Models Mimic Human Falsehoods](https://arxiv.org/abs/2109.07958) — **Lin et al. (2021)** — the factuality benchmark probing resistance to common human misconceptions.
- [FActScore: Fine-grained Atomic Evaluation of Factual Precision](https://arxiv.org/abs/2305.14251) — **Min et al. (2023)** — decompose long-form output into atomic facts, score % supported by a knowledge source.
- [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155) — **Ouyang et al. (2022)** — §3 gives the KL-regularized RLHF reward objective on the page.
- [Learning to summarize from human feedback](https://arxiv.org/abs/2009.01325) — **Stiennon et al. (2020)** — the earlier RLHF formulation (reward model + KL-penalized PPO) the page's RLHF objective also draws on.
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — **Rafailov et al. (2023)** — §4 derives the DPO loss (the closed form behind the page's $\mathcal{L}_{\text{DPO}}$).
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — **Bai et al. (2022)** — RLAIF: align to a written constitution, scaling harmlessness with less human labeling.
- [A General Language Assistant as a Laboratory for Alignment](https://arxiv.org/abs/2112.00861) — **Askell et al. (2021)** — the Helpful/Harmless/Honest (HHH) framing the alignment section is built on.
- [Self-Consistency Improves Chain-of-Thought Reasoning](https://arxiv.org/abs/2203.11171) — **Wang et al. (2022)** — sample many reasoning paths, majority-vote; the sampling-based hallucination-reduction lever.
- [Chain-of-Verification Reduces Hallucination](https://arxiv.org/abs/2309.11495) — **Dhuliawala et al. (2023)** — generate → verify → revise; catches extrinsic errors grounding misses.
- [XSTest: Identifying Exaggerated Safety Behaviours](https://arxiv.org/abs/2308.01263) — **Röttger et al. (2023)** — the over-refusal benchmark: safe prompts that *look* unsafe, measuring the helpful side of the frontier.
- [Knowledge Conflicts for LLMs: A Survey](https://arxiv.org/abs/2403.08319) — **Xu et al. (2024)** — context-vs-parametric conflict: when a model overrides a correct retrieved passage with wrong memory.
- [AlignScore: Evaluating Factual Consistency with a Unified Alignment Function](https://arxiv.org/abs/2305.16739) — **Zha et al. (2023)** — NLI-style faithfulness scoring used to detect un-entailed (hallucinated) claims.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 10 "Large Language Models"](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — the softmax LM head and autoregressive sampling the hallucination argument starts from.
- [Speech and Language Processing, 3rd ed. — Ch. 12 "Model Alignment, Prompting & In-Context Learning"](https://web.stanford.edu/~jurafsky/slp3/) — **Jurafsky & Martin** — alignment, RLHF, and their failure modes.

**In this platform**:
- Concept page (full explanation): [Hallucination & Alignment Basics](20-Hallucination-and-Alignment-Basics.md)
- The mechanism of the fixes: [RLHF & DPO](../15-RLHF-and-DPO/15-RLHF-and-DPO.md) · [RAG & LLM Applications](../../11.%20RAG_and_LLM_Applications/concepts/README.md) · [Decoding & Sampling](../18-Decoding-and-Sampling/18-Decoding-and-Sampling.md) · [Chain-of-Thought Reasoning](../17-Chain-of-Thought-Reasoning/17-Chain-of-Thought-Reasoning.md)
- How it's measured: [LLM Evaluation & Benchmarks](../19-LLM-Evaluation-and-Benchmarks/19-LLM-Evaluation-and-Benchmarks.md)
- Where the disease starts: [Language Modeling Objectives](../01-Language-Modeling-Objectives/01-Language-Modeling-Objectives.md) · [Supervised Fine-Tuning](../13-Supervised-Fine-Tuning/13-Supervised-Fine-Tuning.md) · [Instruction Tuning](../14-Instruction-Tuning/14-Instruction-Tuning.md)
- Capstone context (the full arc): [KV Cache](../05-KV-Cache/05-KV-Cache.md) · [Quantization](../10-Quantization/10-Quantization.md) · [LoRA & PEFT](../12-LoRA-and-PEFT/12-LoRA-and-PEFT.md) · [Long-Context Methods](../08-Long-Context-Methods/08-Long-Context-Methods.md)
- Deeper RL grounding: [Module 6.03 PPO and RLHF](../../../AI-ML-intuition/Module_6_Reinforcement_Learning/6.03_PPO_and_RLHF.md)
