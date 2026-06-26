---
id: "09-llms/llm-evaluation-and-benchmarks/references"
topic: "LLM Evaluation & Benchmarks — References"
parent: "09-llms/llm-evaluation-and-benchmarks"
type: references
updated: 2026-06-27
---

# LLM Evaluation & Benchmarks — references and further reading

> Companion link library for **[LLM Evaluation & Benchmarks](19-LLM-Evaluation-and-Benchmarks.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is free / open-access; every paper is linked by its arXiv ID. Sources cited in the page's `Source / derivation` blockquotes all appear in **Papers** (or **Books**) below.

**Start here — suggested path**:
1. **Get the landscape** — watch [What are LLM Benchmarks?](https://www.youtube.com/watch?v=kDY4TodQwbg) (**IBM Technology**). *MMLU, scoring, and what benchmarks do and don't measure.*
2. **Feel why it's hard** — watch [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) (**Andrej Karpathy**, eval section). *How models are evaluated across the whole pipeline, and where it breaks.*
3. **Do the perplexity math** — read [Speech and Language Processing, Ch. 3 §3.7](https://web.stanford.edu/~jurafsky/slp3/3.pdf) (**Jurafsky & Martin**). *Perplexity = inverse geometric-mean probability = exp(cross-entropy), derived.*
4. **See preference eval** — read [Chatbot Arena (LMSYS)](https://lmsys.org/blog/2023-05-03-arena/) (**LMSYS**). *Elo from human pairwise votes — the live leaderboard.*
5. **Read the judge biases** — [Judging LLM-as-a-Judge (MT-Bench)](https://arxiv.org/abs/2306.05685) (**Zheng et al. 2023**). *LLM judges, their biases, and ~80% human agreement.*
6. **Run an eval** — [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). *The standard tool behind the Open LLM Leaderboard.*

**Videos**:
- [What are Large Language Model (LLM) Benchmarks?](https://www.youtube.com/watch?v=kDY4TodQwbg) — **IBM Technology** — the clearest short intro to benchmarks, scoring, and their limits.
- [Deep Dive into LLMs like ChatGPT](https://www.youtube.com/watch?v=7xTGNNLPyMI) — **Andrej Karpathy** — the full pipeline including how models are evaluated and why benchmarks mislead.
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — capabilities/limits framing that motivates the whole evaluation portfolio.
- [Reinforcement Learning from Human Feedback (RLHF), Clearly Explained](https://www.youtube.com/watch?v=qPN_XZcJf_s) — **StatQuest (Josh Starmer)** — preference data and the Bradley–Terry reward model that Elo evaluation shares.
- [The Elo Rating System, Explained](https://www.youtube.com/watch?v=AsYfbmp0To0) — **Singing Banana (James Grime)** — the chess Elo update and the +400 ⇒ 10× odds convention behind Chatbot Arena.
- [Why Neural Networks Can Be Overconfident (Calibration)](https://www.youtube.com/watch?v=A3iVj9D8mLk) — **DeepFindr** — reliability diagrams and ECE, the calibration lens, visually.

**Interactive & visual**:
- [Chatbot Arena Leaderboard (live)](https://lmarena.ai/) — **LMArena (formerly LMSYS)** — the live pairwise-vote Elo leaderboard; watch ratings update from real human votes.
- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) — **Hugging Face** — the standard benchmark battery (MMLU-Pro, GPQA, MATH, IFEval, BBH, MuSR) run via the eval harness.
- [HELM live results](https://crfm.stanford.edu/helm/) — **Stanford CRFM** — the scenarios × metrics matrix; the clearest interactive demonstration of holistic (not single-number) evaluation.

**Courses (free)**:
- [Stanford CS324 — Large Language Models (Evaluation & Harms)](https://stanford-cs324.github.io/winter2022/) — **Stanford** — how LMs are measured and where benchmarks mislead.
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) — **Stanford** — evaluation within the full LLM build-and-measure stack.
- [Hugging Face — Evaluate library & the Evaluation chapter](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — metrics and a hands-on evaluation workflow.

**Articles / blogs (free, no paywall)**:
- [Chatbot Arena: Benchmarking LLMs in the Wild](https://lmsys.org/blog/2023-05-03-arena/) — **LMSYS** — crowdsourced Elo from pairwise votes, the most-watched preference leaderboard.
- [LLM-as-a-judge: a complete guide](https://www.evidentlyai.com/llm-guide/llm-as-a-judge) — **Evidently AI** — judge methodology and the position/verbosity/self-enhancement biases, with mitigations.
- [HELM: Holistic Evaluation of Language Models](https://crfm.stanford.edu/helm/) — **Stanford CRFM** — the multi-metric, multi-scenario philosophy in prose.
- [Perplexity of fixed-length models](https://huggingface.co/docs/transformers/en/perplexity) — **Hugging Face** — how perplexity is actually computed (sliding window, tokenizer caveats) with runnable code.
- [Evaluating LLMs is a minefield](https://www.cs.princeton.edu/~arvindn/talks/evaluating_llms_minefield/) — **Narayanan & Kapoor (Princeton)** — contamination, construct validity, and why headline scores mislead.
- [A Survey on Evaluation of Large Language Models (companion site)](https://llm-eval.github.io/) — **Chang et al.** — a structured map of what/how/where LLMs are evaluated.

**Key papers** (every paper cited in the page's `Source / derivation` blockquotes is here):
- [Evaluating Large Language Models Trained on Code (Codex / HumanEval)](https://arxiv.org/abs/2107.03374) — **Chen et al. (2021)** — introduces HumanEval and the **unbiased pass@k** estimator $1 - \binom{n-c}{k}/\binom{n}{k}$ (§2.1) the page derives.
- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — **Zheng et al. (2023)** — LLM judges, their position/verbosity/self-enhancement biases, and ~80% agreement with humans.
- [Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference](https://arxiv.org/abs/2403.04132) — **Chiang et al. (2024)** — Bradley–Terry / Elo on crowd-sourced pairwise votes; the source for the Arena rating math.
- [On Calibration of Modern Neural Networks](https://arxiv.org/abs/1706.04599) — **Guo et al. (2017)** — the binned **ECE** estimator and reliability diagrams; the over-confidence finding.
- [Measuring Massive Multitask Language Understanding (MMLU)](https://arxiv.org/abs/2009.03300) — **Hendrycks et al. (2020)** — the 57-subject multiple-choice knowledge benchmark.
- [Training Verifiers to Solve Math Word Problems (GSM8K)](https://arxiv.org/abs/2110.14168) — **Cobbe et al. (2021)** — the grade-school math reasoning benchmark.
- [HellaSwag: Can a Machine Really Finish Your Sentence?](https://arxiv.org/abs/1905.07830) — **Zellers et al. (2019)** — adversarially-filtered commonsense completion.
- [Think you have Solved Question Answering? Try ARC](https://arxiv.org/abs/1803.05457) — **Clark et al. (2018)** — the AI2 Reasoning Challenge (Easy/Challenge splits).
- [Beyond the Imitation Game (BIG-bench)](https://arxiv.org/abs/2206.04615) — **Srivastava et al. (2022)** — 200+ tasks probing capabilities beyond saturated benchmarks.
- [Holistic Evaluation of Language Models (HELM)](https://arxiv.org/abs/2211.09110) — **Liang et al. (2022)** — scenarios × metrics; the case against single-number reporting.
- [A Survey on Evaluation of Large Language Models](https://arxiv.org/abs/2307.03109) — **Chang et al. (2023)** — the what/how/where of LLM evaluation, including pitfalls.
- [A Survey of Large Language Models (§7 Evaluation)](https://arxiv.org/abs/2303.18223) — **Zhao et al. (2023)** — benchmarks, metrics, and contamination in one reference.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 3 "N-gram Language Models" §3.7 (Perplexity)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — **Jurafsky & Martin** — perplexity as inverse geometric-mean probability = exp(cross-entropy), derived from first principles.
- [Speech and Language Processing, 3rd ed. — Ch. 10 "Large Language Models"](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — autoregressive LM evaluation, cross-entropy, and intrinsic vs extrinsic metrics.

**Tools**:
- [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — **EleutherAI** — the standard open framework for running benchmark suites (powers the Open LLM Leaderboard).
- [OpenAI evals](https://github.com/openai/evals) — **OpenAI** — framework for writing and running custom evals, including LLM-as-judge graders.

**In this platform**:
- Concept page (full explanation): [LLM Evaluation & Benchmarks](19-LLM-Evaluation-and-Benchmarks.md)
- Foundations (the *why* behind the math): [NLP Evaluation Metrics (perplexity, BLEU, ROUGE, BERTScore)](../../06.%20NLP/concepts/18-NLP-Evaluation-Metrics.md) · [Information Theory: Entropy & KL Divergence](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Shares the math: [RLHF & DPO](../15-RLHF-and-DPO/15-RLHF-and-DPO.md) (the Bradley–Terry reward model is the same logistic as Elo)
- Related concepts: [Chain-of-Thought Reasoning](../17-Chain-of-Thought-Reasoning/17-Chain-of-Thought-Reasoning.md) (why GSM8K rewards reasoning) · [Decoding & Sampling](../18-Decoding-and-Sampling/18-Decoding-and-Sampling.md) (the sampling that pass@k accounts for) · [Hallucination & Alignment Basics](../20-Hallucination-and-Alignment-Basics/20-Hallucination-and-Alignment-Basics.md) (why calibration matters)
