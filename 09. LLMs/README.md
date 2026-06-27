---
id: "09-llms"
topic: "Large Language Models"
level: advanced
prereqs: ["nlp", "deep-learning"]
updated: 2026-06-20
---

# Large Language Models (LLMs)
> Transformer language models at scale — pretraining, scaling laws, fine-tuning, alignment, and the
> systems that serve them. For *building* one from scratch, see the platform links below.

**⭐ Start here:** [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — the best 1-hour mental model of how LLMs work.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page, a curated
`.references.md` resource card (free, open courses · videos · papers · articles · books · cross-links),
and — for the gold demo chapter — a runnable notebook and `code/`.
> **✅ ready.** New here? Start with the field overview above, then work top to bottom.

### Pretraining & architecture
1. ✅ [Language Modeling Objectives (causal vs masked)](01-Language-Modeling-Objectives/01-Language-Modeling-Objectives.md)
2. ✅ [Pretraining at Scale (data, compute, training dynamics)](02-Pretraining-at-Scale/02-Pretraining-at-Scale.md)
3. ✅ [Scaling Laws (Kaplan → Chinchilla)](03-Scaling-Laws/03-Scaling-Laws.md)
4. ✅ [Decoder-only Architecture (the GPT family)](04-Decoder-only-Architecture/04-Decoder-only-Architecture.md)

### Efficient attention & inference
5. ✅ [KV Cache](05-KV-Cache/05-KV-Cache.md) — *gold demo chapter: page · [notebook](05-KV-Cache/05-KV-Cache.ipynb) · [code/](05-KV-Cache/code/)*
6. ✅ [Efficient Attention (FlashAttention)](06-Efficient-Attention-FlashAttention/06-Efficient-Attention-FlashAttention.md)
7. ✅ [Mixture-of-Experts (MoE)](07-Mixture-of-Experts/07-Mixture-of-Experts.md)
8. ✅ [Long-Context Methods (RoPE scaling, ALiBi, sparse/sliding)](08-Long-Context-Methods/08-Long-Context-Methods.md)
9. ✅ [Inference Optimization & Serving (vLLM · paged attention)](09-Inference-Optimization-and-Serving/09-Inference-Optimization-and-Serving.md)

### Compression
10. ✅ [Quantization (GPTQ · AWQ · GGUF)](10-Quantization/10-Quantization.md)
11. ✅ [Knowledge Distillation](11-Knowledge-Distillation/11-Knowledge-Distillation.md)

### Adaptation & alignment
12. ✅ [LoRA / PEFT (parameter-efficient fine-tuning)](12-LoRA-and-PEFT/12-LoRA-and-PEFT.md)
13. ✅ [Supervised Fine-Tuning (SFT)](13-Supervised-Fine-Tuning/13-Supervised-Fine-Tuning.md)
14. ✅ [Instruction Tuning](14-Instruction-Tuning/14-Instruction-Tuning.md)
15. ✅ [RLHF & DPO (preference alignment)](15-RLHF-and-DPO/15-RLHF-and-DPO.md)

### Prompting, reasoning & decoding
16. ✅ [Prompting & In-Context Learning](16-Prompting-and-In-Context-Learning/16-Prompting-and-In-Context-Learning.md)
17. ✅ [Chain-of-Thought Reasoning](17-Chain-of-Thought-Reasoning/17-Chain-of-Thought-Reasoning.md)
18. ✅ [Decoding & Sampling for LLMs (temperature · top-k · top-p)](18-Decoding-and-Sampling/18-Decoding-and-Sampling.md)

### Evaluation & safety
19. ✅ [LLM Evaluation & Benchmarks](19-LLM-Evaluation-and-Benchmarks/19-LLM-Evaluation-and-Benchmarks.md)
20. ✅ [Hallucination & Alignment basics](20-Hallucination-and-Alignment-Basics/20-Hallucination-and-Alignment-Basics.md)

### Related concepts (canonical home is another section)
> Foundations or applications of LLMs, linked here to avoid repetition.
- **Transformer · Attention · Positional encoding** → [Deep Learning](../05.%20Deep_Learning/concepts/README.md) ([Transformer](../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) · [Attention](../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Positional Encoding](../05.%20Deep_Learning/concepts/17-Positional-Encoding.md))
- **Tokenization & subword (BPE/WordPiece) · Contextual embeddings (BERT)** → [NLP](../06.%20NLP/README.md) ([Tokenization](../06.%20NLP/02-Tokenization-and-Subword-Algorithms/02-Tokenization-and-Subword-Algorithms.md) · [Contextual Embeddings](../06.%20NLP/06-Contextual-Embeddings-ELMo-BERT/06-Contextual-Embeddings-ELMo-BERT.md))
- **Retrieval-Augmented Generation (RAG)** → [RAG & LLM Applications](../11.%20RAG_and_LLM_Applications/README.md)
- **PPO & policy-gradient mechanics** (the RL engine under RLHF) → [Reinforcement Learning](../08.%20Reinforcement_Learning/concepts/README.md)

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
- Foundations: [Transformer Architecture](../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) · [Attention](../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Contextual Embeddings (BERT)](../06.%20NLP/06-Contextual-Embeddings-ELMo-BERT/06-Contextual-Embeddings-ELMo-BERT.md)
