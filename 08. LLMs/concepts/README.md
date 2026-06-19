---
id: "08-llms/concepts"
topic: "LLMs — Concept Index"
parent: "08-llms"
level: advanced
updated: 2026-06-20
---

# LLMs — Concept Index
> Pick a concept to open its resource card — a short guided learning path plus the best **free, open**
> courses, videos, papers, articles, and books for that topic.
> **✅ ready · ⬜ coming soon.** New here? Start with the [field overview](../README.md).

## Pretraining & architecture

1. ✅ [Language Modeling Objectives (causal vs masked)](01-Language-Modeling-Objectives.md)
2. ✅ [Pretraining at Scale (data, compute, training dynamics)](02-Pretraining-at-Scale.md)
3. ✅ [Scaling Laws (Kaplan → Chinchilla)](03-Scaling-Laws.md)
4. ✅ [Decoder-only Architecture (the GPT family)](04-Decoder-only-Architecture.md)

## Efficient attention & inference

5. ✅ [KV Cache](05-KV-Cache.md)
6. ✅ [Efficient Attention (FlashAttention)](06-Efficient-Attention-FlashAttention.md)
7. ✅ [Mixture-of-Experts (MoE)](07-Mixture-of-Experts.md)
8. ✅ [Long-Context Methods (RoPE scaling, ALiBi, sparse/sliding)](08-Long-Context-Methods.md)
9. ✅ [Inference Optimization & Serving (vLLM · paged attention)](09-Inference-Optimization-and-Serving.md)

## Compression

10. ⬜ [Quantization (GPTQ · AWQ · GGUF)](10-Quantization.md)
11. ⬜ [Knowledge Distillation](11-Knowledge-Distillation.md)

## Adaptation & alignment

12. ⬜ [LoRA / PEFT (parameter-efficient fine-tuning)](12-LoRA-and-PEFT.md)
13. ⬜ [Supervised Fine-Tuning (SFT)](13-Supervised-Fine-Tuning.md)
14. ⬜ [Instruction Tuning](14-Instruction-Tuning.md)
15. ⬜ [RLHF & DPO (preference alignment)](15-RLHF-and-DPO.md)

## Prompting, reasoning & decoding

16. ⬜ [Prompting & In-Context Learning](16-Prompting-and-In-Context-Learning.md)
17. ⬜ [Chain-of-Thought Reasoning](17-Chain-of-Thought-Reasoning.md)
18. ⬜ [Decoding & Sampling for LLMs (temperature · top-k · top-p)](18-Decoding-and-Sampling.md)

## Evaluation & safety

19. ⬜ [LLM Evaluation & Benchmarks](19-LLM-Evaluation-and-Benchmarks.md)
20. ⬜ [Hallucination & Alignment basics](20-Hallucination-and-Alignment-Basics.md)

## Related concepts (covered in another section)
> These topics are foundations or applications of LLMs, but their canonical home is another section —
> linked here to avoid repetition.

- **Transformer architecture · Attention · Positional encoding** → [Deep Learning](../../05.%20Deep_Learning/concepts/README.md) ([Transformer](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) · [Attention](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md) · [Positional Encoding](../../05.%20Deep_Learning/concepts/17-Positional-Encoding.md))
- **Tokenization & subword (BPE/WordPiece) · Word & contextual embeddings (BERT)** → [NLP](../../06.%20NLP/concepts/README.md) ([Tokenization](../../06.%20NLP/concepts/02-Tokenization-and-Subword-Algorithms.md) · [Contextual Embeddings](../../06.%20NLP/concepts/06-Contextual-Embeddings-ELMo-BERT.md))
- **Retrieval-Augmented Generation (RAG)** → [RAG & LLM Applications](../../16.%20RAG_and_LLM_Applications/README.md)
- **PPO & policy-gradient mechanics** (the RL engine under RLHF) → [Reinforcement Learning](../../10.%20Reinforcement_Learning/README.md)
