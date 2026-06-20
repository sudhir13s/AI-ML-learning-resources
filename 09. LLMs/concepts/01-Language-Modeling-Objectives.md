---
id: "09-llms/language-modeling-objectives"
topic: "Language Modeling Objectives (causal vs masked)"
parent: "09-llms"
level: intermediate
prereqs: ["softmax", "cross-entropy", "transformer-architecture"]
interview_frequency: very-high
updated: 2026-06-20
---

# Language Modeling Objectives — Causal vs Masked
> The self-supervised training signal behind every LLM: predict missing tokens from context.
> **Causal (autoregressive)** LMs predict the *next* token left-to-right (GPT); **masked** LMs predict
> *blanked-out* tokens from both sides (BERT). The objective you pick decides whether the model
> generates or understands.

**Why it matters:** the classic "why is GPT decoder-only and BERT encoder-only?" question. You'll be
asked to write the cross-entropy loss, explain teacher forcing, contrast next-token vs masked-token
prediction, and say why causal LMs (not masked) became the foundation for generative AI.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [Karpathy: Intro to LLMs](https://www.youtube.com/watch?v=zjkBMFhNj_g). *Frames the whole field around "predict the next token," the core objective.*
2. **See the mechanism** — [3Blue1Brown: Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M). *Shows how the network turns context into a next-token probability distribution.*
3. **Get the math** — read [SLP3 Ch. 10 "Large Language Models"](https://web.stanford.edu/~jurafsky/slp3/10.pdf). *The autoregressive factorization and cross-entropy / perplexity you'll derive.*
4. **Contrast the two** — skim the [BERT paper](https://arxiv.org/abs/1810.04805) §3.1 (masked LM) against the [GPT-3 paper](https://arxiv.org/abs/2005.14165) §2 (causal LM). *Same transformer, opposite objective and attention mask.*
5. **Make it concrete** — code next-token prediction with [Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY). *Implementing the loss cements teacher forcing and the causal mask.*

## 🎓 Courses (free)
- [Stanford CS324 — Lecture: Behavior & capabilities of LMs](https://stanford-cs324.github.io/winter2022/) — **Stanford** — defines the LM objective and what it does/doesn't learn.
- [Hugging Face LLM Course — Ch. 1: Transformer models](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — causal vs masked vs seq2seq objectives, with code.

## 🎥 Videos
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — the 1-hour mental model built entirely on next-token prediction.
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — **3Blue1Brown** — visual: context → logits → next-token distribution.
- [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **Andrej Karpathy** — implements the causal LM loss line by line.
- [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) — **3Blue1Brown** — the causal mask that makes prediction autoregressive.

## 📄 Key Papers
- [Improving Language Understanding by Generative Pre-Training (GPT)](https://arxiv.org/abs/2005.14165) — **Radford et al. / Brown et al. (2020)** — the causal LM objective at scale (GPT-3 paper, §2 covers the objective).
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — **Devlin et al. (2018)** — masked LM + next-sentence prediction.
- [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683) — **Raffel et al. (2020)** — span-corruption objective unifies the two under text-to-text.

## 📰 Articles / Blogs (free, no paywall)
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — **Jay Alammar** — visualizes autoregressive next-token generation.
- [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — **Sebastian Raschka** — situates causal vs masked objectives across the key papers.
- [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) — **Jay Mody** — the forward pass + next-token loss with nothing hidden.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 10 "Large Language Models"**](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — autoregressive LMs, training objective, perplexity.
- [Deep Learning — **Ch. 10 "Sequence Modeling"**](https://www.deeplearningbook.org/) — **Goodfellow, Bengio & Courville** — the probabilistic foundations of sequence modeling.

## 🔗 In this platform
- Concept depth (the *why*): [Module 5.05 Autoregressive Generation & Sampling](../../../AI-ML-intuition/Module_5_Generation/5.05_Autoregressive_Generation_Sampling.md) · [Module 5.01 Information Theory: Entropy & KL](../../../AI-ML-intuition/Module_5_Generation/5.01_Information_Theory_Entropy_KL_Divergence.md)
- Foundations (covered elsewhere): [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) · [Contextual Embeddings (ELMo/BERT)](../../06.%20NLP/concepts/06-Contextual-Embeddings-ELMo-BERT.md) · [N-gram Language Models](../../06.%20NLP/concepts/04-N-gram-Language-Models-and-Smoothing.md)
- Next concepts: [Decoder-only Architecture](04-Decoder-only-Architecture.md) · [Decoding & Sampling](18-Decoding-and-Sampling.md)
