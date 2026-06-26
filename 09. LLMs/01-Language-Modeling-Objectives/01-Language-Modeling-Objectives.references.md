---
id: "09-llms/language-modeling-objectives/references"
topic: "Language Modeling Objectives — References"
parent: "09-llms/language-modeling-objectives"
type: references
updated: 2026-06-26
---

# Language Modeling Objectives — references and further reading

> Companion link library for **[Language Modeling Objectives](01-Language-Modeling-Objectives.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity. All links are free / open, no paywall.

**Start here — suggested path**:
1. **Build the intuition** — watch [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) (**Andrej Karpathy**). *Frames the entire field around "predict the next token" — the objective, made the center of everything.*
2. **See the mechanism** — watch [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) (**3Blue1Brown**). *How context becomes a next-token probability distribution — the thing the loss scores.*
3. **Do the math** — read [Speech and Language Processing, Ch. 10](https://web.stanford.edu/~jurafsky/slp3/10.pdf) (**Jurafsky & Martin**). *The autoregressive factorization, cross-entropy, and perplexity you'll derive.*
4. **Contrast the two objectives** — read [The Illustrated BERT](https://jalammar.github.io/illustrated-bert/) (**Jay Alammar**) against [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/). *Same transformer, opposite objective and attention mask.*
5. **Code it** — watch [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) (**Andrej Karpathy**). *Implements the causal-LM loss, teacher forcing, and the causal mask line by line.*

**Videos**:
- [Intro to Large Language Models](https://www.youtube.com/watch?v=zjkBMFhNj_g) — **Andrej Karpathy** — the 1-hour mental model built entirely on next-token prediction; the best single framing of the objective.
- [Let's build GPT: from scratch, in code](https://www.youtube.com/watch?v=kCc8FmEb1nY) — **Andrej Karpathy** — implements the causal LM loss, the shift, teacher forcing, and the causal mask in PyTorch.
- [Transformers, the tech behind LLMs](https://www.youtube.com/watch?v=wjZofJX0v4M) — **3Blue1Brown** — visual: context → logits → next-token distribution; watch this if softmax-over-vocab still feels abstract.
- [Attention in transformers, step-by-step](https://www.youtube.com/watch?v=eMlx5fFNoYc) — **3Blue1Brown** — the causal mask that makes prediction autoregressive, drawn out.
- [Cross-Entropy, Clearly Explained](https://www.youtube.com/watch?v=6ArSys5qHAU) — **StatQuest (Josh Starmer)** — cross-entropy from first principles, the loss every objective on this page uses.

**Interactive & visual**:
- [LLM Visualizer (3D)](https://bbycroft.net/llm) — **Brendan Bycroft** — walk a single token through a small GPT's full forward pass and *see* the context turn into a next-token distribution.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) — **Polo Club (Georgia Tech)** — type text and watch GPT-2 produce the next-token probabilities live; the causal objective in motion.

**Courses (free)**:
- [Stanford CS324 — Large Language Models](https://stanford-cs324.github.io/winter2022/) — **Stanford** — defines the LM objective and exactly what it does and doesn't learn.
- [Stanford CS336 — Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) — **Stanford** — builds an LM end to end, objective included.
- [Hugging Face LLM Course — Ch. 1: Transformer models](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — causal vs masked vs seq2seq objectives, with code.

**Articles / blogs (free, no paywall)**:
- [The Illustrated GPT-2](https://jalammar.github.io/illustrated-gpt2/) — **Jay Alammar** — visualizes autoregressive next-token generation, the causal objective made pictures.
- [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) — **Jay Alammar** — the masked-LM objective and bidirectional encoding, side by side with causal.
- [Understanding Large Language Models](https://magazine.sebastianraschka.com/p/understanding-large-language-models) — **Sebastian Raschka** — situates causal vs masked objectives across the key papers, with the history of how causal won.
- [GPT in 60 Lines of NumPy](https://jaykmody.com/blog/gpt-from-scratch/) — **Jay Mody** — the forward pass and next-token loss with nothing hidden; the math on this page, in runnable code.
- [Perplexity for LLM Evaluation](https://huggingface.co/docs/transformers/en/perplexity) — **Hugging Face** — how perplexity is computed in practice (sliding window, the shift, padding), and its pitfalls.

**Key papers**:
- [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/v3/bengio03a.html) — **Bengio et al. (2003)** — the first neural language model; introduces the chain-rule factorization $p(x)=\prod_t p(x_t\mid x_{<t})$ learned by a neural network, the objective every modern LLM still trains on.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — the transformer; §3 has the per-position vocabulary projection the loss is computed over.
- [Improving Language Understanding by Generative Pre-Training (GPT)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — **Radford et al. (2018)** — the original causal-LM pre-training paper that started the line.
- [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — **Brown et al. (2020)** — the causal LM objective at scale; §2 covers the objective and training.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — **Devlin et al. (2018)** — the masked-LM objective (and the 80/10/10 recipe) plus next-sentence prediction.
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — **Liu et al. (2019)** — shows the masked-LM recipe matters: drop next-sentence prediction, train longer, and MLM gets much stronger.
- [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683) — **Raffel et al. (2020)** — the span-corruption objective and the text-to-text framing that unifies all three objectives.
- [ELECTRA: Pre-training Text Encoders as Discriminators](https://arxiv.org/abs/2003.10555) — **Clark et al. (2020)** — a more sample-efficient alternative to masked LM (replaced-token detection), directly addressing MLM's 15%-only inefficiency.
- [UL2: Unifying Language Learning Paradigms](https://arxiv.org/abs/2205.05131) — **Tay et al. (2022)** — the primary source for treating all three objectives (causal, masked, span) as one knob — a single "mixture-of-denoisers" pre-training recipe.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 3 "N-gram Language Models"](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — **Jurafsky & Martin** — §3.1 derives the chain-rule sequence factorization and §3.2 defines perplexity as exponentiated cross-entropy.
- [Speech and Language Processing, 3rd ed. — Ch. 10 "Large Language Models"](https://web.stanford.edu/~jurafsky/slp3/10.pdf) — **Jurafsky & Martin** — autoregressive LMs, the cross-entropy training objective, and perplexity, derived cleanly.
- [Deep Learning — Ch. 5 "Machine Learning Basics" (§5.5 Maximum Likelihood)](https://www.deeplearningbook.org/contents/ml.html) — **Goodfellow, Bengio & Courville** — the maximum-likelihood estimation that the cross-entropy / NLL training loss is an instance of.
- [Deep Learning — Ch. 6 "Deep Feedforward Networks" (§6.2.2 Output Units)](https://www.deeplearningbook.org/contents/mlp.html) — **Goodfellow, Bengio & Courville** — the softmax output unit that turns logits into a categorical distribution.
- [Deep Learning — Ch. 10 "Sequence Modeling"](https://www.deeplearningbook.org/contents/rnn.html) — **Goodfellow, Bengio & Courville** — the maximum-likelihood / sequence-modeling foundations the objective rests on.

**In this platform**:
- Concept page (full explanation): [Language Modeling Objectives](01-Language-Modeling-Objectives.md)
- Foundations (the *why* behind transformers and softmax): [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) · [Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md)
- The pre-neural baseline: [N-gram Language Models and Smoothing](../../06.%20NLP/concepts/04-N-gram-Language-Models-and-Smoothing.md) · the masked-LM encoders: [Contextual Embeddings (ELMo / BERT)](../../06.%20NLP/concepts/06-Contextual-Embeddings-ELMo-BERT.md)
- Builds on this: [Decoder-only Architecture](../04-Decoder-only-Architecture/04-Decoder-only-Architecture.md) · [Pretraining at Scale](../02-Pretraining-at-Scale/02-Pretraining-at-Scale.md) · [Scaling Laws](../03-Scaling-Laws/03-Scaling-Laws.md)
- Puts it to work: [Decoding and Sampling](../18-Decoding-and-Sampling/18-Decoding-and-Sampling.md) · [LLM Evaluation and Benchmarks](../19-LLM-Evaluation-and-Benchmarks/19-LLM-Evaluation-and-Benchmarks.md)
