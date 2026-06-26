---
id: "06-nlp/contextual-embeddings/references"
topic: "Contextual Embeddings — References"
parent: "06-nlp/contextual-embeddings"
type: references
updated: 2026-06-27
---

# Contextual Embeddings — references and further reading

> Companion link library for **[Contextual Embeddings (ELMo · BERT)](06-Contextual-Embeddings-ELMo-BERT.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity. Links verified 2026-06-22.

**Start here — suggested path**:
1. **Build intuition** — read [The Illustrated BERT, ELMo & co.](https://jalammar.github.io/illustrated-bert/) (**Jay Alammar**). *The clearest visual story of how contextual representations work — the one explainer to read first.*
2. **Watch it explained** — [BERT Neural Network — EXPLAINED!](https://www.youtube.com/watch?v=xI0HHN5XKDo) (**CodeEmporium**), then [What is BERT?](https://www.youtube.com/watch?v=7kLi8u2dJz0) (**codebasics**). *Two short, complementary takes on masked-LM and bidirectionality.*
3. **Place it in history** — [NLP's ImageNet Moment Has Arrived](https://www.ruder.io/nlp-imagenet/) (**Sebastian Ruder**). *Why pretrained contextual models reset the whole field.*
4. **Read the sources** — [ELMo](https://arxiv.org/abs/1802.05365) → [BERT](https://arxiv.org/abs/1810.04805). *Deep contextualized representations, then masked-LM deep bidirectionality.*
5. **Make it concrete** — [BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/) (**Chris McCormick**). *Extract real contextual vectors and inspect them, layer by layer — mirrors this page's code.*

**Videos**:
- [BERT Neural Network — EXPLAINED!](https://www.youtube.com/watch?v=xI0HHN5XKDo) — **CodeEmporium** — clear intuition for masked-LM and why bidirectional context matters.
- [What is BERT?](https://www.youtube.com/watch?v=7kLi8u2dJz0) — **codebasics** — gentle, example-driven first look.
- [BERT explained: training, inference, BERT vs GPT, fine-tuning, [CLS]](https://www.youtube.com/watch?v=90mGPxR2GgY) — **Umar Jamil** — a deep, careful walkthrough of exactly the topics on this page (MLM, [CLS], BERT vs GPT, fine-tuning).
- [Transformer Models and BERT Model: Overview](https://www.youtube.com/watch?v=hsp1OAcoLBY) — **Google Cloud** — concise official overview tying transformers to BERT.
- [BERT Explained!](https://www.youtube.com/watch?v=OR0wfP2FD3c) — **Connor Shorten** — walks through the paper's key ideas.
- [BERT: Pre-training of Deep Bidirectional Transformers (paper walkthrough)](https://www.youtube.com/watch?v=-9evrZnBorM) — **Yannic Kilcher** — a section-by-section read of the original BERT paper.

**Interactive & visual**:
- [BertViz — visualize attention in BERT](https://github.com/jessevig/bertviz) — **Jesse Vig** — an interactive notebook tool to *see* which tokens each head attends to, layer by layer; the most direct way to watch bidirectional attention build context.
- [A Visual Notebook to Using BERT for the First Time (Colab)](https://colab.research.google.com/github/jalammar/jalammar.github.io/blob/master/notebooks/bert/A_Visual_Notebook_to_Using_BERT_for_the_First_Time.ipynb) — **Jay Alammar** — a runnable, click-through notebook that loads BERT and extracts contextual features step by step.

**Courses (free)**:
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the contextual-representations + pretraining lectures (ELMo → BERT), the canonical academic treatment.
- [Hugging Face LLM Course — Ch. 1: Transformer Models](https://huggingface.co/learn/llm-course/chapter1/1) — **Hugging Face** — how pretrained encoders produce and use contextual representations, code-first.

**Articles / blogs (free, no paywall)**:
- [The Illustrated BERT, ELMo & co.](https://jalammar.github.io/illustrated-bert/) — **Jay Alammar** — the definitive visual explainer of contextual embeddings, ELMo, and BERT.
- [NLP's ImageNet Moment Has Arrived](https://www.ruder.io/nlp-imagenet/) — **Sebastian Ruder** — the historical context for why pretrained contextual models took over.
- [BERT Word Embeddings Tutorial](https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/) — **Chris McCormick** — hands-on extraction of contextual vectors, which layers to pool, subword handling (blog + Colab).
- [BERT 101 — State Of The Art NLP Model Explained](https://huggingface.co/blog/bert-101) — **Hugging Face** — a clean, modern walkthrough of BERT, MLM/NSP, and how to use it.
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — **Harvard NLP** — the encoder internals (self-attention, FFN, layer-norm) behind BERT, implemented line by line.

**Key papers**:
- [Deep Contextualized Word Representations (ELMo)](https://arxiv.org/abs/1802.05365) — **Peters et al. (2018)** — contextual vectors from a deep biLSTM LM and the learned per-task layer weighting.
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — **Devlin et al. (2018)** — masked-LM + NSP; the deep-bidirectional encoder and the pretrain→fine-tune paradigm shift.
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — **Liu et al. (2019)** — more data, no NSP, dynamic masking; "BERT was undertrained."
- [ALBERT: A Lite BERT for Self-supervised Learning](https://arxiv.org/abs/1909.11942) — **Lan et al. (2019)** — cross-layer parameter sharing, factorized embeddings, and SOP in place of NSP.
- [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — **He et al. (2020)** — disentangled content/position attention; topped many encoder leaderboards.
- [ELECTRA: Pre-training Text Encoders as Discriminators](https://arxiv.org/abs/2003.10555) — **Clark et al. (2020)** — replaced-token detection, far more sample-efficient than MLM.
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) — **Reimers & Gurevych (2019)** — why raw BERT sentence vectors fail, and the Siamese fix.
- [BERT Rediscovers the Classical NLP Pipeline](https://arxiv.org/abs/1905.05950) — **Tenney et al. (2019)** — layer probing: BERT learns POS→syntax→semantics bottom-to-top.
- [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108) — **Sanh et al. (2019)** — 40% smaller, ~97% of the quality via knowledge distillation.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 11 "Fine-Tuning and Masked Language Models"](https://web.stanford.edu/~jurafsky/slp3/11.pdf) — **Jurafsky & Martin** — contextual embeddings, BERT, MLM, and fine-tuning in the standard text.
- [Dive into Deep Learning — BERT: pretraining + fine-tuning](https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html) — **Zhang et al.** — BERT built from scratch with runnable code.

**In this platform**:
- Concept page (full explanation): [Contextual Embeddings (ELMo · BERT)](06-Contextual-Embeddings-ELMo-BERT.md)
- Runnable code: [step-by-step teaching notebook](code/06-Contextual-Embeddings-ELMo-BERT.ipynb) · [source-of-truth module](code/contextual_embeddings.py) · [figure generator](code/make_figures_06.py) — the same functions produce the page's numbers, the notebook, and every figure.
- The static limitation it fixes: [Word Embeddings (word2vec · GloVe · fastText)](../05-Word-Embeddings-Word2Vec-GloVe-FastText/05-Word-Embeddings-Word2Vec-GloVe-FastText.md)
- Builds on this: [Sentence & Document Embeddings (Sentence-BERT · USE)](../07-Sentence-and-Document-Embeddings/07-Sentence-and-Document-Embeddings.md) — the naive-`[CLS]` trap and its fix.
- Foundations: [Tokenization & Subword Algorithms](../02-Tokenization-and-Subword-Algorithms/02-Tokenization-and-Subword-Algorithms.md) · [Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) · [Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md)
- The decoder lineage: [KV Cache](../../09.%20LLMs/05-KV-Cache/05-KV-Cache.md) (why encoders don't need one) · the [LLMs concept index](../../09.%20LLMs/README.md)
- Concept depth (the *why* behind dense vectors): [AI-ML-intuition 1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md)
