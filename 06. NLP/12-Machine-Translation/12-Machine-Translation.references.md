---
id: "06-nlp/machine-translation/references"
topic: "Machine Translation — References"
parent: "06-nlp/machine-translation"
type: references
updated: 2026-06-27
---

# Machine Translation — references and further reading

> Companion link library for **[Machine Translation](12-Machine-Translation.md)** (the concept page). This file holds the curated links — external sources *and* internal links to related pages on this platform — kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity, and every link verified.

**Start here — suggested path**:
1. **Build intuition** — watch [Seq2Seq Encoder–Decoder Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=L8HKweZIOmg) (**StatQuest, Josh Starmer**). *The translation pipeline, gently, before any attention.*
2. **See attention fix MT** — read ⭐ [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) (**Jay Alammar**), then watch [Stanford CS224N Lec 7 — Translation, Seq2Seq, Attention](https://www.youtube.com/watch?v=wzfWHP6SXxY). *Watch the single-vector bottleneck appear and attention remove it.*
3. **Get the math (and the metric)** — read [SLP3 Ch. 13 "Machine Translation"](https://web.stanford.edu/~jurafsky/slp3/13.pdf) (**Jurafsky & Martin**). *Encoder–decoder MT, beam search, and BLEU, rigorously and free.*
4. **Understand evaluation** — read the original [BLEU paper](https://aclanthology.org/P02-1040/) (**Papineni et al. 2002**) and the [sacreBLEU "Call for Clarity"](https://aclanthology.org/W18-6319/) (**Post 2018**). *The metric that ran the field, and why you must report it carefully.*
5. **Make it concrete** — run the [opus-mt-fr-en](https://huggingface.co/Helsinki-NLP/opus-mt-fr-en) model and the from-scratch BLEU in this chapter's [notebook](code/12-Machine-Translation.ipynb), or the [TensorFlow NMT-with-attention tutorial](https://www.tensorflow.org/text/tutorials/nmt_with_attention). *Translate, then score.*

**Videos**:
- [Seq2Seq Encoder–Decoder Neural Networks, Clearly Explained](https://www.youtube.com/watch?v=L8HKweZIOmg) — **StatQuest (Josh Starmer)** — the gentlest possible MT intuition.
- [Stanford CS224N Lec 7 — Translation, Seq2Seq, Attention](https://www.youtube.com/watch?v=wzfWHP6SXxY) — **Stanford (Chris Manning / John Hewitt)** — the rigorous NMT + attention + BLEU lecture, with the beam-search and evaluation discussion.
- [Bleu Score (C5W3L06)](https://www.youtube.com/watch?v=DejHQYAGb7Q) — **DeepLearning.AI (Andrew Ng)** — modified n-gram precision and the brevity penalty, built up clearly; the canonical short BLEU explainer.
- [Transformer Neural Networks — EXPLAINED! (Attention is all you need)](https://www.youtube.com/watch?v=TQQlZhbC5ps) — **CodeEmporium** — the architecture that replaced RNN-MT.
- [Encoder–decoder architecture: Overview](https://www.youtube.com/watch?v=zbdong_h-x4) — **Google Cloud Tech** — concise official framing of the seq2seq translation pipeline.

**Courses (free)**:
- [NLP Course for You — Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — **Lena Voita** — the MT chapter, written by a translation researcher, with interactive figures.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — the full NMT + attention lecture and notes.

**Articles / blogs (free, no paywall)**:
- [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) — **Jay Alammar** — the definitive visual explainer of seq2seq + attention for MT.
- [Seq2seq and Attention](https://lena-voita.github.io/nlp_course/seq2seq_and_attention.html) — **Lena Voita** — free, rigorous, MT-focused treatment with diagrams.
- [Neural Machine Translation with attention (TensorFlow tutorial)](https://www.tensorflow.org/text/tutorials/nmt_with_attention) — **TensorFlow** — build and train an NMT model end to end.
- [OPUS-MT — open neural translation models](https://github.com/Helsinki-NLP/Opus-MT) — **Helsinki-NLP (Tiedemann et al.)** — the open MT models (incl. the `opus-mt-fr-en` used on the concept page) and how they're trained.
- [sacreBLEU](https://github.com/mjpost/sacrebleu) — **Matt Post** — the standard, tokenization-fixed BLEU/chrF/TER implementation; read the README on *why* raw BLEU is not comparable. (The from-scratch BLEU/chrF in this chapter's code is verified against it.)
- [IBM Model 1 and 2 — lecture notes](http://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/ibm12.pdf) — **Michael Collins (Columbia)** — the clearest standalone derivation of the EM word-alignment update used on the concept page.
- [Two minutes NLP — BLEU, METEOR, chrF, and friends](https://medium.com/nlplanet/two-minutes-nlp-machine-translation-metrics-bleu-meteor-chrf-and-others-3a5e7d2c5d8c) — **NLPlanet (Fabio Chiusano)** — a quick side-by-side of the MT metrics on the surface→meaning spectrum (free Medium).

**Key papers**:
- [The Mathematics of Statistical Machine Translation: Parameter Estimation (IBM Models)](https://aclanthology.org/J93-2003/) — **Brown et al. (1993)** — the noisy-channel model, word alignment, and IBM Models 1–5 via EM; the source of the alignment math on this page.
- [Statistical Phrase-Based Translation](https://aclanthology.org/N03-1017/) — **Koehn, Och & Marcu (2003)** — the phrase-based SMT that powered production MT for a decade.
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — **Bahdanau, Cho & Bengio (2015)** — attention for translation; the bottleneck fix.
- [Google's Neural Machine Translation (GNMT)](https://arxiv.org/abs/1609.08144) — **Wu et al. (2016)** — production-scale NMT, WordPiece, the ~60% error reduction over phrase-based SMT, and the **length-normalized beam search** ($\alpha\approx0.6$) the decoding section derives.
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — **Vaswani et al. (2017)** — the Transformer, introduced *for* translation; the origin of every modern LLM.
- [Neural Machine Translation of Rare Words with Subword Units (BPE)](https://arxiv.org/abs/1508.07909) — **Sennrich, Haddow & Birch (2016)** — BPE: a finite vocabulary that spells any word, the foundation of shared multilingual vocabularies.
- [Improving NMT Models with Monolingual Data (Back-Translation)](https://arxiv.org/abs/1511.06709) — **Sennrich, Haddow & Birch (2016)** — synthetic parallel data from monolingual text; the low-resource workhorse.
- [Google's Multilingual NMT: Enabling Zero-Shot Translation](https://arxiv.org/abs/1611.04558) — **Johnson et al. (2017)** — one model, shared vocab, a target-language token, and an emergent interlingua enabling zero-shot pairs.
- [BLEU: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040/) — **Papineni et al. (2002)** — the n-gram-overlap metric that defined MT evaluation for 20 years; modified precision, brevity penalty, and geometric mean are all derived here (and re-derived from scratch on this page).
- [A Call for Clarity in Reporting BLEU Scores (sacreBLEU)](https://aclanthology.org/W18-6319/) — **Post (2018)** — *why* raw BLEU is not comparable across tokenizations, and the standard, signed implementation everyone should report.
- [chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — **Popović (2015)** — BLEU's idea over character n-grams; kinder to morphology and valid variation, usually a better human-correlation than BLEU.
- [METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments](https://aclanthology.org/W05-0909/) — **Banerjee & Lavie (2005)** — stem/synonym/paraphrase-aware alignment with a fragmentation penalty.
- [COMET: A Neural Framework for MT Evaluation](https://aclanthology.org/2020.emnlp-main.213/) — **Rei et al. (2020)** — the learned, meaning-aware metric that credits valid paraphrases BLEU misses; the current field standard.
- [BLEURT: Learning Robust Metrics for Text Generation](https://aclanthology.org/2020.acl-main.704/) — **Sellam, Das & Parikh (2020)** — a learned, BERT-based regression metric for translation quality, the other modern learned metric alongside COMET.
- [Correcting Length Bias in Neural Machine Translation](https://aclanthology.org/W18-6322/) — **Murray & Chiang (2018)** — the analysis of beam search's short-output bias that length normalization corrects.
- [Unsupervised Machine Translation Using Monolingual Corpora Only](https://arxiv.org/abs/1711.00043) — **Lample et al. (2018)** — translation from *no* parallel data via shared latent space, denoising, and iterative back-translation.
- [No Language Left Behind: Scaling Human-Centered MT (NLLB)](https://arxiv.org/abs/2207.04672) — **NLLB Team / Meta AI (2022)** — massively multilingual MT for 200 languages, the modern low-resource frontier.

**Books (free chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 13 "Machine Translation"](https://web.stanford.edu/~jurafsky/slp3/13.pdf) — **Jurafsky & Martin** — encoder–decoder MT, beam search, and BLEU, the canonical free textbook treatment.
- [Statistical Machine Translation](https://www.statmt.org/book/) — **Philipp Koehn** — the definitive SMT textbook (noisy channel, alignment, phrase-based decoding); essential for the statistical era.
- [Dive into Deep Learning — Ch. 10 (seq2seq, attention, NMT)](https://d2l.ai/chapter_recurrent-modern/index.html) — **Zhang, Lipton, Li & Smola** — runnable NMT code alongside the theory.

**In this platform**:
- Concept page (full explanation): [Machine Translation](12-Machine-Translation.md)
- Runnable code: [source-of-truth module](code/machine_translation.py) · [teaching notebook](code/12-Machine-Translation.ipynb) · [figure generator](code/make_figures_12.py) — from-scratch BLEU/chrF (matched to sacreBLEU), IBM Model 1 EM, beam length-normalization, and a real opus-mt-fr-en run.
- The architecture MT runs on: [08 Sequence-to-Sequence & Encoder–Decoder](../08-Sequence-to-Sequence-and-Encoder-Decoder/08-Sequence-to-Sequence-and-Encoder-Decoder.md)
- Foundations (the *why* behind attention / the Transformer): [Attention Mechanism](../../05.%20Deep_Learning/15-Attention-Mechanism/15-Attention-Mechanism.md) · [Transformer Architecture](../../05.%20Deep_Learning/16-Transformer-Architecture/16-Transformer-Architecture.md)
- Supporting techniques: [02 Tokenization & Subword Algorithms](../02-Tokenization-and-Subword-Algorithms/02-Tokenization-and-Subword-Algorithms.md) (BPE) · [17 Decoding Strategies](../17-Decoding-Strategies/17-Decoding-Strategies.md) (beam search) · [18 NLP Evaluation Metrics](../18-NLP-Evaluation-Metrics/18-NLP-Evaluation-Metrics.md) (BLEU)
- Related tasks: [13 Text Summarization](../13-Text-Summarization/13-Text-Summarization.md) · [14 Coreference Resolution](../14-Coreference-Resolution/14-Coreference-Resolution.md) (pronoun/gender resolution in document-level MT)
