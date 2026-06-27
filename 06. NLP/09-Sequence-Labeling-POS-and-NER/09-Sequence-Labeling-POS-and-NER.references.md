---
id: "06-nlp/sequence-labeling-pos-ner/references"
topic: "Sequence Labeling — References"
parent: "06-nlp/sequence-labeling-pos-ner"
type: references
updated: 2026-06-27
---

# Sequence Labeling — references and further reading

> Companion link library for **[Sequence Labeling — POS & NER](09-Sequence-Labeling-POS-and-NER.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is from a primary author or a recognized deep explainer — chosen for depth on *this* topic, not popularity.

**Start here — suggested path**:
1. **Frame the task** — read [SLP3 Ch. 17](https://web.stanford.edu/~jurafsky/slp3/17.pdf) intro on POS/NER and BIO tagging (**Jurafsky & Martin**). *Get the task and tagging schemes before any model.*
2. **See HMM + Viterbi** — watch [POS Tagging using the Viterbi algorithm](https://www.youtube.com/watch?v=QYzTTFxcc9I) (**Data Science in your pocket**), then the [fully worked numeric example](https://www.youtube.com/watch?v=OBemI2BapE0). *The classic generative tagger, decoded by hand.*
3. **Get the CRF** — read [SLP3 Ch. 17 §17.4](https://web.stanford.edu/~jurafsky/slp3/17.pdf) and Sutton & McCallum's [CRF tutorial](https://homepages.inf.ed.ac.uk/csutton/publications/crftutv2.pdf). *Why a globally-normalized CRF beats the MEMM — the part interviews probe.*
4. **Read the neural sources** — [biLSTM-CRF](https://arxiv.org/abs/1508.01991) → [Neural NER](https://arxiv.org/abs/1603.01360) → [end-to-end char-CNN-biLSTM-CRF](https://arxiv.org/abs/1603.01354). *How LSTMs + a CRF layer became the standard.*
5. **Make it concrete** — try [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities) or the [HF token-classification tutorial](https://huggingface.co/docs/transformers/en/tasks/token_classification), and score with [seqeval](https://github.com/chakki-works/seqeval). *Tag real text, inspect spans, compute entity F1.*

**Videos**:
- [POS Tagging using the Viterbi algorithm](https://www.youtube.com/watch?v=QYzTTFxcc9I) — **Data Science in your pocket** — HMM + Viterbi for POS, end to end.
- [POS Tagging, Viterbi Algorithm — Solved Problem](https://www.youtube.com/watch?v=OBemI2BapE0) — **Varsha's engineering stuff** — a fully worked numeric Viterbi trace, the kind asked in interviews.
- [POS Tagging, HMM, Viterbi — Emission & Transition matrices](https://www.youtube.com/watch?v=hjFbmosY9y4) — **Varsha's engineering stuff** — the transition/emission probability tables made explicit.
- [CS224N: Named Entity Recognition lecture](https://www.youtube.com/watch?v=8u8sAtbDImI) — **Stanford (Manning)** — NER framed within the deep-learning NLP course.

**Courses (free)**:
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — sequence-modeling and NER lectures grounding tagging tasks.
- [Hugging Face — Token classification chapter](https://huggingface.co/learn/nlp-course/chapter7/2) — **Hugging Face** — fine-tune a transformer for NER, with the subword-label alignment done right.
- [NLP Course for You — Text Classification (token-level models)](https://lena-voita.github.io/nlp_course/text_classification.html) — **Lena Voita** — discriminative models that extend to sequence labeling.

**Articles / blogs (free, no paywall)**:
- [Sequence Labeling chapter (SLP3 draft, Ch. 17)](https://web.stanford.edu/~jurafsky/slp3/17.pdf) — **Jurafsky & Martin** — the free, definitive HMM → MEMM → CRF → neural write-up.
- [An Introduction to Conditional Random Fields](https://homepages.inf.ed.ac.uk/csutton/publications/crftutv2.pdf) — **Sutton & McCallum** — the canonical CRF tutorial (linear-chain, training, inference).
- [Named Entities (spaCy docs)](https://spacy.io/usage/linguistic-features#named-entities) — **spaCy** — how a production NER tagger works, with code.
- [Stanford Named Entity Recognizer (CRF-NER)](https://nlp.stanford.edu/software/CRF-NER.shtml) — **Stanford NLP** — the canonical CRF tagger and its feature design.
- [Token classification (Hugging Face docs)](https://huggingface.co/docs/transformers/en/tasks/token_classification) — **Hugging Face** — `AutoModelForTokenClassification`, subword alignment, and CoNLL training in code.
- [seqeval — sequence labeling evaluation](https://github.com/chakki-works/seqeval) — **chakki-works** — the de-facto Python port of CoNLL `conlleval` for entity-level P/R/F1.

**Key papers**:
- [A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition](https://www.ece.ucsb.edu/Faculty/Rabiner/ece259/Reprints/tutorial%20on%20hmm%20and%20applications.pdf) — **Rabiner (1989)** — the classic HMM + Viterbi + forward–backward reference.
- [The Viterbi Algorithm](https://www2.isye.gatech.edu/~yxie77/ece587/viterbi_algorithm.pdf) — **Forney (1973), Proc. IEEE 61(3):268–278** — the original max-sum trellis decoder; the source of the Viterbi recurrence and backpointer recovery used for HMM/CRF decoding.
- [Maximum Entropy Markov Models for Information Extraction and Segmentation](https://courses.cs.washington.edu/courses/cse517/16wi/papers/mccallum2000.pdf) — **McCallum, Freitag & Pereira (2000)** — the MEMM, the discriminative per-state model (and the source of label bias).
- [Conditional Random Fields: Probabilistic Models for Segmenting and Labeling Sequence Data](https://repository.upenn.edu/handle/20.500.14332/6188) — **Lafferty, McCallum & Pereira (2001)** — the CRF; introduces and diagnoses label bias, fixes it with global normalization.
- [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991) — **Huang, Xu & Yu (2015)** — the biLSTM-CRF that became the standard.
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — **Lample et al. (2016)** — char+word biLSTM-CRF; strong NER without hand features.
- [End-to-end Sequence Labeling via Bi-directional LSTM-CNNs-CRF](https://arxiv.org/abs/1603.01354) — **Ma & Hovy (2016)** — char-CNN variant; the other canonical neural NER architecture.
- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — **Devlin et al. (2018)** — the pretrained encoder + token-classification head that is today's default.
- [Introduction to the CoNLL-2003 Shared Task: Language-Independent NER](https://aclanthology.org/W03-0419/) — **Tjong Kim Sang & De Meulder (2003)** — the benchmark, the BIO data format, and the entity-level F1 metric everyone uses.
- [Design Challenges and Misconceptions in Named Entity Recognition](https://aclanthology.org/W09-1119/) — **Ratinov & Roth (2009)** — why BIOES/BILOU beats BIO, and the practical NER pitfalls.

**Books (free, with chapters)**:
- [Speech and Language Processing, 3rd ed. — Ch. 17 "Sequence Labeling for Parts of Speech and Named Entities"](https://web.stanford.edu/~jurafsky/slp3/17.pdf) — **Jurafsky & Martin** — HMM, MEMM, CRF, neural taggers, Viterbi, and entity F1.
- [Natural Language Processing with Python — Ch. 5 "Categorizing and Tagging Words"](https://www.nltk.org/book/ch05.html) — **Bird, Klein & Loper** — hands-on POS tagging in NLTK.

**In this platform**:
- Concept page (full explanation): [Sequence Labeling — POS & NER](09-Sequence-Labeling-POS-and-NER.md)
- Foundations (the *why* behind probabilities): [Probability & Bayes Theorem](../../../AI-ML-intuition/Module_0_Foundations/0.01_Probability_and_Bayes_Theorem.md)
- The encoder used today: [Contextual Embeddings (ELMo, BERT)](../06-Contextual-Embeddings-ELMo-BERT/06-Contextual-Embeddings-ELMo-BERT.md) — BERT token-classification heads.
- The recurrent backbone of biLSTM-CRF: [RNN, LSTM & GRU](../../05.%20Deep_Learning/14-RNN-LSTM-GRU/14-RNN-LSTM-GRU.md)
- How you score it: [NLP Evaluation Metrics](../18-NLP-Evaluation-Metrics/18-NLP-Evaluation-Metrics.md) — entity-level precision/recall/F1.
- Related task (span extraction): [Question Answering](../11-Question-Answering/11-Question-Answering.md) — extractive QA as span labeling.
- Prior step: [Text Preprocessing & Normalization](../01-Text-Preprocessing-and-Normalization/01-Text-Preprocessing-and-Normalization.md) — tokenization that defines the units to label.
- Sequence-level vs token-level: [Text Classification & Sentiment](../10-Text-Classification-and-Sentiment-Analysis/10-Text-Classification-and-Sentiment-Analysis.md)
