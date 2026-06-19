---
id: "06-nlp/sequence-labeling-pos-ner"
topic: "Sequence Labeling — POS & NER (HMM/CRF → neural)"
parent: "06-nlp"
level: intermediate
prereqs: ["probability", "text-preprocessing", "rnn-lstm"]
interview_frequency: medium
updated: 2026-06-19
---

# Sequence Labeling — POS Tagging & Named Entity Recognition
> Assigning a label to **every token** in a sequence: part-of-speech tags (NOUN, VERB…) or entity
> spans (PERSON, ORG, LOC) using **BIO** tagging. The label depends on neighbors, so the model is the
> sequence, not each token alone — from **HMM** → **CRF** → **BiLSTM-CRF** → transformers.

**Why it matters:** sequence labeling is the canonical structured-prediction task and shows up in
search, IE, and clinical NLP. Be ready to explain **BIO/IOB** tagging, the **HMM** (generative,
Viterbi decoding), why the **CRF** (discriminative, models the whole label sequence) beats it, why a
softmax-per-token model needs a CRF layer on top (to enforce valid tag transitions), and span-level
**F1** as the metric.

**⭐ Start here — suggested path:**

1. **Build intuition** — read [SLP3 Ch. 17](https://web.stanford.edu/~jurafsky/slp3/17.pdf) intro on POS/NER and BIO tagging. *Frame the task before the models.*
2. **See HMM + Viterbi** — watch [POS Tagging using the Viterbi algorithm](https://www.youtube.com/watch?v=QYzTTFxcc9I) (**Data Science in your pocket**). *The classic generative tagger, decoded.*
3. **Get the CRF** — read [SLP3 Ch. 17 §17.4](https://web.stanford.edu/~jurafsky/slp3/17.pdf). *Why discriminative CRFs model the full label sequence — the part interviews probe.*
4. **Read the neural sources** — [BiLSTM-CRF](https://arxiv.org/abs/1508.01991) → [Neural NER](https://arxiv.org/abs/1603.01360). *How LSTMs + a CRF layer became the standard.*
5. **Make it concrete** — try [spaCy NER](https://spacy.io/usage/linguistic-features#named-entities) or the [Stanford NER CRF](https://nlp.stanford.edu/software/CRF-NER.shtml). *Tag real text and inspect spans.*

## 🎓 Courses (free)
- [NLP Course for You — Text Classification (includes token-level models)](https://lena-voita.github.io/nlp_course/text_classification.html) — **Lena Voita** — discriminative models that extend to sequence labeling.
- [Stanford CS224N: NLP with Deep Learning](https://web.stanford.edu/class/cs224n/) — **Stanford (Manning)** — sequence-modeling lectures grounding tagging tasks.

## 🎥 Videos
- [POS Tagging using the Viterbi algorithm](https://www.youtube.com/watch?v=QYzTTFxcc9I) — **Data Science in your pocket** — HMM + Viterbi for POS.
- [POS Tagging, HMM, Viterbi — Emission & Transition matrices](https://www.youtube.com/watch?v=hjFbmosY9y4) — **Varsha's engineering stuff** — the probability matrices made explicit.
- [POS Tagging, Viterbi Algorithm — Solved Problem](https://www.youtube.com/watch?v=OBemI2BapE0) — **Varsha's engineering stuff** — a fully worked numeric example.
- [NLP — Text Preprocessing and Text Classification](https://www.youtube.com/watch?v=nxhCyeRR75Q) — **Machine Learning TV** — token-level features feeding classifiers/taggers.

## 📄 Key Papers
- [Bidirectional LSTM-CRF Models for Sequence Tagging](https://arxiv.org/abs/1508.01991) — **Huang, Xu & Yu (2015)** — the BiLSTM-CRF that became the standard.
- [Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — **Lample et al. (2016)** — char+word BiLSTM-CRF; strong NER without hand features.

## 📰 Articles / Blogs (free, no paywall)
- [Named Entities (spaCy docs)](https://spacy.io/usage/linguistic-features#named-entities) — **spaCy** — how a production NER tagger works, with code.
- [Stanford Named Entity Recognizer (CRF-NER)](https://nlp.stanford.edu/software/CRF-NER.shtml) — **Stanford NLP** — the canonical CRF tagger and its design.
- [Sequence Labeling chapter (SLP3 draft)](https://web.stanford.edu/~jurafsky/slp3/17.pdf) — **Jurafsky & Martin** — the free, definitive HMM→CRF write-up.

## 📚 Books (free, with chapters)
- [Speech and Language Processing, 3rd ed. — **Ch. 17 "Sequence Labeling for Parts of Speech and Named Entities"**](https://web.stanford.edu/~jurafsky/slp3/17.pdf) — **Jurafsky & Martin** — HMM, MEMM, CRF, neural taggers.
- [Natural Language Processing with Python — **Ch. 5 "Categorizing and Tagging Words"**](https://www.nltk.org/book/ch05.html) — **Bird, Klein & Loper** — hands-on POS tagging in NLTK.

## 🔗 In this platform
- Prior step: [01 Text Preprocessing](01-Text-Preprocessing-and-Normalization.md) — tokenization that defines the units to label.
- Concept depth (the *why*): [AI-ML-intuition 0.01 Probability & Bayes (HMM foundation)](../../../AI-ML-intuition/Module_0_Foundations/0.01_Probability_and_Bayes_Theorem.md)
- Related: [10 Text Classification & Sentiment](10-Text-Classification-and-Sentiment-Analysis.md) — sequence-level vs token-level prediction.
