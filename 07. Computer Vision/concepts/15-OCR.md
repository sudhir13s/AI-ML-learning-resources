---
id: "07-computer-vision/ocr"
topic: "Optical Character Recognition (OCR)"
parent: "07-computer-vision"
level: intermediate
prereqs: ["cnns", "rnn-lstm", "object-detection"]
interview_frequency: medium
updated: 2026-06-20
---

# Optical Character Recognition (OCR)
> Turn text *in images* into machine-readable strings. Modern OCR is two stages: **text detection**
> (locate text regions/boxes, e.g. EAST/CRAFT) and **text recognition** (read each crop into characters).
> The classic recognition model is **CRNN** — CNN features → BiLSTM → **CTC** loss — which reads
> variable-length text without character-level alignment. Newer systems use transformer decoders (TrOCR).

**Why it matters:** a focused applied question for document-AI, fintech, and search roles — the
detection+recognition pipeline, why CTC solves the alignment problem (no per-character labels needed),
how CRNN combines convolution and recurrence, and the trade-offs between classic engines (Tesseract),
deep pipelines (EasyOCR), and end-to-end transformers (TrOCR).

**⭐ Start here — suggested path:**

1. **Understand the alignment problem** — watch ⭐ [Connectionist Temporal Classification (CTC) Explained](https://www.youtube.com/watch?v=jDPl1QJGLpE). *Why CTC is the key idea that makes sequence reading work without alignment.*
2. **See the tools** — watch [OCR with Python — Tesseract vs EasyOCR](https://www.youtube.com/watch?v=Q7TTbDZ-KHQ). *The two main practical engines, compared.*
3. **Compare approaches** — watch [Tesseract vs EasyOCR vs Textract](https://www.youtube.com/watch?v=CcC3h0waQ6I). *Where classic vs deep vs cloud OCR each win.*
4. **Read the sources** — ⭐ [CRNN](https://arxiv.org/abs/1507.05717) → [STN-OCR / attention OCR](https://arxiv.org/abs/1707.03985). *The CNN+RNN+CTC architecture and attention-based reading.*
5. **Make it concrete** — run [Tesseract](https://github.com/tesseract-ocr/tesseract) or [EasyOCR](https://github.com/JaidedAI/EasyOCR). *Extract text from a real document end to end.*

## 🎓 Courses (free)
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford** — the CNN + sequence foundations OCR recognition relies on.
- [Tesseract OCR documentation](https://github.com/tesseract-ocr/tesseract) — **Google / open source** — the classic engine with extensive free docs and tutorials.

## 🎥 Videos
- [Connectionist Temporal Classification (CTC) Explained](https://www.youtube.com/watch?v=jDPl1QJGLpE) — **DataMListic** — the alignment-free loss at the heart of CRNN OCR.
- [OCR with Python — Tesseract vs EasyOCR](https://www.youtube.com/watch?v=Q7TTbDZ-KHQ) — **s1n7ax** — the two main engines compared with code.
- [Tesseract vs EasyOCR vs AWS Textract](https://www.youtube.com/watch?v=CcC3h0waQ6I) — **Felipe Tambasco** — where classic, deep, and cloud OCR each win.
- [Using Tesseract OCR in a Python script](https://www.youtube.com/watch?v=HNCypVfeTdw) — **JayMartMedia** — a hands-on pytesseract walkthrough.

## 📄 Key Papers
- [An End-to-End Trainable Neural Network for Image-based Sequence Recognition (CRNN)](https://arxiv.org/abs/1507.05717) — **Shi et al. (2015)** — CNN + BiLSTM + CTC; the canonical recognition model.
- [STN-OCR: A single Neural Network for Text Detection and Recognition](https://arxiv.org/abs/1707.03985) — **Bartz et al. (2017)** — end-to-end detection + recognition with spatial transformers.

## 📰 Articles / Blogs (free, no paywall)
- [Optical Character Recognition](https://en.wikipedia.org/wiki/Optical_character_recognition) — **Wikipedia** — the task, history, and pipeline stages in one page.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) — **open source** — the reference classic engine, with docs.
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) — **JaidedAI** — a ready-to-use deep OCR pipeline (detection + CRNN recognition), free.
- [TrOCR docs](https://huggingface.co/docs/transformers/model_doc/trocr) — **Hugging Face** — the transformer-based end-to-end OCR model, open.

## 📚 Books (free, with chapters)
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 6 (Recognition)**](https://szeliski.org/Book/) — **Richard Szeliski** — text recognition within the recognition landscape, free.
- [Dive into Deep Learning — **Ch. 10 (RNNs)** + **Ch. 9 (Modern RNNs)**](https://d2l.ai/chapter_recurrent-neural-networks/index.html) — **Zhang et al.** — the sequence-modeling machinery CRNN/CTC build on, with code.

## 🔗 In this platform
- Foundation: [Object Detection](07-Object-Detection.md) (text detection stage) · [Deep Learning › RNN / LSTM / GRU](../../05.%20Deep_Learning/concepts/14-RNN-LSTM-GRU.md) (CRNN's recurrent core)
- Foundation: [Deep Learning › CNNs & Convolution](../../05.%20Deep_Learning/concepts/13-CNNs-and-Convolution.md)
- Related domain: [NLP › Sequence Labeling](../../06.%20NLP/09-Sequence-Labeling-POS-and-NER/09-Sequence-Labeling-POS-and-NER.md) (CTC is alignment-free sequence labeling)
