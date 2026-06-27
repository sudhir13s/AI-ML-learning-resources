---
id: "02-data-preprocessing/text-image-overview"
topic: "Text & Image Preprocessing (overview)"
parent: "02-data-preprocessing"
level: beginner
prereqs: ["python", "numpy"]
interview_frequency: medium
updated: 2026-06-20
---

# Text & Image Preprocessing — Overview
> A map of how raw text and pixels become model-ready tensors — cleaning/tokenizing text and
> resizing/normalizing/augmenting images — with pointers to the NLP and Computer Vision domains for depth.

**Why it matters:** unstructured data needs domain-specific preprocessing before any tabular trick applies.
For text: cleaning, tokenization, and (for classical models) stemming/lemmatization. For images: decode →
resize → scale to [0,1] or normalize by channel mean/std → augment. Interviewers expect you to know **why
you normalize images with dataset (or ImageNet) mean/std**, why augmentation only goes on the training set,
and how tokenization choices (word vs subword/BPE) affect everything downstream.

**⭐ Start here — suggested path:**

1. **Text basics** — watch [Text Preprocessing: tokenize/clean/stem/lemmatize](https://www.youtube.com/watch?v=hhjn4HVEdy0). *The classical NLP cleaning pipeline end to end.*
2. **Modern tokenization** — watch [Byte Pair Encoding](https://www.youtube.com/watch?v=tOMjTCO0htA). *The subword tokenization behind today's models.*
3. **Image basics** — watch [Image Preprocessing: normalize pixels](https://www.youtube.com/watch?v=UlQTYXIgIw0). *Why and how to scale/normalize pixel values.*
4. **Augmentation** — watch [PyTorch Data Augmentation (torchvision)](https://www.youtube.com/watch?v=Zvd276j9sZ8). *Train-only transforms that expand the dataset.*
5. **Read the why** — read [CS231n: Data preprocessing](https://cs231n.github.io/neural-networks-2/). *Centering, normalization, and PCA/whitening for images, rigorously.*

## 🎓 Courses (free)
- [Hugging Face NLP Course — Tokenizers](https://huggingface.co/learn/nlp-course/chapter6/1) — **Hugging Face** — the definitive free course on modern text tokenization.
- [Kaggle Learn — Computer Vision](https://www.kaggle.com/learn/computer-vision) — **Kaggle** — hands-on image data loading, augmentation, and CNN basics.

## 🎥 Videos
- [Text Preprocessing — tokenization / cleaning / stemming / stopwords / lemmatization](https://www.youtube.com/watch?v=hhjn4HVEdy0) — **Utsav Aggarwal** — the full classical text pipeline with code.
- [NLP Course — Text Preprocessing (Lecture 3)](https://www.youtube.com/watch?v=6C0sLtw5ctc) — **CampusX** — structured walkthrough of every cleaning step.
- [Byte Pair Encoding](https://www.youtube.com/watch?v=tOMjTCO0htA) — **From Languages to Information (Stanford-style)** — the subword tokenization used by modern LLMs.
- [Image Preprocessing — Normalize Pixel Values](https://www.youtube.com/watch?v=UlQTYXIgIw0) — concise demo of scaling/normalizing image tensors.
- [PyTorch Data Augmentation using Torchvision](https://www.youtube.com/watch?v=Zvd276j9sZ8) — **Aladdin Persson** — train-time augmentation transforms in practice.

## 📄 Key Papers
- [ImageNet Classification with Deep CNNs (AlexNet)](https://papers.nips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf) — **Krizhevsky, Sutskever & Hinton (2012)** — §4 popularized image augmentation + per-channel normalization.
- [Speech and Language Processing (3rd ed.) — **Ch. 2 "Regular Expressions, Text Normalization, Edit Distance"**](https://web.stanford.edu/~jurafsky/slp3/2.pdf) — **Jurafsky & Martin** — the reference treatment of text normalization (PDF chapter).

## 📰 Articles / Blogs (free, no paywall)
- [CS231n — Setting up the data (preprocessing)](https://cs231n.github.io/neural-networks-2/) — **Stanford** — image mean-subtraction, normalization, and PCA/whitening, clearly.
- [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html) — **PyTorch docs** — the standard image preprocessing/augmentation API.
- [NLTK Book — Ch. 3 "Processing Raw Text"](https://www.nltk.org/book/ch03.html) — **Bird, Klein & Loper** — tokenization, normalization, stemming, free online.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 14.1 "Image Augmentation"**](https://d2l.ai/chapter_computer-vision/image-augmentation.html) — **Zhang et al.** — runnable image preprocessing/augmentation; free.
- [Speech and Language Processing (3rd ed.) — **Ch. 2 (text normalization)**](https://web.stanford.edu/~jurafsky/slp3/2.pdf) — **Jurafsky & Martin** — the canonical text-preprocessing chapter.

## 🔗 In this platform
- Full depth lives in the dedicated domains:
  - **Text** — tokenization, normalization, subword algorithms → [06. NLP](../../06.%20NLP/README.md)
  - **Images** — augmentation and vision-specific preprocessing → [07. Computer Vision](../../07.%20Computer%20Vision/concepts/README.md)
- Concept depth (the *why*): [AI-ML-intuition 1.15 Tokenization & BPE](../../../AI-ML-intuition/Module_1_Representation/1.15_Tokenization_and_BPE.md) · [1.17 BoW & TF-IDF](../../../AI-ML-intuition/Module_1_Representation/1.17_BoW_and_TF-IDF.md)
- Next concepts: [06 Feature Engineering](06-Feature-Engineering.md) · [13 Data Pipelines](13-Data-Pipelines.md)
