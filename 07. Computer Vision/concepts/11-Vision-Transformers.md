---
id: "07-computer-vision/vision-transformers"
topic: "Vision Transformers (ViT)"
parent: "07-computer-vision"
level: advanced
prereqs: ["transformers", "attention", "image-classification"]
interview_frequency: very-high
updated: 2026-06-20
---

# Vision Transformers (ViT)
> ViT applies the NLP transformer directly to images: split the image into fixed-size **patches**
> (e.g. 16×16), linearly embed each patch into a token, add positional encodings, prepend a `[CLS]`
> token, and run a standard transformer encoder. With enough data it matches or beats CNNs — trading
> the convolutional locality prior for global self-attention and scale.

**Why it matters:** the defining modern-vision question — how an image becomes a sequence of patch
tokens, why ViT needs large-scale pretraining (it lacks the CNN's built-in locality/translation
priors), how it compares to CNNs on data efficiency, and what hybrids (Swin, hierarchical windows)
add back. Expected for any role touching modern vision or multimodal models.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch ⭐ [Vision Transformers explained](https://www.youtube.com/watch?v=tkZMj1VKD9s) (Code With Aarohi). *Patches → tokens → transformer, the core idea.*
2. **Patchify in detail** — watch [ViT — An Image Is Worth 16×16 Words (Paper Explained)](https://www.youtube.com/watch?v=8phM16htKbU) (Uygar Kurt). *Patch embedding, `[CLS]` token, and positional encodings.*
3. **Hear the original read** — watch [Yannic Kilcher: 16×16 Words](https://www.youtube.com/watch?v=TrdevFK_am4). *A critical paper read covering why scale matters.*
4. **Read the source** — the ⭐ [ViT paper](https://arxiv.org/abs/2010.11929), then [Swin Transformer](https://arxiv.org/abs/2103.14030). *The pure transformer, then the hierarchical/windowed hybrid.*
5. **Make it concrete** — work through [d2l: Vision Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html). *Implement patch embedding + encoder and classify.*

## 🎓 Courses (free)
- [Dive into Deep Learning — Vision Transformer](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html) — **Zhang et al.** — free chapter implementing ViT from scratch with code.
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford** — recent editions include a Vision Transformer lecture in the schedule.

## 🎥 Videos
- [Vision Transformers explained](https://www.youtube.com/watch?v=tkZMj1VKD9s) — **Code With Aarohi** — clear conceptual walkthrough of the ViT pipeline.
- [ViT — An Image Is Worth 16×16 Words (Paper Explained)](https://www.youtube.com/watch?v=8phM16htKbU) — **Uygar Kurt** — patch embedding, `[CLS]` token, and positional encodings in detail.
- [An Image is Worth 16×16 Words (Paper Explained)](https://www.youtube.com/watch?v=TrdevFK_am4) — **Yannic Kilcher** — a critical read of the original paper and why scale matters.
- [Vision Transformer for Image Classification](https://www.youtube.com/watch?v=HZ4j_U3FC94) — **Shusen Wang** — a concise lecture connecting ViT to attention fundamentals.

## 📄 Key Papers
- [An Image is Worth 16×16 Words (ViT)](https://arxiv.org/abs/2010.11929) — **Dosovitskiy et al. (2020)** — the pure-transformer image classifier; the landmark.
- [Swin Transformer](https://arxiv.org/abs/2103.14030) — **Liu et al. (2021)** — hierarchical, windowed attention; a strong general-purpose vision backbone.

## 📰 Articles / Blogs (free, no paywall)
- [Vision Transformer (d2l)](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html) — **Zhang et al.** — explanation plus runnable code, free.
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — **Jay Alammar** — the attention/transformer foundation ViT reuses, free.
- [Hugging Face — Vision Transformer (ViT) docs](https://huggingface.co/docs/transformers/model_doc/vit) — **Hugging Face** — usage and pretrained ViT models, open.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 11.8 (Transformers for Vision)**](https://d2l.ai/chapter_attention-mechanisms-and-transformers/vision-transformer.html) — **Zhang et al.** — ViT with runnable code.
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 5.5 (Transformers in vision)**](https://szeliski.org/Book/) — **Richard Szeliski** — ViT in the architecture landscape, free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.15 The Transformer Block](../../../AI-ML-intuition/Module_4_Stabilization/4D_Nonlinearities/4.15_The_Transformer_Block.md) · [4.08 Multi-Head Attention](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.08_Multi-Head_Attention_Routing.md)
- Foundation: [Deep Learning › Transformer Architecture](../../05.%20Deep_Learning/concepts/16-Transformer-Architecture.md) · [Deep Learning › Attention Mechanism](../../05.%20Deep_Learning/concepts/15-Attention-Mechanism.md)
- Next concepts: [12 Self-Supervised Vision](12-Self-Supervised-Vision.md) (MAE/DINO build on ViT)
