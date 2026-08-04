---
id: "12-multimodal"
topic: "Multimodal Models"
level: advanced
built_from: ["llms", "computer-vision", "diffusion"]
leads_to: ["video-understanding", "agentic-ai"]
updated: 2026-07-14
---

# Multimodal Models
> Models that *understand and reason across modalities* — image+text (VLMs), audio, video,
> and any-to-any architectures. Generation of images lives in
> [Diffusion](../diffusion-models/README.md)/[GenAI](../generative-models/README.md); this topic is
> about joint representation, fusion, and multimodal reasoning.

**⭐ Start here:** [Generalized Visual Language Models (Lil'Log)](https://lilianweng.github.io/posts/2022-06-09-vlm/) — **Lilian Weng** — the canonical map of how vision meets language models.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page.
> **✅ ready · ⬜ coming soon.** Start with contrastive alignment (CLIP), then instruction-tuned VLMs, then fusion architectures.

### Representation alignment
1. ⬜ CLIP & Contrastive Vision-Language Pretraining
2. ⬜ Modern Dual Encoders (SigLIP · retrieval and zero-shot classification)

### Vision-language models (VLMs)
3. ⬜ Interleaved & Few-Shot VLMs (Flamingo lineage · cross-attention fusion)
4. ⬜ Visual Instruction Tuning (LLaVA recipe · projector + LLM)
5. ⬜ Modern Open VLM Architectures (dynamic resolution · native OCR · grounding)

### Fusion & any-to-any
6. ⬜ Fusion Strategies (early vs late · adapters vs native multimodality)
7. ⬜ Unified Token Spaces & Any-to-Any Models (interleaved generation)

### Applications & evaluation
8. ⬜ Document & Chart Understanding (OCR-free reasoning)
9. ⬜ Multimodal RAG (image/table/PDF retrieval into VLM context)
10. ⬜ Multimodal Benchmarks & Evaluation (and their failure modes)

### Related concepts (covered in another section)
> Kept in their canonical home to avoid repetition.
- **Vision Transformers (ViT)** — the visual backbone → [Computer Vision · Vision Transformers](../computer-vision/README.md)
- **Text-to-image generation** → [Diffusion](../diffusion-models/README.md) · [GenAI](../generative-models/README.md)
- **Speech understanding & audio tokens** → [Audio & Speech](../audio-and-speech/README.md)
- **Video-language models** → [Video Understanding](../video-understanding/README.md)
- **LLM decoder architecture, instruction tuning** → [LLMs](../../09.%20LLMs/README.md)

## 🎓 Courses (free)
- [Hugging Face Community Computer Vision Course — Multimodal unit](https://huggingface.co/learn/computer-vision-course) — **Hugging Face** — free, code-first VLM material.

## 🎥 Videos
- [OpenAI CLIP: Connecting Text and Images (paper explained)](https://www.youtube.com/watch?v=T9XSU0pKX2E) — **Yannic Kilcher** — the clearest CLIP walkthrough.

## 📄 Key Papers
- [Learning Transferable Visual Models From Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — **Radford et al. (2021)** — the alignment paper everything builds on.
- [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) — **Alayrac et al. (2022)** — interleaved VLM blueprint.
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) — **Liu et al. (2023)** — the open-VLM recipe.

## 📰 Articles
- [Multimodality and Large Multimodal Models](https://huyenchip.com/2023/10/10/multimodal.html) — **Chip Huyen** — systems-level survey of LMM design.
- [Vision Language Models Explained](https://huggingface.co/blog/vlms) — **Hugging Face** — practical tour of open VLMs.

## 🔗 In this platform
- Backbones: [Computer Vision](../computer-vision/README.md) · Language side: [LLMs](../../09.%20LLMs/README.md)
