---
id: "13-video-understanding"
topic: "Video Understanding"
level: advanced
built_from: ["computer-vision", "multimodal"]
leads_to: ["agentic-ai"]
updated: 2026-07-14
---

# Video Understanding
> Models that *watch* — temporal representation, action recognition, video-language reasoning,
> and the systems tricks that make hour-long video tractable. Video *generation* lives in
> [Diffusion](../diffusion-models/README.md); classic optical flow lives in
> [Computer Vision](../computer-vision/README.md).

**⭐ Start here:** [VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training](https://arxiv.org/abs/2203.12602) — **Tong et al. (2022)** — the self-supervised recipe modern video backbones start from.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page.
> **✅ ready · ⬜ coming soon.** Representations first, then tasks, then long-video systems.

### Temporal representations
1. ⬜ Video Representations & Temporal Modeling (3D CNNs · two-stream · frame sampling)
2. ⬜ Video Transformers (TimeSformer · ViViT · space-time attention trade-offs)
3. ⬜ Self-Supervised Video Pretraining (VideoMAE · contrastive video)

### Core tasks
4. ⬜ Action Recognition & Video Classification
5. ⬜ Temporal Localization & Moment Retrieval
6. ⬜ Video-Language Models (video captioning · VideoQA · video chat)

### Long video & systems
7. ⬜ Long-Video Understanding (memory · keyframe selection · token budgets)
8. ⬜ Efficient Video Inference (streaming · caching · frame-rate scheduling)

### Related concepts (covered in another section)
> Kept in their canonical home to avoid repetition.
- **Optical flow & classic video features** → [Computer Vision · Optical Flow & Video](../computer-vision/README.md)
- **Object tracking & detection backbones** → [Computer Vision](../computer-vision/README.md)
- **Video diffusion / generation** → [Diffusion · Video Diffusion Models](../diffusion-models/README.md)
- **VLM fusion architectures** → [Multimodal](../multimodal-learning/README.md)

## 🎥 Videos
- [CS231n Lecture — Video Understanding](https://www.youtube.com/watch?v=A9D6NXBJdwU) — **Stanford CS231n** — the standard academic overview of temporal architectures.

## 📄 Key Papers
- [Is Space-Time Attention All You Need for Video Understanding? (TimeSformer)](https://arxiv.org/abs/2102.05095) — **Bertasius et al. (2021)** — the video-transformer design space.
- [ViViT: A Video Vision Transformer](https://arxiv.org/abs/2103.15691) — **Arnab et al. (2021)** — factorized space-time attention.
- [VideoMAE](https://arxiv.org/abs/2203.12602) — **Tong et al. (2022)** — self-supervised video pretraining that actually scales down.

## 📰 Articles
- [Video classification with Transformers](https://huggingface.co/docs/transformers/tasks/video_classification) — **Hugging Face** — hands-on fine-tuning path.

## 🔗 In this platform
- Backbones: [Computer Vision](../computer-vision/README.md) · Reasoning side: [Multimodal](../multimodal-learning/README.md)
