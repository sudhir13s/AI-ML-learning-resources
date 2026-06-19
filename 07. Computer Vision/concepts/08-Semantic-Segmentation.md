---
id: "07-computer-vision/semantic-segmentation"
topic: "Semantic Segmentation (FCN, U-Net, DeepLab)"
parent: "07-computer-vision"
level: intermediate
prereqs: ["cnns", "classic-cnn-architectures", "pooling-and-receptive-fields"]
interview_frequency: high
updated: 2026-06-20
---

# Semantic Segmentation — FCN · U-Net · DeepLab
> Assign **every pixel** a class label (road, person, sky) — a dense, per-pixel classification. The key
> architecture is **encoder–decoder**: a CNN downsamples to capture context, then upsamples back to full
> resolution. **FCN** made it fully convolutional, **U-Net** added skip connections for sharp boundaries,
> and **DeepLab** used atrous (dilated) convolutions + CRFs to keep resolution without losing receptive field.

**Why it matters:** a core dense-prediction interview topic — why classification CNNs can't segment
directly, how transposed/upsampling convolutions and skip connections recover spatial detail, what atrous
convolution buys you (large receptive field, no downsampling), and the difference between semantic (no
instances) and instance segmentation. U-Net in particular is ubiquitous in medical imaging.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch ⭐ [U-Net clearly explained](https://www.youtube.com/watch?v=oxcgx75k6yU) (TileStats). *The encoder–decoder + skip-connection idea, from scratch.*
2. **See the landscape** — watch [CS231n Lec 11: Detection & Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo). *FCN, upsampling, and dense prediction in context.*
3. **Go deeper on architecture** — watch [Detection & Segmentation (Michigan, Lec 16)](https://www.youtube.com/watch?v=9AyMR4IhSWQ). *A modern, thorough treatment of segmentation networks.*
4. **Read the sources** — [FCN](https://arxiv.org/abs/1411.4038) → [U-Net](https://arxiv.org/abs/1505.04597) → [DeepLabv3](https://arxiv.org/abs/1706.05587). *Fully convolutional → skip connections → atrous convolution.*
5. **Make it concrete** — work through [d2l: Semantic Segmentation](https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html) + [transposed conv / FCN](https://d2l.ai/chapter_computer-vision/fcn.html). *Build an FCN and see per-pixel predictions.*

## 🎓 Courses (free)
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford** — Lecture 11 covers FCN and dense prediction alongside detection.
- [Dive into Deep Learning — Semantic Segmentation & FCN](https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html) — **Zhang et al.** — free chapters with transposed convolutions and a runnable FCN.

## 🎥 Videos
- [U-Net clearly explained](https://www.youtube.com/watch?v=oxcgx75k6yU) — **TileStats** — the encoder–decoder + skip-connection architecture, beautifully clear.
- [CS231n Lecture 11 — Detection & Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — **Stanford** — FCN, upsampling, and dense prediction in context.
- [Detection & Segmentation (Lecture 16)](https://www.youtube.com/watch?v=9AyMR4IhSWQ) — **Justin Johnson (Michigan)** — a thorough modern treatment of segmentation networks.
- [5-Minute Teaser: U-Net for Biomedical Image Segmentation](https://www.youtube.com/watch?v=81AvQQnpG4Q) — **U-Net authors** — the original short presentation of U-Net.

## 📄 Key Papers
- [Fully Convolutional Networks for Semantic Segmentation (FCN)](https://arxiv.org/abs/1411.4038) — **Long et al. (2015)** — replaced FC layers with convs for dense pixel prediction.
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597) — **Ronneberger et al. (2015)** — symmetric encoder–decoder with skip connections; the workhorse.
- [DeepLab (atrous convolution + CRF)](https://arxiv.org/abs/1606.00915) — **Chen et al. (2016)** — dilated convs keep resolution and receptive field.
- [Rethinking Atrous Convolution (DeepLabv3)](https://arxiv.org/abs/1706.05587) — **Chen et al. (2017)** — atrous spatial pyramid pooling; a strong, widely used baseline.

## 📰 Articles / Blogs (free, no paywall)
- [Semantic Segmentation & dataset (d2l)](https://d2l.ai/chapter_computer-vision/semantic-segmentation-and-dataset.html) — **Zhang et al.** — the task, dataset, and metrics, free.
- [Fully Convolutional Networks (d2l)](https://d2l.ai/chapter_computer-vision/fcn.html) — **Zhang et al.** — transposed convolution and a runnable FCN.
- [Intersection over Union (IoU)](https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/) — **LearnOpenCV** — the overlap metric used for segmentation too, free.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 14.9 (Semantic Segmentation)** + **14.10 (Transposed Conv)** + **14.11 (FCN)**](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang et al.** — dense prediction with runnable code.
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 6.4 (Semantic segmentation)**](https://szeliski.org/Book/) — **Richard Szeliski** — segmentation in context, free.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 4.13 Convolution](../../../AI-ML-intuition/Module_4_Stabilization/4B_Architectural_Motifs/4.13_Convolution.md) — atrous/transposed conv are convolution variants.
- Foundation: [Pooling & Receptive Fields](02-Pooling-and-Receptive-Fields.md) (why we downsample then upsample) · [Classic CNN Architectures](03-Classic-CNN-Architectures.md)
- Next concepts: [09 Instance Segmentation](09-Instance-Segmentation.md) · [10 Detection & Segmentation Metrics](10-Detection-and-Segmentation-Metrics.md)
