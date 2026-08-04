---
id: "07-computer-vision"
topic: "Computer Vision"
level: intermediate
built_from: ["deep-learning", "linear-algebra"]
updated: 2026-06-27
---

# 🖼️ Computer Vision
> The mathematical and deep-learning spine of vision — images as signals, convolution and
> frequency thinking, projective geometry and cameras, deep vision architectures (CNNs · ViTs),
> detection & segmentation, and generative/self-supervised vision.

## 📑 Concept Index
Every chapter is a self-contained folder (`NN-Concept/NN-Concept.md`) with its page.
> **✅ ready · ⬜ coming soon.** New to vision? Start with the field overview below, then work top to bottom.

### Image processing & classical features
1. ✅ [Image Representation & Filtering (edges · HOG · SIFT)](image-representation-and-filtering/image-representation-and-filtering.md)
2. ✅ [Pooling & Receptive Fields](pooling-and-receptive-fields/pooling-and-receptive-fields.md)

### Classification & backbones
3. ✅ [Classic CNN Architectures (LeNet · AlexNet · VGG · ResNet · Inception · EfficientNet)](classic-cnn-architectures/classic-cnn-architectures.md)
4. ✅ [Image Classification](image-classification/image-classification.md)
5. ✅ [Transfer Learning for Vision](transfer-learning-for-vision/transfer-learning-for-vision.md)
6. ✅ [Data Augmentation](data-augmentation/data-augmentation.md)

### Detection & segmentation
7. ✅ [Object Detection (R-CNN family · YOLO · SSD)](object-detection/object-detection.md)
8. ✅ [Semantic Segmentation (FCN · U-Net · DeepLab)](semantic-segmentation/semantic-segmentation.md)
9. ✅ [Instance Segmentation (Mask R-CNN)](instance-segmentation/instance-segmentation.md)
10. ✅ [Detection & Segmentation Metrics (IoU · mAP)](detection-and-segmentation-metrics/detection-and-segmentation-metrics.md)

### Modern & specialized vision
11. ✅ [Vision Transformers (ViT)](vision-transformers/vision-transformers.md)
12. ✅ [Self-Supervised Vision (SimCLR · MAE · DINO)](self-supervised-vision/self-supervised-vision.md)
13. ✅ [Pose Estimation](pose-estimation/pose-estimation.md)
14. ✅ [Optical Flow & Video Understanding](optical-flow-and-video/optical-flow-and-video.md)
15. ✅ [Optical Character Recognition (OCR)](ocr/ocr.md)
16. ✅ [3D & Depth Estimation](3d-and-depth-estimation/3d-and-depth-estimation.md)

### Related concepts (covered in another section)
> These topics are foundational or generative and live in their canonical home to avoid repetition.
- **CNNs & Convolution** (the operation, kernels, stride/padding math) → [Deep Learning › CNNs & Convolution](../../deep-learning/neural-architectures/cnns-and-convolution/cnns-and-convolution.md)
- **Residual / skip connections · Batch & Group Norm** (the motifs deep vision backbones rely on) → [Deep Learning](../../deep-learning/README.md)
- **Image generation** — GANs · Diffusion · text-to-image → [GenAI](../generative-models/README.md)
- **Contrastive / embedding math** (InfoNCE, triplet loss reused by self-supervised vision) → [NLP](../natural-language-processing/README.md) · [Deep Learning](../../deep-learning/README.md)

## 📐 Mathematics curriculum (specialization)

> Elective deep-dive track, absorbed and expanded from the retired `math-for-AIML-Q5`
> CV specialization. Same format as the [main math curriculum](../../foundations/mathematical-foundations/maths-for-ai-ml/README.md):
> what to study → why → best resources → which AI-ML-intuition pages it unlocks.

**Goal:** the mathematical spine of vision — images as signals, convolution and frequency
thinking, projective geometry and cameras, deep vision architectures, and generative vision.

### Core resource backbone
- **Stanford CS231n** — [CNNs for Visual Recognition](https://cs231n.github.io/) (the anchor course)
- **Multiple View Geometry** (Hartley & Zisserman) — the geometric-vision bible
- **First Principles of Computer Vision** (Shree Nayar, Columbia) — [YouTube channel](https://www.youtube.com/@firstprinciplesofcomputerv3258) — exceptional visual lectures on classical CV
- **Szeliski** — [Computer Vision: Algorithms and Applications](https://szeliski.org/Book/) (free)

### Study order & what each module unlocks

| Module | Key sub-topics | Best resources | → AI-ML-intuition |
| :--- | :--- | :--- | :--- |
| **V1. Images as signals** | images as functions, sampling & aliasing, noise models, color spaces | Nayar (First Principles): Image Formation playlist; Szeliski ch. 2 | [0.02 Distributions](../../../AI-ML-intuition/foundational-mental-models/probability-and-belief/distributions-and-gaussians-intuition.md) (noise) |
| **V2. Filtering & convolution** | convolution as local aggregation, edge detectors, stride/padding/pooling, **Fourier intuition**, multi-scale/wavelets | Nayar: Image Processing; 3B1B [Fourier](https://www.youtube.com/watch?v=spUNpyF58BY) + [convolution](https://www.youtube.com/watch?v=KuXjwB4LzSA) | **[4.13 Convolution](../../../AI-ML-intuition/architectural-mechanisms/locality-and-weight-sharing/convolution-intuition.md)** |
| **V3. Geometric vision** | homogeneous coordinates, projective geometry, **camera model**, homographies | Hartley & Zisserman ch. 2–4; Nayar: Imaging playlist | linear maps ([1.05](../../../AI-ML-intuition/representation/dimensionality-and-latent-structure/pca-and-svd-intuition.md) intuitions) |
| **V4. Classical features** | corners/interest points, SIFT/ORB descriptors, optical flow | Nayar: Features playlist; Szeliski ch. 7 | [1.07-1.08 distances](../../../AI-ML-intuition/representation/similarity-and-distance/cosine-vs-euclidean-distance-intuition.md) (descriptor matching) |
| **V5. Deep vision** | CNNs as feature hierarchies, classic architectures (AlexNet→ResNet), norms & residuals in vision, transfer learning | CS231n lectures 5–9 | [4.13](../../../AI-ML-intuition/architectural-mechanisms/locality-and-weight-sharing/convolution-intuition.md), [4.06 Residuals](../../../AI-ML-intuition/training-stability/gradient-health/residual-connections-intuition.md), [4.03 GroupNorm](../../../AI-ML-intuition/training-stability/normalization/group-normalization-intuition.md), [7.03 Transfer](../../../AI-ML-intuition/scaling-adaptation-and-efficiency/adaptation/transfer-learning-and-fine-tuning-intuition.md) |
| **V6. Vision Transformers** | images as patch tokens, self-attention for vision, CNN-ViT hybrids | [ViT paper](https://arxiv.org/abs/2010.11929); CS231n ViT lecture | [4.15 Transformer Block](../../../AI-ML-intuition/architectural-mechanisms/composition/transformer-block-intuition.md), [4.08 MHA](../../../AI-ML-intuition/architectural-mechanisms/attention-and-routing/multi-head-attention-intuition.md) |
| **V7. Detection & segmentation** | bounding-box geometry, IoU/mAP, semantic vs instance segmentation, structured losses | CS231n lecture 11; [Mask R-CNN](https://arxiv.org/abs/1703.06870) | [3.05/3.06 metrics](../../../AI-ML-intuition/objectives-and-evaluation/predictive-evaluation/roc-and-pr-curves-intuition.md) (mAP = AP per class) |
| **V8. 3D & multi-view** | stereo & depth, epipolar geometry, structure-from-motion, (NeRF/Gaussian-splatting overview) | Hartley & Zisserman ch. 9–12; [NeRF](https://arxiv.org/abs/2003.08934) | — (geometry track) |
| **V9. Generative & self-supervised vision** | autoencoders, contrastive pretraining (SimCLR/CLIP), diffusion for images | [CLIP](https://arxiv.org/abs/2103.00020); [SimCLR](https://arxiv.org/abs/2002.05709) | [1.13 Contrastive](../../../AI-ML-intuition/representation/representation-learning/contrastive-learning-intuition.md), [5.02 VAEs](../../../AI-ML-intuition/generation/latent-variable-generation/latent-variable-models-and-elbo-intuition.md), [5.03 Diffusion](../../../AI-ML-intuition/generation/diffusion-and-score-models/diffusion-forward-and-reverse-process-intuition.md) |

### Suggested first pass
1. V2 + V5 first if you're deep-learning-bound (convolution → CNNs → ViTs is the modern spine).
2. V1, V3, V4 for the classical foundation (essential for robotics/AR/3D roles; skippable for pure DL roles).
3. V9 last — it reuses half of AI-ML-intuition's Modules 1 and 5.

**Completion target:** explain convolution's two priors and parameter math, walk a pinhole
camera model, justify ViT patch tokenization, and read a diffusion-vision paper without fear.
