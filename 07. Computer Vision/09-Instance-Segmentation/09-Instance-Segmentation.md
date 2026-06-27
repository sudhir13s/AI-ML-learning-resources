---
id: "07-computer-vision/instance-segmentation"
topic: "Instance Segmentation (Mask R-CNN)"
parent: "07-computer-vision"
level: advanced
prereqs: ["object-detection", "semantic-segmentation"]
interview_frequency: high
updated: 2026-06-20
---

# Instance Segmentation — Mask R-CNN
> Instance segmentation = detection **+** per-pixel masks: not just "there are 3 people" (semantic
> segmentation merges them) but "person #1 is *these* pixels, person #2 is *those*." **Mask R-CNN**
> extends Faster R-CNN with a parallel mask branch and replaces RoIPool with **RoIAlign** (no
> quantization), producing sharp per-instance masks — the standard architecture for this task.

**Why it matters:** the classic "explain Mask R-CNN" question — how it builds on Faster R-CNN, why
**RoIAlign** matters (the misalignment RoIPool introduces and how bilinear sampling fixes it), why the
mask branch is per-class and decoupled from classification, and the precise difference between semantic,
instance, and panoptic segmentation. A favorite for senior CV roles.

**⭐ Start here — suggested path:**

1. **Place it in the landscape** — watch [CS231n Lec 11: Detection & Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo). *Where instance segmentation sits relative to detection and semantic segmentation.*
2. **Get the architecture** — watch ⭐ [Instance Segmentation — Mask R-CNN (UCF)](https://www.youtube.com/watch?v=P2LWNdH3bi8). *RoIAlign vs RoIPool and the mask branch, lecture-quality.*
3. **Read the intuition** — watch [Mask R-CNN Paper Explanation & Intuition](https://www.youtube.com/watch?v=Vbj82MnSjmE). *A guided read of the paper's key ideas.*
4. **Read the sources** — [Faster R-CNN](https://arxiv.org/abs/1506.01497) → ⭐ [Mask R-CNN](https://arxiv.org/abs/1703.06870). *The detector it extends, then the mask branch + RoIAlign.*
5. **Make it concrete** — work through [Detectron2](https://detectron2.readthedocs.io/en/latest/). *Run/train Mask R-CNN on COCO and inspect the masks.*

## 🎓 Courses (free)
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford** — Lecture 11 covers instance segmentation alongside detection and semantic segmentation.
- [Detectron2 documentation & tutorials](https://detectron2.readthedocs.io/en/latest/) — **Meta AI** — the open framework for training Mask R-CNN, with runnable tutorials.

## 🎥 Videos
- [CS231n Lecture 11 — Detection & Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — **Stanford** — situates instance segmentation in the dense-prediction landscape.
- [Instance Segmentation — Mask R-CNN Architecture](https://www.youtube.com/watch?v=P2LWNdH3bi8) — **UCF CRCV** — RoIAlign vs RoIPool and the mask branch, lecture-quality.
- [Mask R-CNN — Paper Explanation & Intuition](https://www.youtube.com/watch?v=Vbj82MnSjmE) — **Sagar Sangodkar** — a guided walkthrough of the paper.
- [Detection & Segmentation (Lecture 16)](https://www.youtube.com/watch?v=9AyMR4IhSWQ) — **Justin Johnson (Michigan)** — modern treatment including instance segmentation.

## 📄 Key Papers
- [Mask R-CNN](https://arxiv.org/abs/1703.06870) — **He et al. (2017)** — the reference instance-segmentation architecture (mask branch + RoIAlign).
- [Faster R-CNN](https://arxiv.org/abs/1506.01497) — **Ren et al. (2015)** — the detector Mask R-CNN extends; the Region Proposal Network.

## 📰 Articles / Blogs (free, no paywall)
- [COCO dataset](https://cocodataset.org/) — **COCO** — the standard instance-segmentation benchmark and annotation format.
- [Detectron2 docs](https://detectron2.readthedocs.io/en/latest/) — **Meta AI** — implementation reference and model zoo for Mask R-CNN.
- [Intersection over Union (IoU)](https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/) — **LearnOpenCV** — the overlap metric used to score masks, free.

## 📚 Books (free, with chapters)
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 6.4 (Instance & semantic segmentation)**](https://szeliski.org/Book/) — **Richard Szeliski** — segmentation taxonomy and methods, free.
- [Dive into Deep Learning — **Ch. 14 (Computer Vision)**](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang et al.** — detection/segmentation foundations Mask R-CNN builds on, with code.

## 🔗 In this platform
- Foundation: [Object Detection](../07-Object-Detection/07-Object-Detection.md) (Mask R-CNN extends Faster R-CNN) · [Semantic Segmentation](../08-Semantic-Segmentation/08-Semantic-Segmentation.md)
- Metrics: [10 Detection & Segmentation Metrics (IoU · mAP)](../10-Detection-and-Segmentation-Metrics/10-Detection-and-Segmentation-Metrics.md)
- Foundation: [Deep Learning › CNNs & Convolution](../../05.%20Deep_Learning/13-CNNs-and-Convolution/13-CNNs-and-Convolution.md)
