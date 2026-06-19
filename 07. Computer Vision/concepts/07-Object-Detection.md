---
id: "07-computer-vision/object-detection"
topic: "Object Detection (R-CNN family, YOLO, SSD)"
parent: "07-computer-vision"
level: intermediate
prereqs: ["cnns", "classic-cnn-architectures", "image-classification"]
interview_frequency: very-high
updated: 2026-06-20
---

# Object Detection — R-CNN family · YOLO · SSD
> Detection answers "what objects are where" — predicting a **class + bounding box** for every object in
> an image. Two paradigms dominate: **two-stage** detectors (R-CNN → Fast → Faster R-CNN) that propose
> regions then classify them, and **one-stage** detectors (YOLO, SSD, RetinaNet) that predict boxes and
> classes in a single pass — faster, and the basis of most real-time systems.

**Why it matters:** the most-asked detection question — explain the R-CNN family's evolution (selective
search → RoI pooling → learned region proposals), contrast two-stage vs one-stage (accuracy vs speed),
describe anchors and NMS, and explain the class-imbalance problem that **focal loss** (RetinaNet) solves.
You should be able to sketch how YOLO turns detection into a single grid regression.

**⭐ Start here — suggested path:**

1. **Build intuition** — watch [How computers recognize objects instantly (TED)](https://www.youtube.com/watch?v=Cgxsv1riJhI) by YOLO's author. *The one-stage detection idea, motivated.*
2. **The full landscape** — watch ⭐ [CS231n Lec 11: Detection & Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo). *R-CNN family + one-stage detectors in one authoritative lecture.*
3. **Understand YOLO** — watch [How YOLO Object Detection Works](https://www.youtube.com/watch?v=svn9-xV7wjk), then read the ⭐ [YOLO paper](https://arxiv.org/abs/1506.02640). *Grid cells, anchors, and single-pass prediction.*
4. **Read the sources** — [R-CNN](https://arxiv.org/abs/1311.2524) → [Fast R-CNN](https://arxiv.org/abs/1504.08083) → [Faster R-CNN](https://arxiv.org/abs/1506.01497) → [SSD](https://arxiv.org/abs/1512.02325) → [RetinaNet (Focal Loss)](https://arxiv.org/abs/1708.02002). *The two-stage→one-stage progression and the imbalance fix.*
5. **Make it concrete** — work through [Ultralytics YOLO docs](https://docs.ultralytics.com/). *Train and run a modern YOLO detector end to end.*

## 🎓 Courses (free)
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford** — Lecture 11 is the definitive tour of the detection landscape.
- [Dive into Deep Learning — Object Detection](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang et al.** — free chapters on anchors, bounding boxes, SSD with runnable code.

## 🎥 Videos
- [How computers recognize objects instantly (TED)](https://www.youtube.com/watch?v=Cgxsv1riJhI) — **Joseph Redmon (YOLO author)** — the motivation and intuition for real-time detection.
- [CS231n Lecture 11 — Detection & Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — **Stanford** — the R-CNN family and one-stage detectors, authoritatively.
- [How YOLO Object Detection Works](https://www.youtube.com/watch?v=svn9-xV7wjk) — **DeepBean** — grid cells, anchors, and the single-pass pipeline, clearly.
- [What is the YOLO algorithm?](https://www.youtube.com/watch?v=ag3DLKsl2vk) — **codebasics** — an accessible, hands-on YOLO walkthrough.

## 📄 Key Papers
- [Rich feature hierarchies (R-CNN)](https://arxiv.org/abs/1311.2524) — **Girshick et al. (2013)** — region proposals + CNN features; started modern detection.
- [Fast R-CNN](https://arxiv.org/abs/1504.08083) — **Girshick (2015)** — RoI pooling; one forward pass per image.
- [Faster R-CNN](https://arxiv.org/abs/1506.01497) — **Ren et al. (2015)** — the learned Region Proposal Network; the two-stage standard.
- [You Only Look Once (YOLO)](https://arxiv.org/abs/1506.02640) — **Redmon et al. (2015)** — detection as single-pass grid regression.
- [Focal Loss for Dense Object Detection (RetinaNet)](https://arxiv.org/abs/1708.02002) — **Lin et al. (2017)** — solves foreground/background imbalance in one-stage detectors.

## 📰 Articles / Blogs (free, no paywall)
- [Intersection over Union (IoU)](https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/) — **LearnOpenCV** — the core box-overlap metric used throughout detection, free.
- [Mean Average Precision (mAP)](https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/) — **LearnOpenCV** — how detectors are scored, with worked examples.
- [Ultralytics YOLO docs](https://docs.ultralytics.com/) — **Ultralytics** — modern YOLO training/inference, fully open.

## 📚 Books (free, with chapters)
- [Dive into Deep Learning — **Ch. 14.3–14.7 (Bounding boxes, Anchors, SSD)**](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang et al.** — detection mechanics with runnable code.
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 6.3 (Object detection)**](https://szeliski.org/Book/) — **Richard Szeliski** — detection in context, free.

## 🔗 In this platform
- Foundation: [Classic CNN Architectures](03-Classic-CNN-Architectures.md) (detectors sit on these backbones) · [Deep Learning › CNNs & Convolution](../../05.%20Deep_Learning/concepts/13-CNNs-and-Convolution.md)
- Metrics: [10 Detection & Segmentation Metrics (IoU · mAP)](10-Detection-and-Segmentation-Metrics.md)
- Next concepts: [08 Semantic Segmentation](08-Semantic-Segmentation.md) · [09 Instance Segmentation](09-Instance-Segmentation.md)
