---
id: "07-computer-vision/object-detection/references"
topic: "Object Detection — References"
parent: "07-computer-vision/object-detection"
type: references
updated: 2026-07-03
---

# Object Detection — references and further reading

> Companion link library for **[Object Detection](07-Object-Detection.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is free / open (no paywall) and from a primary author or a recognized deep explainer — chosen for depth on the *detection task and its core machinery* (IoU, NMS, anchors, Average Precision, two-stage vs one-stage), not popularity.

**Start here — suggested path**:
1. **Build intuition** — watch [How computers learn to recognize objects instantly (TED)](https://www.youtube.com/watch?v=Cgxsv1riJhI) by YOLO's author. *The single-pass detection idea, motivated.*
2. **The full landscape** — watch ⭐ [CS231n Lecture 11 — Detection & Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) (**Stanford**). *The R-CNN family and one-stage detectors in one authoritative lecture.*
3. **Nail the metric** — watch [mAP (mean Average Precision) explained](https://www.youtube.com/watch?v=FppOzcDvaDI) (**Pinecone / James Briggs**), then read the [mAP walkthrough](https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/) (**LearnOpenCV**). *IoU → precision/recall → the PR curve → AP → mAP, with worked examples.*
4. **Read the sources** — [R-CNN](https://arxiv.org/abs/1311.2524) → [Fast R-CNN](https://arxiv.org/abs/1504.08083) → [Faster R-CNN](https://arxiv.org/abs/1506.01497) → [YOLO](https://arxiv.org/abs/1506.02640) → [SSD](https://arxiv.org/abs/1512.02325) → [RetinaNet (Focal Loss)](https://arxiv.org/abs/1708.02002) → [DETR](https://arxiv.org/abs/2005.12872). *The two-stage → one-stage → set-prediction arc.*
5. **Make it concrete** — run this chapter's [notebook](code/07-Object-Detection.ipynb): a real Faster R-CNN on a real photo, with IoU and NMS cross-checked against `torchvision.ops` and AP pinned to a hand-verified 11/12 worked example.

**Videos**:
- [CS231n Lecture 11 — Detection & Segmentation](https://www.youtube.com/watch?v=nDPWywWRIRo) — **Stanford (Fei-Fei Li / Johnson / Yeung)** — the definitive tour of the detection landscape: the R-CNN family, anchors, and one-stage detectors.
- [How computers learn to recognize objects instantly (TED)](https://www.youtube.com/watch?v=Cgxsv1riJhI) — **Joseph Redmon (YOLO author)** — the motivation and intuition for real-time single-pass detection.
- [mean Average Precision (mAP) explained](https://www.youtube.com/watch?v=FppOzcDvaDI) — **Pinecone (James Briggs)** — the clearest short walkthrough of IoU → precision/recall → the PR curve → AP → mAP.
- [YOLO object detection explained](https://www.youtube.com/watch?v=svn9-xV7wjk) — **DeepBean** — grid cells, anchors, and the single-pass pipeline, built up step by step.
- [Non-Maximum Suppression (NMS)](https://www.youtube.com/watch?v=VAo84c1hQX8) — **DeepBean** — the duplicate-removal algorithm and its crowded-scene failure mode, visually.
- [What is the YOLO algorithm?](https://www.youtube.com/watch?v=ag3DLKsl2vk) — **codebasics** — an accessible, hands-on YOLO walkthrough for first exposure.

**Interactive & visual**:
- [Ultralytics YOLO — run a detector in the browser / docs](https://docs.ultralytics.com/) — **Ultralytics** — train and run a modern YOLO detector end to end, fully open.
- [Papers with Code — Object Detection](https://paperswithcode.com/task/object-detection) — **Papers with Code** — live leaderboards, datasets (COCO, PASCAL VOC), and code for the current state of the art (free).
- [COCO dataset explorer](https://cocodataset.org/#explore) — **COCO** — browse the real annotated images and boxes the field is benchmarked on (the photo used in this chapter is COCO val2017).

**Courses (free)**:
- [Stanford CS231n — Deep Learning for Computer Vision](https://cs231n.github.io/) — **Stanford** — Lecture 11 is the canonical detection lecture; the notes situate detection among the recognition tasks.
- [Dive into Deep Learning — Object Detection (Ch. 14)](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang, Lipton, Li & Smola** — bounding boxes, anchors, IoU, multiscale, and SSD, all with runnable code.
- [Practical Deep Learning for Coders](https://course.fast.ai/) — **fast.ai (Jeremy Howard)** — a code-first path to training real detectors.

**Articles / blogs (free, no paywall)**:
- [Intersection over Union (IoU)](https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/) — **LearnOpenCV** — the core box-overlap metric with clear diagrams and code.
- [Mean Average Precision (mAP)](https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/) — **LearnOpenCV** — how detectors are scored, with worked precision-recall examples.
- [A (short) survey of object-detection metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics) — **Rafael Padilla** — the reference open-source implementation of AP/mAP (11-point and all-point), the exact protocol this chapter reproduces.
- [torchvision object detection reference](https://pytorch.org/vision/stable/models.html#object-detection) — **PyTorch** — the pretrained detectors (Faster R-CNN, RetinaNet, SSD) and `torchvision.ops` (box_iou, nms, box_convert) this chapter runs and checks against.
- [Zero to Hero: Faster R-CNN explained](https://d2l.ai/chapter_computer-vision/rcnn.html) — **d2l.ai** — the R-CNN → Fast → Faster progression, with the RoI pooling and RPN mechanics.

**Key papers** (the detection lineage + the metrics standards):
- [Rich feature hierarchies for accurate object detection (R-CNN)](https://arxiv.org/abs/1311.2524) — **Girshick, Donahue, Darrell & Malik (2014)** — region proposals + CNN features; the start of modern detection.
- [Fast R-CNN](https://arxiv.org/abs/1504.08083) — **Girshick (2015)** — RoI pooling; one backbone pass per image.
- [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497) — **Ren, He, Girshick & Sun (2015)** — the learned Region Proposal Network, anchors, and the $(t_x,t_y,t_w,t_h)$ box parametrization; the two-stage standard (the model this chapter runs).
- [You Only Look Once (YOLO)](https://arxiv.org/abs/1506.02640) — **Redmon, Divvala, Girshick & Farhadi (2016)** — detection as single-pass grid regression; real-time one-stage detection.
- [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325) — **Liu et al. (2016)** — multi-scale one-stage detection.
- [Focal Loss for Dense Object Detection (RetinaNet)](https://arxiv.org/abs/1708.02002) — **Lin, Goyal, Girshick, He & Dollár (2017)** — the foreground/background imbalance fix that closed the one-stage accuracy gap.
- [End-to-End Object Detection with Transformers (DETR)](https://arxiv.org/abs/2005.12872) — **Carion, Massa, Synnaeve, Usunier, Kirillov & Zagoruyko (2020)** — detection as set prediction, removing anchors and NMS.
- [The PASCAL Visual Object Classes (VOC) Challenge](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf) — **Everingham, Van Gool, Williams, Winn & Zisserman (2010)** — the AP / mAP evaluation protocol and IoU-matching rule.
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) — **Lin, Maire, Belongie, Hays, Perona, Ramanan, Dollár & Zitnick (2014)** — the benchmark and the stricter mAP@[.5:.95] metric.

**Books (free chapters)**:
- [Dive into Deep Learning — Ch. 14 (Computer Vision: Object Detection)](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang, Lipton, Li & Smola** — bounding boxes, anchors, IoU, NMS, and SSD with runnable code.
- [Computer Vision: Algorithms and Applications, 2nd ed. — Ch. 6.3 (Object detection)](https://szeliski.org/Book/) — **Richard Szeliski** — detection in the broader recognition context; the full PDF is free.

**In this platform**:
- Concept page (full explanation): [Object Detection](07-Object-Detection.md)
- Runnable code: [the module `object_detection.py`](code/object_detection.py) · [the step-by-step notebook](code/07-Object-Detection.ipynb) — a real Faster R-CNN on a real photo, with IoU and NMS from scratch (cross-checked against `torchvision.ops`) and AP pinned to a hand-verified 11/12 worked example; every metric measured, none mocked.
- The task detection builds on: [04 Image Classification](../04-Image-Classification/04-Image-Classification.md) — a detector is a classifier applied to regions plus a box regressor (we *use* classification here; that page derives it).
- The backbone: [05 Deep Learning › CNNs & Convolution](../../05.%20Deep_Learning/13-CNNs-and-Convolution/13-CNNs-and-Convolution.md) — convolution, receptive fields, and the ResNet backbone every detector runs once over the image.
- The metrics, in depth: [10 Detection & Segmentation Metrics (IoU · mAP)](../10-Detection-and-Segmentation-Metrics/10-Detection-and-Segmentation-Metrics.md) — the dedicated treatment of the evaluation protocol this page derives and measures.
- The sibling dense-prediction tasks: [08 Semantic Segmentation](../08-Semantic-Segmentation/08-Semantic-Segmentation.md) (per-pixel classification) · [09 Instance Segmentation](../09-Instance-Segmentation/09-Instance-Segmentation.md) (detection + a mask head; Mask R-CNN).
- Why generalization is the whole game: [00 Basics › Overfitting & Underfitting](../../00.%20Basics/05-Overfitting-and-Underfitting/05-Overfitting-and-Underfitting.md) — the gap augmentation and regularization shrink, which matters for training detectors on scarce boxes.
- Where detection goes next: [11 Vision Transformers](../11-Vision-Transformers/11-Vision-Transformers.md) — the ViT backbone and the DETR set-prediction reformulation that removes anchors and NMS.
- Field overview: [Computer Vision](../README.md)
