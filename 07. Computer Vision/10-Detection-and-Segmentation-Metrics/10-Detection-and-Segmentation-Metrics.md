---
id: "07-computer-vision/detection-and-segmentation-metrics"
topic: "Detection & Segmentation Metrics (IoU, mAP)"
parent: "07-computer-vision"
level: intermediate
prereqs: ["object-detection", "precision-recall"]
interview_frequency: very-high
updated: 2026-06-20
---

# Detection & Segmentation Metrics — IoU · mAP
> **IoU** (intersection over union) measures how well a predicted box/mask overlaps the ground truth; an
> IoU threshold turns a prediction into a true/false positive. **AP** is the area under that class's
> precision–recall curve, and **mAP** averages AP over all classes (and often over IoU thresholds, e.g.
> COCO's mAP@[.5:.95]). These are *the* numbers every detection/segmentation paper reports.

**Why it matters:** an extremely common, easy-to-probe question — define IoU, explain how an IoU
threshold + confidence ranking builds the PR curve, derive AP and mAP, and explain the difference
between Pascal VOC mAP@0.5 and COCO mAP averaged over thresholds. Interviewers love this because it
tests whether you actually understand how your model is scored, not just how it's trained.

**⭐ Start here — suggested path:**

1. **Get IoU first** — read ⭐ [Intersection over Union (LearnOpenCV)](https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/). *The atomic building block of every detection metric.*
2. **Build intuition for mAP** — watch [What is mAP? (Roboflow)](https://www.youtube.com/watch?v=oqXDdxF_Wuw). *From IoU and confidence to a single AP number.*
3. **Work an example** — watch ⭐ [mAP Explained + Implementation (Aladdin Persson)](https://www.youtube.com/watch?v=FppOzcDvaDI). *The full PR-curve → AP → mAP computation, step by step.*
4. **Read the reference** — [COCO detection evaluation](https://cocodataset.org/#detection-eval). *The exact mAP@[.5:.95] protocol modern papers report.*
5. **Connect to PR curves** — read [mAP for object detection (LearnOpenCV)](https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/). *Ties mAP back to the precision/recall foundation you already know.*

## 🎓 Courses (free)
- [Stanford CS231n](https://cs231n.github.io/) — **Stanford** — covers detection evaluation (IoU, mAP) within the detection lectures.
- [COCO detection evaluation](https://cocodataset.org/#detection-eval) — **COCO** — the authoritative metric definitions and protocol used across the field.

## 🎥 Videos
- [What is Mean Average Precision (mAP)?](https://www.youtube.com/watch?v=oqXDdxF_Wuw) — **Roboflow** — clear, visual intro from IoU to mAP.
- [mAP Explained + PyTorch Implementation](https://www.youtube.com/watch?v=FppOzcDvaDI) — **Aladdin Persson** — the full PR-curve → AP → mAP computation, step by step.
- [mAP — Explanation and Implementation](https://www.youtube.com/watch?v=duBGmrxNHS8) — **ExplainingAI** — IoU, NMS, and mAP together with worked examples.
- [Performance Metrics for Object Detection](https://www.youtube.com/watch?v=wq8lnqiUaR4) — **Convolve AI** — precision, recall, AP, and mAP from first principles.

## 📄 Key Papers
- [The PASCAL Visual Object Classes (VOC) Challenge](http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf) — **Everingham et al. (2010)** — defined the AP/mAP@0.5 detection metric.
- [Microsoft COCO: Common Objects in Context](https://arxiv.org/abs/1405.0312) — **Lin et al. (2014)** — introduced the COCO benchmark and the mAP@[.5:.95] protocol.

## 📰 Articles / Blogs (free, no paywall)
- [Intersection over Union (IoU)](https://learnopencv.com/intersection-over-union-iou-in-object-detection-and-segmentation/) — **LearnOpenCV** — the overlap metric, with formulas and code, free.
- [Mean Average Precision (mAP)](https://learnopencv.com/mean-average-precision-map-object-detection-model-evaluation-metric/) — **LearnOpenCV** — AP/mAP derived from precision–recall, with worked examples.
- [COCO detection evaluation](https://cocodataset.org/#detection-eval) — **COCO** — the exact protocol papers report, free.

## 📚 Books (free, with chapters)
- [Computer Vision: Algorithms and Applications, 2nd ed. — **Ch. 6.3 (Detection & evaluation)**](https://szeliski.org/Book/) — **Richard Szeliski** — detection metrics in context, free.
- [Dive into Deep Learning — **Ch. 14 (Computer Vision)**](https://d2l.ai/chapter_computer-vision/index.html) — **Zhang et al.** — detection foundations these metrics evaluate, with code.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 3.06 ROC / AUC / PR Curves](../../../AI-ML-intuition/Module_3_Evaluation/3.06_ROC_AUC_PR_Curves.md) (mAP = AP per class) · [3.05 Precision / Recall / F1](../../../AI-ML-intuition/Module_3_Evaluation/3.05_Classification_Metrics_Precision_Recall_F1.md)
- Foundation: [Object Detection](../07-Object-Detection/07-Object-Detection.md) · [Instance Segmentation](../09-Instance-Segmentation/09-Instance-Segmentation.md) · [Semantic Segmentation](../08-Semantic-Segmentation/08-Semantic-Segmentation.md)
