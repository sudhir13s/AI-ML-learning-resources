---
id: "07-computer-vision/image-classification/references"
topic: "Image Classification — References"
parent: "07-computer-vision/image-classification"
type: references
updated: 2026-07-03
---

# Image Classification — references and further reading

> Companion link library for **[Image Classification](image-classification.md)** (the concept page). External sources *and* internal links to related pages on this platform, kept separate so it can be reused as a standalone reference list. Grouped by type, best-first. Every entry is free / open (no paywall) and from a primary author or a recognized deep explainer — chosen for depth on the *applied classification workflow* (data, transfer learning, augmentation, evaluation), not popularity.

**Start here — suggested path**:
1. **Frame the task** — watch [CS231n Lecture 2: Image Classification](https://www.youtube.com/watch?v=OoUX-nOEjG0) (**Stanford**), then read the [CS231n classification notes](https://cs231n.github.io/classification/). *The data-driven paradigm, the nearest-neighbor baseline, and why we need learned features.*
2. **Get the loss + a CNN classifier** — watch [Image Classification with CNNs](https://www.youtube.com/watch?v=HGwBXDKFk9I) (**StatQuest**), then skim [d2l Ch. 4 — Softmax Regression](https://d2l.ai/chapter_linear-classification/index.html). *Softmax + cross-entropy and how a network turns pixels into a class.*
3. **Learn the move that matters** — read [CS231n — Transfer Learning](https://cs231n.github.io/transfer-learning/) and the [Yosinski et al. paper](https://arxiv.org/abs/1411.1792). *Freeze vs fine-tune, and why low-level features transfer.*
4. **Do it in code** — follow the [fast.ai course](https://course.fast.ai/) (Lesson 1 fine-tunes a pretrained classifier in minutes) and the [PyTorch transfer-learning tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). *The real freeze-then-fine-tune workflow.*
5. **Make it concrete** — run this chapter's [notebook](code/image-classification.ipynb): a frozen ResNet-18 + linear probe beats a from-scratch CNN by a measured margin, augmentation shrinks the overfitting gap, and every metric is cross-checked against scikit-learn.

**Videos**:
- [CS231n Lecture 2 — Image Classification](https://www.youtube.com/watch?v=OoUX-nOEjG0) — **Stanford (Fei-Fei Li / Johnson / Yeung)** — the definitive intro to the classification task, nearest-neighbor baselines, and the data-driven paradigm.
- [Image Classification with Convolutional Neural Networks](https://www.youtube.com/watch?v=HGwBXDKFk9I) — **StatQuest (Josh Starmer)** — a gentle, from-scratch walkthrough of a CNN classifier, one step at a time.
- [Building makemore / Zero to Hero](https://www.youtube.com/watch?v=PaCmpygFfXo) — **Andrej Karpathy** — the training loop, softmax/cross-entropy, and evaluation built by hand (the mechanics under every classifier).
- [Lesson 1: Practical Deep Learning for Coders](https://www.youtube.com/watch?v=8SF_h3xF3cE) — **fast.ai (Jeremy Howard)** — fine-tunes a *pretrained* image classifier to high accuracy in minutes; the transfer-learning workflow, code-first.
- [Transfer Learning](https://www.youtube.com/watch?v=yofjFQddwHE) — **DeepLearning.AI (Andrew Ng)** — the clearest short explanation of *why and when* to transfer, freeze, and fine-tune.

**Interactive & visual**:
- [CNN Explainer](https://poloclub.github.io/cnn-explainer/) — **Georgia Tech (Polo Club)** — a real trained image classifier running in your browser; hover any neuron to watch pixels become class scores, layer by layer.
- [Papers with Code — Image Classification](https://paperswithcode.com/task/image-classification) — **Papers with Code** — live leaderboards, datasets, and code for the current state of the art (free).
- [Know Your Data / TensorFlow Datasets — CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10) — **TensorFlow** — browse the actual dataset used in this chapter, class by class.

**Courses (free)**:
- [Stanford CS231n — Deep Learning for Computer Vision](https://cs231n.github.io/) — **Stanford** — the definitive course; the [classification](https://cs231n.github.io/classification/) and [transfer-learning](https://cs231n.github.io/transfer-learning/) notes are the canonical written references for this page.
- [Practical Deep Learning for Coders](https://course.fast.ai/) — **fast.ai (Jeremy Howard)** — the fastest path to training real classifiers; transfer learning and augmentation are front and center from Lesson 1.
- [Dive into Deep Learning — Linear/Softmax Classification + Modern CNNs](https://d2l.ai/chapter_linear-classification/index.html) — **Zhang, Lipton, Li & Smola** — softmax/cross-entropy then deep classifiers, with runnable code.

**Articles / blogs (free, no paywall)**:
- [CS231n — Image Classification](https://cs231n.github.io/classification/) — **Stanford CS231n** — the data-driven pipeline, k-NN, and the train/val/test discipline.
- [CS231n — Transfer Learning](https://cs231n.github.io/transfer-learning/) — **Stanford CS231n** — the definitive short reference on feature extraction (linear probe) vs fine-tuning, and when to use each.
- [PyTorch — Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) — **PyTorch** — freeze-then-fine-tune a pretrained ResNet on a real dataset, in runnable code (the workflow this chapter measures).
- [A Survey on Image Data Augmentation for Deep Learning](https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0197-0) — **Shorten & Khoshgoftaar (2019, open access)** — a thorough, free survey of augmentation techniques and why they regularize.
- [timm (PyTorch Image Models) documentation](https://huggingface.co/docs/timm/index) — **Ross Wightman / Hugging Face** — the library practitioners actually reach for: hundreds of pretrained backbones and the fine-tuning recipes around them.

**Key papers**:
- [ImageNet: A Large-Scale Hierarchical Image Database](https://www.image-net.org/static_files/papers/imagenet_cvpr09.pdf) — **Deng, Dong, Socher, Li, Li & Fei-Fei (2009)** — the dataset and top-1/top-5 evaluation that defined the classification benchmark.
- [ImageNet Classification with Deep Convolutional Neural Networks (AlexNet)](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) — **Krizhevsky, Sutskever & Hinton (2012)** — the deep-CNN result that made learned features dominant and launched the era.
- [How transferable are features in deep neural networks?](https://arxiv.org/abs/1411.1792) — **Yosinski, Clune, Bengio & Lipson (2014)** — the measurement that justifies transfer learning: lower layers are general, upper layers task-specific, and transfer+fine-tune beats from-scratch.
- [Deep Residual Learning for Image Recognition (ResNet)](https://arxiv.org/abs/1512.03385) — **He, Zhang, Ren & Sun (2016)** — the backbone architecture used in this chapter; skip connections that made very deep, transferable backbones trainable.
- [ImageNet Large Scale Visual Recognition Challenge (ILSVRC)](https://arxiv.org/abs/1409.0575) — **Russakovsky et al. (2015)** — the challenge, its metrics, and the arc of progress that drove every architecture advance.
- [An Image is Worth 16×16 Words (ViT)](https://arxiv.org/abs/2010.11929) — **Dosovitskiy et al. (2020)** — Vision Transformers; where classification backbones go next, and the data-vs-prior trade against CNNs.

**Books (free chapters)**:
- [Dive into Deep Learning — Ch. 4 (Linear Classification) + Ch. 8 (Modern CNNs)](https://d2l.ai/chapter_linear-classification/index.html) — **Zhang, Lipton, Li & Smola** — softmax/cross-entropy through deep classifiers, all with runnable code.
- [Computer Vision: Algorithms and Applications, 2nd ed. — Ch. 6 (Recognition)](https://szeliski.org/Book/) — **Richard Szeliski** — classification situated in the broader recognition landscape; the full PDF is free.

**In this platform**:
- Concept page (full explanation): [Image Classification](image-classification.md)
- Runnable code: [the workflow module `image_classification.py`](code/image_classification.py) · [the step-by-step notebook](code/image-classification.ipynb) — the module runs the real pipeline on CIFAR-10 (frozen ResNet-18 + linear probe vs a from-scratch CNN, an augmentation ablation, and honest evaluation), with every metric cross-checked against scikit-learn; the notebook runs it one measured step at a time.
- The CNN *mechanism* this workflow uses: [05 Deep Learning › CNNs & Convolution](../../../deep-learning/neural-architectures/cnns-and-convolution/cnns-and-convolution.md) — convolution, filters, receptive fields, the LeNet→ResNet lineage (we *use* a CNN here; that page *derives* it).
- The two moves that carry this chapter: [05 Transfer Learning for Vision](../transfer-learning-for-vision/transfer-learning-for-vision.md) (freeze vs fine-tune, in depth) · [06 Data Augmentation](../data-augmentation/data-augmentation.md) (the transform catalogue and theory).
- The loss, derived: [01 Foundations › Cross-Entropy & KL Divergence](../../../foundations/mathematical-foundations/cross-entropy-and-kl-divergence/cross-entropy-and-kl-divergence.md) — softmax cross-entropy as KL from the one-hot label, and its gradient.
- Why generalization is the whole game: [00 Basics › Overfitting & Underfitting](../../../foundations/ai-ml-orientation/overfitting-and-underfitting/overfitting-and-underfitting.md) (the gap augmentation shrinks) · test-set discipline: [02 Data Preprocessing › Data Leakage](../../../foundations/data-preparation/data-leakage/data-leakage.md) · when accuracy lies: [02 Data Preprocessing › Imbalanced Data](../../../foundations/data-preparation/imbalanced-data/imbalanced-data.md)
- Where classification goes next: [07 Object Detection](../object-detection/object-detection.md) (a detector is a classifier over regions) · [10 Detection & Segmentation Metrics](../detection-and-segmentation-metrics/detection-and-segmentation-metrics.md) (mAP, IoU — per-class metrics done right) · [11 Vision Transformers](../vision-transformers/vision-transformers.md) (the post-CNN classification backbone)
- Field overview: [Computer Vision](../README.md)
