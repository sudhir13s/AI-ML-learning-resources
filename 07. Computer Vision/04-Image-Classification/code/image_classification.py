"""Image classification as an applied workflow, on REAL images, with every number MEASURED.

This is not a toy. It runs the real, end-to-end classification pipeline the chapter teaches — data ->
augmentation -> a pretrained backbone -> a new head -> cross-entropy -> honest evaluation — on real RGB
images (CIFAR-10), with real libraries (``torch`` / ``torchvision`` / ``numpy`` / ``scikit-learn``) and real,
reproducible accuracy. The single lesson is the **applied classification workflow**, and in particular the
one move that dominates it in practice: **transfer learning** — reuse the visual features a big model already
learned instead of training from pixels. It is proven several independent ways:

  1. **Transfer beats from-scratch at a small budget (measured).** A frozen ImageNet-pretrained ResNet-18
     backbone with a fresh linear head (a *linear probe*) is trained on a small CIFAR-10 subset and compared,
     at the *same* labeled-data budget, against a small CNN trained from random initialization. The transfer
     model wins by a wide, measured margin — the entire economic case for transfer learning, in one number.

  2. **Augmentation regularizes (measured).** The same from-scratch CNN is trained with and without
     label-preserving augmentation (random crop + horizontal flip). Augmentation sharply shrinks the
     train-minus-test *generalization gap* — the mechanical signature of regularization — measured on real
     held-out data. A hard ``assert`` guards the direction of the effect.

  3. **Metrics from scratch, cross-checked against scikit-learn.** Top-1 accuracy, top-5 accuracy, the
     confusion matrix, and per-class accuracy are each computed from scratch in NumPy *and* cross-checked
     against ``sklearn.metrics`` with a hard ``assert`` (they must match exactly) — so the numbers the chapter
     and figures quote are provably the standard quantities, not a bespoke re-definition.

  4. **Softmax + cross-entropy, from scratch, verified against torch.** The numerically-stable softmax and
     the cross-entropy loss are implemented by hand and checked against ``torch.nn.functional`` to ~1e-6.

Everything is **seeded and CPU-pinned for the reported numbers** so the trace is bit-reproducible on any
machine (GPU kernels are nondeterministic; we detect CUDA/MPS and report it, but pin the measured pipeline to
CPU on purpose). Run standalone::

    python image_classification.py

If CIFAR-10 or the pretrained weights cannot be downloaded (no network), the module *detects* that and falls
back to a real, bundled dataset (scikit-learn's 8x8 digits) with a real from-scratch CNN — still measured,
never mocked — and says so in the banner.

Verified on Python 3.12 / torch 2.12 / torchvision 0.27 / numpy 2.4 / scikit-learn 1.9 (CPU).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

SEED = 0

# CIFAR-10 is 32x32 RGB. The pretrained backbone was trained on ImageNet, so images fed to it are resized to
# 224 and normalized with ImageNet channel statistics (matching the distribution the backbone expects). The
# from-scratch CNN trains at native 32x32 with CIFAR's own statistics.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

_DATA_ROOT = os.environ.get("IC_DATA_ROOT", "./data")


# ================================================================================================
# Device: detect CUDA/MPS for reporting, but PIN the measured pipeline to CPU for a reproducible trace
# ================================================================================================


def detect_accelerator() -> str:
    """Report the best available accelerator (for the banner). We deliberately do NOT train on it: GPU/MPS
    convolution kernels are nondeterministic, and every number this module prints must be bit-reproducible."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


# ================================================================================================
# Softmax + cross-entropy from scratch (the classification loss), verified against torch
# ================================================================================================


def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """Row-wise numerically-stable softmax: subtract the row max before exp so exp never overflows.

    softmax(z)_i = exp(z_i - m) / sum_j exp(z_j - m),  m = max_j z_j  (the shift cancels, exactly).
    """
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def cross_entropy_from_scratch(logits: np.ndarray, y: np.ndarray) -> float:
    """Mean cross-entropy for integer labels: L = -mean_i log softmax(logits_i)[y_i].

    Computed via log-sum-exp for stability (never forms exp of a large number), the same quantity
    ``F.cross_entropy`` returns. Cross-linked to Foundations 23 (Cross-Entropy) for the derivation.
    """
    m = logits.max(axis=1, keepdims=True)
    logsumexp = m[:, 0] + np.log(np.exp(logits - m).sum(axis=1))
    correct = logits[np.arange(len(y)), y]
    return float(np.mean(logsumexp - correct))


@dataclass(frozen=True)
class LossCheck:
    ce_scratch: float
    ce_torch: float
    softmax_vs_torch: float
    ce_vs_torch: float


def verify_softmax_cross_entropy(seed: int = SEED) -> LossCheck:
    """Assert the from-scratch stable softmax and cross-entropy match ``torch.nn.functional`` to ~1e-6."""
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((7, 10)) * 3.0  # deliberately large-ish -> exercises numerical stability
    y = rng.integers(0, 10, size=7)
    p = softmax_stable(logits)
    p_torch = F.softmax(torch.tensor(logits), dim=1).numpy()
    ce = cross_entropy_from_scratch(logits, y)
    ce_torch = float(F.cross_entropy(torch.tensor(logits), torch.tensor(y)))
    softmax_vs = float(np.abs(p - p_torch).max())
    ce_vs = abs(ce - ce_torch)
    assert softmax_vs < 1e-6, f"softmax vs torch too large: {softmax_vs:.2e}"
    assert ce_vs < 1e-6, f"cross-entropy vs torch too large: {ce_vs:.2e}"
    return LossCheck(ce, ce_torch, softmax_vs, ce_vs)


# ================================================================================================
# Metrics from scratch (top-1, top-5, confusion matrix, per-class), cross-checked vs scikit-learn
# ================================================================================================


def top1_accuracy(logits: np.ndarray, y: np.ndarray) -> float:
    """Fraction of images whose single highest-scoring class is the true label."""
    return float((logits.argmax(axis=1) == y).mean())


def topk_accuracy(logits: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """Fraction of images whose true label is among the k highest-scoring classes (top-k accuracy).

    top-k is always >= top-1 and rises with k; it credits a prediction when the answer is 'in the shortlist',
    the standard ImageNet metric for a 1000-way task where the top guess is a demanding bar.
    """
    topk_idx = np.argpartition(-logits, kth=k - 1, axis=1)[:, :k]
    hits = (topk_idx == y[:, None]).any(axis=1)
    return float(hits.mean())


def confusion_matrix_from_scratch(pred: np.ndarray, y: np.ndarray, n_classes: int) -> np.ndarray:
    """C[t, p] = number of images with true class t predicted as class p. Rows sum to each class's support."""
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def per_class_accuracy(cm: np.ndarray) -> np.ndarray:
    """Diagonal over row-sum: the recall of each class (fraction of that class's images classified correctly)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.diag(cm) / cm.sum(axis=1)
    return np.nan_to_num(acc)


@dataclass(frozen=True)
class MetricCheck:
    top1: float
    top5: float
    top1_vs_sklearn: float
    top5_vs_sklearn: float
    confusion_matches_sklearn: bool


def verify_metrics(logits: np.ndarray, y: np.ndarray, n_classes: int) -> MetricCheck:
    """Cross-check the from-scratch top-1, top-5, and confusion matrix against scikit-learn (hard asserts)."""
    from sklearn.metrics import accuracy_score, confusion_matrix, top_k_accuracy_score

    pred = logits.argmax(axis=1)
    t1 = top1_accuracy(logits, y)
    t5 = topk_accuracy(logits, y, k=5)
    cm = confusion_matrix_from_scratch(pred, y, n_classes)

    t1_sk = float(accuracy_score(y, pred))
    t5_sk = float(top_k_accuracy_score(y, softmax_stable(logits), k=5, labels=np.arange(n_classes)))
    cm_sk = confusion_matrix(y, pred, labels=np.arange(n_classes))

    d1, d5 = abs(t1 - t1_sk), abs(t5 - t5_sk)
    cm_match = bool(np.array_equal(cm, cm_sk))
    assert d1 < 1e-9, f"top-1 vs sklearn mismatch: {d1:.2e}"
    assert d5 < 1e-9, f"top-5 vs sklearn mismatch: {d5:.2e}"
    assert cm_match, "confusion matrix disagrees with sklearn"
    return MetricCheck(t1, t5, d1, d5, cm_match)


# ================================================================================================
# Data: a balanced CIFAR-10 subset (real RGB images); a graceful, real fallback to sklearn digits
# ================================================================================================


@dataclass
class Dataset:
    name: str
    images_train: np.ndarray  # uint8 [N, H, W, 3] (or [N, H, W] grayscale for the fallback)
    labels_train: np.ndarray  # int64 [N]
    images_test: np.ndarray
    labels_test: np.ndarray
    classes: list[str]
    is_rgb: bool
    transfer_available: bool  # True only when a pretrained backbone could be loaded (CIFAR path)


def _balanced_indices(labels: np.ndarray, per_class: int, n_classes: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx: list[int] = []
    for c in range(n_classes):
        members = np.where(labels == c)[0]
        idx.extend(rng.choice(members, size=per_class, replace=False).tolist())
    idx_arr = np.array(idx)
    rng.shuffle(idx_arr)
    return idx_arr


def try_load_cifar(train_per_class: int, test_per_class: int, seed: int = SEED) -> Dataset | None:
    """Load a balanced CIFAR-10 subset AND confirm the pretrained backbone weights are reachable.

    Returns ``None`` (triggering the fallback) if either the dataset or the ImageNet weights can't be fetched.
    """
    try:
        import torchvision
        from torchvision.models import ResNet18_Weights, resnet18

        tr = torchvision.datasets.CIFAR10(root=_DATA_ROOT, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root=_DATA_ROOT, train=False, download=True)
        # Confirm weights are actually available (this is what makes transfer learning possible).
        resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    except Exception:  # noqa: BLE001 -- any failure (offline, disk) should trigger the real fallback
        return None

    tr_labels, te_labels = np.array(tr.targets), np.array(te.targets)
    tr_idx = _balanced_indices(tr_labels, train_per_class, 10, seed)
    te_idx = _balanced_indices(te_labels, test_per_class, 10, seed + 1)
    return Dataset(
        "CIFAR-10",
        tr.data[tr_idx], tr_labels[tr_idx], te.data[te_idx], te_labels[te_idx],
        list(tr.classes), is_rgb=True, transfer_available=True,
    )


def load_digits_fallback(seed: int = SEED) -> Dataset:
    """Real offline fallback: scikit-learn's 8x8 handwritten digits (bundled, no download). No pretrained
    backbone exists offline, so the transfer-learning contrast is skipped and only the from-scratch CNN runs —
    still real, measured accuracy on real images."""
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    d = load_digits()
    x = (d.images / 16.0 * 255).astype(np.uint8)  # [N,8,8] grayscale 0..255
    x_tr, x_te, y_tr, y_te = train_test_split(x, d.target, test_size=0.25, random_state=seed, stratify=d.target)
    return Dataset(
        "sklearn-digits (fallback)",
        x_tr, y_tr.astype(np.int64), x_te, y_te.astype(np.int64),
        [str(i) for i in range(10)], is_rgb=False, transfer_available=False,
    )


# ================================================================================================
# The pretrained backbone -> frozen feature extractor (the reusable ImageNet features)
# ================================================================================================


def _to_backbone_batch(images_uint8: np.ndarray, size: int = 224) -> torch.Tensor:
    """Resize real RGB uint8 images to the backbone's expected size and apply ImageNet normalization.

    [N,H,W,3] uint8 -> [N,3,size,size] float, (x - ImageNet_mean) / ImageNet_std.
    """
    x = torch.tensor(images_uint8).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def extract_features(images_uint8: np.ndarray, batch_size: int = 128, size: int = 224) -> torch.Tensor:
    """Run every image through a FROZEN ImageNet-pretrained ResNet-18 (fc replaced by identity) and return the
    512-D penultimate feature per image. This is the reusable representation transfer learning stands on: the
    backbone's weights are never updated; we only read out its features. CPU-pinned for reproducibility.
    """
    from torchvision.models import ResNet18_Weights, resnet18

    backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    backbone.fc = torch.nn.Identity()  # keep the 512-D global-average-pooled features, drop the 1000-way head
    backbone.eval()
    feats: list[torch.Tensor] = []
    for start in range(0, len(images_uint8), batch_size):
        batch = _to_backbone_batch(images_uint8[start : start + batch_size], size)
        feats.append(backbone(batch))
    return torch.cat(feats)


# ================================================================================================
# Models: a linear head (transfer / linear probe) and a small from-scratch CNN
# ================================================================================================


class SmallCNN(torch.nn.Module):
    """A compact conv->BN->ReLU->pool stack with a global-average-pooled linear head. Trained FROM SCRATCH
    (random init) as the honest baseline transfer learning is measured against. Works on RGB (CIFAR) or a
    single grayscale channel (the digits fallback)."""

    def __init__(self, in_ch: int = 3, n_classes: int = 10) -> None:
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, 32, 3, padding=1), torch.nn.BatchNorm2d(32), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 3, padding=1), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.head = torch.nn.Linear(128, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---- Transfer learning: train a linear head on the frozen features (a "linear probe") -----------


@dataclass
class ProbeResult:
    logits: np.ndarray  # test-set logits [N_test, n_classes]
    n_head_params: int
    train_curve: list[float]  # train cross-entropy per epoch
    val_curve: list[float]  # test top-1 accuracy per epoch


def train_linear_probe(
    feat_train: torch.Tensor, y_train: np.ndarray, feat_test: torch.Tensor, y_test: np.ndarray,
    *, epochs: int = 40, lr: float = 1e-3, seed: int = SEED,
) -> ProbeResult:
    """Freeze the backbone, train ONLY a linear classifier on top of its cached features. This is the fastest,
    strongest transfer baseline (a 'linear probe'): the ImageNet features are so general that a single linear
    layer reads CIFAR classes off them. Everything is CPU + seeded for reproducibility."""
    torch.manual_seed(seed)
    n_classes = int(max(y_train.max(), y_test.max())) + 1
    head = torch.nn.Linear(feat_train.shape[1], n_classes)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    xt, yt = feat_train, torch.tensor(y_train)
    train_curve, val_curve = [], []
    for _ in range(epochs):
        head.train()
        opt.zero_grad()
        loss = F.cross_entropy(head(xt), yt)
        loss.backward()
        opt.step()
        train_curve.append(loss.item())
        head.eval()
        with torch.no_grad():
            val_curve.append(top1_accuracy(head(feat_test).numpy(), y_test))
    head.eval()
    with torch.no_grad():
        logits = head(feat_test).numpy()
    return ProbeResult(logits, count_params(head), train_curve, val_curve)


# ---- From-scratch CNN training, with an optional augmentation pipeline ---------------------------


def _normalize_images(images_uint8: np.ndarray, mean: tuple[float, ...], std: tuple[float, ...], is_rgb: bool) -> torch.Tensor:
    if is_rgb:
        x = torch.tensor(images_uint8).permute(0, 3, 1, 2).float() / 255.0
        m = torch.tensor(mean).view(1, 3, 1, 1)
        s = torch.tensor(std).view(1, 3, 1, 1)
    else:
        x = torch.tensor(images_uint8).unsqueeze(1).float() / 255.0  # [N,1,H,W]
        m = torch.tensor(mean[:1]).view(1, 1, 1, 1)
        s = torch.tensor(std[:1]).view(1, 1, 1, 1)
    return (x - m) / s


@dataclass
class ScratchResult:
    logits: np.ndarray  # test logits
    n_params: int
    train_curve: list[float]  # train cross-entropy per epoch
    val_curve: list[float]  # test top-1 accuracy per epoch
    train_acc: float  # final training accuracy (to expose the generalization gap)
    test_acc: float


def train_from_scratch(
    data: Dataset, *, augment: bool, epochs: int = 40, lr: float = 2e-3, batch_size: int = 128, seed: int = SEED,
) -> ScratchResult:
    """Train ``SmallCNN`` from random initialization on the real images. With ``augment=True`` each mini-batch is
    passed through label-preserving random crop + horizontal flip (regularization). CPU + fully seeded."""
    _seed_everything(seed)
    mean, std = (CIFAR_MEAN, CIFAR_STD) if data.is_rgb else ((0.5,) * 3, (0.5,) * 3)
    x_train = _normalize_images(data.images_train, mean, std, data.is_rgb)
    x_test = _normalize_images(data.images_test, mean, std, data.is_rgb)
    y_train = torch.tensor(data.labels_train)
    in_ch = 3 if data.is_rgb else 1
    n_classes = len(data.classes)

    model = SmallCNN(in_ch=in_ch, n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    aug = None
    if augment:
        import torchvision.transforms.v2 as T

        pad = 4 if data.is_rgb else 2
        aug = T.Compose([T.RandomCrop(x_train.shape[-1], padding=pad), T.RandomHorizontalFlip()])

    n = x_train.shape[0]
    train_curve, val_curve = [], []
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            xb = x_train[idx]
            if aug is not None:
                xb = aug(xb)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), y_train[idx])
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        train_curve.append(epoch_loss / n)
        model.eval()
        with torch.no_grad():
            val_curve.append(top1_accuracy(model(x_test).numpy(), data.labels_test))

    model.eval()
    with torch.no_grad():
        logits = model(x_test).numpy()
        train_acc = top1_accuracy(model(x_train).numpy(), data.labels_train)
    return ScratchResult(
        logits, count_params(model), train_curve, val_curve, train_acc, top1_accuracy(logits, data.labels_test)
    )


# ================================================================================================
# Honest predictions for the sample grid (deliberately include correct AND wrong cases)
# ================================================================================================


@dataclass(frozen=True)
class SamplePredictions:
    images: np.ndarray  # [K, H, W, 3] or [K, H, W]
    true_labels: np.ndarray
    pred_labels: np.ndarray
    confidences: np.ndarray  # softmax probability of the predicted class
    classes: list[str]


def sample_predictions(
    data: Dataset, logits: np.ndarray, n_correct: int = 8, n_wrong: int = 4, seed: int = SEED
) -> SamplePredictions:
    """Pick a mix of correctly- and incorrectly-classified test images so the figure is HONEST — a classifier
    that only ever showed its wins would be a lie. Confidence = softmax prob of the predicted class."""
    rng = np.random.default_rng(seed)
    probs = softmax_stable(logits)
    pred = logits.argmax(axis=1)
    correct = np.where(pred == data.labels_test)[0]
    wrong = np.where(pred != data.labels_test)[0]
    take_c = rng.choice(correct, size=min(n_correct, len(correct)), replace=False)
    take_w = rng.choice(wrong, size=min(n_wrong, len(wrong)), replace=False)
    idx = np.concatenate([take_c, take_w])
    return SamplePredictions(
        data.images_test[idx], data.labels_test[idx], pred[idx],
        probs[idx, pred[idx]], data.classes,
    )


# ================================================================================================
# The full experiment, bundled (so figures and the notebook reuse one measured run)
# ================================================================================================


@dataclass
class Experiment:
    dataset_name: str
    classes: list[str]
    n_train: int
    n_test: int
    accelerator: str
    # transfer vs scratch
    transfer: ProbeResult | None
    transfer_top1: float | None
    transfer_top5: float | None
    scratch_noaug: ScratchResult
    scratch_aug: ScratchResult
    transfer_delta: float | None  # transfer_top1 - scratch_noaug.test_acc
    aug_gap_noaug: float  # train_acc - test_acc without augmentation
    aug_gap_aug: float  # train_acc - test_acc with augmentation
    aug_test_delta: float  # scratch_aug.test_acc - scratch_noaug.test_acc
    # evaluation of the best (transfer if available, else scratch) model
    best_name: str
    best_logits: np.ndarray = field(repr=False)
    confusion: np.ndarray = field(repr=False)
    per_class: np.ndarray = field(repr=False)
    metric_check: MetricCheck = field(repr=False)
    samples: SamplePredictions = field(repr=False)
    loss_check: LossCheck = field(repr=False)


def run_experiment(train_per_class: int = 300, test_per_class: int = 100, epochs: int = 40, seed: int = SEED) -> Experiment:
    """Run the whole measured pipeline once and return every quantity the chapter, figures, and notebook cite."""
    _seed_everything(seed)
    accelerator = detect_accelerator()
    loss_check = verify_softmax_cross_entropy(seed)

    data = try_load_cifar(train_per_class, test_per_class, seed)
    if data is None:
        data = load_digits_fallback(seed)

    # --- transfer learning (linear probe on frozen ImageNet features) ---
    transfer: ProbeResult | None = None
    transfer_top1 = transfer_top5 = transfer_delta = None
    if data.transfer_available:
        feat_tr = extract_features(data.images_train)
        feat_te = extract_features(data.images_test)
        transfer = train_linear_probe(feat_tr, data.labels_train, feat_te, data.labels_test, epochs=epochs, seed=seed)
        transfer_top1 = top1_accuracy(transfer.logits, data.labels_test)
        transfer_top5 = topk_accuracy(transfer.logits, data.labels_test, k=5)

    # --- from-scratch CNN, with and without augmentation ---
    scratch_noaug = train_from_scratch(data, augment=False, epochs=epochs, seed=seed)
    scratch_aug = train_from_scratch(data, augment=True, epochs=epochs, seed=seed)

    if transfer_top1 is not None:
        transfer_delta = transfer_top1 - scratch_noaug.test_acc

    aug_gap_noaug = scratch_noaug.train_acc - scratch_noaug.test_acc
    aug_gap_aug = scratch_aug.train_acc - scratch_aug.test_acc
    aug_test_delta = scratch_aug.test_acc - scratch_noaug.test_acc

    # --- honest evaluation of the best available model ---
    if transfer is not None:
        best_name, best_logits = "transfer (linear probe)", transfer.logits
    else:
        best_name, best_logits = "from-scratch CNN", scratch_aug.logits
    n_classes = len(data.classes)
    metric_check = verify_metrics(best_logits, data.labels_test, n_classes)
    cm = confusion_matrix_from_scratch(best_logits.argmax(axis=1), data.labels_test, n_classes)
    per_class = per_class_accuracy(cm)
    samples = sample_predictions(data, best_logits, seed=seed)

    return Experiment(
        data.name, data.classes, len(data.labels_train), len(data.labels_test), accelerator,
        transfer, transfer_top1, transfer_top5, scratch_noaug, scratch_aug,
        transfer_delta, aug_gap_noaug, aug_gap_aug, aug_test_delta,
        best_name, best_logits, cm, per_class, metric_check, samples, loss_check,
    )


# ================================================================================================
# Report
# ================================================================================================


def main() -> None:
    import torchvision

    exp = run_experiment()
    print(
        f"torch {torch.__version__} | torchvision {torchvision.__version__} | numpy {np.__version__} "
        f"(reported on CPU, seed={SEED}; accelerator available: {exp.accelerator})\n"
    )

    print(f"=== Dataset: {exp.dataset_name} ===")
    print(f"  {exp.n_train} train / {exp.n_test} test, {len(exp.classes)} classes: {exp.classes}\n")

    print("=== Softmax + cross-entropy from scratch vs torch ===")
    lc = exp.loss_check
    print(f"  CE(scratch)={lc.ce_scratch:.6f}  CE(torch)={lc.ce_torch:.6f}  "
          f"|softmax-torch|={lc.softmax_vs_torch:.1e}  |CE-torch|={lc.ce_vs_torch:.1e}\n")

    if exp.transfer_top1 is not None:
        print("=== Transfer learning (frozen ResNet-18 backbone + linear head) vs from-scratch CNN ===")
        print(f"  transfer   : top-1 = {exp.transfer_top1:.4f}   top-5 = {exp.transfer_top5:.4f}   "
              f"(head params = {exp.transfer.n_head_params:,}, backbone FROZEN)")
        print(f"  from-scratch: top-1 = {exp.scratch_noaug.test_acc:.4f}                    "
              f"(all {exp.scratch_noaug.n_params:,} params trained from random init)")
        print(f"  -> transfer beats from-scratch by {exp.transfer_delta * 100:+.1f} points at the SAME "
              f"{exp.n_train}-image budget\n")
    else:
        print("=== Transfer learning skipped (offline fallback: no pretrained backbone) ===\n")

    print("=== Data augmentation ablation (same from-scratch CNN, +/- random crop & flip) ===")
    print(f"  no aug: train acc = {exp.scratch_noaug.train_acc:.4f}  test acc = {exp.scratch_noaug.test_acc:.4f}  "
          f"gap = {exp.aug_gap_noaug * 100:.1f} pts")
    print(f"  aug   : train acc = {exp.scratch_aug.train_acc:.4f}  test acc = {exp.scratch_aug.test_acc:.4f}  "
          f"gap = {exp.aug_gap_aug * 100:.1f} pts")
    print(f"  -> augmentation cuts the train-test gap {exp.aug_gap_noaug * 100:.1f} -> {exp.aug_gap_aug * 100:.1f} pts "
          f"(test acc delta {exp.aug_test_delta * 100:+.1f} pts)\n")

    print(f"=== Metrics from scratch on the best model ({exp.best_name}), cross-checked vs scikit-learn ===")
    mc = exp.metric_check
    print(f"  top-1 = {mc.top1:.4f}  top-5 = {mc.top5:.4f}   "
          f"(|top1-sklearn|={mc.top1_vs_sklearn:.1e}  |top5-sklearn|={mc.top5_vs_sklearn:.1e}  "
          f"confusion==sklearn: {mc.confusion_matches_sklearn})")
    order = np.argsort(exp.per_class)
    worst = ", ".join(f"{exp.classes[i]} {exp.per_class[i]:.2f}" for i in order[:3])
    best = ", ".join(f"{exp.classes[i]} {exp.per_class[i]:.2f}" for i in order[::-1][:3])
    print(f"  best classes : {best}")
    print(f"  worst classes: {worst}\n")

    # --- hard asserts on the headline relationships (raise, not print, if the lesson breaks) ---
    if exp.transfer_delta is not None and exp.transfer_delta < 0.10:
        raise AssertionError(f"transfer should beat from-scratch by >10 pts, got {exp.transfer_delta * 100:.1f}")
    if exp.aug_gap_aug >= exp.aug_gap_noaug:
        raise AssertionError(
            f"augmentation should shrink the train-test gap: {exp.aug_gap_aug:.3f} !< {exp.aug_gap_noaug:.3f}"
        )
    print("All checks passed (transfer > from-scratch; augmentation shrinks the generalization gap; "
          "from-scratch metrics == scikit-learn).")


if __name__ == "__main__":
    main()
