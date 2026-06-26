"""From-scratch knowledge distillation: see the dark knowledge, then prove KD beats hard-only.

Three things, end to end, on a tiny synthetic classification task:

  1. SOFT TARGETS -- print one teacher distribution at T=1 vs T=4 so the "dark knowledge"
     (the off-target probability mass on similar classes) becomes visible.
  2. THE KD LOSS -- implement L = alpha * T^2 * KL(student_T || teacher_T) + (1-alpha) * CE,
     with the temperature division, the KL term, and the T^2 rescale each commented inline.
  3. THE PAYOFF -- train a student two ways (hard-labels-only vs KD) from the SAME init and
     show the KD student agrees with the teacher MORE (assert KD agreement >= hard-only).

Correctness (the dark-knowledge print and the agreement assert) is established before any
timing. Device-agnostic (CUDA / MPS / CPU): the device is DETECTED, but the reproducible
training run is PINNED to CPU and the print says so honestly -- the small nets and fixed seed
make CPU both fast and deterministic. The same verified code lives in the teaching notebook.

Verified on Python 3.12 / torch 2.x.

Run:
    python knowledge_distillation.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Hyperparameters (the two that define a KD run, hoisted and named) ---------------
TEMPERATURE = 4.0   # T: softmax temperature for the soft targets; T>1 flattens the distribution to expose dark knowledge
ALPHA = 0.9         # alpha: weight on the soft (distillation) term; (1-alpha) goes to the hard-label term
N_CLASSES = 5       # a 5-class toy problem
N_FEATURES = 20     # input dimensionality
NOISE_STD = 0.55    # blob spread: small enough that a strong teacher is genuinely accurate, so its soft targets carry real signal
N_TEACHER_TRAIN = 4_000  # the teacher sees ABUNDANT data -> it becomes accurate
N_STUDENT_TRAIN = 150    # the student sees a SMALL labelled subset -> the classic data-starved regime where soft targets add information the few hard labels can't
N_TEST = 2_000      # synthetic test points
TEACHER_HIDDEN = 256  # the teacher is wide -> accurate but expensive
STUDENT_HIDDEN = 8    # the student is narrow -> cheap, 32x fewer hidden units
TEACHER_EPOCHS = 200
STUDENT_EPOCHS = 300
LEARNING_RATE = 0.01
SEED = 0

# Detect the best accelerator for honesty, but PIN the reproducible run to CPU so the
# fixed-seed training is deterministic and the printed device matches where it actually runs.
DETECTED_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
DEVICE = torch.device("cpu")  # the reproducible trace runs on CPU regardless of what's detected


def make_synthetic_data(
    n_samples: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """A Gaussian-blob classification task with a deliberate similarity structure.

    Each class has a mean vector; classes 0 ("cat") and 1 ("dog") sit CLOSE together so a
    good teacher learns they are confusable -- that confusability is the dark knowledge KD
    transfers. Returns features (n_samples, N_FEATURES) and integer labels (n_samples,).
    """
    # Class means: 0 and 1 are near each other (similar), the rest spread out (distinct).
    means = torch.zeros(N_CLASSES, N_FEATURES, device=DEVICE)
    means[0, :3] = torch.tensor([2.0, 2.0, 0.0], device=DEVICE)   # "cat"
    means[1, :3] = torch.tensor([2.0, 1.4, 0.5], device=DEVICE)   # "dog": close to cat on purpose -> confusable
    means[2, :3] = torch.tensor([-2.0, 2.0, 0.0], device=DEVICE)  # distinct
    means[3, :3] = torch.tensor([0.0, -2.5, 1.0], device=DEVICE)  # distinct
    means[4, :3] = torch.tensor([-2.0, -2.0, -1.0], device=DEVICE)  # distinct
    labels = torch.randint(0, N_CLASSES, (n_samples,), generator=generator, device=DEVICE)
    noise = torch.randn(n_samples, N_FEATURES, generator=generator, device=DEVICE) * NOISE_STD  # smaller spread -> a strong teacher is achievable
    features = means[labels] + noise  # blob around the class mean
    return features, labels


class MLP(nn.Module):
    """A one-hidden-layer classifier. Width is the only thing separating teacher from student."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, hidden),
            nn.ReLU(),
            nn.Linear(hidden, N_CLASSES),  # returns raw LOGITS (no softmax) -- KD needs logits
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_teacher(
    features: torch.Tensor, labels: torch.Tensor, generator: torch.Generator
) -> MLP:
    """Train the wide teacher on hard labels until it is accurate. Returns the trained net."""
    torch.manual_seed(SEED)  # re-seed so teacher init is fixed
    teacher = MLP(TEACHER_HIDDEN).to(DEVICE)
    optimizer = torch.optim.Adam(teacher.parameters(), lr=LEARNING_RATE)
    for _ in range(TEACHER_EPOCHS):
        optimizer.zero_grad()
        loss = F.cross_entropy(teacher(features), labels)  # plain CE on hard labels
        loss.backward()
        optimizer.step()
    teacher.eval()
    return teacher


def show_dark_knowledge(teacher: MLP, features: torch.Tensor, labels: torch.Tensor) -> None:
    """Print ONE teacher distribution at T=1 vs T=4 so the dark knowledge is visible."""
    # Pick a CORRECTLY-classified "cat" (class 0) that the teacher is NOT maximally saturated
    # on, so the softened dog/lynx mass is visible rather than rounded to zero. We take cats
    # the teacher gets right, rank them by confidence, and choose a mid-confidence one --
    # representative of a real example, not a cherry-picked extreme.
    with torch.no_grad():
        logits_all = teacher(features)
    correct_cat = (labels == 0) & (logits_all.argmax(dim=-1) == 0)  # cats the teacher gets right
    cat_logits = logits_all[correct_cat]
    cat_confidence = F.softmax(cat_logits, dim=-1)[:, 0]  # P(cat) at T=1 for each
    order = cat_confidence.argsort()  # ascending confidence
    idx = order[len(order) // 4]  # 25th-percentile confidence: correct but not saturated -> dark knowledge shows
    logits = cat_logits[idx]
    names = ["cat", "dog", "lynx", "car", "plane"]  # same class names the page figure uses
    p1 = F.softmax(logits, dim=-1)                    # T=1: the sharp distribution training uses
    p4 = F.softmax(logits / TEMPERATURE, dim=-1)      # T=4: the softened distribution KD uses
    print("Dark knowledge -- one teacher 'cat' example, softened by temperature:")
    print(f"  {'class':>8} | {'T=1 prob':>9} | {'T=4 prob':>9}")
    print("  " + "-" * 32)
    for name, a, b in zip(names, p1.tolist(), p4.tolist()):
        print(f"  {name:>8} | {a:>9.3f} | {b:>9.3f}")
    # The takeaway: at T=1 'dog' mass is tiny; at T=4 it lifts to a clearly visible value --
    # that lift is the class-similarity signal the one-hot hard label has zero of.
    print(f"  -> at T=1 'dog' carries {p1[1].item():.3f}; at T=4 it lifts to {p4[1].item():.3f} "
          f"(the dark knowledge a one-hot label discards)\n")


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    hard_labels: torch.Tensor,
    temperature: float = TEMPERATURE,
    alpha: float = ALPHA,
) -> torch.Tensor:
    """The Hinton KD loss: L = alpha * T^2 * KL(student_T || teacher_T) + (1-alpha) * CE.

    student_logits, teacher_logits: (batch, N_CLASSES) raw logits. hard_labels: (batch,) ints.
    """
    # --- soft term: match the teacher's softened distribution -------------------------
    # Divide logits by T BEFORE softmax: larger T flattens both distributions, exposing the
    # off-target structure. log_softmax for the student (KL wants log-probs of the input arg).
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)  # student_T as log-probs
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)          # teacher_T as probs (the target)
    # KL(teacher_T || student_T): "how far is the student's softened dist from the teacher's".
    # batchmean averages over the batch (the mathematically correct KL reduction).
    soft_kl = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
    # The T^2 factor: the soft-target gradient scales as 1/T^2 (Hinton's high-T expansion),
    # so multiplying by T^2 restores it to a magnitude comparable to the hard-CE gradient --
    # without it, raising T silently shrinks the soft term's influence on the update.
    soft_term = (temperature**2) * soft_kl

    # --- hard term: still fit the ground-truth labels ---------------------------------
    hard_ce = F.cross_entropy(student_logits, hard_labels)  # standard CE on the true labels (T=1)

    # --- mix: alpha on soft, (1-alpha) on hard ----------------------------------------
    return alpha * soft_term + (1.0 - alpha) * hard_ce


def train_student_hard(
    features: torch.Tensor, labels: torch.Tensor, init_state: dict[str, torch.Tensor]
) -> MLP:
    """Baseline: train the student on HARD LABELS ONLY (no teacher). Same init as the KD student."""
    student = MLP(STUDENT_HIDDEN).to(DEVICE)
    student.load_state_dict(init_state)  # identical starting weights -> a fair comparison
    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    for _ in range(STUDENT_EPOCHS):
        optimizer.zero_grad()
        loss = F.cross_entropy(student(features), labels)  # only the hard labels; no dark knowledge
        loss.backward()
        optimizer.step()
    student.eval()
    return student


def train_student_kd(
    features: torch.Tensor,
    labels: torch.Tensor,
    teacher: MLP,
    init_state: dict[str, torch.Tensor],
) -> MLP:
    """Train the student with the KD loss (soft teacher targets + hard labels). Same init."""
    student = MLP(STUDENT_HIDDEN).to(DEVICE)
    student.load_state_dict(init_state)  # SAME starting weights as the hard-only student
    optimizer = torch.optim.Adam(student.parameters(), lr=LEARNING_RATE)
    with torch.no_grad():
        teacher_logits = teacher(features)  # teacher logits are fixed targets -> compute once
    for _ in range(STUDENT_EPOCHS):
        optimizer.zero_grad()
        loss = distillation_loss(student(features), teacher_logits, labels)  # the KD objective
        loss.backward()
        optimizer.step()
    student.eval()
    return student


def accuracy(model: MLP, features: torch.Tensor, labels: torch.Tensor) -> float:
    """Fraction of examples whose argmax prediction matches the true label."""
    with torch.no_grad():
        preds = model(features).argmax(dim=-1)
    return (preds == labels).float().mean().item()


def agreement_with_teacher(student: MLP, teacher: MLP, features: torch.Tensor) -> float:
    """Fraction of examples where the student's argmax matches the TEACHER's argmax.

    This is the KD-specific metric: not 'is the student right' but 'does the student behave
    like the teacher' -- exactly what distillation is trying to transfer.
    """
    with torch.no_grad():
        student_preds = student(features).argmax(dim=-1)
        teacher_preds = teacher(features).argmax(dim=-1)
    return (student_preds == teacher_preds).float().mean().item()


def main() -> None:
    print(f"device: {DEVICE} (detected {DETECTED_DEVICE}; pinned to CPU for reproducibility)")
    print("torch:", torch.__version__)
    print()

    # One generator drives all data sampling so the run is fully reproducible.
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    teacher_x, teacher_y = make_synthetic_data(N_TEACHER_TRAIN, generator)  # ABUNDANT data for the teacher
    student_x, student_y = make_synthetic_data(N_STUDENT_TRAIN, generator)  # a SMALL labelled subset for the students
    test_x, test_y = make_synthetic_data(N_TEST, generator)

    # 1) Train the teacher on abundant data and report its accuracy (the ceiling to chase).
    teacher = train_teacher(teacher_x, teacher_y, generator)
    teacher_acc = accuracy(teacher, test_x, test_y)
    print(f"teacher test accuracy: {teacher_acc:.3f}  "
          f"(wide net: {TEACHER_HIDDEN} hidden units, {N_TEACHER_TRAIN} train pts)\n")

    # 2) Show the dark knowledge: T=1 vs T=4 on one teacher example.
    show_dark_knowledge(teacher, teacher_x, teacher_y)

    # 3) Build ONE shared student init so hard-only vs KD differ ONLY in the loss.
    torch.manual_seed(SEED + 1)  # fixed, distinct from the teacher seed
    init_student = MLP(STUDENT_HIDDEN).to(DEVICE)
    init_state = {k: v.clone() for k, v in init_student.state_dict().items()}

    # 4) Train both students on the SMALL student subset and measure.
    print(f"both students train on the SAME small subset of {N_STUDENT_TRAIN} points "
          f"(narrow net: {STUDENT_HIDDEN} hidden units)\n")
    student_hard = train_student_hard(student_x, student_y, init_state)
    student_kd = train_student_kd(student_x, student_y, teacher, init_state)

    hard_acc = accuracy(student_hard, test_x, test_y)
    kd_acc = accuracy(student_kd, test_x, test_y)
    hard_agree = agreement_with_teacher(student_hard, teacher, test_x)
    kd_agree = agreement_with_teacher(student_kd, teacher, test_x)

    print(f"{'student':>16} | {'test acc':>9} | {'agreement w/ teacher':>21}")
    print("-" * 54)
    print(f"{'hard-labels-only':>16} | {hard_acc:>9.3f} | {hard_agree:>21.3f}")
    print(f"{'KD (soft+hard)':>16} | {kd_acc:>9.3f} | {kd_agree:>21.3f}")
    print(f"{'teacher (ref)':>16} | {teacher_acc:>9.3f} | {'1.000':>21}")
    print()

    # 5) The claim, asserted: KD makes the student behave more like the teacher.
    assert kd_agree >= hard_agree, (
        f"KD agreement ({kd_agree:.3f}) should be >= hard-only ({hard_agree:.3f})"
    )
    print(f"PASS: KD student agreement {kd_agree:.3f} >= hard-only {hard_agree:.3f} "
          f"(+{(kd_agree - hard_agree) * 100:.1f} points) -- distillation transferred the "
          f"teacher's behaviour, not just the labels.")


if __name__ == "__main__":
    main()
