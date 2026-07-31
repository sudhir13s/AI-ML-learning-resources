"""Animated (GIF) intuition figure for 03-Embedding-Models-for-Retrieval.

Companion to the static PNGs. Where those show final states, this brings the *mechanism* to life:
as the contrastive bi-encoder trains, you watch the toy sentences move in 2D — each paraphrase
passage migrating toward its query (positives pulled together) while unrelated sentences drift
apart (negatives pushed away). The "meaning → nearby coordinates" lesson, learned step by step.

    python make_animation_03.py

The GIF is written to ../../images/ and uses matplotlib's PillowWriter (no ffmpeg needed). The
trajectory is the chapter's OWN: the same canonical DenseBiEncoder + InfoNCE loss from
embedding_models.py, trained with the same seed, snapshotting embedding positions over steps.

Produced:
  rag03_contrastive_pull.gif -- query/passage pairs migrating together under contrastive training;
                                positives converge, unrelated pairs separate.

Verified on Python 3.12 / matplotlib 3.x / numpy 2.x / torch 2.x / Pillow (PillowWriter).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: render frames to a file, never open a window
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FuncAnimation, PillowWriter

from embedding_models import (
    PARAPHRASE_PAIRS,
    DenseBiEncoder,
    LEARNING_RATE,
    SEED,
    bow_tensor,
    build_vocab,
    info_nce_loss,
)

# ---- Palette (matches the chapter's muted Mermaid classDefs) -------------------------------
GREEN = "#2E7A5A"
INK = "#1C2530"
GRID = "#D4D9DF"
CLUSTER_COLORS = ["#3A6B96", "#2E7A5A", "#7A6528", "#5D4A8A"]  # one per paraphrase pair

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 95
FPS = 10
SNAPSHOT_EVERY = 12  # capture embedding positions every N training steps
TOTAL_STEPS = 480  # enough to see convergence in the toy
HOLD_FRAMES = 12  # dwell on the converged state


def _project_2d(vectors: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project (n, d) onto a FIXED 2D basis so frames are comparable (no per-frame PCA flips)."""
    return vectors @ basis.T


def build_animation() -> None:
    queries = [q for q, _ in PARAPHRASE_PAIRS]
    passages = [p for _, p in PARAPHRASE_PAIRS]
    vocab = build_vocab(queries + passages)

    torch.manual_seed(SEED)
    model = DenseBiEncoder(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    q_in = torch.stack([bow_tensor(q, vocab) for q in queries])
    p_in = torch.stack([bow_tensor(p, vocab) for p in passages])

    # train fully once to get a STABLE 2D basis (final-state PCA), then reuse it for every frame
    for _ in range(TOTAL_STEPS):
        loss = info_nce_loss(model(q_in), model(p_in))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = torch.cat([model(q_in), model(p_in)]).numpy()
    centered = final - final.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:2]  # fixed 2D basis from the converged geometry

    # re-train from scratch, snapshotting positions projected onto the fixed basis
    torch.manual_seed(SEED)
    model = DenseBiEncoder(len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    snapshots: list[np.ndarray] = []
    steps_at: list[int] = []
    for step in range(TOTAL_STEPS + 1):
        loss = info_nce_loss(model(q_in), model(p_in))
        if step % SNAPSHOT_EVERY == 0:
            with torch.no_grad():
                coords = _project_2d(torch.cat([model(q_in), model(p_in)]).numpy(), basis)
            snapshots.append(coords)
            steps_at.append(step)
        if step < TOTAL_STEPS:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    frames = list(range(len(snapshots))) + [len(snapshots) - 1] * HOLD_FRAMES

    n = len(queries)
    fig, ax = plt.subplots(figsize=(6.8, 6.2))

    def update(frame_idx: int):
        coords = snapshots[frames[frame_idx]]
        step = steps_at[frames[frame_idx]]
        ax.clear()
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_title(f"Contrastive training pulls paraphrases together   ·   step {step}",
                     fontsize=11.5, color=INK)
        # connector + markers for each pair (query=circle, passage=square, same colour)
        for i in range(n):
            qx, qy = coords[i]
            px, py = coords[i + n]
            color = CLUSTER_COLORS[i]
            ax.plot([qx, px], [qy, py], color=color, linewidth=1.3, alpha=0.5, zorder=2)
            ax.scatter(qx, qy, s=190, color=color, marker="o", edgecolors=INK, linewidths=1.0, zorder=3)
            ax.scatter(px, py, s=190, color=color, marker="s", edgecolors=INK, linewidths=1.0, zorder=3)
        ax.text(0, -1.07, "circle = query   ·   square = its paraphrase passage   ·   colour = pair",
                ha="center", va="center", fontsize=8.5, color=INK)
        return ax.collections

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / FPS, blit=False)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "rag03_contrastive_pull.gif"
    anim.save(out_path, writer=PillowWriter(fps=FPS), dpi=DPI)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    build_animation()


if __name__ == "__main__":
    main()
