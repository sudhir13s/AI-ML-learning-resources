"""Reproducible figure generator for 13-Supervised-Fine-Tuning.

Produces every embedded PNG for the chapter from the SAME numbers used on the page and in the
notebook -- so the figures cannot silently drift from the prose. Run:

    python make_figures_13.py

Each figure is written to ../images/ at 150 dpi. The palette matches the chapter's Mermaid
diagrams (muted, white text on coloured fills). Numbers are recomputed here from the actual
demo (supervised_fine_tuning.py), never hardcoded from memory:
  - the prompt-mask example (9 tokens, 7 prompt + 2 response),
  - the loss-on-all vs loss-on-response bars (2.825 vs 3.399 at init),
  - the SFT training-loss curve (2.92 -> 0.0045 over 60 steps),
  - the data-quality (LIMA ~1000 curated) illustration,
  - the chat-template token-structure strip.

Verified on Python 3.12 / torch 2.x / matplotlib 3.x.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: write files, never open a window
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

# Reuse the EXACT demo machinery so figures and the script can never disagree.
from supervised_fine_tuning import (
    DEMOS,
    EOS,
    IGNORE_INDEX,
    LEARNING_RATE,
    N_SFT_STEPS,
    SEED,
    TinyCausalLM,
    build_example,
    build_tokenizer,
    causal_lm_loss,
    encode,
    format_chat,
)

# ---- Palette (matches the chapter Mermaid classDefs) --------------------------------
BLUE = "#3A6B96"
PURPLE = "#5D4A8A"
GREEN = "#2E7A5A"
RED = "#8B3B4A"
SLATE = "#4A5B6E"
AMBER = "#7A6528"
NAVY = "#2A5B80"
INK = "#1C2530"  # near-black for axis text
GRID = "#D4D9DF"

OUT_DIR = Path(__file__).resolve().parent.parent.parent / "images"
DPI = 150


def _style_axis(ax: plt.Axes) -> None:
    """Consistent muted styling: light grid, no top/right spines, ink-coloured labels."""
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(SLATE)
    ax.tick_params(colors=INK)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _save(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.name}")


# =====================================================================================
# Figure 1 -- the prompt mask: which tokens carry loss (the single most important visual)
# =====================================================================================
def fig_prompt_mask() -> None:
    """One (prompt, response) stream, coloured by whether each token contributes to the loss."""
    instruction, response = DEMOS[0]  # "translate hello to french" -> "bonjour"
    stoi, itos = build_tokenizer(DEMOS)
    input_ids, labels, n_prompt = build_example(instruction, response, stoi, "cpu")
    toks = [itos[i] for i in input_ids.tolist()]
    is_response = [labels[i].item() != IGNORE_INDEX for i in range(len(toks))]

    fig, ax = plt.subplots(figsize=(11.0, 2.7))
    ax.set_xlim(0, len(toks))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for idx, (tok, resp) in enumerate(zip(toks, is_response)):
        colour = GREEN if resp else SLATE
        rect = mpatches.FancyBboxPatch(
            (idx + 0.06, 0.30), 0.88, 0.42,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=colour, edgecolor="white", linewidth=1.5, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(idx + 0.5, 0.51, tok, ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold", zorder=3)
        label_txt = "loss" if resp else "-100"
        ax.text(idx + 0.5, 0.16, label_txt, ha="center", va="center",
                color=GREEN if resp else RED, fontsize=8.5, fontweight="bold")
        ax.text(idx + 0.5, 0.80, str(idx), ha="center", va="center", color=INK, fontsize=8)

    ax.text(n_prompt / 2, 0.93, "PROMPT  (masked, label = -100)", ha="center",
            color=SLATE, fontsize=10.5, fontweight="bold")
    ax.text(n_prompt + (len(toks) - n_prompt) / 2 + 0.15, 0.93, "RESPONSE  (loss here only)",
            ha="center", color=GREEN, fontsize=10.5, fontweight="bold")
    ax.axvline(n_prompt, color=AMBER, linewidth=2.0, linestyle="--", zorder=1)
    ax.set_title("SFT prompt masking: loss is computed ONLY on response tokens",
                 fontsize=12, fontweight="bold", color=INK, pad=8)
    _save(fig, "sft_prompt_mask.png")


# =====================================================================================
# Figure 2 -- loss on ALL tokens vs loss on RESPONSE only (numbers from the demo)
# =====================================================================================
def fig_loss_all_vs_response() -> None:
    """At init, the two objectives give different losses; the masked one is the SFT objective."""
    torch.manual_seed(SEED)
    instruction, response = DEMOS[0]
    stoi, _ = build_tokenizer(DEMOS)
    vocab_size = len(stoi)
    input_ids, labels, _ = build_example(instruction, response, stoi, "cpu")
    model = TinyCausalLM(vocab_size)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0))
        loss_all = causal_lm_loss(logits, input_ids.unsqueeze(0)).item()
        loss_resp = causal_lm_loss(logits, labels.unsqueeze(0)).item()

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    _style_axis(ax)
    bars = ax.bar(
        ["loss on ALL tokens\n(prompt + response)", "loss on RESPONSE only\n(SFT, label = -100)"],
        [loss_all, loss_resp],
        color=[SLATE, GREEN], edgecolor="white", linewidth=1.5, width=0.6, zorder=3,
    )
    for bar, val in zip(bars, [loss_all, loss_resp]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.05, f"{val:.3f}",
                ha="center", va="bottom", color=INK, fontsize=11, fontweight="bold")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_ylim(0, max(loss_all, loss_resp) * 1.18)
    ax.set_title("The prompt mask changes the objective",
                 fontsize=12, fontweight="bold", color=INK)
    _save(fig, "sft_loss_all_vs_response.png")


# =====================================================================================
# Figure 3 -- SFT training-loss curve (response-token loss, from the actual training loop)
# =====================================================================================
def fig_training_curve() -> None:
    """Reproduce supervised_fine_tuning.py's training loop and plot the response-loss drop."""
    torch.manual_seed(SEED)
    stoi, _ = build_tokenizer(DEMOS)
    vocab_size = len(stoi)
    examples = [build_example(i, r, stoi, "cpu") for i, r in DEMOS]
    max_len = max(ex[0].size(0) for ex in examples)
    pad_id = stoi[EOS]  # SAME pad_id as supervised_fine_tuning.py + the notebook (cross-file consistency)
    batch_inputs = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    batch_labels = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long)
    for row, (ids, labs, _) in enumerate(examples):
        batch_inputs[row, : ids.size(0)] = ids
        batch_labels[row, : labs.size(0)] = labs

    # build the model RIGHT after the seed above (no intervening forward pass), identical to
    # the .py training block -> RNG-identical init -> the SAME trace in all four places.
    model = TinyCausalLM(vocab_size)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    steps, losses = [], []
    for step in range(N_SFT_STEPS + 1):
        logits = model(batch_inputs)
        loss = causal_lm_loss(logits, batch_labels)
        steps.append(step)
        losses.append(loss.item())
        if step == N_SFT_STEPS:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    _style_axis(ax)
    ax.plot(steps, losses, color=BLUE, linewidth=2.4, zorder=3)
    ax.scatter([0, N_SFT_STEPS], [losses[0], losses[-1]], color=[RED, GREEN], s=60, zorder=4)
    ax.annotate(f"start: {losses[0]:.2f}", (0, losses[0]), textcoords="offset points",
                xytext=(12, 0), color=RED, fontsize=10, fontweight="bold", va="center")
    ax.annotate(f"end: {losses[-1]:.4f}", (N_SFT_STEPS, losses[-1]), textcoords="offset points",
                xytext=(-10, 20), color=GREEN, fontsize=10, fontweight="bold", ha="right")
    ax.set_xlabel("SFT step")
    ax.set_ylabel("response-token loss (nats)")
    ax.set_title("SFT learns the demonstrations: response loss collapses",
                 fontsize=12, fontweight="bold", color=INK)
    _save(fig, "sft_training_curve.png")


# =====================================================================================
# Figure 4 -- data quality > quantity (LIMA: ~1000 curated beats massive noisy)
# =====================================================================================
def fig_data_quality() -> None:
    """Illustrative: a small curated set can match/beat a large noisy one (LIMA's finding)."""
    setups = ["1,000 curated\n(LIMA)", "52,000 synthetic\n(Alpaca-style)", "noisy web\nscrape"]
    # Illustrative win-rate-style quality scores -- the SHAPE is the point (curated wins),
    # labelled illustrative since exact numbers depend on the eval. LIMA reports parity/wins
    # vs much larger SFT sets on its human preference eval.
    quality = [0.82, 0.70, 0.41]
    sizes = [1_000, 52_000, 500_000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2))
    _style_axis(ax1)
    bars = ax1.bar(setups, quality, color=[GREEN, BLUE, RED], edgecolor="white",
                   linewidth=1.5, width=0.6, zorder=3)
    for bar, val in zip(bars, quality):
        ax1.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}",
                 ha="center", va="bottom", color=INK, fontsize=10.5, fontweight="bold")
    ax1.set_ylabel("response quality (illustrative)")
    ax1.set_ylim(0, 1.0)
    ax1.set_title("Quality of curation drives quality of output", fontsize=11.5,
                  fontweight="bold", color=INK)

    _style_axis(ax2)
    bars2 = ax2.bar(setups, sizes, color=[GREEN, BLUE, RED], edgecolor="white",
                    linewidth=1.5, width=0.6, zorder=3)
    ax2.set_yscale("log")
    for bar, val in zip(bars2, sizes):
        ax2.text(bar.get_x() + bar.get_width() / 2, val * 1.15, f"{val:,}",
                 ha="center", va="bottom", color=INK, fontsize=9.5, fontweight="bold")
    ax2.set_ylabel("number of examples (log scale)")
    ax2.set_title("...and it is NOT proportional to dataset size", fontsize=11.5,
                  fontweight="bold", color=INK)
    fig.suptitle("Data quality ≫ quantity for SFT (LIMA, Zhou et al. 2023)",
                 fontsize=12.5, fontweight="bold", color=INK, y=1.02)
    _save(fig, "sft_data_quality.png")


# =====================================================================================
# Figure 5 -- per-response-token loss BEFORE vs AFTER SFT (where the learning landed)
# =====================================================================================
def fig_per_token_before_after() -> None:
    """Per-response-token loss for one example, at init vs after SFT -- learning is on the response."""
    torch.manual_seed(SEED)
    stoi, itos = build_tokenizer(DEMOS)
    vocab_size = len(stoi)
    instruction, response = DEMOS[1]  # "capital of france" -> "paris"
    input_ids, labels, n_prompt = build_example(instruction, response, stoi, "cpu")

    def per_response_losses(model: TinyCausalLM) -> tuple[list[str], list[float]]:
        model.eval()
        with torch.no_grad():
            logits = model(input_ids.unsqueeze(0))
            shift_logits = logits[:, :-1, :]
            shift_labels = input_ids[1:]
            per_tok = F.cross_entropy(
                shift_logits.reshape(-1, vocab_size), shift_labels, reduction="none"
            )
        toks, vals = [], []
        for j in range(shift_labels.size(0)):
            if j + 1 >= n_prompt:  # response position
                toks.append(itos[int(shift_labels[j])])
                vals.append(per_tok[j].item())
        return toks, vals

    # init model
    model = TinyCausalLM(vocab_size)
    toks, before = per_response_losses(model)

    # train it (same recipe as the script, single example for a crisp before/after)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    for _ in range(N_SFT_STEPS):
        logits = model(input_ids.unsqueeze(0))
        loss = causal_lm_loss(logits, labels.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _, after = per_response_losses(model)

    x = np.arange(len(toks))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    _style_axis(ax)
    ax.bar(x - width / 2, before, width, label="before SFT", color=RED,
           edgecolor="white", linewidth=1.2, zorder=3)
    ax.bar(x + width / 2, after, width, label="after SFT", color=GREEN,
           edgecolor="white", linewidth=1.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"'{t}'" for t in toks])
    ax.set_xlabel("response token (predicted)")
    ax.set_ylabel("cross-entropy loss (nats)")
    ax.set_title("SFT drives the RESPONSE-token loss to ~0",
                 fontsize=12, fontweight="bold", color=INK)
    ax.legend(frameon=False)
    _save(fig, "sft_per_token_before_after.png")


# =====================================================================================
# Figure 6 -- chat-template token structure (roles + special tokens)
# =====================================================================================
def fig_chat_template() -> None:
    """Show how one turn is wrapped in role tags + special tokens (the SFT data format)."""
    instruction, response = DEMOS[0]
    prompt_text, full_text = format_chat(instruction, response)
    toks = full_text.split()
    n_prompt = len(prompt_text.split())

    def kind(tok: str) -> str:
        if tok in ("<s>", "</s>"):
            return "special"
        if tok in ("<user>", "<assistant>"):
            return "role"
        return "content"

    colours = {"special": NAVY, "role": PURPLE, "content": SLATE}
    fig, ax = plt.subplots(figsize=(11.0, 2.4))
    ax.set_xlim(0, len(toks))
    ax.set_ylim(0, 1)
    ax.axis("off")
    for idx, tok in enumerate(toks):
        c = colours[kind(tok)]
        # mark the response span with a green underline bar
        rect = mpatches.FancyBboxPatch(
            (idx + 0.06, 0.34), 0.88, 0.40,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=c, edgecolor="white", linewidth=1.5, zorder=2,
        )
        ax.add_patch(rect)
        ax.text(idx + 0.5, 0.54, tok, ha="center", va="center", color="white",
                fontsize=9.5, fontweight="bold", zorder=3)
        if idx >= n_prompt:
            ax.add_patch(mpatches.Rectangle((idx + 0.06, 0.24), 0.88, 0.05,
                                            facecolor=GREEN, edgecolor="none", zorder=2))
    legend = [
        mpatches.Patch(color=NAVY, label="special token (BOS/EOS)"),
        mpatches.Patch(color=PURPLE, label="role tag"),
        mpatches.Patch(color=SLATE, label="content"),
        mpatches.Patch(color=GREEN, label="response span (trained)"),
    ]
    ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.18),
              ncol=4, frameon=False, fontsize=9)
    ax.set_title("Chat template: one turn wrapped in roles + special tokens",
                 fontsize=12, fontweight="bold", color=INK, pad=6)
    _save(fig, "sft_chat_template.png")


def main() -> None:
    print(f"writing figures to {OUT_DIR}")
    fig_prompt_mask()
    fig_loss_all_vs_response()
    fig_training_curve()
    fig_data_quality()
    fig_per_token_before_after()
    fig_chat_template()
    print("done.")


if __name__ == "__main__":
    main()
