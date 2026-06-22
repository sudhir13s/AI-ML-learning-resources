"""Sequence-to-Sequence & Encoder-Decoder concept-page diagrams.

Muted palette, parallel matplotlib scale. Four visuals for
06. NLP/concepts/08-Sequence-to-Sequence-and-Encoder-Decoder.md:

  1. seq2seq_unrolled.png       -- the ARCHITECTURE: encoder RNN unrolled into a
     single context vector c, decoder RNN unrolled generating the target
     autoregressively, conditioned on c and its own previous outputs.
  2. seq2seq_bottleneck.png     -- the BOTTLENECK: MEASURED accuracy vs source
     length for a toy copy/reverse task, no-attention (degrades) vs
     attention (holds) -- the BLEU-vs-length drop reproduced.
  3. seq2seq_alignment.png      -- ATTENTION as soft alignment: a MEASURED
     attention heatmap (source x target) from the trained attention model on
     the reverse task -- the anti-diagonal alignment it learns.
  4. seq2seq_transformer.png    -- the MODERN seq2seq: the Transformer
     encoder-decoder block (self-attn replaces recurrence, cross-attn replaces
     Bahdanau attention).

The toy task (copy/reverse of a digit string) is trained for diagrams 2 and 3
so the numbers are real, not illustrative. Run with the ml-py312 torch env.
"""
import os, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

OUT = os.path.join(os.path.dirname(__file__), "..", "06. NLP", "concepts", "images")
OUT = os.path.abspath(OUT)
os.makedirs(OUT, exist_ok=True)

BLUE, PURPLE, GREEN, RED, SLATE, AMBER, NAVY = (
    "#3A6B96", "#5D4A8A", "#2E7A5A", "#8B3B4A", "#4A5B6E", "#7A6528", "#2A5B80")
plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _box(ax, x, y, w, h, color, text, fs=10, tc="white", lw=1.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.04",
                                fc=color, ec="white", lw=lw, mutation_aspect=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold", zorder=5)


def _arrow(ax, x0, y0, x1, y1, color=SLATE, lw=1.8, style="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                 mutation_scale=13, color=color, lw=lw, zorder=3))


# ---- 1. Encoder-decoder unrolled --------------------------------------------
def unrolled():
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.set_xlim(0, 14.6); ax.set_ylim(0, 7.0); ax.axis("off")
    ax.set_title("Encoder–Decoder unrolled: source → context vector c → target (autoregressive)",
                 fontsize=14, fontweight="bold", y=1.02)

    src = ["le", "chat", "noir", "<eos>"]
    tgt_in = ["<bos>", "the", "black", "cat"]
    tgt_out = ["the", "black", "cat", "<eos>"]

    # --- Encoder row (top-left) ---
    ex = 0.35
    enc_centers = []
    for i, w in enumerate(src):
        x = ex + i * 1.5
        _box(ax, x, 5.1, 1.15, 0.7, BLUE, f"$h_{i+1}$", fs=11)
        ax.text(x + 0.575, 6.0, w, ha="center", fontsize=9.5, color=NAVY, fontweight="bold")
        if i > 0:
            _arrow(ax, x - 0.35, 5.45, x, 5.45, color=BLUE, lw=2.0)
        enc_centers.append(x + 0.575)
    ax.text(ex - 0.05, 6.5, "ENCODER (reads source left→right)",
            fontsize=10.5, color=BLUE, fontweight="bold")

    # --- Context vector (between encoder end and decoder start, clear column) ---
    cx = enc_centers[-1] + 1.15
    _box(ax, cx, 5.05, 1.55, 0.85, PURPLE, "context\nvector  c", fs=10)
    _arrow(ax, enc_centers[-1] + 0.35, 5.45, cx, 5.45, color=PURPLE, lw=2.4)

    # --- Decoder row (bottom, shifted right of the encoder) ---
    dx = 6.6
    dec_centers = []
    for j in range(4):
        x = dx + j * 1.95
        _box(ax, x, 2.0, 1.35, 0.7, GREEN, f"$s_{j+1}$", fs=11)
        # input token feeding step j (teacher forcing / prev output)
        ax.text(x + 0.675, 1.05, tgt_in[j], ha="center", fontsize=9, color=SLATE, fontweight="bold")
        _arrow(ax, x + 0.675, 1.35, x + 0.675, 2.0, color=SLATE, lw=1.5)
        # output token from step j
        _box(ax, x + 0.12, 3.1, 1.15, 0.55, AMBER, tgt_out[j], fs=9)
        _arrow(ax, x + 0.675, 2.7, x + 0.675, 3.1, color=AMBER, lw=1.6)
        if j > 0:
            _arrow(ax, x - 0.6, 2.35, x, 2.35, color=GREEN, lw=2.0)
        dec_centers.append(x + 0.675)
    ax.text(dx - 0.1, 0.55, "DECODER (generates target token-by-token; each output feeds the next step)",
            fontsize=10.5, color=GREEN, fontweight="bold")

    # c initializes / conditions decoder (clean diagonal into s1, no crossings)
    _arrow(ax, cx + 0.55, 5.05, dec_centers[0], 2.7, color=PURPLE, lw=2.4)
    ax.text(cx + 1.85, 4.05, "initializes &\nconditions\nthe decoder", fontsize=9,
            color=PURPLE, fontweight="bold", ha="left")

    fig.tight_layout(); fig.savefig(f"{OUT}/seq2seq_unrolled.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote seq2seq_unrolled.png")


# ===========================================================================
#  Toy seq2seq: COPY a digit string (target = source).  Trained two ways:
#  (a) no attention -- decoder is conditioned ONLY on the final encoder state
#      (the single context vector -- the bottleneck), so all source info must
#      survive one fixed vector.
#  (b) Bahdanau attention -- the decoder attends over ALL encoder states each
#      step, so a long source never has to fit one vector.
#  We measure free-running exact-match accuracy vs source length, and extract a
#  real alignment matrix.  Copy is the classic task where the bottleneck shows:
#  accuracy collapses with length for (a) but holds for (b).
# ===========================================================================
def _train_toy(seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed); np.random.seed(seed)
    V = 10                      # digits 0..9
    PAD, BOS, EOS = 10, 11, 12
    NV = 13
    H = 128
    LMAX_TRAIN = 16
    dev = "cpu"

    def make_batch(bs, Lmin, Lmax):
        L = np.random.randint(Lmin, Lmax + 1, size=bs)
        Lm = int(L.max())
        src = np.full((bs, Lm + 1), PAD); tgt = np.full((bs, Lm + 2), PAD)
        for b in range(bs):
            seq = np.random.randint(0, V, size=L[b])
            src[b, :L[b]] = seq; src[b, L[b]] = EOS
            tgt[b, 0] = BOS; tgt[b, 1:1 + L[b]] = seq; tgt[b, 1 + L[b]] = EOS  # COPY
        return torch.tensor(src), torch.tensor(tgt)

    class Enc(nn.Module):
        # bidirectional GRU; final state -> single context vector for the
        # no-attention decoder, all states -> attention pool for the attn one.
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(NV, H, padding_idx=PAD)
            self.rnn = nn.GRU(H, H, batch_first=True, bidirectional=True)
            self.bridge = nn.Linear(2 * H, H)
        def forward(self, src):
            e = self.emb(src)
            out, h = self.rnn(e)                 # out:(B,S,2H); h:(2,B,H)
            out = self.bridge(out)               # (B,S,H) per-position states
            ctx = self.bridge(torch.cat([h[0], h[1]], -1)).unsqueeze(0)  # (1,B,H)
            return out, ctx

    class DecNoAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(NV, H, padding_idx=PAD)
            self.rnn = nn.GRU(H, H, batch_first=True)
            self.out = nn.Linear(H, NV)
        def forward(self, tgt_in, enc_out, ctx):
            e = self.emb(tgt_in)
            o, _ = self.rnn(e, ctx)              # conditioned ONLY on ctx (1 vector)
            return self.out(o), None

    class DecAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(NV, H, padding_idx=PAD)
            self.rnn = nn.GRU(H + H, H, batch_first=True)
            self.Wa = nn.Linear(H, H, bias=False)   # Bahdanau additive score:
            self.Ua = nn.Linear(H, H, bias=False)   #  e_ij = v^T tanh(Wa s + Ua h_j)
            self.va = nn.Linear(H, 1, bias=False)
            self.out = nn.Linear(H, NV)
        def forward(self, tgt_in, enc_out, ctx):
            e = self.emb(tgt_in)                     # (B,T,H)
            T = e.shape[1]
            s = ctx                                  # (1,B,H) decoder state
            logits, attns = [], []
            for t in range(T):
                dec_h = s[-1].unsqueeze(1)           # (B,1,H)
                score = self.va(torch.tanh(self.Wa(dec_h) + self.Ua(enc_out)))  # (B,S,1)
                a = torch.softmax(score, dim=1)      # (B,S,1) alignment weights
                cvec = (a * enc_out).sum(1, keepdim=True)  # (B,1,H) per-step context
                inp = torch.cat([e[:, t:t + 1], cvec], dim=-1)
                o, s = self.rnn(inp, s)
                logits.append(self.out(o)); attns.append(a.squeeze(-1))
            return torch.cat(logits, 1), torch.stack(attns, 1)  # attn:(B,T,S)

    def train(use_attn, steps=4000):
        enc = Enc(); dec = DecAttn() if use_attn else DecNoAttn()
        params = list(enc.parameters()) + list(dec.parameters())
        opt = torch.optim.Adam(params, lr=1e-3)
        lossf = nn.CrossEntropyLoss(ignore_index=PAD)
        for _ in range(steps):
            src, tgt = make_batch(96, 1, LMAX_TRAIN)
            enc_out, ctx = enc(src)
            logits, _ = dec(tgt[:, :-1], enc_out, ctx)   # teacher forcing
            loss = lossf(logits.reshape(-1, NV), tgt[:, 1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
        return enc, dec

    @torch.no_grad()
    def eval_acc(enc, dec, L, n=400):
        # free-running (autoregressive) exact-match accuracy at fixed length L
        src = np.full((n, L + 1), PAD); gold = []
        for b in range(n):
            seq = np.random.randint(0, V, size=L)
            src[b, :L] = seq; src[b, L] = EOS; gold.append(seq)
        src = torch.tensor(src)
        enc_out, ctx = enc(src)
        ys = torch.full((n, 1), BOS); emit = []
        for _ in range(L):
            logits, _ = dec(ys, enc_out, ctx)            # re-run whole prefix (simple, correct)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], 1); emit.append(nxt)
        pred = torch.cat(emit, 1).numpy()
        ok = sum(np.array_equal(pred[b], gold[b]) for b in range(n))
        return ok / n

    @torch.no_grad()
    def alignment(enc, dec, seq):
        src = np.full((1, len(seq) + 1), PAD); src[0, :len(seq)] = seq; src[0, len(seq)] = EOS
        src = torch.tensor(src)
        enc_out, ctx = enc(src)
        tgt = torch.tensor([[BOS] + list(seq) + [EOS]])
        _, attn = dec(tgt[:, :-1], enc_out, ctx)         # (1,T,S)
        return attn[0].numpy()

    enc_a, dec_a = train(True)
    enc_n, dec_n = train(False)
    Ls = list(range(2, 26, 2))
    acc_a = [eval_acc(enc_a, dec_a, L) for L in Ls]
    acc_n = [eval_acc(enc_n, dec_n, L) for L in Ls]
    al = alignment(enc_a, dec_a, np.array([3, 1, 4, 1, 5, 9, 2]))
    return Ls, acc_a, acc_n, al


_CACHE = {}
def _toy():
    if "r" not in _CACHE:
        _CACHE["r"] = _train_toy()
    return _CACHE["r"]


# ---- 2. The bottleneck: accuracy vs source length ---------------------------
def bottleneck():
    Ls, acc_a, acc_n, _ = _toy()
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(Ls, np.array(acc_n) * 100, color=RED, lw=2.6, marker="o", ms=5,
            label="No attention (single context vector c)")
    ax.plot(Ls, np.array(acc_a) * 100, color=GREEN, lw=2.6, marker="s", ms=5,
            label="Bahdanau attention (read all encoder states)")
    ax.axvspan(16, 25.5, color=AMBER, alpha=0.10)
    ax.text(20.5, 12, "lengths LONGER\nthan training\n(1–16)", color=AMBER,
            fontsize=9, fontweight="bold", ha="center")
    ax.set_xlabel("Source length (digits to copy)")
    ax.set_ylabel("Exact-match accuracy (%)  — free-running")
    ax.set_title("The fixed-context bottleneck, measured: no-attention collapses with length",
                 fontsize=13.5, fontweight="bold")
    ax.set_ylim(0, 103); ax.set_xlim(1, 25.5)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5); _despine(ax)
    fig.tight_layout(); fig.savefig(f"{OUT}/seq2seq_bottleneck.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote seq2seq_bottleneck.png")
    return Ls, acc_a, acc_n


# ---- 3. Attention alignment heatmap (measured) ------------------------------
def alignment_map():
    _, _, _, al = _toy()
    seq = [3, 1, 4, 1, 5, 9, 2]
    src_lbl = [str(d) for d in seq] + ["<eos>"]
    tgt_lbl = [str(d) for d in seq] + ["<eos>"]
    al = al[:len(tgt_lbl), :len(src_lbl)]
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    im = ax.imshow(al, cmap="viridis", aspect="auto", vmin=0, vmax=al.max())
    ax.set_xticks(range(len(src_lbl))); ax.set_xticklabels(src_lbl)
    ax.set_yticks(range(len(tgt_lbl))); ax.set_yticklabels(tgt_lbl)
    ax.set_xlabel("Source position (input digits)")
    ax.set_ylabel("Target position (generated)")
    ax.set_title("Learned soft alignment: attention discovers the copy mapping",
                 fontsize=13, fontweight="bold")
    for i in range(al.shape[0]):
        for j in range(al.shape[1]):
            if al[i, j] > 0.18:
                ax.text(j, i, f"{al[i,j]:.2f}", ha="center", va="center",
                        color="white", fontsize=8, fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("attention weight $\\alpha_{ij}$")
    fig.tight_layout(); fig.savefig(f"{OUT}/seq2seq_alignment.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote seq2seq_alignment.png")


# ---- 4. The Transformer encoder-decoder block -------------------------------
def transformer_block():
    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 7.4); ax.axis("off")
    ax.set_title("Modern seq2seq: the Transformer encoder–decoder (T5 / BART)",
                 fontsize=14, fontweight="bold")

    # Encoder stack (left)
    ex, ew = 0.6, 3.4
    _box(ax, ex, 0.5, ew, 0.55, BLUE, "Source embeddings + positions", fs=9)
    _box(ax, ex, 1.5, ew, 0.8, PURPLE, "Self-attention\n(bidirectional)", fs=9.5)
    _box(ax, ex, 2.7, ew, 0.7, GREEN, "Feed-forward", fs=9.5)
    ax.text(ex + ew / 2, 3.7, "×N encoder layers", ha="center", fontsize=9,
            color=SLATE, fontweight="bold")
    _arrow(ax, ex + ew / 2, 1.05, ex + ew / 2, 1.5, color=SLATE)
    _arrow(ax, ex + ew / 2, 2.3, ex + ew / 2, 2.7, color=SLATE)
    ax.text(ex + ew / 2, 4.05, "ENCODER", ha="center", fontsize=11,
            color=BLUE, fontweight="bold")

    # Decoder stack (right)
    dx, dw = 8.0, 3.4
    _box(ax, dx, 0.5, dw, 0.55, BLUE, "Target embeddings + positions", fs=9)
    _box(ax, dx, 1.5, dw, 0.8, PURPLE, "Masked self-attention\n(causal)", fs=9)
    _box(ax, dx, 2.6, dw, 0.8, AMBER, "Cross-attention\n(Q=decoder, K,V=encoder)", fs=8.8)
    _box(ax, dx, 3.7, dw, 0.7, GREEN, "Feed-forward", fs=9.5)
    _box(ax, dx, 4.8, dw, 0.6, RED, "Linear + softmax → next token", fs=8.6)
    ax.text(dx + dw / 2, 5.7, "×N decoder layers", ha="center", fontsize=9,
            color=SLATE, fontweight="bold")
    for y0, y1 in [(1.05, 1.5), (2.3, 2.6), (3.4, 3.7), (4.4, 4.8)]:
        _arrow(ax, dx + dw / 2, y0, dx + dw / 2, y1, color=SLATE)
    ax.text(dx + dw / 2, 6.05, "DECODER", ha="center", fontsize=11,
            color=GREEN, fontweight="bold")

    # cross-attention bridge: encoder output -> decoder cross-attn
    _arrow(ax, ex + ew, 3.05, dx, 3.0, color=AMBER, lw=2.6, style="-|>")
    ax.text((ex + ew + dx) / 2, 3.35, "encoder states\n(K, V)", ha="center",
            fontsize=8.6, color=AMBER, fontweight="bold")
    ax.text((ex + ew + dx) / 2, 1.2,
            "self-attention\nreplaces recurrence;\ncross-attention\nreplaces Bahdanau",
            ha="center", fontsize=8.2, color=NAVY, fontweight="bold")

    fig.tight_layout(); fig.savefig(f"{OUT}/seq2seq_transformer.png", dpi=150, bbox_inches="tight")
    plt.close(fig); print("wrote seq2seq_transformer.png")


if __name__ == "__main__":
    unrolled()
    Ls, acc_a, acc_n = bottleneck()
    alignment_map()
    transformer_block()
    print("\n--- MEASURED NUMBERS (for the page) ---")
    print("Length:      ", Ls)
    print("Attn acc %:  ", [round(a*100, 1) for a in acc_a])
    print("NoAttn acc %:", [round(a*100, 1) for a in acc_n])
