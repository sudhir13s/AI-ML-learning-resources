"""A decoder-only transformer stack, from scratch: build it, trace every shape, prove no leakage.

This is the runnable companion to the *Decoder-only Architecture* concept page. It builds the
exact block the page draws -- token+positional embedding -> N x [ pre-norm -> causal multi-head
self-attention -> residual -> pre-norm -> MLP -> residual ] -> final norm -> weight-tied LM head
-> next-token logits -- on tiny dimensions you can hold in your head (d_model=32, 4 heads, 2
layers), and prints the tensor shape at every step so the data flow is concrete.

The headline check is the **no-leakage test**: changing a *future* token must leave the logits
at every *earlier* position bit-for-bit identical. That is the causal mask doing its one job, and
it is what makes teacher-forced parallel training legal (one forward pass = T next-token
predictions, with no position able to peek at its own answer). We assert it before anything else.

Verified on Python 3.12 / torch 2.12.0. Device-agnostic (CUDA / MPS / CPU); the forward pass and
the no-leakage guarantee hold identically on any device. Seeded for reproducibility on CPU.

Run:
    python decoder_only_architecture.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Model dimensions for the tiny demo stack (small enough to trace by eye) ---
VOCAB_SIZE = 50  # toy vocabulary; real GPT-2 uses 50,257 sub-word tokens
D_MODEL = 32  # model width: every token is a 32-d vector through the whole stack
N_HEADS = 4  # attention heads; each head works in a D_MODEL // N_HEADS = 8-d subspace
HEAD_DIM = D_MODEL // N_HEADS  # 8; n_heads * head_dim must equal d_model
N_LAYERS = 2  # depth: how many identical decoder blocks we stack
FFN_EXPANSION = 4  # MLP hidden width = 4 * d_model, the classic Transformer choice
SEQ_LEN = 8  # tokens in the demo sequence
BATCH = 1  # one sequence; the math is identical per batch element
SEED = 0  # fixed so CPU runs are bit-for-bit reproducible

# Attention scale 1/sqrt(head_dim): without it, dot products grow like head_dim and push
# softmax into its saturated, near-zero-gradient region. Hoisted out of the hot path.
ATTENTION_SCALE = HEAD_DIM**-0.5

# Token offset for the leakage probe: any non-zero shift to a future token is fine; we just
# need to change it to *something else* and confirm earlier positions don't move.
LEAK_PROBE_OFFSET = 7

# Run on the best available accelerator; CPU is the universal, reproducible fallback.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def causal_mask(seq_len: int, device: str = DEVICE) -> torch.Tensor:
    """Additive causal mask: 0 on/below the diagonal, -inf strictly above it.

    Adding this to the raw scores *before* softmax sends every future key to exp(-inf) = 0
    attention weight, so a query at position i can only attend to positions <= i. We add it
    pre-softmax (not multiply post-softmax) so the surviving allowed keys still renormalize
    to sum to 1 -- the future gets *no* weight, not a small one.
    """
    # triu(..., diagonal=1) keeps the STRICTLY-upper triangle (the future j > i) and zeros the
    # rest; filling that triangle with -inf is exactly the forbidden-future pattern.
    return torch.triu(
        torch.full((seq_len, seq_len), float("-inf"), device=device), diagonal=1
    )


class DecoderBlock(nn.Module):
    """One decoder-only block: pre-norm causal self-attention + pre-norm MLP, each residual.

    Pre-norm means we normalize the *input* to each sub-layer and add the sub-layer's output
    back onto the un-normalized residual stream (x = x + sublayer(norm(x))). That keeps a clean
    identity "highway" for gradients, which is why deep decoder stacks train stably (Xiong et
    al. 2020). The block is shape-preserving (T, d) -> (T, d), which is the whole reason a stack
    of L identical blocks composes like Lego.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert n_heads * (d_model // n_heads) == d_model, "n_heads must divide d_model"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # One fused projection produces Q, K, V together (3 * d_model wide), then we split it;
        # fusing is the standard trick -- one matmul instead of three, identical math.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)  # W_O: mixes the heads back together
        self.norm_attn = nn.LayerNorm(d_model)  # pre-norm before attention
        self.norm_ffn = nn.LayerNorm(d_model)  # pre-norm before the MLP
        # Position-wise MLP: widen to 4*d, non-linearity, project back. Applied to each token
        # independently -- attention moves information BETWEEN tokens, the MLP computes WITHIN one.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, FFN_EXPANSION * d_model),
            nn.GELU(),
            nn.Linear(FFN_EXPANSION * d_model, d_model),
        )

    def self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Causal multi-head self-attention. x: (batch, seq_len, d_model) -> same shape."""
        batch, seq_len, d_model = x.shape
        q, k, v = self.qkv(x).split(d_model, dim=-1)  # each (batch, seq_len, d_model)

        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, head_dim): view splits d_model
        # into (heads, head_dim); transpose puts heads next so each head attends independently.
        def to_heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = to_heads(q), to_heads(k), to_heads(v)

        # q . kᵀ over head_dim = one alignment score per (query, key) pair; *scale tames it.
        scores = (q @ k.transpose(-1, -2)) * ATTENTION_SCALE  # (batch, n_heads, seq_len, seq_len)
        scores = scores + causal_mask(seq_len, device=str(x.device))  # forbid the future
        weights = F.softmax(scores, dim=-1)  # over keys -> a distribution per query row
        context = weights @ v  # (batch, n_heads, seq_len, head_dim): weighted sum of values

        # Merge heads back: transpose heads beside head_dim FIRST, then reshape -- doing the
        # reshape without the transpose would interleave heads and corrupt the d_model vector.
        merged = context.transpose(1, 2).reshape(batch, seq_len, d_model)
        return self.out_proj(merged)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.norm_attn(x))  # pre-norm attention + residual
        x = x + self.ffn(self.norm_ffn(x))  # pre-norm MLP + residual
        return x


class DecoderOnlyLM(nn.Module):
    """Token+positional embedding -> L decoder blocks -> final norm -> weight-tied LM head."""

    def __init__(
        self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_seq_len: int
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)  # E: (V, d), id -> vector
        self.position_embedding = nn.Embedding(max_seq_len, d_model)  # learned absolute positions
        self.blocks = nn.ModuleList(
            [DecoderBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # d -> V logits
        # Weight tying: the LM head IS the token-embedding matrix. The vector that represents a
        # token going in and the vector that scores it coming out share one space -- saves V*d
        # parameters and mildly improves quality (Press & Wolf 2017).
        self.lm_head.weight = self.token_embedding.weight

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (batch, seq_len) of ints -> logits (batch, seq_len, vocab_size)."""
        _, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device)
        # Self-attention is permutation-invariant, so position must be injected explicitly;
        # here we ADD a learned positional vector to each token's embedding (GPT-2 style).
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        for block in self.blocks:
            x = block(x)  # each block is (batch, seq_len, d_model) -> same shape
        return self.lm_head(self.final_norm(x))


def build_model() -> DecoderOnlyLM:
    """Construct the seeded tiny demo model on DEVICE."""
    torch.manual_seed(SEED)  # seed before parameter init so CPU runs reproduce bit-for-bit
    model = DecoderOnlyLM(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        max_seq_len=SEQ_LEN,
    )
    return model.to(DEVICE).eval()


def trace_shapes(model: DecoderOnlyLM, token_ids: torch.Tensor) -> torch.Tensor:
    """Run a forward pass and print the shape at every stage; return the logits."""
    _, seq_len = token_ids.shape
    positions = torch.arange(seq_len, device=token_ids.device)
    embedded = model.token_embedding(token_ids) + model.position_embedding(positions)
    print(f"token ids                : {tuple(token_ids.shape)}  (batch, seq_len)")
    print(f"after embedding + position: {tuple(embedded.shape)}  (batch, seq_len, d_model)")

    x = embedded
    for layer_index, block in enumerate(model.blocks):
        x = block(x)
        print(f"after decoder block {layer_index}    : {tuple(x.shape)}  (shape preserved)")

    logits = model.lm_head(model.final_norm(x))
    print(f"next-token logits        : {tuple(logits.shape)}  (batch, seq_len, vocab)")
    return logits


def assert_no_leakage(model: DecoderOnlyLM, token_ids: torch.Tensor) -> float:
    """Change a FUTURE token and assert earlier-position logits are bit-for-bit identical.

    This is the causal mask's entire guarantee: position i's output depends only on positions
    <= i. If it holds, teacher forcing is legal -- one parallel forward pass scores every
    next-token prediction with no position able to see its own answer. Returns the max abs
    diff over the unchanged prefix (must be exactly 0.0).
    """
    last_position = token_ids.shape[1] - 1
    with torch.no_grad():
        logits_original = model(token_ids)
        perturbed = token_ids.clone()
        # Flip ONLY the last (future-most) token to a different id; everything before is untouched.
        perturbed[0, last_position] = (token_ids[0, last_position] + LEAK_PROBE_OFFSET) % VOCAB_SIZE
        logits_perturbed = model(perturbed)

    # Compare logits at every position BEFORE the changed token -- they must not have moved.
    prefix_original = logits_original[0, :last_position]
    prefix_perturbed = logits_perturbed[0, :last_position]
    max_diff = (prefix_original - prefix_perturbed).abs().max().item()
    assert torch.equal(prefix_original, prefix_perturbed), (
        f"LEAKAGE: changing the future token moved earlier logits by {max_diff:.2e}"
    )
    return max_diff


def main() -> None:
    print(f"device: {DEVICE}")
    print(f"torch : {torch.__version__}\n")

    model = build_model()
    total_params = sum(p.numel() for p in model.parameters())
    tied = model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()
    print(
        f"tiny decoder-only LM: d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}, "
        f"vocab={VOCAB_SIZE}"
    )
    print(f"parameters: {total_params:,}  | LM head tied to token embedding: {tied}\n")

    # Deterministic token ids (seeded) so the run is reproducible on CPU.
    torch.manual_seed(SEED)
    token_ids = torch.randint(0, VOCAB_SIZE, (BATCH, SEQ_LEN), device=DEVICE)

    print("--- forward pass: shape at every step ---")
    trace_shapes(model, token_ids)

    # Prove the mask works BEFORE anything else -- correctness first, then everything else.
    print("\n--- no-leakage check (causal mask) ---")
    max_diff = assert_no_leakage(model, token_ids)
    print(
        f"changed the LAST token; logits at all earlier positions unchanged "
        f"(max abs diff: {max_diff:.2e})"
    )
    print("PASS: a future token cannot influence an earlier position's prediction.")


if __name__ == "__main__":
    main()
