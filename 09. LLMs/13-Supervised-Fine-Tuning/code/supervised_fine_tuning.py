"""From-scratch Supervised Fine-Tuning (SFT): the prompt-masked cross-entropy loss.

The single idea that makes SFT *supervised* (and not just "more pretraining") is the
**prompt mask**: the (prompt, response) pair is one token stream, but the loss is computed
ONLY on the response tokens. Prompt-token labels are set to the ignore index (-100), so the
model is never rewarded for predicting the prompt it was handed -- only for producing the
response. This script:

  1. builds a tiny from-scratch GPT-style decoder LM (real next-token cross-entropy, no library),
  2. formats a few (instruction, response) pairs with a minimal chat template,
  3. builds labels with -100 on prompt tokens ("completion-only" masking),
  4. computes the masked cross-entropy and PROVES, numerically, that it ignores prompt tokens
     (loss-on-all-tokens vs loss-on-response-only differ; the masked loss equals a hand-rolled
     response-only average),
  5. runs a few SFT steps and shows the response-token loss drop.

Device-agnostic (CUDA / MPS / CPU) the same way kv_cache.py is. Training is pinned to CPU and
reported honestly so the printed loss trace is reproducible regardless of the detected device.

Run:
    python supervised_fine_tuning.py
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Hyperparameters (hoisted; one source of truth for the page + notebook + figures) ----
IGNORE_INDEX = -100  # PyTorch cross-entropy skips any target position equal to this
D_MODEL = 64  # tiny embedding/hidden width -- enough to train in seconds on CPU
N_HEADS = 4
HEAD_DIM = D_MODEL // N_HEADS  # 16
N_LAYERS = 2
MAX_LEN = 64  # max sequence length the toy model supports (positional table size)
SEED = 0
N_SFT_STEPS = 60  # SFT steps for the loss-drop demo
LEARNING_RATE = 3e-3
ASSERT_ATOL = 1e-6  # masked CE must equal hand-rolled response-only average to float tolerance

# Detect the best accelerator the same way kv_cache.py does; CPU is the universal fallback.
DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# ---- A minimal chat template (roles + special tokens), exactly as real SFT uses ----------
# Real templates (Llama-3, ChatML) wrap each turn in role tags; we mirror the STRUCTURE with
# short tokens so the masking logic is identical to production, just readable by eye.
BOS = "<s>"
USER_TAG = "<user>"
ASSISTANT_TAG = "<assistant>"
EOS = "</s>"

# A handful of instruction -> response demonstrations (the SFT dataset).
DEMOS: list[tuple[str, str]] = [
    ("translate hello to french", "bonjour"),
    ("capital of france", "paris"),
    ("opposite of hot", "cold"),
    ("two plus two", "four"),
]


def format_chat(instruction: str, response: str) -> tuple[str, str]:
    """Return (prompt_text, full_text) under the chat template.

    prompt_text is everything the model is GIVEN (and must NOT be trained to predict);
    full_text appends the response and EOS (the part the loss is computed on).
    """
    prompt = f"{BOS} {USER_TAG} {instruction} {ASSISTANT_TAG}"  # the model is handed this
    full = f"{prompt} {response} {EOS}"  # ... and must learn to continue it with this
    return prompt, full


def build_tokenizer(demos: list[tuple[str, str]]) -> tuple[dict[str, int], dict[int, str]]:
    """Build a tiny word-level tokenizer over the chat-formatted corpus."""
    tokens: set[str] = {BOS, USER_TAG, ASSISTANT_TAG, EOS}
    for instruction, response in demos:
        _, full = format_chat(instruction, response)
        tokens.update(full.split())
    # sorted() makes the vocab deterministic across runs -> reproducible token ids
    stoi = {tok: i for i, tok in enumerate(sorted(tokens))}
    itos = {i: tok for tok, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int], device: str) -> torch.Tensor:
    """Whitespace-tokenize text into a 1-D LongTensor of ids on `device`."""
    ids = [stoi[tok] for tok in text.split()]
    return torch.tensor(ids, dtype=torch.long, device=device)


def build_example(
    instruction: str, response: str, stoi: dict[str, int], device: str
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return (input_ids, labels, n_prompt_tokens) for one SFT example.

    labels are input_ids shifted in the loss (handled by the model), with the first
    `n_prompt_tokens` positions set to IGNORE_INDEX so the loss skips the prompt. This is the
    "completion-only" / response-masking construction -- the heart of SFT.
    """
    prompt_text, full_text = format_chat(instruction, response)
    input_ids = encode(full_text, stoi, device)
    n_prompt = len(prompt_text.split())  # how many leading tokens are the prompt
    labels = input_ids.clone()
    labels[:n_prompt] = IGNORE_INDEX  # mask the prompt -> loss is computed only on the response
    return input_ids, labels, n_prompt


class TinyCausalLM(nn.Module):
    """A minimal GPT-style decoder LM: token + positional embeddings -> transformer blocks -> logits.

    Small enough to train on CPU in seconds, but the loss path (causal self-attention +
    next-token cross-entropy) is the real thing, so the SFT masking demonstration is faithful.
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)  # id -> vector
        self.pos_emb = nn.Embedding(MAX_LEN, D_MODEL)  # position -> vector (learned)
        layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=4 * D_MODEL,
            batch_first=True,  # tensors are (batch, seq, d_model)
            activation="gelu",
        )
        # We use an encoder STACK but feed it a causal mask, making it a decoder (GPT-style).
        self.blocks = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.ln_f = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size)  # hidden -> vocab logits

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """input_ids: (batch, seq) -> logits: (batch, seq, vocab_size)."""
        seq_len = input_ids.shape[1]
        positions = torch.arange(seq_len, device=input_ids.device)  # 0..seq-1
        x = self.token_emb(input_ids) + self.pos_emb(positions)  # (batch, seq, d_model)
        # Causal mask: position i may attend only to <= i (True = blocked) -> no peeking ahead.
        causal = torch.triu(
            torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1
        )
        x = self.blocks(x, mask=causal)
        return self.head(self.ln_f(x))  # (batch, seq, vocab_size)


def causal_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Standard next-token cross-entropy with label masking.

    Shapes: logits (batch, seq, vocab); labels (batch, seq). We predict token t+1 from
    position t, so logits are sliced [:-1] and labels [1:] (the canonical teacher-forcing
    shift). cross_entropy then SKIPS every position whose (shifted) label == ignore_index,
    which is exactly how the prompt tokens are excluded from the SFT loss.
    """
    shift_logits = logits[:, :-1, :].contiguous()  # drop the last position's prediction (no t+1 target)
    shift_labels = labels[:, 1:].contiguous()  # drop the first label (no t-1 input to predict it)
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),  # (batch*(seq-1), vocab)
        shift_labels.view(-1),  # (batch*(seq-1),)
        ignore_index=ignore_index,  # -100 positions contribute zero loss and zero gradient
    )


def response_only_loss_by_hand(
    logits: torch.Tensor, input_ids: torch.Tensor, n_prompt: int
) -> torch.Tensor:
    """Re-derive the response-only loss WITHOUT ignore_index, to prove masking is doing it.

    We compute per-token cross-entropy across the whole sequence, then average ONLY the
    positions whose target is a response token. If this equals the ignore_index loss, the
    mask is provably skipping the prompt -- not approximating it.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()  # UN-masked labels (real ids everywhere)
    per_token = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",  # keep one loss value per position
    )  # (batch*(seq-1),)
    # A target at shifted index j corresponds to original position j+1. It is a response token
    # iff j+1 >= n_prompt. Build that boolean mask and average the response positions only.
    seq_minus_1 = shift_labels.size(1)
    target_positions = torch.arange(1, seq_minus_1 + 1, device=logits.device)  # original index of each target
    response_mask = target_positions >= n_prompt
    return per_token[response_mask].mean()


def main() -> None:
    torch.manual_seed(SEED)
    train_device = "cpu"  # pin training to CPU so the printed loss trace is reproducible
    detected = DEVICE
    print(f"device: {train_device} (detected {detected}; pinned to CPU for reproducibility)")
    print("torch:", torch.__version__)
    print()

    stoi, itos = build_tokenizer(DEMOS)
    vocab_size = len(stoi)
    print(f"toy vocab size: {vocab_size} tokens")

    # ---- Show the chat template + the prompt mask for ONE example (the key visual in code) ----
    instruction, response = DEMOS[0]
    prompt_text, full_text = format_chat(instruction, response)
    input_ids, labels, n_prompt = build_example(instruction, response, stoi, train_device)
    print(f"\nprompt : {prompt_text}")
    print(f"full   : {full_text}")
    print(f"\nper-token label mask (-100 = prompt, ignored by loss):")
    print(f"{'pos':>4} | {'token':<14} | {'label':>6} | trained?")
    print("-" * 44)
    for pos, tok_id in enumerate(input_ids.tolist()):
        label = labels[pos].item()
        trained = "no (prompt)" if label == IGNORE_INDEX else "YES (response)"
        print(f"{pos:>4} | {itos[tok_id]:<14} | {label:>6} | {trained}")

    # ---- Prove: masked CE == hand-rolled response-only average; and != loss-on-all-tokens ----
    model = TinyCausalLM(vocab_size).to(train_device)
    model.eval()
    with torch.no_grad():
        logits = model(input_ids.unsqueeze(0))  # (1, seq, vocab)
        loss_masked = causal_lm_loss(logits, labels.unsqueeze(0))
        loss_all = causal_lm_loss(logits, input_ids.unsqueeze(0))  # labels = real ids everywhere -> trains on prompt too
        loss_hand = response_only_loss_by_hand(logits, input_ids.unsqueeze(0), n_prompt)

    print(f"\nloss on ALL tokens        : {loss_all.item():.4f}  (prompt + response -- the WRONG objective)")
    print(f"loss on RESPONSE only (-100): {loss_masked.item():.4f}  (SFT: the masked objective)")
    print(f"same, re-derived by hand   : {loss_hand.item():.4f}  (should match the masked loss)")
    # The masked loss must EXACTLY equal the hand-rolled response-only average -- that is proof
    # ignore_index is skipping the prompt, not merely down-weighting it.
    assert torch.allclose(loss_masked, loss_hand, atol=ASSERT_ATOL), "mask is not response-only!"
    # And it must DIFFER from loss-on-all -- otherwise masking would be a no-op here.
    assert not torch.allclose(loss_masked, loss_all, atol=1e-3), "masking changed nothing?!"
    print("assert OK: masked loss == response-only average, and != loss-on-all-tokens")

    # ---- Run a few SFT steps; watch the RESPONSE-token loss drop -------------------------
    # Re-seed and build a FRESH model right before training so the printed loss trace is
    # identical across this script, the notebook, and make_figures_13.py regardless of how many
    # forward passes ran above (dropout advances the global RNG). This makes the trace the
    # single canonical SFT result quoted on the page, in the figure caption, and the notebook.
    torch.manual_seed(SEED)
    model = TinyCausalLM(vocab_size).to(train_device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    examples = [build_example(i, r, stoi, train_device) for i, r in DEMOS]
    # Pad to a common length so we can batch; pad labels with IGNORE_INDEX so padding is free.
    max_len = max(ex[0].size(0) for ex in examples)
    pad_id = stoi[EOS]  # any real id works for inputs; labels are masked so it is never trained
    batch_inputs = torch.full((len(examples), max_len), pad_id, dtype=torch.long, device=train_device)
    batch_labels = torch.full((len(examples), max_len), IGNORE_INDEX, dtype=torch.long, device=train_device)
    for row, (ids, labs, _) in enumerate(examples):
        batch_inputs[row, : ids.size(0)] = ids
        batch_labels[row, : labs.size(0)] = labs

    print(f"\nSFT training ({N_SFT_STEPS} steps, response-token loss):")
    print(f"{'step':>5} | {'response-loss':>13}")
    print("-" * 23)
    for step in range(N_SFT_STEPS + 1):
        logits = model(batch_inputs)
        loss = causal_lm_loss(logits, batch_labels)  # masked: only response tokens contribute
        if step % 10 == 0:
            print(f"{step:>5} | {loss.item():>13.4f}")
        if step == N_SFT_STEPS:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # ---- Show the model now completes a prompt with the trained response -----------------
    model.eval()
    instruction, gold = DEMOS[1]  # "capital of france" -> "paris"
    prompt_text, _ = format_chat(instruction, gold)
    prompt_ids = encode(prompt_text, stoi, train_device).unsqueeze(0)
    with torch.no_grad():
        next_logits = model(prompt_ids)[0, -1]  # logits for the token AFTER the prompt
        predicted = itos[int(next_logits.argmax())]
    print(f"\nafter SFT, prompt '{instruction}' -> model's next token: '{predicted}' (gold: '{gold}')")


if __name__ == "__main__":
    main()
