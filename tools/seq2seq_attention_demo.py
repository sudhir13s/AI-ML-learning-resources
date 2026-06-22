"""Seq2seq with vs without attention on a COPY task, end to end, from scratch.

This is the runnable companion to the concept page's Code section. It trains
two tiny GRU encoder-decoders on "copy this digit string":
  - no-attention: the decoder is initialized from ONE context vector (the
    encoder's final state) -- the fixed-context bottleneck.
  - Bahdanau attention: the decoder attends over ALL encoder states each step.
Then it reports free-running (autoregressive) exact-match accuracy at a SHORT
length (in-distribution) and a LONG length (stress) so you can watch the
bottleneck appear: attention holds, no-attention collapses on long inputs.

Verified on Python 3.12 (torch 2.12), CPU, ~30 s.
"""
import numpy as np, torch, torch.nn as nn

torch.manual_seed(0); np.random.seed(0)
V, PAD, BOS, EOS, NV, H = 10, 10, 11, 12, 13, 128
LTRAIN = 16


def batch(bs, lo, hi):
    L = np.random.randint(lo, hi + 1, bs); Lm = int(L.max())
    src = np.full((bs, Lm + 1), PAD); tgt = np.full((bs, Lm + 2), PAD)
    for b in range(bs):
        s = np.random.randint(0, V, L[b])
        src[b, :L[b]] = s; src[b, L[b]] = EOS
        tgt[b, 0] = BOS; tgt[b, 1:1 + L[b]] = s; tgt[b, 1 + L[b]] = EOS   # COPY
    return torch.tensor(src), torch.tensor(tgt)


class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NV, H, padding_idx=PAD)
        self.rnn = nn.GRU(H, H, batch_first=True, bidirectional=True)
        self.bridge = nn.Linear(2 * H, H)

    def forward(self, src):
        out, h = self.rnn(self.emb(src))
        states = self.bridge(out)                                  # (B,S,H)
        ctx = self.bridge(torch.cat([h[0], h[1]], -1)).unsqueeze(0)  # (1,B,H)
        return states, ctx


class DecNoAttn(nn.Module):
    """Decoder conditioned ONLY on the single context vector ctx."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NV, H, padding_idx=PAD)
        self.rnn = nn.GRU(H, H, batch_first=True)
        self.out = nn.Linear(H, NV)

    def forward(self, ti, states, ctx):
        o, _ = self.rnn(self.emb(ti), ctx)
        return self.out(o)


class DecAttn(nn.Module):
    """Bahdanau additive attention over all encoder states, per decode step."""
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(NV, H, padding_idx=PAD)
        self.rnn = nn.GRU(H + H, H, batch_first=True)
        self.Wa = nn.Linear(H, H, bias=False)
        self.Ua = nn.Linear(H, H, bias=False)
        self.va = nn.Linear(H, 1, bias=False)
        self.out = nn.Linear(H, NV)

    def forward(self, ti, states, ctx):
        e = self.emb(ti); s = ctx; logits = []
        for t in range(e.shape[1]):
            score = self.va(torch.tanh(self.Wa(s[-1].unsqueeze(1)) + self.Ua(states)))
            a = torch.softmax(score, 1)                  # alignment over source
            cvec = (a * states).sum(1, keepdim=True)     # per-step context
            o, s = self.rnn(torch.cat([e[:, t:t + 1], cvec], -1), s)
            logits.append(self.out(o))
        return torch.cat(logits, 1)


def train(dec, steps=4000):
    enc = Enc()
    params = list(enc.parameters()) + list(dec.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    lf = nn.CrossEntropyLoss(ignore_index=PAD)
    for _ in range(steps):
        src, tgt = batch(96, 1, LTRAIN)
        states, ctx = enc(src)
        logits = dec(tgt[:, :-1], states, ctx)           # teacher forcing
        loss = lf(logits.reshape(-1, NV), tgt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
    return enc, dec


@torch.no_grad()
def acc(enc, dec, L, n=300):
    src = np.full((n, L + 1), PAD); gold = []
    for b in range(n):
        s = np.random.randint(0, V, L); src[b, :L] = s; src[b, L] = EOS; gold.append(s)
    src = torch.tensor(src); states, ctx = enc(src)
    ys = torch.full((n, 1), BOS)
    for _ in range(L):
        nxt = dec(ys, states, ctx)[:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], 1)
    pred = ys[:, 1:].numpy()
    return sum(np.array_equal(pred[b], gold[b]) for b in range(n)) / n


if __name__ == "__main__":
    enc_n, dec_n = train(DecNoAttn())
    enc_a, dec_a = train(DecAttn())
    print(f"{'task':>23} | {'short (L=6)':>11} | {'long (L=18)':>11}")
    print("-" * 53)
    print(f"{'no attention (1 vector)':>23} | {acc(enc_n, dec_n, 6)*100:9.1f}% | {acc(enc_n, dec_n, 18)*100:9.1f}%")
    print(f"{'Bahdanau attention':>23} | {acc(enc_a, dec_a, 6)*100:9.1f}% | {acc(enc_a, dec_a, 18)*100:9.1f}%")
