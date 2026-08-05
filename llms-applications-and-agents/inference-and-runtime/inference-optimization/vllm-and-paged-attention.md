---
id: vllm-and-paged-attention
title: vLLM and Paged Attention
minutes: 20
category: inference-and-runtime
---
# LLM serving — vLLM + paged attention

> Curated concept note, migrated from the in-app practice library into the canonical repo so the content lives once at the source. Original wording.

## Problem statement

You are serving an open-weight 70B chat model behind a `/v1/chat/completions` endpoint. The product needs a sub-200 ms **TTFT** (time to first token), a steady token stream, and the lowest possible cost per million tokens. A naive deployment — one request at a time, contiguous KV-cache allocation — wastes most of an expensive H100 and serves a handful of users. The interviewer will push on *why* LLM inference is slow, where the memory goes, and how vLLM's **PagedAttention** plus **continuous batching** turn a memory-bound bottleneck into a high-throughput serving engine.

## Core concepts

- **Autoregressive decode is the bottleneck.** Generation is one token per forward pass: to emit token *n* the model must have emitted tokens *1…n-1*. You cannot parallelize across the time axis of a single sequence.
- Decode is **memory-bandwidth bound**, not compute bound. Each step reads the full model weights and the whole **KV cache** from HBM to produce one token — the GPU's compute units sit mostly idle waiting on memory.
- The **KV cache** stores the key and value tensors for every past token so attention does not recompute them. It **grows linearly with sequence length** and is the resource that bounds concurrency.
- **PagedAttention** (vLLM) treats the KV cache like paged virtual memory — fixed-size blocks instead of one contiguous buffer — eliminating the fragmentation that otherwise wastes 60–80% of VRAM.
- **Continuous (in-flight) batching** lets requests join and leave the batch at any decode step, keeping the GPU busy instead of stalling on the slowest sequence in a static batch.

## Deep dive

### The autoregressive decode bottleneck

A transformer forward pass produces exactly one new token. To generate a 300-token answer you run 300 sequential forward passes. Two phases differ sharply:

- **Prefill** processes the whole prompt in one pass — all prompt tokens go through attention together, so it is **compute-bound** and parallelizes well across tokens.
- **Decode** generates one token per pass, each reading every weight and every cached K/V from HBM. With batch size 1 the arithmetic intensity is tiny, so decode is **memory-bandwidth bound**: an H100 with ~3 TB/s of HBM bandwidth spends its time moving bytes, not doing math. This is *why* batching matters — it amortizes the weight read across many concurrent sequences.

### The KV cache and its memory math

For every token already in a sequence, attention needs that token's **key** and **value** vectors at every layer. Recomputing them each step would be quadratic, so we cache them. The cache size for one sequence is:

```
kv_bytes ≈ 2 · n_layers · d_model · seq_len · bytes_per_elem
```

The `2` is for K and V. So memory scales with **layers × model width × sequence length**. Worked example for a 70B-class model (80 layers, d_model ≈ 8192, FP16 = 2 bytes):

```
per token ≈ 2 · 80 · 8192 · 2 ≈ 2.6 MB
4,000-token conversation ≈ ~10 GB of KV
```

Two consequences fall out of this: the cache **grows with context length** (a long conversation costs more memory than a short one even on the same model), and on a fixed VRAM budget the KV cache — not the weights — is what caps **how many sequences you can serve at once**. In the ChatGPT-style reference, a GPT-4o instance leaves ~67 GB per H100 for KV after weights, at ~0.5 MB/token (FP8), fitting only ~67 concurrent conversations per GPU. Concurrency is a memory-budget question.

### PagedAttention — KV cache as paged virtual memory

The naive approach pre-allocates one **contiguous** KV buffer per request sized to `max_seq_len`. This is doubly wasteful: a request that generates 50 tokens but reserved 4,096 wastes the rest (**internal fragmentation**), and freed-then-reallocated buffers leave unusable gaps (**external fragmentation**). Measured waste is 60–80% of VRAM.

**PagedAttention** borrows the OS virtual-memory trick. The KV cache is split into fixed-size **blocks** (e.g. 16 tokens each). A per-sequence **block table** maps logical token positions to physical blocks, which need not be contiguous. Blocks are allocated **on demand** as the sequence grows and **freed immediately** when a request completes. The result:

- Fragmentation waste drops from 60–80% to **under 4%** — roughly doubling the number of concurrent sequences per GPU.
- Identical prefixes (e.g. a shared system prompt across millions of requests) can **share physical blocks** — *prefix caching* — saving ~30% more KV. The sharp edge: editing the system prompt invalidates the shared prefix and spikes memory, so you pre-warm on deploy.
- When memory is exhausted, low-priority sequences are **preempted** (their KV swapped to CPU or recomputed) instead of OOM-killing the process.

### Continuous (in-flight) batching vs static batching

**Static batching** assembles a fixed group of requests, runs them together, and returns when the *last* one finishes. The problem: sequences finish at different lengths, so a slot that completed at token 20 sits idle while a 500-token request runs — the GPU runs at the pace of the slowest member, and new arrivals wait for the whole batch to drain.

**Continuous batching** (a.k.a. in-flight or iteration-level batching) schedules at the granularity of a **single decode step**. After each step the scheduler evicts finished sequences, frees their KV blocks, and admits queued requests into the freed slots. Because every iteration repacks the batch, the GPU stays near-saturated. This is the single biggest **throughput** lever in modern serving — it raises tokens/sec dramatically — though larger effective batches can raise per-request latency, so prefill and decode are tuned as separate knobs.

### Tensor vs pipeline parallelism for big models

When a model does not fit on one GPU you split it:

- **Tensor parallelism (TP):** shard each layer's weight matrices across GPUs; every forward pass does an **AllReduce** to combine partial results. TP communicates on *every layer*, so the GPUs must be co-located — **NVLink** AllReduce is ~0.5 ms versus 3–8 ms cross-node over InfiniBand, a 6–16× penalty. Keep a TP group on one node (`STRICT_PACK` placement).
- **Pipeline parallelism (PP):** split the model into stage *ranges* of layers across GPUs/nodes; activations pass stage-to-stage. PP communicates far less, tolerating cross-node links, but introduces **pipeline bubbles** (idle stages waiting for upstream) unless you micro-batch.

Rule of thumb: TP within a node for latency-critical serving, PP across nodes only when the model is too large for one node's GPUs. Getting this wrong — splitting a TP group across nodes — produces high latency with *dropping* GPU utilization, a silent topology bug.

## Common pitfalls

- **KV-cache OOM.** Concurrent sequences exceed the KV budget and the GPU process is OOM-killed, dropping *every* live generation on that GPU. Mitigation: PagedAttention preemption (swap to CPU, don't die), alert at `kv_cache_utilization > 92%` before the cliff, cap max concurrent sequences per GPU, and scale out. The per-token KV math is what sizes this budget — it is not academic.
- **Head-of-line blocking.** One long-context or high-`max_tokens` request monopolizes a batch slot and stalls short interactive requests behind it. Mitigation: priority scheduling (paid > free, short > long) and **chunked prefill** so a giant prompt does not block decodes.
- **Prefill vs decode imbalance.** Prefill is compute-bound and bursty; decode is memory-bandwidth-bound and steady. Mixing a 4096-token prefill into a batch of decodes spikes that iteration's latency. Mitigation: chunked prefill (piggyback decode tokens onto prefill steps) or disaggregate prefill and decode onto separate pools — they bottleneck on different hardware.
- **Tuning TTFT and throughput as one knob.** They optimize in opposite directions — bigger batches lift tokens/sec but delay any single request's first token. Tune them separately, per model pool.
- **Forgetting prefix-cache invalidation.** Prefix caching saves ~30% KV only while the shared prefix is stable; a system-prompt edit silently invalidates it and spikes memory. Pre-warm on deploy.

## Discussion prompts

1. Why is decode memory-bandwidth bound while prefill is compute bound? What does that imply for batching?
2. Compute the KV-cache size for a 70B model at 8k context. What caps your concurrency on an 80 GB GPU — weights or KV?
3. Explain PagedAttention to someone who knows OS virtual memory. Where does the 60–80% → <4% waste reduction come from?
4. Static vs continuous batching: walk through why a finished sequence wastes a GPU slot under static batching, and how continuous batching reclaims it.
5. You serve a 70B with TP=4 and TTFT regresses 4× while GPU utilization *drops*. What is your first hypothesis?

## Things to defend

- [ ] State why autoregressive decode is sequential and memory-bandwidth bound, and why batching is the fix.
- [ ] Write the KV-cache memory formula (`2 · layers · d_model · seq · bytes`) and reason about concurrency from a VRAM budget.
- [ ] Explain PagedAttention as paged virtual memory and quantify the fragmentation win.
- [ ] Contrast static vs continuous batching and name continuous batching as the main throughput lever.
- [ ] Distinguish tensor parallelism (co-locate, AllReduce per layer) from pipeline parallelism (cross-node, bubbles).
- [ ] Name at least three failure modes (KV OOM, head-of-line blocking, prefill/decode imbalance) and one mitigation each.
