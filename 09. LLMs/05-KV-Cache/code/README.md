# KV-Cache demo code

A from-scratch, single-layer attention that runs the decode loop **both ways** — recomputing
K/V for every past token (the `O(n^2)` trap) versus keeping a cache (`O(n)`) — and:

1. asserts the two paths produce **identical** outputs to floating-point tolerance, and
2. times them across **growing sequence lengths** so the speedup is seen *widening* with `N`.

This is the runnable companion to [`05-KV-Cache.md`](../05-KV-Cache.md) and the step-by-step
[`05-KV-Cache.ipynb`](../05-KV-Cache.ipynb). CPU-only, a few seconds, no GPU needed.

## Run

```bash
python kv_cache.py
```

## Expected output

A table where the `identical` column is `True` at every length (the cache changes nothing about
*what* the model produces) while the `speedup` column **grows** as the sequence lengthens — the
`O(n^2) → O(n)` curve made visible. Exact milliseconds vary by machine; trust the *shape*:

```
     N |   no-cache |   kv-cache |  speedup | identical
----------------------------------------------------------
   256 |     58.7ms |     48.2ms |    1.2x | True
   512 |    161.4ms |    113.7ms |    1.4x | True
  1024 |    483.0ms |    288.8ms |    1.7x | True
  2048 |   1611.3ms |    800.0ms |    2.0x | True
```

## Requirements

- Python 3.10+
- `torch` (CPU build is sufficient)
