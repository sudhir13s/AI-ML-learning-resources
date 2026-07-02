---
id: "11-rag-and-llm-apps/vector-databases-ann-indexes"
topic: "Vector Databases & ANN Indexes (HNSW · IVF)"
parent: "11-rag-and-llm-apps"
level: intermediate
prereqs: ["embedding-models-for-retrieval", "cosine-similarity", "k-means-clustering"]
interview_frequency: high
template: concept-deep
updated: 2026-07-02
---

# Vector Databases & ANN Indexes: searching millions of vectors in milliseconds

[Chapter 3](../03-Embedding-Models-for-Retrieval/03-Embedding-Models-for-Retrieval.md) ended with retrieval reframed as pure geometry: embed the query, find the nearest passage vectors. It even promised this chapter — "makes 'find the nearest vectors' *fast* at scale." That promise hides a brutal cost problem, and this chapter is how real-world systems beat it.

Everything below is **real and measured**. We embed **30,000 real Wikipedia passages** with a real sentence-transformer (`BAAI/bge-small-en-v1.5`, 384-dim), build **real [FAISS](https://github.com/facebookresearch/faiss) indexes** (exact `IndexFlatIP`, `IndexIVFFlat`, `IndexHNSWFlat`, `IndexIVFPQ`), run real semantic queries, and *measure* what every ANN engineer measures: **recall@10 against exact search** and **real query latency** as we turn the recall/speed knob. There are no hand-made toy vectors and no faked library calls — every number on this page comes from the [executed teaching notebook](code/04-Vector-Databases-and-ANN-Indexes.ipynb) over that real corpus, and the notebook mirrors the [runnable module `code/vector_indexes.py`](code/vector_indexes.py) step by step. (Because it's real, wall-clock latency varies run-to-run; we report the shipped run's medians and flag what's inherently variable.)

Here's the wall. The obvious way to find a query's nearest neighbours is to **compare it to every vector** — compute the distance to all $N$ passages and keep the smallest. That's **exact** ("flat" or "brute-force") search, and it's $O(N \cdot d)$ per query. For a few thousand vectors it's instant. But RAG corpora are big: at **10 million passages × 768-dimensional embeddings**, a single query is $10{,}000{,}000 \times 768 \approx \mathbf{7.7\ billion}$ multiply-adds — *per query*. Run that for every user, at every keystroke of an autocomplete, and you've built a space heater, not a search engine. RAG needs **sub-millisecond** retrieval over corpora this size.

The fix is **Approximate Nearest Neighbour (ANN)** search: build an **index** that lets you *skip* almost all the vectors — route the query to the right neighbourhood and only scan what's nearby — trading a tiny, controllable amount of **recall** for **orders-of-magnitude** speed. I'll build this the way I'd actually tune a vector store: feel the exact-search cost, then the "skip most of the haystack" intuition, then the two dominant index families (IVF and HNSW) with the math **derived in full** — the k-means partition, the graph's layer pyramid, and the product-quantization codebook — followed by a real measured walkthrough where you *watch* recall trade for speed, the gotchas that bite (the recall cliff, filtering, the curse of dimensionality), and how to choose and operate a vector database. By the end you'll be able to:

- explain why exact search is $O(N \cdot d)$ and when it's still the right answer;
- describe **IVF** (k-means cells + probe `nprobe`) and **HNSW** (navigable graph + greedy descent), and their knobs, well enough to re-derive them;
- derive the IVF cost and recall↔`nprobe` tradeoff, the HNSW $O(\log N)$ navigation from its layer pyramid, and PQ's encode/decode and compression math;
- read a **real recall cliff and recall/latency frontier** and pick a tuning point against a recall SLO;
- avoid the silent failures — the recall cliff, metadata-filtering traps, index-build blowups, stale graphs.

> **Note:** ANN is a *systems* optimization, not a *modeling* one. It changes **how fast** you find neighbours and **how much memory** the index costs — not what "near" means (that's the embedder, chapter 3). A great index over bad embeddings still retrieves the wrong things, fast.

---

## The problem: exact search doesn't scale

To feel why ANN exists, count the work exact search does.

Flat search answers "what are the $k$ nearest vectors to $\mathbf{q}$?" by computing the distance from $\mathbf{q}$ to **every** vector and sorting. Each distance is $O(d)$ (a $d$-dimensional dot product on our unit-norm vectors), and there are $N$ of them, so one query is $O(N \cdot d)$. Concretely, our real corpus ($N = 30{,}000$, $d = 384$) does $30{,}000 \times 384 = \mathbf{11{,}520{,}000}$ multiply-adds per query. Scale to a real-world corpus and it explodes:

$$
\text{flat cost per query} = N \cdot d \;\;\xrightarrow{\;N=10^7,\; d=768\;}\;\; 7.68 \times 10^9 \approx \mathbf{7.7\ \text{billion multiply-adds}}.
$$

> **Source / derivation:** [Johnson, Douze & Jégou (2017/2019), *Billion-scale similarity search with GPUs* (arXiv:1702.08734)](https://arxiv.org/abs/1702.08734) — the FAISS GPU paper; §2 lays out the exact (flat) search cost $O(N \cdot d)$ that ANN indexes are built to avoid, and the brute-force baseline the library still ships as `IndexFlat`.

The cost grows **linearly with $N$**: double the corpus, double every query. That linear wall is the whole motivation for ANN.

![Exact search cost O(N·d) (red) grows linearly with corpus size N — reaching ~7.7 billion multiply-adds per query at a 10M×768 real-world corpus — while an IVF index (green) that scans √N centroids plus a few cells' worth of vectors stays orders of magnitude below it. The amber line anchors our *real* measured corpus (N=30,000, d=384), where exact search takes ~1 ms/query on one core. Both axes log scale; the curves are the asymptotic shapes, the N-anchor is measured. Generated by `code/make_figures_04.py`.](../images/rag04_bruteforce_growth.png)

**A crucial, honest caveat about "milliseconds."** FAISS's exact search is extraordinarily well-optimized (SIMD dot products, cache-friendly layout). On our real 30k×384 corpus, a single query against `IndexFlatIP` takes only **~1 ms on one core** — already fast. So at *this* scale the win from ANN is real but modest; the dramatic $O(N)$ blow-up bites at 10M–1B vectors, where exact search crosses from milliseconds into tens-to-hundreds of milliseconds *per query on one core*, and where the memory (raw float32 vectors) stops fitting in RAM. You could throw hardware at it (more cores, GPUs — what FAISS does), but that only buys a constant factor; the $O(N)$ growth still wins eventually. The real fix is algorithmic: **stop comparing the query to vectors that obviously can't be its neighbours.** Even at 30k, we'll *measure* that skipping most of the corpus gives a genuine **~15–30× speedup** at high recall — and at real-world scale that factor becomes the difference between a viable product and a space heater.

---

## Intuition first: don't search the whole haystack

Here's the mental model that holds up.

Imagine finding the closest restaurant in a country. The brute-force way: measure the distance to *every* restaurant in the nation and take the smallest. Insane. What you actually do: **go to the right city first**, then search only within it. You skip 99.9% of restaurants because they're in cities you never visit. You might *occasionally* miss a restaurant just across a city line that's technically closer than anything in your city — but you accept that small risk for an enormous speedup.

That's ANN exactly. Build an index that **organizes vectors by neighbourhood** (cluster them, or wire them into a graph), then at query time **only look in the neighbourhoods near the query**. The "approximate" is the city-line risk: a true neighbour sitting just outside the regions you searched gets missed. That's the one thing you give up — and it's *tunable*: search more neighbourhoods to recover those misses, at the cost of more time.

Push on the analogy — it survives, and where it bends, it teaches:

- **"What exactly do I lose?"** Recall. If a true top-$k$ neighbour lives in a cell (or graph region) you didn't probe, you never see it — so **recall < 100%**. ANN reports "approximate" results; how approximate is a knob (`nprobe` for IVF, `efSearch` for HNSW). We *measure* this loss below, on real data.
- **"Can I always just search more neighbourhoods?"** Yes, and at the limit (probe every cell) you recover exact search — but then you've paid the full brute-force cost. The art is finding the smallest search that gives the recall you need.
- **"Does this get easier or harder in high dimensions?"** Harder. In high-dimensional space, distances concentrate (everything is roughly equidistant — the *curse of dimensionality*), so "neighbourhoods" are less cleanly separated and you must probe more to hit the same recall. ANN over 384- or 768-dim embeddings is genuinely hard; the indexes below are what make it work anyway.

The mapping is exact: **organizing vectors by neighbourhood is the index build (k-means cells or a graph), going to the right city is query routing, and the city-line risk is the recall you trade for speed.** Two index families dominate, differing only in *how* they organize the neighbourhoods.

![Animated — the recall/speed knob, turning. The query (star) lands in its Voronoi cell; at nprobe=1 only that one cell is scanned and several true neighbours are still missed (red rings). As nprobe grows the probed region expands to cover the neighbouring cells, the misses turn to hits (green), and recall climbs to 1.0 — at the cost of scanning more vectors. This 2D schematic shows the *routing*; the real, steep high-dimensional cliff over our 384-D corpus is measured in the recall-cliff figure below. Generated by `code/make_animation_04.py`.](../images/rag04_nprobe_growth.gif)

---

## The mechanism: IVF and HNSW (and a flat baseline)

```mermaid
graph TD
    Q(["query vector q"]):::amber --> ROUTE{"which index?"}:::process

    ROUTE -->|"baseline"| FLAT["FLAT (brute force)<br/>scan ALL N vectors<br/>exact, O(N·d)"]:::danger
    ROUTE -->|"cluster-based"| IVF["IVF<br/>k-means → nlist cells<br/>probe nprobe nearest cells"]:::process
    ROUTE -->|"graph-based"| HNSW["HNSW<br/>navigable small-world graph<br/>greedy descent, ~O(log N)"]:::process

    IVF --> SCAN["scan only the<br/>probed cells' vectors"]:::out
    HNSW --> HOP["hop neighbour→neighbour<br/>toward q"]:::out
    FLAT --> ALL["sort all N distances"]:::out
    SCAN --> TOPK(["top-k neighbours<br/>(approximate)"]):::amber
    HOP --> TOPK
    ALL --> EXACT(["top-k neighbours<br/>(exact)"]):::amber

    PQ["Product Quantization (PQ)<br/>compress vectors → codes<br/>~32× smaller memory"]:::frozen -.->|optional, on IVF/HNSW| SCAN

    classDef amber fill:#7A6528,stroke:#6A5518,color:#fff
    classDef process fill:#5D4A8A,stroke:#4D3A7A,color:#fff
    classDef out fill:#2E7A5A,stroke:#1E6A4A,color:#fff
    classDef danger fill:#8B3B4A,stroke:#7B2B3A,color:#fff
    classDef frozen fill:#4A5B6E,stroke:#3A4B5E,color:#fff
```

**Flat (the baseline).** Store the vectors, scan them all, sort. Exact, simple, and the *right choice* below ~10k vectors (an index's overhead isn't worth it at that scale). It's also, always, the **ground truth**: its top-k is what we score every approximate index's recall against. In code this is `IndexFlatIP` — inner-product, because our vectors are unit-normalised (more on that in a moment). Everything else trades a little of flat's perfect recall for speed.

**IVF (Inverted File — cluster-based).** Build: run **k-means** to partition the vectors into `nlist` cells (Voronoi regions), and store an *inverted list* mapping each cell to the vectors in it. Query: find the `nprobe` cell **centroids** nearest the query, then exact-search only the vectors in those cells. You scan roughly `(N / nlist) × nprobe` vectors instead of all $N$ — a big cut when `nprobe ≪ nlist`. The miss: a true neighbour in an unprobed cell is invisible. (IVF builds on [k-means](../../04.%20Unsupervised_Learning/01-K-Means-Clustering/01-K-Means-Clustering.md) — the partitioning *is* k-means. FAISS trains the real k-means for us.)

![SCHEMATIC (2D projection for intuition — the real index lives in 384-D): IVF partitions the corpus into Voronoi cells (coloured by k-means centroid, the dark plus-marks). For a query (star), only the 3 nearest cells are probed (saturated colours); the rest of the corpus (faded) is skipped entirely. The query's true top-10 neighbours (green rings) all fall in the probed cells here — but a neighbour just over a cell boundary would be missed, which is the recall cost. Generated by `code/make_figures_04.py`.](../images/rag04_voronoi_cells.png)

**HNSW (Hierarchical Navigable Small World — graph-based).** Build: wire each vector to its nearest neighbours as a **graph**, with *multiple layers* — a sparse top layer of long-range links for fast travel, dense lower layers for precision (like express vs local subway lines). Query: start at an entry point in the top layer and **greedily hop** to whichever neighbour is closer to the query; when you can't get closer, drop to the next layer down and repeat, until the dense base layer. The hierarchy means each query touches only $\approx O(\log N)$ nodes instead of $N$.

![SCHEMATIC of HNSW's layered graph: a sparse top layer (long links), a medium layer, and the dense base layer holding all points. Greedy descent (amber arrows) starts at the top, hops to the node nearest the query, then drops a layer and repeats — reaching the query's neighbourhood in a few hops instead of scanning everything. Illustrative. Generated by `code/make_figures_04.py`.](../images/rag04_hnsw_layers.png)

> **Note:** IVF vs HNSW in one line — **IVF partitions space (cells you probe); HNSW connects points (a graph you walk).** HNSW usually gives higher recall at a given speed and is the default in most modern vector DBs, but it costs more memory (the graph links) and is harder to update (deletes degrade the graph). IVF is lighter and trivially updatable but needs the `nprobe` tuned. Many systems offer both. We'll *measure* HNSW's advantage on our real corpus below.

---

## The math, derived in full

Three mechanisms, each derived from first principles — the partition (IVF), the graph (HNSW), and the codebook (PQ) — plus the distance metric that underlies all of them. This is the section the earlier version skimped; here we take our time.

### 0. The distance metric: why inner product *is* cosine here

Before any index, fix what "near" means. Retrieval ranks passages by **cosine similarity** $\cos(\mathbf q,\mathbf v)=\dfrac{\mathbf q\cdot\mathbf v}{\lVert\mathbf q\rVert\,\lVert\mathbf v\rVert}$. The embedding step (`embed_corpus.py`) **L2-normalises every vector** to unit length, so $\lVert\mathbf q\rVert=\lVert\mathbf v\rVert=1$ and the cosine collapses to a plain dot product:

$$
\cos(\mathbf q,\mathbf v)=\mathbf q\cdot\mathbf v \qquad\text{(for unit vectors).}
$$

That is why every index in the code is built with `faiss.METRIC_INNER_PRODUCT` and the exact index is `IndexFlatIP` (IP = inner product): maximising inner product and maximising cosine are the *same ranking*, so we get cosine search at the price of a dot product. It also connects the two common FAISS metrics: for unit vectors, squared-L2 distance is $\lVert\mathbf q-\mathbf v\rVert^2 = 2 - 2\,\mathbf q\cdot\mathbf v$, so *minimising* L2 and *maximising* IP give identical neighbours — the choice is a convention, and we take IP to keep "bigger score = more similar." The notebook verifies the vectors really are unit-norm before trusting this.

### 1. IVF — the k-means partition, the routing, and the recall↔nprobe knob (derived)

**The build is k-means.** IVF's `nlist` cells are the clusters of a k-means run over the corpus. k-means alternates two steps until the centroids stop moving (see the [k-means chapter](../../04.%20Unsupervised_Learning/01-K-Means-Clustering/01-K-Means-Clustering.md)): **assign** every vector to its nearest centroid, then **update** each centroid to the mean of its assigned vectors. The result is a set of `nlist` centroids $\{\mathbf c_1,\dots,\mathbf c_{n_{\text{list}}}\}$ that induce a **Voronoi partition** — space carved into cells, where cell $j$ is every point closer to $\mathbf c_j$ than to any other centroid. In FAISS this is the single call `index.train(x)`; the follow-up `index.add(x)` assigns each vector to its cell and appends its id to that cell's **inverted list** (hence "inverted file"). So the index is a *coarse quantizer* (the centroids, themselves stored in a little `IndexFlatIP`) plus `nlist` id-lists.

We can *look inside* this partition on the real corpus — it is not a black box:

![REAL IVF partition on our corpus: a histogram of the 256 inverted-list lengths (vectors per cell). k-means gives roughly balanced cells around the mean of ~117 (= N/nlist), but real Wikipedia clusters topically, so there is a genuine tail out to 592 — a reminder that cells are not uniform and that a query landing in a big cell scans more. Generated by `code/make_figures_04.py`.](../images/rag04_ivf_cell_sizes.png)

**The routing.** At query time, IVF computes the query→centroid similarities $\mathbf q\cdot\mathbf c_j$ for all `nlist` centroids (cheap: `nlist` $\ll N$ dot products), picks the `nprobe` nearest cells, and exact-searches **only the vectors in those cells' inverted lists**. Everything in the other $n_{\text{list}}-n_{\text{probe}}$ cells is skipped entirely.

**Why nprobe trades recall for speed — the coverage-vs-cost derivation.** Let the true top-$k$ neighbours of $\mathbf q$ be spread across cells according to how many of them land in the query's own cell, its second-nearest cell, and so on. Probing `nprobe` cells retrieves *exactly* the true neighbours that happen to live in those `nprobe` cells; any true neighbour sitting in an unprobed cell is invisible. So

$$
\text{recall}(n_{\text{probe}}) \;=\; \frac{\mathbb{E}\big[\#\{\text{true top-}k \text{ neighbours in the } n_{\text{probe}} \text{ nearest cells}\}\big]}{k},
$$

a **non-decreasing** function of `nprobe` that starts below 1 (some neighbours straddle cell boundaries into unprobed cells) and reaches exactly 1 at $n_{\text{probe}}=n_{\text{list}}$ (you've probed every cell = exact search). The *cost*, meanwhile, is the number of vectors scanned, which on average is $n_{\text{probe}}\cdot N/n_{\text{list}}$ — **linear in `nprobe`**. Recall rises with *diminishing* returns (each extra cell you probe is farther from $\mathbf q$ and holds fewer of its true neighbours) while cost rises linearly — which is exactly why the recall/`nprobe` curve is a **cliff then a plateau**, and why a *sweet spot* exists at the smallest `nprobe` that clears your recall target. We measure this shape below.

**The cost model.** Putting the two phases together:

$$
\text{IVF cost} \;\approx\; \underbrace{n_{\text{list}} \cdot d}_{\text{scan centroids (routing)}} \;+\; \underbrace{\frac{N}{n_{\text{list}}} \cdot n_{\text{probe}} \cdot d}_{\text{scan the probed cells}}.
$$

> **Source / derivation:** [Jégou, Douze & Schmid (2011), *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI (DOI 10.1109/TPAMI.2010.57)](https://www.semanticscholar.org/paper/Product-Quantization-for-Nearest-Neighbor-Search-J%C3%A9gou-Douze/4748d22348e72e6e06c2476486afddbc76e5eca7) — §IV introduces the inverted-file (IVFADC) structure: a coarse quantizer (k-means cells) you probe, with the cost split into the centroid scan plus the probed inverted lists.

Symbols: $N$ vectors, $d$ dimensions, $n_{\text{list}}$ cells, $n_{\text{probe}}$ cells probed. **Choosing `nlist`.** Differentiating the cost w.r.t. $n_{\text{list}}$ and setting it to zero gives $n_{\text{list}}^\star \propto \sqrt{N\,n_{\text{probe}}}$ — so the common rule of thumb $n_{\text{list}}\approx\sqrt N$ (up to a small constant) *balances the two terms*, making both $\approx\sqrt N\cdot d$ and the whole query **sub-linear** in $N$ — the entire point. Too few cells and each cell is huge (you scan a lot per probe); too many and the centroid scan itself dominates *and* cells become underpopulated. We use $n_{\text{list}}=256$ over 30k vectors (a few × $\sqrt{30000}\approx 173$), which the histogram above shows fills cells to a healthy ~117 on average.

### 2. HNSW — the layer pyramid, greedy descent, and why it's $O(\log N)$ (derived)

**The build is incremental, and the key trick is the layer assignment.** HNSW has no separate "train" step; `index.add(x)` inserts vectors one at a time. When a node is inserted, it is assigned a **maximum layer** $\ell$ drawn from a geometric/exponential distribution: $\ell = \lfloor -\ln(u)\cdot m_L \rfloor$ for a uniform $u\in(0,1)$, where $m_L$ is a level-multiplier constant (FAISS's default is $m_L=1/\ln M$). A node present at layer $\ell$ is inserted into every layer from $\ell$ down to 0, connecting to its $M$ nearest neighbours *within each layer*. Because $\Pr[\ell \ge L]$ decays like $e^{-L/m_L}$, each layer up holds a **roughly constant fraction** of the nodes below it: going up one level multiplies the survivor count by $e^{-1/m_L}$, and with the default $m_L = 1/\ln M$ that fraction is $e^{-\ln M} = 1/M$ (here $1/32 \approx 0.031$). So the layers form a **pyramid**: the base (layer 0) holds all $N$ nodes; each level up keeps only $\approx 1/M$ of the one below; the top holds a handful.

This is not hand-waving — we read the *real* pyramid straight off the built graph:

![REAL HNSW layer pyramid on our 30k corpus: node counts per level plotted on a log x-axis (base at the bottom). Layer 0 holds all 30,000 nodes; layer 1 holds 944; layer 2, 23; layer 3, just 2. Each level up is a small fraction of the one below — the geometric decay from the exponential layer assignment. This shrinking pyramid is exactly what makes greedy descent ~O(log N). Generated by `code/make_figures_04.py`.](../images/rag04_hnsw_pyramid.png)

**Greedy descent, and the $O(\log N)$ argument.** A query enters at the single top-layer node and, on that layer, **greedily hops** to whichever neighbour is closer to $\mathbf q$, stopping when no neighbour improves (a local optimum for that layer). It then drops to the next layer down and repeats from where it landed, until the dense base layer, where it does a final broadened search. Why is the total work $\approx O(\log N)$? Two multiplied factors:

1. **Number of layers.** With each layer holding $\approx 1/M$ of the nodes below, the number of non-empty layers is $\approx \log_M N$ — i.e. $O(\log N)$ layers to descend. (Our pyramid: $30000\to944\to23\to2$, four levels, and $\log_{32} 30000 \approx 3$, in line with the four measured levels.)
2. **Work per layer is $O(1)$ in expectation.** On a navigable small-world graph, the greedy walk within a layer reaches the local optimum in a bounded (roughly constant) number of hops, because the long-range links let each hop cover a constant *fraction* of the remaining distance — the small-world property.

Multiply them: $O(\log N)$ layers $\times\ O(1)$ hops/layer $= O(\log N)$ total — the hallmark that makes HNSW scale like binary search rather than a linear scan. The build is $O(N\log N)$: each of $N$ insertions runs a greedy search to find its neighbours.

> **Source / derivation:** [Malkov & Yashunin (2016/2018), *Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs* (arXiv:1603.09320)](https://arxiv.org/abs/1603.09320) — introduces HNSW; the multi-layer structure and the exponentially-decaying level assignment give the polylogarithmic search complexity, and it defines the build parameters $M$ (links per node) and `efConstruction` / `efSearch` (candidate-list sizes).

**What the knobs actually control.** **`M`** is the graph degree — the number of links each node keeps per layer. More links = more paths to any neighbourhood = higher recall, but more memory (the links dominate HNSW's overhead) and slower build. **`efConstruction`** is the size of the candidate list the *build* keeps while searching for each new node's neighbours — bigger = a better-connected graph = higher recall ceiling, at the cost of a slower build. **`efSearch`** is the size of the candidate list the *query* keeps during the base-layer walk — the **recall↔speed knob at query time**, HNSW's analogue of `nprobe`: bigger explores more of the neighbourhood before committing (higher recall, slower), smaller commits faster. We build with `M=32`, `efConstruction=200`, and sweep `efSearch` below.

> **Note (build cost is real):** $O(N \log N)$ build is not free. On our 30k corpus, single-threaded, the HNSW graph takes **~9 seconds** to construct while the IVF k-means + add takes **~0.1 s** and Flat is instant. HNSW buys query speed with build time and memory — a trade you'll see plotted below.

### 3. Product Quantization — encode, decode, and the compression math (derived)

Storing $N$ raw vectors costs $N \cdot d \cdot 4$ bytes (float32). At a billion 768-dim vectors that's ~3 TB — too much for RAM. **Product Quantization (PQ)** compresses each vector into a handful of bytes. The construction has three moving parts; the code exposes all of them.

**Encode.** Split the $d$-dimensional vector into $m$ contiguous **subvectors** of $d/m$ dimensions each ($m$ must divide $d$; for us $384/48 = 8$ dims per subvector). For each of the $m$ subspaces, run k-means to learn a **codebook** of $k = 2^{\text{nbits}}$ centroids (with $\text{nbits}=8$, that's $256$ centroids per subspace). To encode a vector, replace each subvector by the **id of its nearest sub-centroid** — a number in $[0, 255]$. The concatenation of those $m$ ids **is** the code: for us, $48$ bytes.

**Decode.** To reconstruct, look up each stored id in its subspace codebook and concatenate the chosen centroids back into a $d$-dim vector. This is *lossy* — the reconstruction is the nearest codebook centroid per subspace, not the original — and the gap is the **quantization error**.

![REAL PQ encode/decode. Left: one vector is split into m=48 subvectors of 8 dims; each is replaced by the id of its nearest centroid in that subspace's 256-entry codebook — the actual first six ids of passage 0 are shown (181, 47, 187, 218, 42, 58, …). The 48 ids ARE the code: 48 bytes vs 1,536 bytes raw. Right: the real distribution of L2 reconstruction error over all 30,000 vectors (mean ≈ 0.386) — the quantization loss that costs a little recall. Generated by `code/make_figures_04.py`.](../images/rag04_pq_encoding.png)

**Search on codes — asymmetric distance.** The clever part: at query time the *query stays full-precision*; only the database vectors are the coarse decodes. FAISS precomputes, for each subspace, the distance from the query's subvector to all $256$ sub-centroids (a small $m\times 256$ table), then a database vector's distance is just the **sum of $m$ table look-ups** indexed by its code — no decompression, and more accurate than quantizing the query too. This is *asymmetric distance computation* (ADC).

**The compression math.** Raw vs code, per vector:

$$
\text{raw bits} = d \cdot 32, \qquad \text{PQ bits} = m \cdot \text{nbits}, \qquad \text{ratio} = \frac{32\,d}{m\cdot\text{nbits}}.
$$

> **Source / derivation:** [Jégou, Douze & Schmid (2011), *Product Quantization for Nearest Neighbor Search*, IEEE TPAMI (DOI 10.1109/TPAMI.2010.57)](https://www.semanticscholar.org/paper/Product-Quantization-for-Nearest-Neighbor-Search-J%C3%A9gou-Douze/4748d22348e72e6e06c2476486afddbc76e5eca7) — §III defines PQ (decompose into $m$ subspaces, quantize each with a $2^{\text{nbits}}$-entry codebook, store the codes) and §III-A the asymmetric-distance trick.

For our real corpus ($d = 384$, $m = 48$, nbits $= 8$): raw $= 384 \times 4 = \mathbf{1{,}536}$ bytes; PQ $= 48 \times 8 / 8 = \mathbf{48}$ bytes — a **32× memory cut** (46.1 MB → 1.44 MB for the whole corpus). We measured the real mean reconstruction error at ~0.386 (cosine ≈ 0.922 between original and decode), which is why PQ costs some recall (below). It's usually combined with IVF (the `IndexIVFPQ` family) for billion-scale search.

![Product Quantization memory per vector on our real corpus: a raw float32 vector (384 dims × 4 bytes = 1,536 bytes) versus its PQ code (48 subquantizers × 8 bits = 48 bytes) — a 32× reduction, shrinking the 30,000-vector corpus from 46 MB to 1.4 MB. Log scale. Generated by `code/make_figures_04.py`.](../images/rag04_pq_memory.png)

**Is that all the math?** For this chapter, yes — these three derivations (partition, graph, codebook) plus the IP↔cosine↔L2 metric relation cover what you need to *reason about and tune* every index we build. The natural next layers — the theory of the recall/latency Pareto frontier, or graph ANN that spills to SSD (DiskANN) — are pointers in the references, not required to operate FAISS.

---

## Worked example: the real pipeline, measured step by step

This is the heart of the chapter — the same code the [teaching notebook](code/04-Vector-Databases-and-ANN-Indexes.ipynb) runs step by step, over the real embedded Wikipedia corpus. The runnable module is [`code/vector_indexes.py`](code/vector_indexes.py) (a typed FAISS wrapper) and the corpus is built once by [`code/embed_corpus.py`](code/embed_corpus.py). Every number below comes from the shipped notebook run — nothing is hand-typed. I walk through *what each piece of code does and why*, so nothing in the implementation is left unexplained.

> **One implementation reality worth naming.** On macOS, `faiss` and `torch` both link `libomp`, and importing them in the *same* process double-initialises OpenMP and **crashes** (even with `KMP_DUPLICATE_LIB_OK=TRUE`). So the pipeline is split into two processes: `embed_corpus.py` (torch / sentence-transformers) embeds the corpus and writes plain `.npy` vectors; `vector_indexes.py` and the notebook (numpy + faiss, **never torch**) load those vectors and do all the indexing. That split is also correct real-world architecture — embedding is a batch job, the index is a serving system. We also measure latency **one query at a time on one core** (via `_median_query_latency_ms`, which times individual `index.search` calls and takes the median), because that's the honest per-request serving cost — FAISS's default batch search parallelises across queries and cores and would flatter every index.

**Load the corpus and build the exact baseline.** `load_corpus()` reads the cached `.npy` vectors and passage text; `build_flat()` constructs an `IndexFlatIP` and calls `.add()` (Flat needs no training — it just copies the vectors in). `exact_topk` runs `flat.search(queries, k)` to get the ground-truth neighbour ids, and `exact_latency_ms` times a single query on one core:

```python
from vector_indexes import load_corpus, build_flat, exact_topk, exact_latency_ms, TOP_K

corpus = load_corpus()                       # 30,000 real passages × 384-dim (bge-small-en-v1.5)
flat = build_flat(corpus.embeddings)         # IndexFlatIP — exact, and our recall ground truth
ground_truth = exact_topk(flat, corpus.queries, TOP_K)
print(exact_latency_ms(flat, corpus.queries))   # ~1.0 ms/query over 30,000 vectors, one core
```

The retrieved neighbours are genuinely on-topic — for a "USB" query, exact search returns the USB passages (usability at similarity 1.0, then durability, compatibility, standards at 0.76–0.80). That's the point of using real data: you can *read* that retrieval works before we start approximating it. The notebook also verifies the vectors are unit-norm (min/max L2 norm = 1.0000), confirming the IP-is-cosine reasoning from the math section.

**Build IVF, and look inside the partition.** `build_ivf()` calls `index.train(x)` (the k-means that learns the `nlist=256` centroids) then `index.add(x)` (assign each vector to its cell, build the inverted lists). The helpers `ivf_centroids` and `ivf_cell_sizes` expose the real partition — the centroid matrix and the inverted-list lengths:

```python
from vector_indexes import build_ivf, ivf_cell_sizes, ivf_centroids
ivf = build_ivf(corpus.embeddings)           # train() = k-means; add() = fill inverted lists
sizes = ivf_cell_sizes(ivf)                  # vectors per cell: mean 117, min 20, max 592
```

The cells hold ~117 vectors on average (= $N/n_{\text{list}}$) with a real tail to 592 — the histogram figure above. That mean is the concrete payoff: probing one cell scans ~117 vectors instead of 30,000, **~256× fewer distance computations**.

**One IVF query, then the `nprobe` sweep — the recall cliff.** A single query at `nprobe=8` scans the 8 nearest cells (~850 vectors, 2.8% of the corpus) and, for this query, recovers recall 1.0. Then `sweep_ivf` sets `index.nprobe` to each value, searches all 500 queries, and records the real recall@10 (vs the exact ground truth) and the real single-query latency:

```python
from vector_indexes import sweep_ivf
for p in sweep_ivf(ivf, corpus.queries, ground_truth):   # measured on the real index
    print(p.knob, p.recall, p.latency_ms)
```

```
 nprobe | recall@10 | ms/query | speedup vs exact
 -------------------------------------------------
      1 |     0.674 |    0.017 |          58.5x
      2 |     0.791 |    0.021 |          46.7x
      4 |     0.876 |    0.030 |          32.7x
      8 |     0.933 |    0.051 |          19.1x
     16 |     0.970 |    0.088 |          11.1x
     32 |     0.987 |    0.158 |           6.2x
     64 |     0.997 |    0.317 |           3.1x
    128 |     0.999 |    0.603 |           1.6x
    256 |     1.000 |    1.172 |           0.8x
```

Read it top to bottom — this is the **recall cliff** the coverage-vs-cost math predicted. At **`nprobe=1`** the index is **~58× faster** than exact but recall is only **0.674** — it misses a third of the true neighbours, because they live in cells it didn't probe. As `nprobe` climbs, recall recovers with diminishing returns: **0.674 → 0.791 → 0.876 → 0.933 → 0.970** — and cost rises linearly. The **sweet spot is `nprobe=16`**: recall **0.970** while still **~11× faster** than exact. Past that, you pay a lot more latency for almost no recall gain (at `nprobe=256` you've scanned every cell — recall 1.0, but *slower* than exact because of the centroid-scan overhead). *That* shape — steep climb then plateau — is exactly what the derivation said, and what you tune against.

![Recall@10 (green) vs nprobe on our REAL IVF index over 30k×384 Wikipedia vectors: a steep cliff from 0.67 at nprobe=1 up to 1.0, with per-query latency (red, the cost) rising in counterpoint and the exact-flat latency (~1 ms) marked as the dotted reference. The sweet spot (nprobe=16) reaches recall 0.97 at ~11× faster than exact; beyond it, latency rises for negligible recall gain. (The annotated speedup is from the figure's own measurement pass; run-to-run latency varies by ±10%, so the notebook table above may read a hair different — recall is identical.) Generated by `code/make_figures_04.py`.](../images/rag04_ivf_recall_cliff.png)

**Build HNSW, and read the layer pyramid.** `build_hnsw()` sets `M=32`, `efConstruction=200` and calls `.add()` (the incremental graph construction — no train step). `hnsw_level_counts` reads the real per-layer node counts off the built graph:

```python
from vector_indexes import build_hnsw, hnsw_level_counts, sweep_hnsw
hnsw = build_hnsw(corpus.embeddings)         # M=32, efConstruction=200 — the real graph
print(hnsw_level_counts(hnsw))               # [30000, 944, 23, 2] — the pyramid, ~1/M decay
```

The counts — `30000 → 944 → 23 → 2` — are the $\approx 1/M$ geometric decay the math predicted ($944/30000 \approx 1/32$), giving $\approx \log_M N \approx 3$–$4$ layers to descend — the pyramid that makes the walk $O(\log N)$. Then `sweep_hnsw` sweeps `efSearch` the same way `sweep_ivf` swept `nprobe`:

```
 efSearch | recall@10 | ms/query | speedup vs exact
 --------------------------------------------------
        8 |     0.979 |    0.034 |          28.5x
       16 |     0.991 |    0.051 |          19.2x
       32 |     0.996 |    0.082 |          11.9x
       64 |     0.998 |    0.147 |           6.6x
      128 |     0.999 |    0.274 |           3.6x
      256 |     1.000 |    0.480 |           2.0x
      512 |     1.000 |    0.985 |           1.0x
```

Notice how much *higher and flatter* HNSW's curve starts: even at `efSearch=8` it's already at **recall 0.979 while ~29× faster than exact** — a point IVF never reaches (IVF's fastest 0.97-recall setting is only ~11× faster). That's HNSW's reputation in one table.

![Recall@10 (green) vs efSearch on our REAL HNSW graph: recall starts high (0.98 at efSearch=8) and saturates to 1.0, while latency (red) climbs — at efSearch=8 HNSW is roughly 30× faster than exact search. efSearch is HNSW's recall/speed knob — the graph analogue of IVF's nprobe. (The annotated speedup is the figure run's own measurement; run-to-run latency varies by ±10%, so it may read a hair different from the notebook table above — recall is identical.) Generated by `code/make_figures_04.py`.](../images/rag04_hnsw_efsearch.png)

**Compare them honestly — the recall/latency frontier.** A single number lies; the real comparison is the **frontier**: at a target recall, which index is faster? The notebook's `fastest_at` helper filters each sweep to the points clearing a recall target and takes the fastest. At recall ≥ 0.95: IVF's best is `nprobe=16` (recall 0.970, ~11× vs exact); HNSW's is `efSearch=8` (recall 0.979, ~29× vs exact) — **HNSW wins**. This is exactly what [ANN-Benchmarks](https://ann-benchmarks.com/) plots.

![The REAL recall/latency frontier over our 30k corpus: HNSW (amber, sweeping efSearch) sits *up and to the left* of IVF (purple, sweeping nprobe) — higher recall at lower latency — with exact flat (~1 ms, recall 1.0) marked. Up-and-to-the-left wins; this is why HNSW is the default in most modern vector DBs. The result is corpus- and hardware-dependent, but the shape is characteristic. Generated by `code/make_figures_04.py`.](../images/rag04_recall_vs_latency.png)

On this corpus HNSW dominates — higher recall at lower latency across the board. That's *why* it's the default in most modern vector DBs. The cost shows up elsewhere: build time and memory.

![Left: REAL index build time as N grows (real subsets of our corpus). HNSW's graph construction (amber) grows super-linearly to ~9 s at 30k while Flat and IVF stay near-instant. Right: index memory — the raw float32 vectors are the floor (blue), and HNSW adds its graph links on top (amber). HNSW buys query speed with build time and memory. Generated by `code/make_figures_04.py`.](../images/rag04_build_memory.png)

**Product Quantization — encode, decode, and combine, measured.** `pq_encode_decode` trains a real `ProductQuantizer`, encodes every vector to its 48-byte code, decodes it back, and returns the per-vector reconstruction error. Then `build_ivfpq` fuses IVF routing with PQ codes:

```python
from vector_indexes import pq_encode_decode, build_ivfpq, recall_at_k
codes, recon, errors = pq_encode_decode(corpus.embeddings)   # 48 uint8 ids/vector; mean err ≈ 0.386
ivfpq = build_ivfpq(corpus.embeddings); ivfpq.nprobe = 16
_s, ids = ivfpq.search(corpus.queries, TOP_K)
print(recall_at_k(ids, ground_truth))                        # ~0.69 — PQ's approximate distances cost recall
```

The real code for passage 0 is 48 small integers (`[181, 47, 187, 218, 42, 58, …]`) — a whole embedding in 48 bytes. The reconstruction error (mean 0.386, cosine 0.922 vs original) is why `IndexIVFPQ` at `nprobe=16` gives **recall ~0.688** where plain IVF gave 0.970 — PQ's approximate distances add a second source of error on top of the cell misses, a **0.28 recall cost** for **32× less memory** (46 MB → 1.44 MB). At billion scale you pay that gladly because the raw vectors simply don't fit in RAM; you usually recover some recall by over-fetching candidates and re-ranking the survivors on their exact vectors.

**The library one-liners.** The code above *is* the library — this is exactly how you use FAISS in the real world. For completeness, the same in Postgres via **pgvector**: `CREATE INDEX ... USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64)` (its documented defaults), or `USING ivfflat (...) WITH (lists = 100)`, tuning `SET hnsw.ef_search = 40` (default) per session.

---

## Pitfalls and failure modes

These are where vector search quietly breaks in the real world.

**1. The recall cliff at too-low `nprobe`/`efSearch`.** The most common ANN mistake: set the search knob too low to "make it fast," and silently lose neighbours.

- *Failing:* you ship `nprobe=1` because latency looked great in testing; recall is **0.674** (our real table) and users get visibly worse answers, with no error to alert you.
- *Fix:* **measure recall against exact ground truth** on a held-out query set and pick the smallest `nprobe`/`efSearch` that clears your recall target — the sweet spot (nprobe=16 → recall 0.97 here), not the fastest setting.

**2. Metadata filtering + ANN (the trap).** You often want "nearest vectors *where* `tenant_id = X` and `date > Y`." Combining filters with ANN is genuinely hard, and both naive approaches fail:

- *Post-filter* (ANN first, then drop non-matching results): if the filter is selective, the top-k ANN results may *all* get filtered out, **starving** you of results — you asked for 10 and got 2.
- *Pre-filter* (restrict to matching vectors, then search): this **breaks the graph/cells** — HNSW's links assume the full graph, and removing most nodes destroys navigability; IVF cells may be nearly empty.
- *Fix:* use a vector DB with **native filtered search** (Qdrant's filterable HNSW, Weaviate's, pgvector's row filters with the index) that integrates the filter into traversal, or **over-fetch** (retrieve far more than $k$ then post-filter) when filters are mild. Never assume naive post-filtering is safe.

**3. Index build time + memory blow-up.** HNSW's graph links and PQ's codebooks aren't free. We *measured* it: HNSW built in ~9 s at 30k (single-threaded) vs ~0.1 s for IVF; a large `M` or `efConstruction` makes both worse, and HNSW memory can be *multiples* of the raw vectors.

- *Failing:* you set `M=64, efConstruction=512` for "max recall," and the index takes far longer to build and OOMs the box.
- *Fix:* start with sane defaults (HNSW `M=16`, `efConstruction=64` — pgvector's defaults; we used `M=32` here), measure recall, and raise only if needed; use **PQ** (or a quantized index) when memory is the constraint, accepting its recall loss (our IVFPQ dropped from 0.97 to ~0.69).

**4. Updates and deletes degrading a graph index.** HNSW doesn't delete cleanly — removing a node leaves dangling links, and graphs that churn heavily drift from optimal.

- *Failing:* a high-churn corpus (frequent doc updates) on HNSW slowly loses recall as deletes pile up as tombstones.
- *Fix:* use **soft-deletes + periodic rebuilds**, or an index that supports updates better (IVF re-assigns more cleanly); many vector DBs handle this for you — know your store's update story before you pick it.

> **Gotcha:** notice the curse of dimensionality lurking under all of this — in high-$d$ space, distances concentrate, so cells and graph neighbourhoods are less separable and you need *more* `nprobe`/`efSearch` for the same recall than a 2D intuition suggests. Our real cliff is already steep at $d=384$; at $d=768$ (or $d=1536$ for large embedders) it's a real engineering effort to keep recall high *and* latency low. That tension is the job.

---

## Where it matters, and when flat is right

**The one problem ANN solves:** finding the (approximately) nearest vectors in a large corpus in **sub-millisecond** time, by indexing the vectors so a query scans a tiny, relevant fraction instead of all $N$ — trading a tunable sliver of recall for orders-of-magnitude speed at scale, and (with PQ) a large memory saving. It's the **retrieval-serving** layer of RAG: the embedder (chapter 3) defines *what's near*; the index makes *finding it* fast.

**When flat (exact) is the right answer — don't index:** below **~10,000 vectors**, brute force is already sub-millisecond (we measured ~1 ms even at 30k), and an ANN index only adds build time, memory, and a recall penalty for little real speed benefit. FAISS ships `IndexFlat` precisely for this. *Reach for an index when $N$ is large enough that the linear scan hurts, or when the raw vectors stop fitting in RAM* — not before.

**The core tradeoffs you're always balancing:**

| Lever | More of it → | Cost |
|---|---|---|
| `nprobe` (IVF) / `efSearch` (HNSW) | higher recall | slower query |
| `M`, `efConstruction` (HNSW) | better graph, higher recall | slower build, more memory |
| `nlist` (IVF) | finer cells (less to scan per cell) | more centroids to scan, harder to fill |
| PQ (compression) | far less memory | extra recall loss (approximate distances) |

---

## In production

Real systems, with **verified** specifics:

- **FAISS** (Meta) — the foundational library, and the one we used on this page: `IndexFlat` (exact), `IndexIVFFlat`, `IndexHNSWFlat`, and `IndexIVFPQ` (the billion-scale workhorse — IVF routing + PQ compression). The reference for index choice and tuning; powers many of the DBs below under the hood.
- **pgvector** (Postgres extension) — HNSW and IVFFlat indexes inside your existing database. Documented defaults: **HNSW `m = 16`, `ef_construction = 64`**, query-time **`ef_search = 40`**; IVFFlat `lists` guidance is **`rows/1000`** up to 1M rows and **`√rows`** beyond, with a starting **`probes ≈ √lists`**. The pragmatic choice when you don't want a separate vector store.
- **Pinecone, Weaviate, Qdrant, Milvus** — managed/dedicated vector DBs, almost all **HNSW**-based, adding the operational layer ANN libraries lack: **native metadata filtering**, horizontal sharding, replication, and live updates. Qdrant and Weaviate are known for **filterable HNSW** (the pitfall-2 fix); Milvus scales to billions with IVF/PQ + GPU.
- **Chroma** — a lightweight embedded store (HNSW via hnswlib) ideal for prototypes and small corpora — and a reminder that **for small $N$, an in-process flat/HNSW index is plenty**; you don't need a cluster.

**When to reach for which:** prototype with a flat scan or **Chroma** (small $N$); add **pgvector** when your data already lives in Postgres; graduate to **Pinecone/Weaviate/Qdrant/Milvus** when you need scale, filtering, and operations. Whatever you choose, the discipline is the same one we practised on this page: **measure recall against exact ground truth, then tune the search knob to the smallest value that meets your recall SLO.**

> **Note:** the through-line continues. Chapters 1–3 built the retrieval *quality* stack (pipeline, chunks, embeddings); this chapter made search *fast at scale*. Next, [chapter 5](../05-Hybrid-Search-BM25-and-Dense/05-Hybrid-Search-BM25-and-Dense.md) combines this dense ANN search with lexical (BM25) search for the best of both, and [chapter 6](../06-Re-ranking-Cross-Encoders/06-Re-ranking-Cross-Encoders.md) reranks the candidates this index returns — and, as we saw with IVFPQ, re-ranking on exact vectors is also how you recover recall lost to compression. ANN gets you the *right neighbourhood, fast*; the rest of the stack sharpens *which* of those candidates wins.

---

## Recap and rapid-fire

**If you remember nothing else:** exact ("flat") nearest-neighbour search is $O(N \cdot d)$ — fine for thousands (~1 ms even at our 30k×384 corpus), hopeless at 10M×768 (~7.7B ops/query). **ANN indexes** skip almost all vectors by organizing them into neighbourhoods — **IVF** (k-means cells you probe with `nprobe`) or **HNSW** (a navigable graph whose layer pyramid gives ~$O(\log N)$ descent) — trading a tunable bit of **recall** for orders-of-magnitude speed. The recall↔search-knob tradeoff is a **cliff**, which we derived (coverage vs linear cost) and measured on real data: low `nprobe`/`efSearch` is fast but misses neighbours (IVF recall 0.674 at nprobe=1), recovering to ~1.0 as you probe more (IVF sweet spot nprobe=16: recall 0.97 at ~11× faster; HNSW hits recall 0.98 at ~29× faster — it dominates the frontier). **PQ** encodes each vector into $m$ codebook ids (48 bytes, 32× smaller; 46 MB → 1.4 MB here) at a real recall cost (0.97 → ~0.69) from its ~0.386 reconstruction error. Below ~10k vectors, **just use flat**.

**Quick-fire — say these out loud:**

- *Why doesn't exact search scale?* It's $O(N \cdot d)$ — linear in corpus size; 10M×768 ≈ 7.7B ops per query, and the raw vectors stop fitting in RAM.
- *IVF in one sentence?* k-means the vectors into cells; at query time probe only the `nprobe` nearest cells' inverted lists.
- *Why does nprobe trade recall for speed?* Recall = fraction of true neighbours living in the probed cells (rises with diminishing returns); cost = vectors scanned $\approx n_{\text{probe}}\,N/n_{\text{list}}$ (rises linearly) — hence a cliff then a plateau.
- *HNSW in one sentence?* A multi-layer navigable graph whose layers decay ~1/M per level (with $m_L=1/\ln M$); greedily hop neighbour→neighbour toward the query, ~$O(\log N)$ (≈ $\log_M N$ layers × $O(1)$ hops/layer).
- *What's the recall cliff?* Too-low `nprobe`/`efSearch` is fast but silently misses neighbours (IVF recall 0.67 at nprobe=1 in our measured sweep).
- *IVF vs HNSW?* IVF partitions space (lighter, updatable, tune `nprobe`); HNSW connects points (dominates the recall/latency frontier, but more memory, ~9 s build, harder to update).
- *What does PQ buy and cost?* 32× less memory (m=48 codebook ids vs raw floats); costs recall from approximate (asymmetric) distances (we measured 0.97 → 0.69).
- *Why is filtering + ANN hard?* Post-filter can starve results; pre-filter breaks the graph — use native filtered search or over-fetch.
- *When NOT to use an ANN index?* Below ~10k vectors — flat is already ~1 ms; the index only adds build time, memory, and recall loss.
- *How do you tune it?* Measure recall vs exact ground truth; pick the smallest search knob that meets your recall SLO — read it off the recall/latency frontier.

---

## References and further reading

The curated link library for this topic — videos, courses, articles, papers, books, and internal cross-links — lives in a companion file so it can be reused as a standalone reference list:

**→ [Vector Databases & ANN Indexes — references and further reading](04-Vector-Databases-and-ANN-Indexes.references.md)**
