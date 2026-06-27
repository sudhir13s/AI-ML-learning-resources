---
id: "11-rag-and-llm-apps/caching-and-cost-optimization"
topic: "Caching & Cost Optimization for LLM Apps"
parent: "11-rag-and-llm-apps"
level: intermediate
prereqs: ["rag-fundamentals", "llm-app-orchestration"]
interview_frequency: medium
updated: 2026-06-20
---

# Caching & Cost Optimization for LLM Apps
> LLM apps are billed per token and measured in latency — both explode at scale. The big levers:
> **prompt (prefix) caching** (reuse the unchanging system prompt / retrieved context — 50–90% input
> cost cut), **semantic caching** (serve a cached answer when a *new* query is semantically close to a
> past one), **model routing** (cheap model for easy queries), plus shrinking context (rerank/compress)
> and batching. Knowing these is what separates a prototype from an affordable product.

**Why it matters:** the "how do you make this affordable and fast in production?" question. You'll
distinguish prompt/prefix caching (exact-prefix reuse) from semantic caching (embedding-similarity
reuse), discuss cache invalidation and false-hit risk, and combine routing, context compression, and
batching to hit a cost/latency budget — while protecting quality.

**⭐ Start here — suggested path:**

1. **Cut input cost first** — watch [Prompt Caching Guide](https://www.youtube.com/watch?v=RDjaUJz-uWo). *How OpenAI/Anthropic/Google prefix-cache the static parts of a prompt for 50–90% savings.*
2. **Read the mechanics** — read [Anthropic: Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) + [OpenAI: Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching). *Cache breakpoints, automatic vs explicit, what's cacheable.*
3. **Add semantic caching** — watch [Semantic Caching Explained (Redis)](https://www.youtube.com/watch?v=NrqvtsnjIHU). *Serve cached answers for *similar* (not identical) queries via embeddings.*
4. **Build a semantic cache** — watch [GPTCache — Save Cost on LLMs](https://www.youtube.com/watch?v=Yug3gObpX-g), and read the [GPTCache repo](https://github.com/zilliztech/GPTCache). *A working embedding-based response cache.*
5. **Combine strategies** — watch [Caching Strategies to Slash Your LLM Bill](https://www.youtube.com/watch?v=j9wVKM89XFU). *Prompt + semantic caching together, with a cost demo.*

## 🎓 Courses (free)
- [LangChain for LLM Application Development](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/) — **DeepLearning.AI × LangChain** — covers caching and efficient chains as part of building real apps.
- [HF: LLM Inference Optimization (guide)](https://huggingface.co/docs/transformers/en/llm_optims) — **Hugging Face** — free, hands-on guide to latency/cost optimizations (caching, batching, quantization pointers).

## 🎥 Videos
- [Prompt Caching Guide (non-technical)](https://www.youtube.com/watch?v=RDjaUJz-uWo) — **Dan Cleary (PromptHub)** — how prefix caching works across OpenAI/Anthropic/Google.
- [Semantic Caching Explained: Reduce AI API Costs with Redis](https://www.youtube.com/watch?v=NrqvtsnjIHU) — **Nariman Codes** — embedding-similarity caching at the gateway.
- [GPTCache — Save Cost on LLMs](https://www.youtube.com/watch?v=Yug3gObpX-g) — **Fahd Mirza** — installing and using a semantic response cache locally.
- [Caching Strategies to Slash Your LLM Bill](https://www.youtube.com/watch?v=j9wVKM89XFU) — **MadeForCloud** — prompt + semantic caching combined, with a cost demo.

## 📄 Key Papers
- [GPT Semantic Cache: Reducing LLM Costs and Latency via Semantic Embedding Caching](https://arxiv.org/abs/2411.05276) — **Regmi & Pun (2024)** — embedding-based caching; up to ~69% fewer API calls.
- [MeanCache: User-Centric Semantic Caching for LLM Web Services](https://arxiv.org/abs/2403.02694) — **Gill et al. (2024)** — privacy-aware, federated semantic cache design.
- [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) — **Kwon et al. (2023)** — the KV-cache/serving optimization behind cheap high-throughput inference.

## 📰 Articles / Blogs (free, no paywall)
- [Anthropic: Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — **Anthropic** — cache breakpoints and what to cache.
- [OpenAI: Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — **OpenAI** — automatic prefix caching and how to structure prompts for hits.
- [zilliztech/GPTCache (GitHub)](https://github.com/zilliztech/GPTCache) — **Zilliz** — the reference semantic-cache library to read and run.
- [What is Semantic Caching?](https://redis.io/blog/what-is-semantic-caching/) — **Redis** — how similarity-based caching works and when it pays off.

## 📚 Books (free, with chapters)
- [Hugging Face Transformers — **"LLM inference optimization" guide**](https://huggingface.co/docs/transformers/en/llm_optims) — **Hugging Face** — the closest free book-length reference: caching, batching, and the latency/cost levers, fully open.

## 🔗 In this platform
- Concept depth (the *why*): [AI-ML-intuition 8.02 Retrieval-Augmented Generation](../../../AI-ML-intuition/Module_8_LLMs_and_Agentic_Systems/8.02_Retrieval_Augmented_Generation.md) · [1.02 Dense Embeddings](../../../AI-ML-intuition/Module_1_Representation/1.02_Dense_Embeddings.md) (the basis of semantic caching)
- Prereqs: [01 RAG Fundamentals](../01-RAG-Fundamentals/01-RAG-Fundamentals.md) · [15 LLM App Orchestration](../15-LLM-App-Orchestration/15-LLM-App-Orchestration.md)
- Related domain: [09. LLMs — KV Cache](../../09.%20LLMs/05-KV-Cache/05-KV-Cache.md) · [09. LLMs — Inference Optimization & Serving (vLLM)](../../09.%20LLMs/09-Inference-Optimization-and-Serving/09-Inference-Optimization-and-Serving.md)
