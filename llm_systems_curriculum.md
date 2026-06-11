# LLM Systems Engineering Curriculum (Personal Study Notebook)

## Chapter 1 — The Inference Stack
### 1.1 Text → Token Pipeline
- BPE, SentencePiece, WordPiece
- Vocabulary design
- Token length vs compute cost

### 1.2 Embedding Layer
- Token vs positional embeddings
- Memory layout

### 1.3 Forward Pass Pipeline
- Layer-wise computation
- Residual connections
- Logits generation

### 1.4 Autoregressive Generation Loop
- Greedy, top-k, top-p, temperature
- Streaming generation

### 1.5 Logits Processing
- Softmax optimization
- Logit biasing
- Output filtering

---

## Chapter 2 — Transformer Deep Dive
### 2.1 Attention Mechanisms
- MHA, MQA, GQA

### 2.2 Attention Complexity
- O(n²) bottleneck
- Sparse/linear attention (overview)

### 2.3 Positional Encoding
- Sinusoidal, RoPE, ALiBi

### 2.4 Feed Forward Networks
- MLP layers
- GELU, SwiGLU

### 2.5 Architecture Variants
- Decoder-only, encoder-decoder, hybrid

---

## Chapter 3 — Prefill, Decode & KV Cache
### 3.1 Prefill Phase
- Parallel computation

### 3.2 Decode Phase
- Token-by-token generation

### 3.3 KV Cache Fundamentals
- Storage mechanics
- Memory analysis

### 3.4 KV Cache Optimization
- Prefix caching
- Prompt caching
- Chunked prefill

### 3.5 KV Compression
- GQA
- Token eviction
- H2O
- StreamingLLM

---

## Chapter 4 — GPU Architecture & Roofline
### 4.1 GPU Fundamentals
- SMs, warps, tensor cores

### 4.2 Memory Hierarchy
- Global, shared, registers

### 4.3 Roofline Model
- Compute vs memory bound

### 4.4 LLM Workload Mapping
- Prefill vs decode

---

## Chapter 5 — Quantization
### 5.1 Precision Formats
- FP32, FP16, BF16, INT8, INT4

### 5.2 Techniques
- Post-training
- Quantization-aware

### 5.3 Advanced Methods
- GPTQ
- AWQ

### 5.4 Model Formats
- GGUF

### 5.5 Trade-offs
- Accuracy vs latency

---

## Chapter 6 — Speculative Decoding
### 6.1 Core Concept
- Draft vs target model

### 6.2 Algorithms
- n-gram
- EAGLE
- Medusa

### 6.3 Acceptance Sampling
- Validation

---

## Chapter 7 — FlashAttention & Inference Optimization
### 7.1 FlashAttention
- IO-aware tiling
- Online softmax

### 7.2 Kernel Fusion
- Operator fusion

### 7.3 Batching
- Static, continuous, in-flight

### 7.4 Scheduling
- Token-level scheduling

---

## Chapter 8 — MoE & Model Parallelism
### 8.1 MoE Basics
- Sparse activation
- Expert routing

### 8.2 Routing
- Top-k gating
- Load balancing

### 8.3 Capacity
- Capacity factor
- Token dropping

### 8.4 Parallelism
- Tensor, pipeline, expert

---

## Chapter 9 — Edge Deployment
### 9.1 Constraints
- Memory, power, latency

### 9.2 CPU Inference
- ARM optimization

### 9.3 Apple Silicon
- MLX, Metal

### 9.4 Formats
- GGUF

### 9.5 Optimization
- Threading
- Cache locality

---

## Chapter 10 — Voice Pipeline
### 10.1 ASR
- Whisper basics
- Streaming

### 10.2 LLM Integration
- Conversational loop

### 10.3 TTS
- Piper basics

### 10.4 End-to-End
- Real-time latency

---

## Chapter 11 — Multimodal Inference
### 11.1 Vision Encoders
- CNN, ViT

### 11.2 Cross-Modal Attention
- Fusion

### 11.3 Vision-Language Models
- CLIP, LLaVA

### 11.4 Image Generation
- Diffusion

### 11.5 Video Generation
- Temporal attention

---

## Chapter 12 — Production Systems
### 12.1 Deployment
- Blue-green, canary

### 12.2 Cold Start
- Warm-up

### 12.3 Routing
- Cache-aware routing

### 12.4 Guardrails
- Filtering, safety

### 12.5 Modes
- Batch vs online

---

## Chapter 13 — Structured Output & Evaluation
### 13.1 Structured Generation
- JSON schema
- Guided decoding

### 13.2 Logit Control
- Biasing

### 13.3 Evaluation
- Benchmarks

### 13.4 Profiling
- Latency, throughput

### 13.5 Observability
- Logging, metrics

---

## Chapter 14 — Fine-Tuning & Distillation
### 14.1 PEFT
- LoRA, QLoRA

### 14.2 Distillation
- Teacher-student

### 14.3 Pipeline
- Train → quantize → deploy

### 14.4 Trade-offs
- Size vs performance
