# Technical Architecture Deep Dive

This document provides an in-depth exploration of vLLM's architecture, design
choices, implementation details, optimizations, and the reasoning behind them.

[TOC]

---

## System Architecture Overview

vLLM is a high-throughput, memory-efficient inference engine for Large Language
Models. The system is built around a multi-process architecture that separates
API serving, scheduling, and GPU execution into distinct processes connected via
ZMQ sockets.

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          CLIENT LAYER                                   │
│   HTTP/REST ─── gRPC ─── Python SDK (LLM class) ─── CLI (vllm serve)  │
└──────────┬──────────┬──────────────────┬───────────────────┬────────────┘
           │          │                  │                   │
           ▼          ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      API SERVER PROCESS(ES)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────────┐   │
│  │  FastAPI      │  │  Input       │  │  Output Processor           │   │
│  │  Router       │  │  Processor   │  │  (Detokenization, Streaming)│   │
│  │  (OpenAI API) │  │  (Tokenize,  │  │                             │   │
│  │              │  │   MM Load)   │  │                             │   │
│  └──────┬───────┘  └──────┬───────┘  └──────────────┬──────────────┘   │
└─────────┼─────────────────┼──────────────────────────┼──────────────────┘
          │                 │                          │
          └────────┬────────┘                          │
                   │  ZMQ Socket                       │  ZMQ Socket
                   ▼                                   ▲
┌─────────────────────────────────────────────────────────────────────────┐
│                      ENGINE CORE PROCESS                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────────┐  │
│  │  Scheduler    │  │  KV Cache    │  │  Structured Output Manager  │  │
│  │  (FCFS/      │  │  Manager     │  │  (Grammar, JSON Schema,     │  │
│  │   Priority)  │  │  (Paged      │  │   Regex Constraints)        │  │
│  │              │  │   Attention) │  │                              │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────────────────────────┘  │
│         │                 │                                             │
│         └────────┬────────┘                                             │
│                  ▼                                                       │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                    MODEL EXECUTOR                                │   │
│  │  Dispatches work to GPU workers; manages distributed execution  │   │
│  └──────┬──────────┬──────────┬──────────┬─────────────────────────┘   │
└─────────┼──────────┼──────────┼──────────┼──────────────────────────────┘
          │          │          │          │
          ▼          ▼          ▼          ▼
┌────────────┐┌────────────┐┌────────────┐┌────────────┐
│ GPU Worker ││ GPU Worker ││ GPU Worker ││ GPU Worker │
│ (Rank 0)   ││ (Rank 1)   ││ (Rank 2)   ││ (Rank 3)   │
│            ││            ││            ││            │
│ ┌────────┐ ││ ┌────────┐ ││ ┌────────┐ ││ ┌────────┐ │
│ │ Model  │ ││ │ Model  │ ││ │ Model  │ ││ │ Model  │ │
│ │ Runner │ ││ │ Runner │ ││ │ Runner │ ││ │ Runner │ │
│ └───┬────┘ ││ └───┬────┘ ││ └───┬────┘ ││ └───┬────┘ │
│     │      ││     │      ││     │      ││     │      │
│ ┌───┴────┐ ││ ┌───┴────┐ ││ ┌───┴────┐ ││ ┌───┴────┐ │
│ │ KV     │ ││ │ KV     │ ││ │ KV     │ ││ │ KV     │ │
│ │ Cache  │ ││ │ Cache  │ ││ │ Cache  │ ││ │ Cache  │ │
│ │ (GPU)  │ ││ │ (GPU)  │ ││ │ (GPU)  │ ││ │ (GPU)  │ │
│ └────────┘ ││ └────────┘ ││ └────────┘ ││ └────────┘ │
└────────────┘└────────────┘└────────────┘└────────────┘
      ▲              ▲              ▲              ▲
      └──────────────┴──────NCCL───┴──────────────┘
            Tensor Parallel / Pipeline Parallel
```

### Process Count Formula

For a deployment with `N` GPUs, tensor parallelism `TP`, pipeline parallelism
`PP`, and data parallelism `DP`:

| Process Type | Count | Role |
|---|---|---|
| API Server | `DP` (configurable) | HTTP handling, tokenization, media loading |
| Engine Core | `DP` | Scheduling, KV cache management |
| GPU Worker | `N` = `DP × PP × TP` | Model forward passes |
| DP Coordinator | 1 if `DP > 1` | Load balancing across DP ranks |

---

## Design Choices: The "Why"

### Why PagedAttention?

**Problem**: Traditional LLM serving allocates contiguous GPU memory per
sequence for KV cache. With variable-length sequences, this leads to severe
**internal fragmentation** (up to 60-80% memory waste) and **external
fragmentation** (unusable gaps between allocations).

**Solution**: PagedAttention borrows the concept of **virtual memory paging**
from operating systems. KV cache is divided into fixed-size **blocks** (default
16 tokens each), allocated non-contiguously via **block tables** (indirection
tables). This achieves near-zero memory waste and enables:

- **Memory sharing** across sequences (e.g., parallel sampling, beam search)
- **Dynamic allocation** without pre-reserving maximum sequence length
- **Copy-on-write** for forked sequences

**Impact**: 2-4× throughput improvement over static allocation schemes.

### Why Continuous Batching?

**Problem**: Static batching waits for the longest sequence in a batch to
finish before processing new requests. Short requests are blocked by long ones,
leading to low GPU utilization.

**Solution**: vLLM implements **continuous batching** (also called iteration-level
scheduling), where the scheduler can add new requests or remove finished
requests at every decode step. Combined with **chunked prefill**, long prompts
are processed in chunks interleaved with decode tokens from other requests.

**Impact**: Up to 23× throughput improvement over static batching in
high-traffic scenarios.

### Why a Multi-Process Architecture?

**Problem**: Python's GIL prevents true parallelism in a single process.
Tokenization, scheduling, and GPU execution have different resource needs
(CPU-bound vs GPU-bound) and blocking characteristics.

**Solution**: V1 separates concerns into distinct processes:

- **API Server Process**: Handles I/O-bound HTTP serving and tokenization
- **Engine Core Process**: Runs the scheduling hot loop without GIL contention
- **GPU Worker Processes**: One per GPU, dedicated to model execution

Processes communicate via **ZMQ sockets** (low-latency IPC) with
**msgspec**-based serialization (zero-copy where possible).

### Why Custom CUDA Kernels?

**Problem**: Standard PyTorch operations don't exploit LLM-specific access
patterns (paged KV cache, fused operations, mixed-precision compute).

**Solution**: vLLM ships 50+ custom CUDA/C++ kernels for:

- **Paged Attention**: Block-indexed KV cache access with shared memory
  optimization
- **Fused MoE**: Single-kernel routing + expert computation
- **Quantization**: Marlin, CUTLASS W8A8, FP8, NvFP4 format-specific kernels
- **Activation Fusion**: SiLU/GELU fused with quantization
- **Custom AllReduce**: Shared-memory intra-node collective operations

**Impact**: 2-5× kernel-level speedups over unfused alternatives.

### Why Prefix Caching?

**Problem**: Many production workloads share common prefixes (system prompts,
few-shot examples, RAG context). Without caching, identical prefixes are
recomputed for every request.

**Solution**: vLLM hashes KV cache blocks by their token content. When a new
request arrives, the scheduler checks for matching block hashes and reuses
cached blocks. This is **automatic** (no user configuration required beyond
enabling the feature) and operates at block granularity.

**Impact**: Up to 10× reduction in time-to-first-token for workloads with
shared prefixes.

---

## Core Engine Design: The "What"

### Request Lifecycle

Every request flowing through vLLM follows this path:

```
User Input (text/tokens + params)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ 1. INPUT PROCESSING (API Server Process)            │
│    ├─ Tokenize prompt text → token IDs              │
│    ├─ Process multimodal inputs (images/audio/video) │
│    ├─ Apply chat template (Jinja2)                  │
│    ├─ Resolve LoRA adapter                          │
│    └─ Create EngineCoreRequest (msgspec.Struct)     │
└──────────────────────┬──────────────────────────────┘
                       │ ZMQ
                       ▼
┌─────────────────────────────────────────────────────┐
│ 2. SCHEDULING (Engine Core Process)                 │
│    ├─ Add to request queue (WAITING state)          │
│    ├─ Compute block hashes for prefix caching       │
│    ├─ Schedule loop:                                │
│    │   ├─ Check RUNNING requests (decode tokens)    │
│    │   ├─ Check WAITING requests (new prefills)     │
│    │   ├─ Allocate KV cache blocks                  │
│    │   ├─ Preempt if out of memory (LRU eviction)   │
│    │   └─ Build SchedulerOutput                     │
│    └─ Dispatch to executor                          │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ 3. MODEL EXECUTION (GPU Worker Processes)           │
│    ├─ Prepare attention metadata (block tables)     │
│    ├─ Forward pass through model layers:            │
│    │   ├─ Embedding lookup                          │
│    │   ├─ Transformer blocks (attention + MLP)      │
│    │   ├─ KV cache read/write via PagedAttention    │
│    │   └─ Final LayerNorm + output projection       │
│    ├─ Sample next token(s) from logits              │
│    └─ Return ModelRunnerOutput                      │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│ 4. OUTPUT PROCESSING (Engine Core → API Server)     │
│    ├─ Append new token to request state             │
│    ├─ Check stop conditions (EOS, max_tokens, etc.) │
│    ├─ Update request status                         │
│    ├─ Detokenize incrementally                      │
│    └─ Stream response to client (SSE)               │
└─────────────────────────────────────────────────────┘
```

### Request States

```
                    ┌────────────────┐
                    │    WAITING     │ ◄── New request arrives
                    └───────┬────────┘
                            │ Scheduled for execution
                    ┌───────▼────────┐
              ┌────►│    RUNNING     │◄────┐
              │     └───────┬────────┘     │
              │             │              │
    Resumed   │    ┌────────┼────────┐     │ Re-scheduled
              │    │        │        │     │
              │    ▼        ▼        ▼     │
         ┌────┴────┐  ┌────────┐  ┌─┴──────────┐
         │PREEMPTED│  │FINISHED│  │WAITING_FOR_ │
         │(evicted)│  │(*stop*)│  │FSM/KVs/...  │
         └─────────┘  └────────┘  └─────────────┘
```

Terminal states: `FINISHED_STOPPED`, `FINISHED_LENGTH_CAPPED`,
`FINISHED_ABORTED`, `FINISHED_ERROR`, `FINISHED_REPETITION`

### Scheduler Algorithm

The V1 scheduler uses a **unified scheduling model** where each request tracks
`num_computed_tokens` (what's been processed) and `num_tokens` (total tokens
including new outputs). The scheduler advances computation by allocating token
budget:

```python
def schedule(self) -> SchedulerOutput:
    token_budget = self.max_num_scheduled_tokens  # e.g., 2048

    # Phase 1: Schedule RUNNING requests (continuing generation)
    for request in self.running:
        num_new = request.num_tokens - request.num_computed_tokens
        num_new = min(num_new, token_budget)  # Chunked prefill

        blocks = self.kv_cache_manager.allocate_slots(request, num_new)
        if blocks is None:
            # Out of memory → preempt lowest priority request
            victim = self._select_preempt_victim()
            self._preempt(victim)  # Move to WAITING, free blocks
            # Retry allocation
        token_budget -= num_new

    # Phase 2: Schedule WAITING requests (new arrivals)
    for request in self.waiting:
        if token_budget <= 0:
            break
        # Similar allocation logic with prefix cache lookup
```

**Key scheduling features**:

- **Chunked prefill**: Long prompts processed incrementally (chunks limited by
  `max_num_batched_tokens`), allowing decode tokens to be interleaved
- **Preemption**: When KV cache is full, the scheduler evicts the lowest
  priority request (FCFS or priority-based)
- **Async scheduling**: Overlaps scheduling of batch N+1 with execution of
  batch N to hide latency

### KV Cache Management

The KV cache is the heart of vLLM's memory efficiency. It uses a
**BlockPool** with an LRU free list:

```
┌──────────────────────────────────────────────────────────┐
│                    KV CACHE MANAGER                       │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │                  BLOCK POOL                       │    │
│  │                                                   │    │
│  │  Total GPU Blocks: N (determined by profiling)   │    │
│  │  Block Size: 16 tokens (configurable)            │    │
│  │                                                   │    │
│  │  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐     ┌─────┐   │    │
│  │  │ B0  │ │ B1  │ │ B2  │ │ B3  │ ... │ BN  │   │    │
│  │  │alloc│ │alloc│ │free │ │free │     │cache│   │    │
│  │  └─────┘ └─────┘ └──┬──┘ └──┬──┘     └──┬──┘   │    │
│  │                      │       │            │       │    │
│  │         Free Block Queue (LRU doubly-linked list) │    │
│  │         B2 ◄──► B3 ◄──► ... ◄──► BN              │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              PREFIX CACHE (optional)              │    │
│  │                                                   │    │
│  │  BlockHash → KVCacheBlock mapping                │    │
│  │  hash(tokens[0:16])  → Block 5                   │    │
│  │  hash(tokens[0:32])  → Block 12                  │    │
│  │  hash(tokens[0:48])  → Block 7                   │    │
│  │                                                   │    │
│  │  Hash algorithms: SHA256, xxHash, CBOR           │    │
│  └──────────────────────────────────────────────────┘    │
│                                                          │
│  ┌──────────────────────────────────────────────────┐    │
│  │              BLOCK TABLES (per request)           │    │
│  │                                                   │    │
│  │  Request A: [B0, B5, B12, B7, B1]               │    │
│  │  Request B: [B5, B12, B8, B3]  ← shares prefix  │    │
│  │                                                   │    │
│  │  Each entry maps logical → physical block ID     │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

**Block allocation flow**:

1. Compute block hashes for the request's token sequence
2. Look up prefix cache for matching blocks (O(1) per block)
3. Allocate new blocks from the free queue for remaining tokens
4. If free queue is empty, evict LRU cached blocks
5. If still insufficient, signal preemption to the scheduler

**Multi-group support**: Models with different attention patterns (e.g., full
attention + sliding window) maintain separate block coordinators per KV cache
group.

---

## Model Execution Pipeline: The "How"

### Model Registry and Loading

vLLM supports 200+ model architectures through a dynamic registry system:

```
HuggingFace Config (e.g., "LlamaForCausalLM")
    │
    ▼
Model Registry (dict mapping arch name → module path)
    │
    ▼
Dynamic Import (importlib)
    │
    ▼
Model Initialization
    ├─ Phase 1: Create model on meta device
    ├─ Phase 2: Load weights (safetensors/bin/GGUF)
    │   ├─ Multi-threaded weight iteration (8 threads default)
    │   ├─ Tensor parallel sharding during load
    │   └─ Quantization-aware weight packing
    └─ Phase 3: Post-load processing
        ├─ Online quantization (if configured)
        ├─ KV cache scale computation
        └─ Platform-specific finalization
```

### Layer Abstractions

vLLM provides custom `nn.Module` layers that abstract away parallelism and
quantization:

| Layer | Purpose | Parallelism |
|---|---|---|
| `ColumnParallelLinear` | Split output features across TP ranks | All-gather output |
| `RowParallelLinear` | Split input features across TP ranks | All-reduce output |
| `MergedColumnParallelLinear` | Fused QKV / gate-up projections | All-gather output |
| `QKVParallelLinear` | Specialized Q/K/V with head splitting | All-gather output |
| `ReplicatedLinear` | Full replication across all TP ranks | No communication |
| `VocabParallelEmbedding` | Vocabulary split across TP ranks | All-reduce output |

Each layer delegates to a **`LinearMethodBase`** that handles quantization:

```python
class LinearMethodBase:
    def create_weights(self, layer, ...):
        """Allocate weight storage (packed format for quantized)."""

    def apply(self, layer, x, ...):
        """Forward pass with optional dequantization."""

    def process_weights_after_loading(self, layer, ...):
        """Post-load processing (online quantization, repacking)."""
```

### Quantization Support

vLLM supports 25+ quantization methods with format-specific kernels:

| Method | Bits | Kernel Backend | Key Feature |
|---|---|---|---|
| AWQ | 4 | Marlin | Group quantization, fast dequant |
| GPTQ | 2/3/4/8 | Marlin, CUTLASS | Layerwise calibration |
| FP8 (E4M3) | 8 | CUTLASS, cuBLAS | Per-tensor/per-token/per-block scales |
| INT8 (W8A8) | 8 | CUTLASS | Symmetric quantization |
| BitsAndBytes | 4/8 | BNB library | Online quantization (no calibration) |
| GGUF | Mixed | Custom | Portable, embedded metadata |
| NvFP4/MXFP4 | 4 | Custom CUDA | NVIDIA Blackwell-optimized |
| Compressed-Tensors | Mixed | CUTLASS | Sparsity + quantization combined |

### Attention Mechanisms

**Backend selection flow**:

```
Model Config + Hardware Detection
    │
    ▼
Platform.get_attn_backend_cls()
    │
    ├─ Check compute capability (SM 8.0+ for FlashAttention)
    ├─ Check head size compatibility
    ├─ Check dtype support (FP16, BF16, FP8)
    ├─ Check feature requirements (MLA, sliding window, etc.)
    │
    ▼
Selected Backend (FlashAttention / FlashInfer / Triton / ROCm / CPU)
    │
    ▼
Attention Layer
    ├─ Prefill: Variable-length flash attention (O(N) memory)
    └─ Decode: Paged attention with block tables (O(1) per token)
```

**Available backends and their strengths**:

| Backend | Hardware | Strengths |
|---|---|---|
| FlashAttention V3 | NVIDIA SM 8.0+ | Optimized tiling, varlen support |
| FlashInfer | NVIDIA SM 8.0+ | TRT-LLM integration, paged wrappers |
| Triton | Any GPU | Portable, custom kernels |
| ROCm/AITER | AMD GPUs | Native AMD optimization |
| MLA Backends | NVIDIA SM 10.0+ | Multi-head Latent Attention (DeepSeek) |
| CPU Attention | x86/ARM | Fallback for CPU-only inference |

**PagedAttention kernel architecture** (two versions):

- **V1**: Single-pass kernel. Grid: `(num_heads, num_seqs, 1)`. Computes
  attention logits in shared memory, applies softmax, and accumulates output.
  Best for shorter sequences.

- **V2**: Two-pass partitioned kernel. Grid: `(num_heads, num_seqs,
  max_partitions)`. First pass computes partial attention per partition;
  second pass reduces with numerical stability (log-sum-exp correction).
  Best for longer sequences.

### Torch Compilation and CUDA Graphs

vLLM provides multiple compilation levels:

| Level | Name | Description |
|---|---|---|
| 0 | `NONE` | Full eager execution (debugging) |
| 1 | `STOCK_TORCH_COMPILE` | Standard `torch.compile()` |
| 2 | `DYNAMO_TRACE_ONCE` | Single Dynamo trace, no recompilation |
| 3 | `VLLM_COMPILE` | Custom Inductor backend with graph caching and piecewise compilation |

**CUDA graph modes**:

| Mode | Description |
|---|---|
| `NONE` | No CUDA graphs |
| `PIECEWISE` | Individual kernel fusion |
| `FULL` | End-to-end graph capture |
| `FULL_DECODE_ONLY` | Separate graphs for prefill (eager) and decode (graphed) |

CUDA graphs eliminate kernel launch overhead by replaying recorded GPU command
sequences. vLLM captures graphs for common batch sizes and replays them during
decode, reducing per-step overhead from ~1ms to ~0.1ms.

---

## Distributed Execution

### Parallelism Strategies

vLLM supports four complementary parallelism strategies:

```
┌─────────────────────────────────────────────────────────────────┐
│              8-GPU Deployment Example (TP=2, PP=2, DP=2)       │
│                                                                 │
│  DP Rank 0                        DP Rank 1                    │
│  ┌─────────────────────────┐     ┌─────────────────────────┐   │
│  │ PP Stage 0              │     │ PP Stage 0              │   │
│  │ ┌─────────┬───────────┐ │     │ ┌─────────┬───────────┐ │   │
│  │ │ GPU 0   │  GPU 1    │ │     │ │ GPU 4   │  GPU 5    │ │   │
│  │ │ (TP=0)  │  (TP=1)   │ │     │ │ (TP=0)  │  (TP=1)   │ │   │
│  │ │ Layers  │  Layers   │ │     │ │ Layers  │  Layers   │ │   │
│  │ │ 0-15    │  0-15     │ │     │ │ 0-15    │  0-15     │ │   │
│  │ └────┬────┴─────┬─────┘ │     │ └────┬────┴─────┬─────┘ │   │
│  │      │ AllGather│       │     │      │ AllGather│       │   │
│  │      └────┬─────┘       │     │      └────┬─────┘       │   │
│  │           │ PP Send     │     │           │ PP Send     │   │
│  │           ▼             │     │           ▼             │   │
│  │ PP Stage 1              │     │ PP Stage 1              │   │
│  │ ┌─────────┬───────────┐ │     │ ┌─────────┬───────────┐ │   │
│  │ │ GPU 2   │  GPU 3    │ │     │ │ GPU 6   │  GPU 7    │ │   │
│  │ │ (TP=0)  │  (TP=1)   │ │     │ │ (TP=0)  │  (TP=1)   │ │   │
│  │ │ Layers  │  Layers   │ │     │ │ Layers  │  Layers   │ │   │
│  │ │ 16-31   │  16-31    │ │     │ │ 16-31   │  16-31    │ │   │
│  │ └─────────┴───────────┘ │     │ └─────────┴───────────┘ │   │
│  └─────────────────────────┘     └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

#### Tensor Parallelism (TP)

**What**: Splits individual weight matrices across GPUs horizontally. Each GPU
holds a shard of every layer.

**Why**: Enables serving models that don't fit in a single GPU's memory while
maintaining low latency (all GPUs work in parallel on the same token).

**How**: Uses NCCL all-gather/reduce-scatter after column/row parallel linear
layers. Implemented via `GroupCoordinator` with dual communication channels
(NCCL for GPU tensors, Gloo for CPU metadata).

#### Pipeline Parallelism (PP)

**What**: Splits model layers across GPUs vertically. Each GPU holds a
contiguous subset of layers.

**Why**: Reduces inter-GPU communication bandwidth (only activations between
stages, not all-reduce). Enables scaling beyond what TP alone can achieve.

**How**: Point-to-point send/receive between pipeline stages. The last PP rank
broadcasts sampled tokens to all other ranks. vLLM uses a **batch queue** to
overlap scheduling of batch N+1 with execution of batch N, reducing pipeline
bubbles.

#### Expert Parallelism (EP)

**What**: Distributes MoE (Mixture of Experts) model experts across GPUs.

**Why**: MoE models have many experts (e.g., 64-256) but only activate a few
per token (e.g., 8). Distributing experts maximizes memory capacity without
wasting compute.

**How**: All2All communication routes tokens to the GPU hosting their selected
experts. Two implementations:

- **NaiveAll2AllManager**: All-reduce based dispatch/combine
- **AgRsAll2AllManager**: All-gather dispatch + reduce-scatter combine
  (optimized for large expert counts)

Includes **Expert Parallelism Load Balancing (EPLB)** for dynamic rebalancing
of expert replicas across devices.

#### Data Parallelism (DP)

**What**: Runs independent model replicas processing different requests.

**Why**: Scales throughput linearly with the number of replicas. Each replica
handles a separate request stream.

**How**: A DP Coordinator process distributes incoming requests across DP ranks
using load-balanced routing. Each rank has its own Engine Core and GPU workers.

### Communication Infrastructure

```
┌──────────────────────────────────────────────────────┐
│              GroupCoordinator                          │
│  ┌────────────────┐  ┌────────────────────────────┐  │
│  │  Device Group   │  │  CPU Group                 │  │
│  │  (NCCL)        │  │  (Gloo)                    │  │
│  │                 │  │                             │  │
│  │  all_reduce()  │  │  broadcast() for metadata  │  │
│  │  all_gather()  │  │  barrier()                 │  │
│  │  reduce_scatter│  │  gather/scatter            │  │
│  └────────────────┘  └────────────────────────────┘  │
│                                                       │
│  ┌────────────────────────────────────────────────┐  │
│  │  Custom AllReduce (intra-node optimization)    │  │
│  │  ├─ GPU peer-to-peer shared memory             │  │
│  │  ├─ World size ∈ {2, 4, 6, 8}                 │  │
│  │  └─ Max 8MB tensor size (configurable)        │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

---

## Speculative Decoding

Speculative decoding accelerates autoregressive generation by **drafting
multiple candidate tokens** with a fast method and **verifying them in parallel**
with the full model.

### Why Speculative Decoding?

Autoregressive LLM generation is **memory-bandwidth bound** during decode:
each step reads the full model weights from GPU memory but only processes one
token. The compute units are underutilized. Speculative decoding converts this
into a **compute-bound** verification step, improving hardware utilization.

### Supported Strategies

| Strategy | Draft Method | Speed | Quality |
|---|---|---|---|
| **Draft Model** | Separate smaller model | Fast | High (learned) |
| **Eagle** | Neural network on hidden states | Very fast | High |
| **Medusa** | Auxiliary heads on target model | Fast | Medium-High |
| **N-gram** | Pattern matching in prompt tokens | Instant | Variable |
| **Suffix Decoding** | Suffix tree lookup | Instant | Variable |
| **MLP Speculator** | Single MLP layer | Very fast | Medium |

### How It Works

```
Step 1: DRAFT (fast)              Step 2: VERIFY (parallel)
┌────────────────────┐           ┌────────────────────────────┐
│ Draft model/method │           │ Target model forward pass   │
│ generates K tokens │           │ on ALL K+1 positions        │
│                    │           │ simultaneously               │
│ Input: "The cat"   │           │                              │
│ Draft: "sat on     │─────────►│ Verify: "sat" ✓ "on" ✓     │
│         the mat"   │           │         "the" ✓ "mat" ✗    │
│ (K=4 candidates)  │           │         Accept 3, reject 1  │
└────────────────────┘           └────────────────────────────┘

Result: Generated 3 tokens in 1 target model step instead of 3 steps
```

**Key implementation details**:

- **Triton kernels** for efficient batch padding/unpadding of draft tokens
- **Rejection sampling** ensures output distribution matches the target model
  exactly
- **Variable draft length** per request (suffix decoding adapts based on
  prefix matches)
- **Bonus token**: On rejection, the correct token from the target model is
  always accepted as a "bonus"

---

## API and Serving Layer

### OpenAI-Compatible API

vLLM implements a fully compatible OpenAI API surface:

| Endpoint | Description |
|---|---|
| `POST /v1/chat/completions` | Chat completion (GPT-style) |
| `POST /v1/completions` | Text completion (legacy) |
| `POST /v1/embeddings` | Text embeddings |
| `GET /v1/models` | List available models |
| `POST /v1/audio/transcriptions` | Speech-to-text |

**Extensions beyond OpenAI**:

- **Structured outputs**: JSON Schema, regex, EBNF grammar, choice constraints
- **LoRA switching**: Per-request adapter selection via `lora_request`
- **Prefix caching salt**: Per-user cache isolation via `cache_salt`
- **Priority scheduling**: Request priority via `priority` field
- **Reasoning outputs**: Extended thinking with separate token tracking
- **Tool calling**: Streaming function calls with parallel execution
- **Multimodal inputs**: Images, audio, video in chat messages

### Streaming Architecture

```
Client ◄─── SSE Stream ───┐
                           │
                    ┌──────┴──────┐
                    │  Async      │
                    │  Generator  │
                    │             │
                    │  For each   │
                    │  new token: │
                    │  ┌────────┐ │
                    │  │ Delta  │ │
                    │  │ text + │ │
                    │  │ logprobs│ │
                    │  │ + tool │ │
                    │  │ calls  │ │
                    │  └────────┘ │
                    │             │
                    │  Final:     │
                    │  finish_    │
                    │  reason +   │
                    │  usage      │
                    └─────────────┘
```

Streaming uses Server-Sent Events (SSE) with `data: {json}\n\n` framing.
Each chunk contains delta content, optional logprobs, tool call fragments,
and usage statistics.

### Multimodal Processing

```
Multimodal Input (image URL, audio bytes, video frames)
    │
    ▼
MediaConnector (download/convert/validate)
    │
    ▼
BaseMultiModalProcessor._apply_hf_processor()
    │  (HuggingFace processor with LRU caching)
    ▼
MultiModalFeatureSpec (tensor specs for model)
    │
    ▼
Model-specific processing
    ├─ Vision encoder (e.g., CLIP, SigLIP)
    ├─ Audio encoder (e.g., Whisper)
    └─ Merged with text token embeddings
```

### LoRA Adapter Support

**What**: Low-Rank Adaptation enables serving multiple fine-tuned model
variants from a single base model.

**How**:

1. Base model weights remain frozen in GPU memory
2. Per-request LoRA adapters are loaded on demand: `output = base(x) + (A @ B) * scaling`
3. LRU cache manages loaded adapters (automatic eviction when capacity exceeded)
4. Adapters can be resolved from filesystem or HuggingFace Hub via plugin system

**Key benefit**: Serve 10-100+ fine-tuned variants with minimal additional
memory (LoRA adapters are typically <1% of base model size).

---

## Complete Feature Set

### Core Inference

| Feature | Description |
|---|---|
| Continuous Batching | Add/remove requests at every decode step |
| PagedAttention | Non-contiguous KV cache with block tables |
| Chunked Prefill | Process long prompts in interleaved chunks |
| Prefix Caching | Automatic block-level KV cache reuse |
| Speculative Decoding | 6 draft strategies for latency reduction |
| CUDA Graphs | Pre-recorded GPU command sequences for decode |
| Torch Compilation | 4 compilation levels with custom Inductor backend |

### Model Support

| Feature | Description |
|---|---|
| 200+ Architectures | Llama, Qwen, Mixtral, Gemma, Phi, DeepSeek, etc. |
| Encoder-Decoder | T5, BART, Whisper, Florence |
| Mixture of Experts | Mixtral, DBRX, DeepSeek-V2/V3, Qwen-MoE |
| Multi-head Latent Attention | DeepSeek models with MLA optimization |
| Embedding Models | BERT, RoBERTa, E5, BGE for retrieval |
| Cross-Encoder/Reranking | Score query-document pairs |
| State Space Models | Mamba, Jamba (SSM + Attention hybrids) |

### Quantization

| Feature | Description |
|---|---|
| FP8 (E4M3/E5M2) | Per-tensor, per-token, per-block scaling |
| INT8 (W8A8) | CUTLASS symmetric quantization |
| AWQ (4-bit) | Activation-aware weight quantization |
| GPTQ (2/3/4/8-bit) | Post-training quantization with Marlin kernels |
| BitsAndBytes | Online 4/8-bit quantization (no calibration) |
| GGUF | Portable format with embedded metadata |
| NvFP4/MXFP4 | NVIDIA Blackwell 4-bit formats |
| Compressed-Tensors | Combined sparsity + quantization |
| TorchAO | Native PyTorch quantization |

### Distributed Execution

| Feature | Description |
|---|---|
| Tensor Parallelism | Split weight matrices across GPUs |
| Pipeline Parallelism | Split model layers across GPUs |
| Expert Parallelism | Distribute MoE experts across GPUs |
| Data Parallelism | Independent replicas for throughput |
| Context Parallelism | Split long sequences across GPUs (decode + prefill) |
| Custom AllReduce | Shared-memory intra-node collectives |
| Elastic EP | Dynamic scaling of expert replicas |
| Multi-node | Ray-based distributed execution |

### Multimodal

| Feature | Description |
|---|---|
| Vision | Images via CLIP, SigLIP, InternVL, etc. |
| Audio | Speech-to-text via Whisper, Qwen-Audio |
| Video | Frame sequences via LLaVA-Video, Qwen-VL |
| Embeddings | Pre-computed vision/audio bypass |
| Multi-image | Multiple images per conversation turn |

### Serving

| Feature | Description |
|---|---|
| OpenAI API | Full chat/completion/embedding compatibility |
| gRPC Server | High-performance RPC serving |
| Tool Calling | Streaming function calls with parallel execution |
| Structured Outputs | JSON Schema, regex, grammar constraints |
| Reasoning | Extended thinking with separate token tracking |
| LoRA | Per-request adapter switching with LRU cache |
| Priority Scheduling | Request priority with preemption |
| Prefix Cache Salt | Per-user cache isolation |
| Prometheus Metrics | Comprehensive observability |
| OpenTelemetry | Distributed tracing |

### Platform Support

| Platform | Backend | Features |
|---|---|---|
| NVIDIA CUDA | NCCL, cuBLAS, CUTLASS | Full feature support |
| AMD ROCm | RCCL, hipBLAS | Core features + AITER kernels |
| Intel XPU | oneCCL, IPEX | Core inference |
| Google TPU | libtpu, Pathways | Core inference |
| CPU (x86/ARM) | Gloo, OpenMP | Fallback inference |

---

## Optimizations Deep Dive

### Memory Optimizations

**1. PagedAttention Block Allocation**

- **O(1) allocation/deallocation** via doubly-linked free list with sentinel
  nodes
- **Reference counting** for copy-on-write block sharing
- **LRU eviction** of cached blocks when memory is exhausted
- Near-zero memory waste vs. 60-80% with contiguous allocation

**2. GPU Memory Profiling**

On startup, vLLM profiles GPU memory to determine how many KV cache blocks
can fit. The formula:

```
available_memory = total_gpu_memory × gpu_memory_utilization - model_weights - activation_memory
num_blocks = available_memory / (block_size × num_layers × 2 × num_kv_heads × head_size × dtype_size)
```

The `gpu_memory_utilization` parameter (default 0.9) controls the trade-off
between KV cache capacity and memory safety margin.

**3. FP8 KV Cache**

Storing KV cache in FP8 instead of FP16/BF16 halves memory usage, doubling
the number of sequences that can be served concurrently. Per-head/per-token
quantization scales maintain accuracy.

### Compute Optimizations

**1. Fused Kernels**

| Fusion | Operations Combined | Benefit |
|---|---|---|
| SiLU+Multiply | Activation + gate multiply | Eliminates intermediate tensor |
| RMSNorm+Quant | Normalization + FP8 quantization | Single memory pass |
| RoPE | Position encoding in attention | Fused rotation |
| Fused MoE | Router + expert compute | Single kernel launch |

**2. CUDA Graphs**

CUDA graphs capture and replay GPU command sequences, eliminating per-kernel
launch overhead (~10μs per kernel). For a 32-layer model with ~200 kernels per
step, this saves ~2ms per decode step (20% of total latency at small batch
sizes).

**3. Custom AllReduce**

For intra-node tensor parallelism, vLLM implements a custom all-reduce using
GPU peer-to-peer shared memory, bypassing NCCL overhead for small tensors
(<8MB). This reduces all-reduce latency by 2-5× for typical hidden state sizes.

**4. Speculative Decoding Acceptance Rates**

The effectiveness depends on draft-target agreement. vLLM supports multiple
strategies to maximize acceptance:

- **Eagle/Medusa**: Trained on target model's representations (80-90% acceptance)
- **N-gram**: Best for repetitive/templated outputs (variable)
- **Draft model**: Language-model quality drafts (70-85% acceptance)

### Scheduling Optimizations

**1. Async Scheduling**

By scheduling batch N+1 while executing batch N, vLLM hides the scheduling
overhead (Python execution time) behind GPU computation:

```
Time ──────────────────────────────────────────►

Without async scheduling:
  [Schedule₁][Execute₁][Schedule₂][Execute₂]...
                        ▲ GPU idle

With async scheduling:
  [Schedule₁][Execute₁────────][Execute₂────────]...
              [Schedule₂]      [Schedule₃]
              ▲ Overlapped     ▲ Overlapped
```

**2. Batch Queue for Pipeline Parallelism**

When PP > 1, the batch queue allows multiple batches to be in-flight
simultaneously, reducing pipeline bubble overhead from `O(PP)` idle stages to
overlapped execution.

**3. Chunked Prefill Interleaving**

Long prompts are split into chunks that are interleaved with decode tokens.
This prevents a single long prompt from monopolizing the GPU and starving
decode requests:

```
Without chunked prefill:
  [──── Prefill (2048 tokens) ────][Decode][Decode]...
                                    ▲ Decode requests wait

With chunked prefill:
  [Prefill chunk₁ + Decode][Prefill chunk₂ + Decode]...
   ▲ Decode requests run continuously
```

### I/O Optimizations

**1. Multi-threaded Weight Loading**

Model weights are loaded from safetensors files using 8 parallel threads,
overlapping file I/O with tensor allocation and sharding.

**2. Memory-Mapped Loading**

The `fastsafetensors` iterator memory-maps weight files, enabling the OS
to manage efficient page-level I/O without explicit read calls.

**3. Incremental Detokenization**

Rather than detokenizing the full output at every step, vLLM detokenizes
incrementally (only the new token), reducing detokenization overhead from
O(n²) to O(n) total.

---

## Custom CUDA Kernel Inventory

The `csrc/` directory contains vLLM's performance-critical C++/CUDA kernels:

| Directory | Kernels | Purpose |
|---|---|---|
| `attention/` | PagedAttention V1/V2, MLA (CUTLASS SM100) | KV cache-efficient attention |
| `quantization/` | Marlin, CUTLASS W8A8/FP8, NvFP4, MXFP4, Machete | Weight dequantization GEMMs |
| `moe/` | Router GEMM, TopK/Softmax, Permute, Align | MoE token routing and compute |
| `activation_kernels.cu` | SiLU, GELU with FP8 fusion | Fused activation functions |
| `layernorm_kernels.cu` | RMSNorm, LayerNorm with quant | Fused normalization |
| `pos_encoding_kernels.cu` | RoPE, batched RoPE | Position encoding |
| `cache_kernels.cu` | Swap, copy, reshape blocks | KV cache management |
| `sampler.cu` | TopK, TopP, multinomial | Token sampling |
| `custom_all_reduce.cu` | Shared-memory all-reduce | Intra-node collectives |
| `sparse/` | Sparse CUTLASS GEMMs | Structured sparsity |
| `mamba/` | Selective scan forward | State space model kernels |

---

## Observability and Debugging

### Metrics (Prometheus)

vLLM exports comprehensive metrics for production monitoring:

- **Request metrics**: Latency, throughput, queue depth, time-to-first-token
- **KV cache metrics**: Utilization, hit rate, eviction count
- **GPU metrics**: Memory usage, compute utilization
- **Scheduling metrics**: Batch size, prefill/decode ratio, preemption count
- **Model metrics**: Token generation rate, per-layer timing

### Tracing (OpenTelemetry)

vLLM integrates with OpenTelemetry for distributed tracing:

- Span creation for key operations (scheduling, forward pass, sampling)
- W3C TraceContext propagation for end-to-end request tracing
- OTLP export (gRPC/HTTP) to backends like Jaeger, Zipkin, Datadog

### Profiling

- **Layerwise profiler**: Hierarchical event tree showing per-layer CUDA kernel
  timing
- **Torch profiler integration**: Standard PyTorch profiling with trace export
- **Configurable iteration thresholds**: Profile only after warm-up

---

## Benchmarking Suite

vLLM includes a comprehensive benchmarking suite in the `benchmarks/` directory:

| Benchmark | Measures | Use Case |
|---|---|---|
| `benchmark_serving.py` | End-to-end throughput and latency | Production capacity planning |
| `benchmark_throughput.py` | Raw tokens per second | Maximum throughput measurement |
| `benchmark_latency.py` | First-token and inter-token latency | Latency-sensitive tuning |
| `benchmark_prefix_caching.py` | Cache hit rates and savings | Prefix caching effectiveness |
| Kernel benchmarks | Per-kernel performance | Kernel optimization |
| Multi-turn benchmarks | Conversation serving | Chat workload simulation |
| Disaggregated benchmarks | Prefill/decode separation | Disaggregated serving tuning |

---

## Summary: Design Philosophy

vLLM's architecture embodies several key design principles:

1. **Memory efficiency first**: PagedAttention, prefix caching, and FP8 KV
   cache maximize the number of concurrent sequences per GPU.

2. **Throughput through batching**: Continuous batching, chunked prefill, and
   async scheduling keep GPUs maximally utilized.

3. **Latency through speculation**: Multiple speculative decoding strategies
   reduce time-per-token for latency-sensitive workloads.

4. **Modularity through abstraction**: Pluggable backends (attention, quantization,
   platform), plugin systems (LoRA resolvers, IO processors), and layer
   abstractions enable extensibility without core changes.

5. **Production readiness**: OpenAI API compatibility, Prometheus metrics,
   OpenTelemetry tracing, and comprehensive configuration make vLLM
   deployment-ready.

6. **Hardware flexibility**: Platform abstraction layer with custom kernels
   per hardware target (NVIDIA, AMD, Intel, TPU, CPU).

These design choices work together to deliver what vLLM is known for:
**the highest throughput LLM inference engine** with production-grade
reliability and broad model/hardware support.
