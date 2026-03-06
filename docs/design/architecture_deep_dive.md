# vLLM Architecture Deep Dive

This document is a deep technical companion to the [Architecture Overview](./arch_overview.md).
It focuses on the architecture at subsystem level and explains the **why / what / how**
for major design decisions, core feature sets, and performance optimizations.

[TOC]

## 1) System Architecture at a Glance

```mermaid
flowchart TD
    C[Clients / SDKs] -->|OpenAI-compatible HTTP| API[API Server Entrypoints]
    C -->|Python API| LLM[LLM / AsyncLLMEngine Entrypoints]

    API --> ENG[V1 Engine Core]
    LLM --> ENG

    ENG --> SCH[Scheduler]
    ENG --> KVC[KV Cache Manager]
    ENG --> EXEC[Executor]

    SCH --> EXEC
    KVC --> EXEC

    EXEC --> WRK[GPU Worker Processes]
    WRK --> MR[Model Runner]
    MR --> MOD[Model + Layers + Kernels]
    MOD --> KRN[CUDA / HIP / Custom Kernels]

    ENG --> DIST[Distributed Coordination\n(TP/PP/DP/EP)]
    DIST --> WRK

    API --> MM[Tokenizer + Multi-modal Input Processing]
    MM --> ENG
```

## 2) Design Principles (Why)

### Throughput-first serving

vLLM is designed around maximizing tokens/sec under mixed workloads. Instead of
strict per-request execution, the scheduler batches work continuously across many
requests to keep accelerator utilization high.

### Memory efficiency as a first-class concern

Large language model serving is typically KV-cache bound. vLLM prioritizes memory
efficiency through paged KV management, prefix reuse, and optimized attention paths.
This directly increases effective batch size and therefore throughput.

### Extensibility for fast-moving model ecosystems

Model support and hardware backends evolve rapidly. vLLM uses layered abstractions
(entrypoints, engine, executor/worker, model runner, kernels) to isolate changes and
enable adding new model architectures and execution backends with minimal disruption.

### Deployment flexibility

The project supports offline inference, API serving, and multiple distributed
topologies. This allows a single stack to scale from local experimentation to
production multi-node deployments.

## 3) Subsystem Deep Dive (What + How)

### 3.1 Entrypoints and APIs

- **What**: User-facing interfaces for offline generation and online serving.
- **How**:
  - Offline: `vllm/entrypoints/llm.py`
  - Online: `vllm/entrypoints/openai/api_server.py`
  - CLI integration: `vllm/entrypoints/cli/main.py`
- **Why**: Separate user interaction concerns from core scheduling and model execution.

### 3.2 Engine Core

- **What**: Request lifecycle orchestration: queueing, scheduling, decoding, and output.
- **How**:
  - Legacy/v0 style orchestration: `vllm/engine/llm_engine.py`,
    `vllm/engine/async_llm_engine.py`
  - V1 orchestration path: `vllm/v1/engine/core.py`
- **Why**: Centralized coordination enables global optimization (batching, cache reuse,
  fairness, latency/throughput tradeoffs).

### 3.3 Scheduler

- **What**: Chooses which requests/tokens are executed each step.
- **How**: Balances prefill and decode work while respecting memory limits and active
  sequence groups.
- **Why**: Scheduling policy is the dominant control knob for throughput, tail latency,
  and fairness under load.

### 3.4 KV Cache Management

- **What**: Manages attention key/value state across active and completed sequences.
- **How**: Uses block/page-based memory management and reuses pages where possible.
- **Why**: Avoids fragmentation and dramatically improves effective cache utilization.

### 3.5 Executor and Workers

- **What**: Executes model forward passes and handles device-specific runtime logic.
- **How**:
  - Multiprocess execution: `vllm/v1/executor/multiproc_executor.py`
  - GPU worker runtime: `vllm/v1/worker/gpu_worker.py`
- **Why**: Process isolation per accelerator simplifies failure containment and enables
  scalable TP/PP/DP layouts.

### 3.6 Model Runner and Model Implementations

- **What**: Bridges generic engine requests to model-specific tensor execution.
- **How**:
  - Model execution glue and helpers in `vllm/model_executor/`
  - Model definitions in `vllm/model_executor/models/`
  - Core layers in `vllm/model_executor/layers/`
- **Why**: Keeps core serving logic model-agnostic while allowing per-model optimization.

### 3.7 Kernel Layer

- **What**: High-performance kernels for attention, quantization, and tensor transforms.
- **How**:
  - Python-facing kernel wrappers in `vllm/kernels/`
  - Native implementations in `csrc/`
- **Why**: Custom kernels and fused paths reduce memory traffic and kernel launch overhead.

### 3.8 Distributed Inference Layer

- **What**: Coordinates tensor/pipeline/data/expert parallel execution.
- **How**: Runtime coordination and communication logic in `vllm/distributed/`.
- **Why**: Enables serving models larger than a single device and scaling throughput.

### 3.9 Multi-modal and Adapter Layers

- **What**: Supports image/video inputs and parameter-efficient adapters (LoRA).
- **How**:
  - Multi-modal processing in `vllm/multimodal/`
  - LoRA support in `vllm/lora/`
- **Why**: Expands serving coverage without requiring separate serving stacks.

## 4) End-to-End Request Lifecycle

1. **Ingress**: Request enters via Python API or OpenAI-compatible server.
2. **Preprocessing**: Tokenization and optional multi-modal preprocessing.
3. **Admission**: Engine enqueues request with metadata and sampling parameters.
4. **Scheduling**: Scheduler selects prefill/decode work based on current resource state.
5. **Memory planning**: KV blocks/pages are allocated or reused.
6. **Execution**: Workers run model forward pass via model runner and optimized kernels.
7. **Postprocess**: Token IDs are decoded; logprobs/metadata are assembled.
8. **Streaming/return**: Partial or final outputs are emitted to caller.
9. **Cleanup/reuse**: KV cache pages are reclaimed or reused for future compatible requests.

## 5) Feature Set (What Exists)

Core serving features include:

- Offline inference and high-throughput API serving
- Continuous batching
- Paged KV-cache management
- Prefix caching
- Chunked prefill
- Speculative decoding
- Streaming output
- Multi-LoRA support
- Quantization modes (e.g., INT4/INT8/FP8 and integration-dependent options)
- Distributed execution (TP/PP/DP/EP)
- Multi-modal model support
- OpenAI-compatible server semantics

## 6) Optimization Playbook (What + Why + How)

### PagedAttention and paged KV cache

- **Why**: Prevents contiguous-allocation bottlenecks and reduces fragmentation.
- **How**: Cache is partitioned into fixed-size blocks/pages that can be mapped/reused
  across sequence growth and request churn.

### Continuous batching

- **Why**: Static batches underutilize hardware in bursty, variable-length workloads.
- **How**: The scheduler admits and retires requests continuously each decoding step.

### Prefix caching

- **Why**: Repeated prompts are common in production workloads.
- **How**: Reuses compatible KV prefix state to skip redundant prefill compute.

### Chunked prefill

- **Why**: Long prompts can monopolize compute and increase tail latency.
- **How**: Splits prefill into chunks so decode traffic can interleave with long-context
  requests.

### CUDA/HIP graph capture and optimized kernels

- **Why**: Launch overhead and framework overhead become significant at small decode steps.
- **How**: Captures stable execution graphs and routes core operations to tuned kernels.

### Quantization

- **Why**: Increases throughput and reduces memory bandwidth/footprint.
- **How**: Uses quantized weights/activations where supported by model and backend.

### Distributed parallelism

- **Why**: Fits larger models and scales aggregate throughput.
- **How**:
  - Tensor parallelism splits intra-layer tensor ops.
  - Pipeline parallelism splits layers across devices.
  - Data parallelism replicates engines for request-level scale-out.
  - Expert parallelism maps MoE experts efficiently across hardware.

## 7) Key Design Tradeoffs

### Throughput vs latency

Aggressive batching improves throughput but can increase per-request latency under light
load. vLLM's scheduling policies balance this dynamically by mixing prefill/decode work.

### Generality vs peak specialization

A broad model/hardware abstraction can hide specialized fast paths. vLLM addresses this
with layered abstractions plus backend-specific kernel integrations where needed.

### Memory reuse vs management complexity

Paged KV and prefix reuse improve efficiency but require robust cache bookkeeping and
compatibility checks. Centralized cache management in the engine keeps this complexity
contained.

## 8) Practical Guidance for Contributors

- For API behavior: start in `vllm/entrypoints/`.
- For scheduling and request lifecycle: inspect `vllm/engine/` and `vllm/v1/engine/`.
- For runtime/device behavior: inspect `vllm/v1/executor/` and `vllm/v1/worker/`.
- For model integration: inspect `vllm/model_executor/models/` and related layers.
- For performance debugging: inspect `vllm/kernels/`, `csrc/`, and scheduler/cache paths.

When adding features, preserve the separation between entrypoint logic, scheduling policy,
execution runtime, and kernel-level optimization. This separation is the core reason vLLM
can evolve quickly without destabilizing the serving stack.
