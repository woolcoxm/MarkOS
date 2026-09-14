# Pi-7b — MarkOS' own AX8850 implementation plan

Goal: MarkOS runs GGUF models on the LLM8850 **bare-metal** — no Linux, no
llama.cpp, no vendor driver — following the architecture proven in
[woolcoxm/Axera-AX8850-GGUF-Support](https://github.com/woolcoxm/Axera-AX8850-GGUF-Support).
Per the design rule: the card exists to make tokens/second go up while the
Pi's cores stay free for orchestration/sampling.

## What we build (three components)

### 1. MarkOS control plane (port of what llama.cpp does today)
- GGUF parse — done (Pi-3c). Extend: tokenizer (Qwen3 BPE, from the GGUF's
  vocab), sampler (greedy + temperature/top-p), detokenize/stream.
- Decode orchestration: per generated token, issue 28 layer-engine calls +
  1 post call; hidden state stays on-card (bf16, ping-ponged buffer);
  K/V rows land in on-card caches via the engines' K_cache_out/V_cache_out
  bindings; attention indices + mask tensors uploaded once per token
  (hoisted: identical across layers — RE repo's biggest host win).
- Runs on the execution pool (Pi-4); CPU cost target ~0 (RE repo proved
  the envelope: ~2 ms host time per token outside the engines).

### 2. Engine artifacts (offline, x86 toolchain — unchanged from RE repo)
- Engine set: pulsar2 `llm_build2 -w s4` (or vendor w8a16) — 28 layers +
  post, per architecture, built ONCE.
- Weight patching: the installer bakes patched engines into the SD image
  (port `gemm/gguf_patch_w8.py` + `layout_v4.bin` sidecar into the
  installer pipeline; the layout is tied to the engine build, not the
  GGUF). MarkOS the appliance never compiles anything — it loads engines.
- Optional later phase: on-device patcher in Rust for GGUF swaps without a
  re-install (the bf16 template path is simple scatter; w8a16 is the
  decoded nibble-plane layout).

### 3. axcl-lite (the genuinely new code)
A bare-metal implementation of the AXCL Runtime subset the RE repo
actually uses (semantic spec: `axcl-docs-en` API reference; canonical call
sequence: `axcl-samples/ax_model_runner/ax_model_runner_axcl.cpp`):

```
device:   init, SetDevice, GetVersion, RebootDevice (wedge rescue)
memory:   Malloc/Free (card CMM), MallocHost (pinned staging),
          Memcpy H2D/D2H/D2D, MemFlush/MemInvalidate
engine:   LoadFromMem, Unload, CreateContext, GetIOInfo,
          GetShapeGroupsCount, Get{Input,Output}SizeByIndex,
          CreateIO, Set{Input,Output}BufferByIndex,
          Execute, ExecuteAsync + stream sync
```

Scope discipline: ~40 calls, no codec/native API, no PPL, single device,
single stream to start. The whole-layer decode loop uses exactly this
surface (proven in the RE repo; nothing else is load-bearing).

## The critical path: the PCIe transport

`libaxcl.so` + `axcl_host.ko` are closed binaries; the wire protocol (BAR
register map, command framing, DMA descriptors/rings, CMM allocator
protocol) is undocumented anywhere. Reimplementing it is static
reverse-engineering work — no hardware required to read the binaries:

1. **Artifact collection**: `axclhost` package from the M5Stack apt pool
   (`repo.llm.m5stack.com`) — kernel modules + `libaxcl*.so` + headers.
   The RE repo's `driver-good/` backup is the matched pair to document.
2. **Kernel-module RE**: `ax_pcie_host_dev.ko` / `axcl_host.ko` — file
   ops + ioctl table, BAR map, DMA descriptor format, interrupt path,
   mailbox/doorbell semantics. objdump + decompiler; card-side log strings
   (axcl runtime prints) as anchors.
3. **Userspace-lib RE**: `libaxcl_rt.so` — how each axclrt* call frames
   into kernel ioctls / ring commands; the CMM allocator's bookkeeping.
4. **Cross-check**: the RE repo's `vendor_trace.c` (LD_PRELOAD tracer)
   captures the vendor runtime's API-level IO on Linux; on-hardware PCIe
   captures (later, on the user's rig) validate the reconstructed
   protocol before MarkOS ever talks to the card.

## Phases

| Phase | Work | Needs hardware? |
|---|---|---|
| Pi-7b-0 | artifact collection, symbol/ioctl inventory, RE tooling | no |
| Pi-7b-1 | transport RE: BAR map, command framing, DMA rings → `docs/axcl-transport.md` | no |
| Pi-7b-2 | MarkOS: BCM2712 PCIe RC bring-up (ECAM base confirm, link train, AX8850 0650 visible via our pcie.rs) | yes (Pi 5 + card) |
| Pi-7b-3 | transport MVP: device init → malloc → upload one .axmodel → LoadFromMem → one Execute with IO bindings; compare vs golden IO from the RE repo's `engine_dump.c` harness | yes |
| Pi-7b-4 | whole-model decode (28+1 engines, hidden ping-pong, KV/mask bindings) — first tokens bare-metal | yes |
| Pi-7b-5 | performance parity: async chain (one sync after layer 27), pinned staging DMA, deferred KV write-back, chunked-prefill ladder | yes |

MarkOS control protocol grows `GEN`/`PROMPT`/`STREAM` opcodes over the
existing authenticated TCP transport (Pi-5b); the UDOT CPU path stays as
the no-card fallback and the correctness reference (same greedy
agreement methodology the RE repo uses against its CPU reference).

## Risks / rules (inherited from the RE repo's scars)

- Matched driver/firmware pairing matters to RE: extract protocol
  constants from the SAME axclhost build we capture traffic against.
- Card wedges on abnormal termination with work in flight: MarkOS must
  implement the fail-fast + `axclrtRebootDevice` discipline from day one,
  never submit after OFFLINE.
- Engine IO conventions are contractual (mask values, 2 KB alignment,
  shape-group quirks like m<64 groups never writing `output`): encode them
  as constants/tests, not folklore.
- s4 64-token ladder binding bug in the RE repo's backend is unsolved
  upstream — start MarkOS chunked-prefill with vendor 128-token sets only.
