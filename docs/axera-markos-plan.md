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

## The transport: known, not reverse-engineered

The M5Stack `axclhost` package is DKMS — it **ships the kernel-driver
source**. The PCIe transport (BAR map, handshake, kfifo rings, mailbox
doorbells, CMM allocator ioctls) is documented from that source in
[docs/axcl-transport.md](docs/axcl-transport.md). The only closed piece
remaining is the *command payload* layer inside `libaxcl_rt.so` — and
frames are visible at the open kernel boundary, so a printk patch on the
rig while running `axcl_run_model` dumps the command set. What was scoped
as binary RE is now: read open source + capture command frames.

## Phases

| Phase | Work | Needs hardware? |
|---|---|---|
| Pi-7b-0 | artifact collection (axclhost 3.6.5-m5stack1 .deb, sha256-verified) — **done** | no |
| Pi-7b-1 | transport doc from driver source (handshake/rings/mailbox/mmb) — **done**, see [axcl-transport.md](axcl-transport.md); command-frame capture pending rig access | no |
| Pi-7b-2 | MarkOS: BCM2712 PCIe RC bring-up (ECAM base confirm, link train, AX8850 `1f4b:0650` visible via our pcie.rs), handshake + rings in bare metal | yes (Pi 5 + card) |
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
