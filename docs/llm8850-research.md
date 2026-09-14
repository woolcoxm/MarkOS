# M5Stack LLM8850 (Axera AX8850) — research status

Status: **architecture decided; bare-metal transport RE is the critical
path.** Primary research asset: the user's reverse-engineering repo
[woolcoxm/Axera-AX8850-GGUF-Support](https://github.com/woolcoxm/Axera-AX8850-GGUF-Support)
("the RE repo" below), plus AXERA-TECH's open docs/samples and M5Stack's
card documentation.

## What the card is

M.2 M-Key card (Pi 5 PiHat kit) around the Axera AX8850: 8x Cortex-A55,
4/8 GB LPDDR4x, 24 TOPS int8 NPU, **PCIe 2.0 x2 operating as an Endpoint**
running its own firmware (identifies on lspci as "Axera Semiconductor
Device 0650"). The host never touches the NPU registers — everything goes
through a firmware request/response transport over PCIe.

## Layer map (their working Linux stack → what MarkOS must own)

```
llama.cpp (tokenizer, sampling, graph)          → MarkOS control plane
  ggml-axcl.cpp  (REPO: engine orchestration,
                  GGUF weight patching, KV mgmt) → MarkOS inference engine
    libaxcl.so     (CLOSED userspace runtime)   → MarkOS "axcl-lite"
      axcl_host.ko + ax_pcie_host_dev.ko        → MarkOS PCIe transport
        (CLOSED kernel drivers)                   (BAR regs, rings, DMA)
          PCIe → AX8850 EP firmware + NPU       → stays (card firmware)
```

## What the RE repo established (do not re-derive)

- **Whole-layer engine architecture**: pulsar2 `llm_build`/`llm_build2`
  compiles one NPU engine per transformer layer (+ a post engine for final
  norm + lm_head). Per token: 28 engine calls + 1 post call; hidden state
  (bf16) never leaves the card; K/V live in on-card caches passed as IO.
  Exact IO contract per engine (K_cache/V_cache bf16 [1,2048,1024],
  indices u32, input [1,1,1024] bf16, mask [1,1,2049], 10 shape groups,
  64/128-token chunk ladders) documented in NOTES-DYNAMIC-WEIGHTS.md.
- **.axmodel / npu_params byte layout fully decoded**: weights stored at
  deterministic offsets; w8a16 int8 = per-row symmetric scales (960
  clusters) in two nibble planes (coarse byte per element pair holds both
  top nibbles; fine byte 18 positions earlier holds both low nibbles),
  anchor columns, norm entries — decoded via Pulsar2 5.2 marker builds
  (`gemm/decode_v52_*.py`), validated on-card (cosine 0.997).
- **GGUF → engine weight patching works**: `gemm/gguf_patch_w8.py`
  dequantizes GGUF (Q8_0/Q4_K/Q6_K), requantizes against the engine's own
  scales, patches only differing elements → 19.5 t/s at 96% token
  agreement with the CPU reference. The GGUF is the only model artifact.
- **Card ops**: `axclrtRebootDevice` = software EP reset (rescues the
  classic wedge); driver/firmware are a matched pair (M5Stack axclhost
  3.6.5-m5stack1 + ax650_card.pac); card state survives host crashes.
- **Performance envelope**: ~23-30 t/s decode, 700-1276 t/s chunked
  prefill, Pi CPU ~0%; decode is weight-stream-bound (~25 GB/s of 34.1
  peak; ~457 us fixed per engine call).

## What AXERA-TECH's open repos give us

- `axcl-docs-en`: the complete **AXCL Runtime API reference** (~90
  functions) — device/context/stream lifecycle, Memory API (Malloc /
  MallocHost / Memcpy / Flush / Invalidate), Engine API (LoadFromMem,
  IOInfo/shape-groups/IO sizes/dims, CreateIO, Set{Input,Output}Buffer,
  Execute / ExecuteAsync + sync, RebootDevice). This is the semantic spec
  a bare-metal implementation must satisfy.
- `axcl-samples`: canonical call sequences. `ax_model_runner_axcl.cpp`
  (514 lines) is the minimal orchestration: malloc → H2D model →
  LoadFromMem → CreateContext → GetIOInfo → CreateIO per group → bind
  buffers → H2D inputs → Execute → D2H outputs.
- `ax-llm`: the vendor's on-card LLM runner (reference for conventions:
  1.0 gate masks, 0/-65536 attention masks, ping-pong state).

What none of the open repos contain: **the PCIe wire protocol** (BAR
register map, command framing, DMA ring layouts between libaxcl.so and
the card firmware). The kernel driver + userspace lib ship only as
binaries (M5Stack apt pool `repo.llm.m5stack.com`, package `axclhost`).

## Decision

MarkOS implements its **own** stack in the layer map above: GGUF-driven
control plane + whole-layer engines + axcl-lite, with the closed PCIe
transport reverse-engineered from the shipped binaries (static RE needs
no hardware). Until Pi-7b lands, the Pi 5 CPU (NEON UDOT, per-core pool)
remains the primary inference path.

Sources: [RE repo](https://github.com/woolcoxm/Axera-AX8850-GGUF-Support) ·
[axcl-docs-en](https://github.com/AXERA-TECH/axcl-docs-en) ·
[axcl-samples](https://github.com/AXERA-TECH/axcl-samples) ·
[ax-llm](https://github.com/AXERA-TECH/ax-llm) ·
[M5Stack card setup](https://docs.m5stack.com/en/guide/ai_accelerator/llm-8850/m5_llm_8850_software_install) ·
[CNX-Software card analysis](https://www.cnx-software.com/2025/10/03/m5stack-llm-8850-card-an-m-2-m-key-ai-accelerator-module-based-on-axera-ax8850-24-tops-soc/)
