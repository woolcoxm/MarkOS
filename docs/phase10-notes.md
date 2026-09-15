# Phase 10 — throughput & latency instrumentation notes

## What is measured

The GEN decode loop accumulates, per generated token, the elapsed
`uptime_ms` between step start and step end (norm → argmax → stream →
feed-forward). STATS exposes:

- `gen_tokens` — served tokens since boot
- `gen_ms` / `tps_milli` — total decode time and tokens × 1e6 / ms
  (tokens/sec with three decimals)
- `mean_gen_ms`, `gen_min_ms`, `gen_max_ms` — per-token latency stats
  (min = a final step, which skips the feed-forward pass; max = a step
  that ran all 28 layers again)
- `run_ms` — full-request wall time (prefill + decode), for the client
  cross-check

## Measured (QEMU TCG, cortex-a76 model, Qwen3-0.6B q8_0)

| quantity | value |
|---|---|
| decode throughput | **16 milli-tok/s (≈ 0.016 tok/s, ~61 s/token)** |
| per-token latency | mean 60.7 s, min 10.0 s, max 111.4 s |
| full 2-token request | 349.6 s (prefill dominates: ~237 s) |
| client-vs-kernel wall | 349,732 ms vs 349,634 ms — **98 ms apart** |

## Why TCG numbers are not the throughput claim

QEMU TCG interprets every guest instruction — roughly three orders of
magnitude slower than real Cortex-A76 silicon — so these numbers prove
the *accounting* and the serving loop, not the machine's speed. The
brief's throughput criterion ("competitive with the same CPU under
Linux") is a hardware measurement: the same Qwen3-0.6B q8_0 GGUF, the
same Pi 5, llama.cpp vs `kernel_2712.img`, greedy tokens/sec — recorded
as a Pi-8 deliverable. The compute-side work that closes the gap (NEON
q8_0 matvec, pool-parallel layers, weight-read caching) is tracked in
the roadmap perf phase; the OS-overhead advantages (no syscalls, no
scheduler, no copies) are structural and already in place.

## Allocation policy (Phase 11 leak argument)

The GEN path performs **zero heap allocations**: all engine scratch is
static `.bss` (activations, KV cache, weight chunks, metadata window),
and the control path formats into stack/static buffers. The sustained
soak therefore checks: deterministic token outputs across runs, stable
per-run latency (drift bound 3×), and an unbroken connection — there is
no allocator to leak.

## Soak measurement (Phase 11, TCG dev loop)

 runs 2 sequential full GEN requests (identical workload)
with per-run wall/accounting cross-checks and a 3x drift bound on
per-run mean latency. On real Pi 5 hardware the same gate scales: each
run is ~0.2 s of compute, so a 24 h soak fits trivially and the drift
statistic becomes the meaningful signal.
