# MarkOS performance — measured numbers, ceilings, and how they were won

All numbers below are **on-target measurements** on the reference appliance:
Raspberry Pi 5 16 GB (d0 stepping, 2026 production board) + M5Stack LLM-8850
(Axera AX8850, 24 TOPS) in the M.2 slot, Buildroot image `markos-sd`, engine
0.1.0, CPU governor `performance`, ~39 °C (no throttling). Measured
2026-09-17, after the perf deep dive described here.

## Final numbers

| model | tier | quant | decode t/s | prefill tok/s | first-request TTFT | warm multi-turn TTFT | quality |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B-Instruct | CPU/NEON (non-matching geometry) | Q4_K_M | **23.8** (median 23.4 non-stream) | **94** | **0.3–0.6 s** | **~0.3 s** | 8/8 evals |
| Qwen3-0.6B (thinking) | NPU whole-layer (matching engine set) | Q8_0 | **16.1** | **135** | **0.3 s** | **~0.3 s** | coherent reasoning |

Definitions, so the numbers stay honest:

- **decode t/s**: streaming SSE generation of a fresh single-turn prompt,
  tokens after the first. Non-streaming median (whole-request wall time
  including prefill) is within ~1 t/s of the same figure at 128 tokens.
- **prefill tok/s**: `prompt_tokens / (total_request_time − decode_time)`.
  A 454-token prompt pre-fills in ~5.2 s on the CPU tier, 425 tokens in
  ~3.5 s through the NPU whole-layer path.
- **first-request TTFT**: first streamed content chunk after a fresh engine
  boot with auto-load + warmup enabled (models resident before serving).
- **warm multi-turn TTFT**: third turn of a growing conversation — this is
  the prompt-cache path (only the new suffix is pre-filled).

Before this work the same box measured **CPU 1.0–1.9 t/s decode with a
~13 s time-to-first-token on every request**, and **NPU 7.4–9.7 t/s**.

## Ceilings: what the raw stack can do

`scripts/build-llama-bench.sh` cross-builds a **CPU-only `llama-bench`**
(no axcl backend linked) from the exact pinned fork with the exact Buildroot
toolchain and flags. On the same box, same GGUF (Qwen2.5-0.5B Q4_K_M):

| test | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| tg64 (decode) | 16.2 t/s | **22.6 t/s** | 16.6 t/s |
| pp512 (prefill) | 23.5 t/s | — | **84.5 t/s** (FA on: identical) |

The engine now sits **on top of** these numbers (23.8 t/s streaming decode,
94 tok/s prefill) instead of underneath them. That equivalence is the real
acceptance test for the serving layer: MarkOS adds no measurable overhead
over raw llama.cpp on this hardware.

The Pi 5 decode curve is bandwidth-bound: 2 threads beat 4. This is why the
engine splits threads (below) instead of exposing one knob.

## Root causes of the old slowness (all fixed)

1. **Stale fork pin — the dominant one.** `os/package/markos-llama` pinned
   `c8d226b`, one commit *before* the fork's opt-in device-registration fix
   (`1bddded`). With the card up, the axcl backend registered as a compute
   device unconditionally and — sharing the CPU buffer type, so the
   scheduler cannot tell them apart — received **CPU-tier graphs**. Its
   host fallback path computes single-threaded: on-device thread sampling
   during decode showed one core at 100% and zero worker threads, and a
   later graph-build even routed the *prefill* through per-op matmul
   probing (each probe a PCIe round trip). Every shipped image built this
   way ran ~10× under the hardware's capability.
   **Fix:** pin bumped to `1bddded` (`markos-llama.mk` + `.hash`); the
   registration gate (`GGML_AXCL_LAYER`) is now part of every image.
2. **One thread count for two very different workloads.** Decode on the Pi 5
   wants 2 threads; prefill wants 4 (see ceilings). A single `threads`
   config forced a bad compromise for both.
   **Fix:** new `threads_decode` per-model config, default = half of
   `threads` (≥1); `threads` now means prefill threads. The shim accepts
   separate `n_threads` / `n_threads_batch`. This also lifted the **NPU**
   tier 9.7 → 16.1 t/s: whole-layer serving still runs host-side KV staging
   work, which stopped oversubscribing the cores.
3. **Full re-prefill of every chat turn.** The context was reused but its KV
   was cleared per request, so a 5-turn conversation paid the whole-prompt
   prefill five times.
   **Fix:** prompt-prefix KV reuse on the CPU tier via a new
   `markos_llama_memory_seq_rm` shim: keep the shared prefix's cells, trim
   from the first divergence, pre-fill only the suffix (a repeated 454-token
   prompt: 5.3 s → 0.4 s). The whole-layer NPU path has no partial-removal
   primitive and still clears; correctness is guarded per tier.
4. **Cold first request.** NPU engine loading (28 layer templates + post)
   and the KV/graph allocation landed on whoever sent the first request.
   **Fix:** the inference context is created once per model slot and kept;
   auto-loaded models run a 1-token warmup generation at boot (measured:
   ~130–430 ms per model).
5. **Per-request host waste in the serving layer.** `resolve_model` re-walked
   the full GGUF header — including materializing the 151,936-entry
   tokenizer arrays into a joined string — on **every** request;
   `update_resident_metrics` did it again after each request; the
   stop-string scan copied the whole output every token (O(n²) overall).
   **Fix:** GGUF metadata cached by (mtime, size) with vocab-sized arrays
   skipped (the serving layer never reads them — llama.cpp's tokenizer owns
   that data); stop scanning is a bounded tail window; no per-token copies.
6. **Misleading prefill math in the old bench.** `bench.py` computed
   `prompt_tokens / total_request_time`, so as the engine got faster, the
   reported "prefill" number was dominated by decode time. `scripts/bench.py`
   now subtracts the decode component; `scripts/bench_final.py` is the
   authoritative suite (first-request, decode, prefill, multi-turn TTFT).

## Verification

- `scripts/bench_final.py` — numbers above (run against the live box).
- `scripts/bench.py` — classic suite + 8-prompt quality eval: **8/8 CPU**
  (with the prompt cache active), coherent Qwen3 thinking-mode reasoning on
  the NPU tier.
- `scripts/e2e_test.py` — 31/31 API/UI/auth/TLS/validation/security tests
  green against the final deployed binary.
- Host unit tests: 31/31 green; `--features llama` type-checks clean.
- Raw-stack cross-check: engine vs CPU-only `llama-bench` from the same
  fork (table above) — parity within noise.
- Clean-build path: `markos-llama-dirclean` + `markos-engine-rebuild` from
  the pre-seeded tarball verifies the hash and relinks the engine against
  the fresh `1bddded` tree.

## Reproducing / deploying

```sh
# build + deploy the engine to the Pi without reflashing (WSL)
wsl -- bash -c "tr -d '\r' < scripts/engine-build.sh > /tmp/eb.sh && bash /tmp/eb.sh"
scripts/deploy-engine.sh          # scp to /data/bin + repoint the service

# raw-stack ceiling for comparison
wsl -- bash -c "tr -d '\r' < scripts/build-llama-bench.sh > /tmp/bb.sh && bash /tmp/bb.sh"
# → os/output/llama-bench.pi; push to the box, then:
#   /data/llama-bench -m <gguf> -p 512 -n 64 -t 1,2,4 [-fa 0|1]

# benchmarks against the appliance
python scripts/bench_final.py --host 10.0.0.69
python scripts/bench.py --host 10.0.0.69
python scripts/e2e_test.py --host 10.0.0.69
```

## Remaining headroom (NPU tier, template quality — not engine code)

The engine no longer masks the card. What is left to reach the PoC's
24–30+ t/s is in `Axera-AX8850-GGUF-Support/PERF-PLAN.md`, in order:

1. **s4/GPTQ Pulsar2-tuned templates** for the qwen3-0.6B set
   (~1.17 ms/layer vs the vendor Q8_0 set's ~1.9 ms/layer) — biggest lever;
2. **kv1024 engine set** — ~28 t/s at ctx ≤ 1024;
3. **trimmed post engine** (90.9k → ~55k kept rows) — −3 to −5 ms/token;
4. host-path diet (getenv storm, claim table, KV-flush spread) — ~0.5–1
   ms/token; fold into the same fork push as the template work;
5. speculative verification — the 2–3× endgame at longer contexts.

Also worth knowing: at Q4_K_M the CPU tier (23.8 t/s) currently *beats* the
NPU tier with vendor templates (16.1 t/s) — the geometry-matching router
still prefers the NPU (frees the CPU, keeps latency stable under load), but
if a model exists in both quants and you want raw tokens, the CPU path is
the faster one until the tuned template sets are installed.
