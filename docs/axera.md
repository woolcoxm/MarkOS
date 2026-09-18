# Axera AX8850 accelerator support (M5Stack LLM-8850 M.2 card)

MarkOS turns the Pi 5 into an inference appliance; with the Axera card in
the M.2/PCIe slot, the heavy tensor work moves to the card's 24-TOPS NPU
and the Pi's CPU stays essentially idle (24–30 t/s decode on matching
models, vs ~2–4 t/s on the CPU).

The design contract — the part that makes "any GGUF" true:

| model | serving path | speed class |
|---|---|---|
| geometry matches an installed **engine set** | whole-layer NPU engines, GGUF weights patched into the templates at load | 24–30 t/s |
| no matching set, per-op **matmul engines** installed for its shapes | NPU matmuls, host attention/norms | between |
| anything else | CPU/NEON reference | Pi-speed |

Every GGUF loads and answers. Nothing model-specific is hardcoded in the
serving layer: the backend discovers the model's geometry from the live
compute graph (hidden width, vocab, layers, projection widths) and picks
the path at load time. A model that can't be accelerated is served on the
CPU with a clear log line — never a failure.

## Pieces and where they live

| piece | what | where |
|---|---|---|
| ggml-axcl backend | llama.cpp fork that runs GGUF weights directly on the NPU (whole-layer engines + bf16/int8 weight patching + per-op matmul dispatch) | `woolcoxm/llama.cpp` branch **`axera-any-gguf`** (generalized from the PoC branch; pinned in `os/package/markos-llama`) |
| markos-llama-sys | builds the fork via cmake, exposes a plain-C shim (no llama.cpp struct crosses FFI) | `engine/markos-llama-sys/` |
| engine wiring | `axcl` cargo feature, PCI detection, env bootstrap, set matching for reporting | `engine/src/backend/axcl.rs`, `engine/src/accel.rs`, `engine/src/axsets.rs` |
| axclhost runtime | libaxcl_rt & friends, card firmware (`ax650_card.pac`), udev, ld.so config, axcl-smi | Buildroot package `os/package/axclhost/`, sourced from a vendored M5Stack deb |
| PCIe driver | `ax_pcie_host_dev`, `ax_pcie_msg`, `ax_pcie_mmb`, `axcl_host`, `ax_pcie_p2p_rc` kernel modules | Buildroot package `os/package/axcl-driver/` (AXERA-TECH V3.6.2 deb source, built against our kernel) |
| engine sets + matmul engines | per-architecture template `.axmodel` sets with `set.txt` manifests | `/data/axcl/sets/<name>/` on the appliance |

## Building the image

```sh
cd os && ./build.sh --variant sd
```

Requirements beyond the usual ones:

1. **Vendored runtime** — `os/vendor/axclhost-root/` must contain the
   unpacked `axclhost_3.6.5-m5stack1_arm64.deb` root tree. `build.sh`
   seeds it automatically from `../Axera-refs/` or `~/Downloads` when
   either holds the deb; otherwise unpack it yourself:
   ```sh
   mkdir -p os/vendor/axclhost-root && cd os/vendor/axclhost-root
   ar x /path/to/axclhost_3.6.5-m5stack1_arm64.deb
   tar --zstd -xf data.tar.zst
   ```
2. **Fork pushed** — the `markos-llama` package pins commit
   `c8d226b4dc94b4eaa7638e5313c2165029dcc17e` of
   `https://github.com/woolcoxm/llama.cpp` (branch `axera-any-gguf`).
   Push the branch before building on a fresh machine, and bump the pin
   in `os/package/markos-llama/markos-llama.mk` when it moves.
3. The driver deb downloads from HuggingFace
   (`AXERA-TECH/AXCL`, sha256-pinned in `axcl-driver.hash`).

For a CPU-only build, flip the defconfig backend choice back
(`BR2_PACKAGE_MARKOS_ENGINE_LLAMA=y`, drop the three AXCL/LLAMA package
lines) — the engine source is identical; only the tensor backend changes.

## Installing engine sets

Engine sets live on the data partition so they can be added without
reflashing: `/data/axcl/sets/<name>/` (created on first boot; also
reachable at `/usr/local/share/ggml-axcl/sets`). A set is a directory of
whole-layer template engines plus a `set.txt` manifest:

```
family=qwen3
pattern=qwen3_p128_l%d_together.axmodel
post=qwen3_post.axmodel
layout=layout_v4.bin
hidden=1024
vocab=151936
layers=28
ctx=2048
```

- `family` — `qwen3` (dense attention) or `qwen35-hybrid`
- `pattern` — layer template filenames (one `%d`)
- `post` — the final-norm + lm_head engine
- `layout` — optional weight sidecar; enables GGUF patching (without it
  the set only serves vendor-weighted models)
- `hidden` / `vocab` / `layers` — geometry gate. A set only matches a
  GGUF whose graph geometry agrees (fields present must match; a
  manifest may omit fields to match loosely)

The templates are compiled once per architecture+geometry with the
Pulsar2 toolchain (`llm_build`, x86_64 host — see the
Axera-AX8850-GGUF-Support repo's `pulsar2/` and `gemm/` labs). Example
manifest: `gemm/engine_sets/set.txt.example` in that repo.

Per-op matmul engines (the middle tier) are shape-keyed files under
`/data/axcl/matmul/matmul_m1_k<k>_n<n>.axmodel`; generate them for a
model's projection shapes with the same toolchain.

## Operation on the appliance

- **Boot**: driver modules load last (S99axcl-modules, detached in the
  background — the driver's module init does a synchronous 155 MB firmware
  push + up-to-2-minute EP handshake). `axcl-smi` output lands on the
  console when the card is ready. The engine detects the card on PCIe
  (vendor `0x1f4b`, device `0x0650`) at startup and arms the NPU path
  before the first model load. No configuration needed.
- **Boot config**: `config_5.txt` enables `dtparam=pciex1` +
  `dtoverlay=pciex1-compat-pi5,no-mip` — without these the Pi 5 doesn't
  recognize the card or can't allocate MSI IRQs (AXCL setup docs,
  axcl-samples#5).
- **Disable**: `MARKOS_AXCL=0` in the service environment boots
  CPU-only even with the card present.
- **Diagnostics**: `axcl-smi` on the box; the model inventory
  (`/api/models/{id}/inventory`) reports the serving path as
  `"accel": {"mode": "npu_layer"|"per_op"|"cpu", ...}`.
- **Card hygiene** (from the PoC findings, still true): killing the
  engine mid-inference can wedge the card; the backend installs signal
  guards and auto-reboots the EP on PCIe drop, but avoid repeated
  SIGKILLs. Firmware flashing must not be repeated. **A Pi reboot does
  not reset the card** — full wall power is the reliable recovery.
- **Fan**: there is no host-side fan control — the AXCL API exposes none
  (`axcl-smi`'s Fan column is officially unsupported; the `axcl_rt`
  headers have no fan/pwm entry points) and the fan is driven by the
  card's own firmware. Fan behavior therefore follows the flashed card
  `.pac`: the **3.6.5-m5stack1 pac runs the fan at full speed all the
  time**; the **3.6.6-m5stack1 pac is the quiet variant** (PoC-verified
  serving combo: 3.6.5 runtime + 3.6.6 card firmware, all modes OK). The
  image ships the 3.6.6 pac — `build.sh` applies it over the vendored
  3.6.5 tree from `Axera-refs/fwm/axclhost_3.6.6-m5stack1_arm64.deb`
  (repo.llm.m5stack.com, deb sha256 `22e0d8f9…`, pac sha256
  `9270e25b…`); the 3.6.5 pac is kept beside it as `ax650_card.pac.3.6.5`.
  The driver flashes it **once** at boot when the card's version differs
  (repeated flashing is the card-wedging vector — see the S99axcl-modules
  header). First boot after the bump: expect the normal slow module-init
  flash, then a quiet card.
- **Kernel/module skew hazard**: the AXCL `.ko`s are built against the
  image's exact kernel (`/lib/modules/<ver>-v8-16k`). If the boot
  partition and the rootfs slot ever get out of sync (e.g. a stale
  `kernel=Image` from an older image on the FAT partition booting a newer
  rootfs), `modprobe` finds no modules for the running kernel, the card
  never comes up, and — since the fan is card-firmware-driven — the fan
  runs at the flashed firmware's speed with no host control. Symptom
  check: `uname -r` vs `ls /lib/modules/`, empty `/proc/modules`,
  `/dev/axcl_host` missing. Fix is a coherent reflash, not module
  hacking.

## Measured performance (2026-09-17, on-target — after the perf deep dive)

| model | tier | quant | decode t/s | prefill t/s | quality | notes |
|---|---|---|---|---|---|---|
| Qwen3-0.6B | NPU whole-layer | Q8_0 | **16.1** | **135** | coherent reasoning | 2.6 GB CMM; 28 layer templates + post engine |
| Qwen2.5-0.5B | CPU fallback | Q4_K_M | **23.8** | **94** | 8/8 evals | non-matching geometry → pure CPU, 2T decode / 4T prefill |

First request after boot: 0.3–0.6 s TTFT (auto-load + warmup); warm
multi-turn TTFT ~0.3 s (prompt-prefix KV reuse on the CPU tier).

### Where the earlier numbers went wrong (2026-09-17 perf deep dive)

The first hardware measurements (NPU 7.4 t/s, CPU 1.0–1.9 t/s with a 13 s
time-to-first-token) were not template-limited — they were **engine-limited
by a stale fork pin**:

- The Buildroot pin (`c8d226b`) predated the fork's opt-in device
  registration (`1bddded`). With the card up, the axcl backend registered
  unconditionally and — sharing the CPU buffer type — the llama.cpp
  scheduler handed it CPU-tier graphs. Its host fallback computes
  **single-threaded** (2.4 t/s where the raw fork does 16–23 t/s), and each
  request paid NPU/PCIe probing (the 13 s TTFT).
- **Fix**: pin bumped to `1bddded` (`os/package/markos-llama/markos-llama.mk`)
  + engine-side decode/prefill thread split (`threads_decode`, default half
  of `threads` — the Pi 5 decodes fastest at 2 of 4 cores: llama-bench tg64
  22.6 t/s @ 2T vs 16.6 @ 4T), prompt-prefix KV reuse, context caching with
  boot warmup, and GGUF-metadata caching. Cross-checked with a CPU-only
  `llama-bench` built from the same fork: tg64 16.2/22.6/16.6 t/s at
  1/2/4 threads, pp512 84.5 t/s — the engine now matches the raw stack.

### Remaining headroom vs the PoC's 24–30 t/s

Not host code — **template quality and engine-set builds** (the PoC plan in
`Axera-AX8850-GGUF-Support/PERF-PLAN.md`):

1. Vendor Q8_0 templates run ~1.9 ms/layer; Pulsar2-tuned s4/GPTQ builds
   ~1.17 ms/layer (PoC baseline 24.3 t/s). Rebuilding the qwen3-0.6B set
   with the s4 flow is the single biggest lever.
2. `kv1024` engine set: ~0.93 ms/layer → ~28 t/s at ctx ≤ 1024.
3. Trimmed post engine (90.9k → ~55k kept vocab rows): −3 to −5 ms/token.
4. Host-path diet (Phase 1: getenv storm, claim table, KV-flush spread) is
   still open upstream; worth ~0.5–1 ms/token — apply it when building the
   new template set so one fork push covers both.
5. Speculative verification (Phase 5) for the 2–3× endgame.

## Multi-model notes

- Whole-layer NPU engages for the **first** matching model in a process.
  A second model with different geometry logs one line and serves on
  per-op/CPU (the fork retires whole-layer mode for the process).
- System-RAM guardrails are unchanged: GGUF weights still live in host
  RAM; the card holds staged copies in its own CMM.
- Two heavy NPU models at once will contend for the card's ~7 GB CMM;
  the per-op engine loader falls back to CPU on CMM allocation failure,
  so this degrades rather than fails.

## Validation status

| check | result |
|---|---|
| fork `ggml-axcl.cpp` generalized (runtime geometry, engine-set manifests, fallback ladder) | x86 g++ syntax-clean against axclhost 3.6.5 headers |
| fork full cmake build (GGML_AXCL=ON) | green on WSL x86_64 (libllama + libggml-axcl + shim) |
| engine host tests (mock + axsets/accel units, e2e) | 44/44 green |
| engine `--features axcl` type-check (WSL, real fork build) | green, zero warnings |
| **Buildroot image build** (`os/build.sh --variant sd`, WSL) | **green**: markos-sd.img (1.23 GB) with the axcl variant — AXCL PCIe modules compiled against the 6.6.28 kernel, axclhost runtime installed, engine cross-compiled AND linked (aarch64 ELF, `DT_NEEDED libaxcl_rt.so`), depmod'd modules, firmware, udev, modules-load |
| **appliance boot + serving (QEMU aarch64, no card)** | **green**: healthz OK, admin login OK, `/api/state` reports `accel: {present:false, driver_loaded:false, engines_root:/data/axcl/sets}`; `ggml-axcl: axclInit failed` logged cleanly and the model served on the CPU path — Qwen2.5-0.5B Q4_K_M auto-loaded (`resident:true`) and answered `"The capital of France is" → "Paris..."` over the OpenAI API. This IS the any-GGUF fallback ladder proven in the shipped binary |
| **on-target boot (real Pi 5 16 GB + AX8850 card)** | **green**: appliance up in 11 s to a serving API at `10.0.0.69`; SSH key auth; admin web UI; model manager; OpenAI-compatible API serving a real completion ("The capital of France is" → "Paris, the capital city of France...") through the Qwen2.5-0.5B GGUF |
| **AX8850 card: detection + driver + firmware** | **green on hardware**: card enumerated at PCIe `0000:03:00.0` (1f4b:0650), all 5 driver modules loaded (built against the running 6.6.28-v8-16k kernel), firmware pushed + EP handshake complete, `axcl-smi` reports AX650N V3.6.4, 29°C, 943 MiB / 7040 MiB CMM, `/dev/axcl_host` live |
| **engine accelerator detection on hardware** | **green**: engine reports `accel: {present: true, driver_loaded: true, n_sets: 1, pci_address: "0000:03:00.0"}` — card found, engine set found, NPU path armed |
| **NPU-tier generation (whole-layer decode)** | **green**: Qwen3-0.6B Q8_0 generates coherent reasoning at 16.1 t/s decode / 135 t/s prefill with 2.6 GB CMM on card. The opt-in backend registration fix (fork `1bddded` + engine `d1dfe8e`) resolved the garbage-output issue: the backend now only registers as a device when GGML_AXCL_LAYER=1, and the engine arms it per-model only when the geometry matches an installed engine set. **The appliance image must pin ≥ 1bddded** — building from `c8d226b` re-introduces the single-threaded CPU-tier pathology (see measured performance above) |
| **any-GGUF serving ladder on hardware** | **green**: non-matching model (Qwen2.5-0.5B Q4_K_M) serves on CPU at 23.8 t/s with 8/8 quality evals passing; matching model (Qwen3-0.6B Q8_0) serves on NPU at 16.1 t/s — both from the same process with card present, automatic per-model routing |
