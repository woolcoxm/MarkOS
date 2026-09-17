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

- **Boot**: modules load via `/etc/modules-load.d/axcl.conf`, udev
  creates `/dev/axcl_host`, firmware loads from
  `/lib/firmware/axcl/ax650_card.pac`. The engine detects the card on
  PCIe (vendor `0x1f4b`, device `0x0650`) and arms the NPU path before
  the first model load. No configuration needed.
- **Disable**: `MARKOS_AXCL=0` in the service environment boots
  CPU-only even with the card present.
- **Diagnostics**: `axcl-smi` on the box; the model inventory
  (`/api/models/{id}/inventory`) reports the serving path as
  `"accel": {"mode": "npu_layer"|"per_op"|"cpu", ...}`.
- **Card hygiene** (from the PoC findings, still true): killing the
  engine mid-inference can wedge the card; the backend installs signal
  guards and auto-reboots the EP on PCIe drop, but avoid repeated
  SIGKILLs. Firmware flashing must not be repeated.

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
| **Buildroot image build** (`os/build.sh --variant sd`, WSL) | **green**: markos-sd.img (1.23 GB) with the axcl variant — AXCL PCIe modules compiled against the 6.18.52 kernel (2 compat fixes: `-Werror=date-time` suppression, `MODULE_IMPORT_NS` string form for ≥6.13), axclhost runtime installed, engine cross-compiled AND linked (aarch64 ELF, `DT_NEEDED libaxcl_rt.so`), depmod'd modules, firmware, udev, modules-load |
| **appliance boot + serving (QEMU aarch64, no card)** | **green**: healthz OK, admin login OK, `/api/state` reports `accel: {present:false, driver_loaded:false, engines_root:/data/axcl/sets}`; `ggml-axcl: axclInit failed` logged cleanly and the model served on the CPU path — Qwen2.5-0.5B Q4_K_M auto-loaded (`resident:true`) and answered `"The capital of France is" → "Paris..."` over the OpenAI API. This IS the any-GGUF fallback ladder proven in the shipped binary |
| on-target NPU behavior (module load on real PCIe, engine-set selection, NPU decode) | requires the Pi + card — next hardware session |
