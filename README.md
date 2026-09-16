# MarkOS — a single-purpose LLM inference appliance

MarkOS turns a **Raspberry Pi 5 (16 GB)** into a headless, maintenance-free
LLM inference box: boot it, find it at `http://pi-inference.local`, and hit
an OpenAI-compatible API from anything on your LAN. No desktop, no unrelated
services, no cloud dependency — every component is custom-built for this one
job.

```
┌────────────────────────────────────────────────────────────────────┐
│  install.toml / GUI  →  markos-installer (Windows/Linux, Rust)     │
│        │  validates, injects provision file into image             │
│        ▼                                                           │
│  markos-sd.img / markos-ssd.img  ←  os/build.sh (Buildroot)        │
│        flashed to SD / USB SSD / NVMe                              │
│        ▼                                                           │
│  Pi 5 boots → runit supervises markos-engine                       │
│        ├── OpenAI-compatible API  :8080  (SSE streaming)           │
│        ├── Web configuration UI   :80    (embedded, airgap-safe)   │
│        ├── rescue UI              169.254.9.1:4444                 │
│        └── guardrails, A/B updates, watchdog, factory reset        │
└────────────────────────────────────────────────────────────────────┘
```

## Language

**Rust, everywhere** — the engine, the installer, the OS userspace. Zero-cost
abstractions compile to the same machine code as C on the NEON hot path
(the tensor kernels are ggml's, codegen'd for the Cortex-A76), with no GC, no
runtime, and single executables: the Windows installer is one `.exe`, the
engine is one aarch64 binary. Full rationale: [docs/design.md](docs/design.md#2-implementation-language-and-why-it-satisfies-blazing-speeds).

## Repository layout

| path | what it is |
|---|---|
| `docs/design.md` | **the design document** — OS base choice, storage strategy per media, A/B updates, recovery, custom-vs-reused boundary |
| `engine/` | markos-engine: custom HTTP/1.1 serving layer, OpenAI-compatible API (SSE), admin control plane, embedded web UI (`engine/web/`), GGUF reader, memory guardrails, model manager (mock backend default; `llama` feature = llama.cpp/ggml) |
| `installer/` | markos-installer: egui GUI + scriptable CLI, config validation, **custom FAT32 writer** for config injection, raw-disk flasher (`\\.\PhysicalDriveN`, `/dev/sdX`) |
| `os/` | Buildroot external tree: defconfig, kernel fragment, board overlay (runit services, firewall, watchdog, first-boot provisioner, recovery, `markos-update`), genimage layouts, `build.sh` |
| `scripts/make_gguf_test.py` | synthetic GGUF generator (ported from the repo's earlier bare-metal kernel work) |

## Quickstart

**1. Build the appliance image** (Linux/WSL2, ~1 hr first run):

```sh
cd os && ./build.sh --variant sd     # → os/output/markos-sd.img + .sha256
```

**2. Configure + flash** (Windows or Linux; GUI = run `markos-installer` with
no args, or use the CLI):

```sh
cargo build --release -p markos-installer
markos-installer --config install.toml.example --image os/output/markos-sd.img \
                 --target sd --out configured.img
# then: markos-installer --config install.toml.example --image configured.img \
#                        --write \\.\PhysicalDrive3 --verify
```

**3. Use it**: boot the Pi (27 W USB-C PD, active cooler recommended), open
`http://pi-inference.local`, add a model (HF repo + quant), hit the API:

```sh
curl http://pi-inference.local:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0-6b","messages":[{"role":"user","content":"hello"}],"profile":"coding"}'
```

## Design invariants

- **CPU/NEON only.** The VideoCore VII GPU is not a compute target; no GPU
  code path exists anywhere.
- **Guardrails, not crashes.** The engine estimates weights + KV + compute
  buffers from GGUF metadata and refuses loads that would OOM — with numbers,
  in the UI and via HTTP 409.
- **Concurrency honesty.** One active generation per resident model, bounded
  queue, `429` on overflow. Up to 2 models resident when memory allows
  (LRU eviction otherwise).
- **Rollback-safe updates.** A/B slots + firmware `tryboot`; commit only
  after the engine proves healthy for 3 minutes. Manual on-demand only.
- **Always a way back in.** Link-local rescue UI (`169.254.9.1:4444`),
  GPIO26 factory-reset button, serial console — see
  [docs/recovery.md](docs/recovery.md).

## Validation status

| component | validated |
|---|---|
| engine API/auth/guardrails/queue/templates | 22 host unit tests, green |
| engine end-to-end (boots real binary, provision → login → OpenAI JSON → SSE) | integration test, green |
| **real tensor backend (`--features llama`)** | **compiles against llama.cpp (WSL, cmake+bindgen); real Qwen2.5-0.5B Q4_K_M served: correct answer, `finish_reason: stop`, 28-chunk SSE stream** |
| **OS image build (`os/build.sh --variant sd`, WSL2 Ubuntu)** | **`markos-sd.img` produced: FAT32 boot (kernel/dtb/firmware), squashfs A/B slots, data partition, aarch64 engine cross-compiled with llama.cpp+TLS, full appliance overlay (init stages, s6 services, firewall, update/recovery tools) — verified by unpacking the image** |
| installer config validation, FAT32 injection round-trip | unit + e2e green; **injection verified against the real Buildroot image (provision files read back via FAT32 walk)** |
| installer → engine provision handoff (cross-component) | manual e2e, green |
| **appliance boot gate (QEMU aarch64)** | **`markos-sd.img` booted headless on `qemu-system-aarch64 -M virt` (virtio kernel fragment): s6-supervised engine healthy in ~35 s, installer-provisioned admin login (`{"ok":true,"role":"Admin"}`, wrong password 401), hardcoded Qwen2.5-0.5B Q4_K_M auto-loaded and served — `"The capital of France is Paris."`, `finish_reason: stop`** — procedure in [docs/build.md](docs/build.md) |
| **USB SSD / NVMe variant (`--variant ssd`)** | **`markos-ssd.img` built (ext4 root + A/B slots + data partition); engine, init stages, s6 services and inittab verified inside the ext4 root by loop-mount** |
| on-target behavior (real Pi 5: thermal, NVMe EEPROM boot order) | requires physical hardware |

## Non-goals (v1)

No GPU/accelerator path · no multi-Pi clustering (noted as future work in the
design doc) · no general server dashboard · no telemetry, no auto-updates,
no runtime dependencies beyond user-initiated model downloads.
