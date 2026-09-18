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
| `scripts/` | benchmark + e2e suites (`bench_final.py`, `bench.py`, `e2e_test.py`), engine build/deploy (`engine-build.sh`, `deploy-engine.sh`), raw-stack ceiling (`build-llama-bench.sh`), synthetic GGUF generator (`make_gguf_test.py`) |
| `docs/axera.md` | **Axera AX8850 accelerator support** — the M5Stack LLM-8850 M.2 card: NPU serving ladder, engine sets, driver/runtime packages |

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
`https://pi-inference.local` (self-signed cert — accept the warning), add a model (HF repo + quant), hit the API:

```sh
curl http://pi-inference.local:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"qwen3-0-6b","messages":[{"role":"user","content":"hello"}],"profile":"coding"}'
```

## Installing MarkOS — step by step

There are three ways to get MarkOS onto a Pi 5: the **CLI installer**, the
**GUI installer**, or a **hand-flashed prebuilt image**. All of them consume
the same artifacts (`os/output/markos-sd.img` for SD boot, `markos-ssd.img`
for USB SSD/NVMe) and the same configuration surface (`install.toml`, or the
equivalent GUI fields). The installer *refuses* to proceed on: unset
`boot_media`, a config with no way in (no admin password, no SSH key, serial
console off), an admin password shorter than 8 chars, an invalid static IP,
or a bad WPA PSK.

### 0. Prerequisites

- **Raspberry Pi 5** (any RAM; d0-stepping/2026 boards are fully supported —
  the boot firmware is vendored from the official Raspberry Pi OS image at
  build time, see `os/board/markos/post-image.sh`).
- **27 W USB-C PD supply (5 V/5 A)** and an active cooler. Underpowered
  boards throttle or brown out under inference load.
- A microSD card (≥ 8 GB; the data partition grows to fill the card on
  first boot) or a USB SSD / NVMe HAT for the SSD variant.
- To **build**: Linux or WSL2 with the Buildroot prerequisites
  (`make gcc g++ patch perl tar unzip cpio rsync bc wget file cmake xz fdisk
  mcopy`) plus `rustup` (the engine cross-compiles through it).
- To **flash** from Windows: build `markos-installer.exe` (`cargo build
  --release -p markos-installer`) — raw-disk writes need an elevated shell.

### 1. Build the image (Linux / WSL2)

```sh
git clone https://github.com/woolcoxm/MarkOS.git && cd MarkOS/os
./build.sh --variant sd        # SD-card image: read-only squashfs root
# or: ./build.sh --variant ssd # USB-SSD/NVMe image: writable ext4 root
# or: ./build.sh --variant both
```

Artifacts land in `os/output/`: `markos-sd.img` (+ `.sha256`) and
`markos-ssd.img`. First run downloads Buildroot, the Raspberry Pi kernel
(`rpi-6.18.y`), the official Pi OS boot environment (cached under `os/dl/`),
and cross-compiles the engine with llama.cpp — about an hour; subsequent
builds are incremental (minutes). Validate a build in QEMU first if you
like: see [docs/build.md](docs/build.md).

### 2. Write your configuration

Copy `install.toml.example` to `install.toml` (git-ignored) and edit:

| block | what it controls |
|---|---|
| `boot_media` | `"sd"` \| `"usb-ssd"` \| `"nvme"` — image layout + wear strategy |
| identity | `hostname`, `timezone` |
| `admin_user` / `admin_password` | web UI account, created on first boot (argon2id-hashed in the provision file) |
| `active_cooling` | `false` → installer warns about throttling |
| `[network]` | `dhcp = true`, or static: `ip`, `gateway`, `dns` (applied as /24). `mdns_name` → `http://<name>.local`. Optional `wifi_ssid`/`wifi_psk` (WPA2/WPA3-SAE; omit both for Ethernet) |
| `[ssh]` | `enabled` (key-only, root, password logins never), `authorized_keys`, `serial_console` |
| `update_channel` | `"stable"` \| `"manual"` — nothing phones home either way |
| `[model_preseed]` | optional HF `repo` + `quant` (or direct `url`) so the box downloads a model on first boot |

**Windows note:** `active_cooling` and all root-level keys must precede any
`[table]` header — TOML scoping.

### 3. Flash — CLI

Two equivalent flows; both validate the config, inject the provision files
into the image's boot FAT, and (on `--write`) verify the written card
byte-for-byte.

**Produce a configured image file** (flash it later with any tool):

```sh
markos-installer --config install.toml --image os/output/markos-sd.img \
                 --target sd --out configured.img
```

**Write directly to the card** (list targets first; on Windows use
`--list-drives` for `\\.\PhysicalDriveN`, on Linux `/dev/sdX`):

```sh
markos-installer --list-drives
markos-installer --config install.toml --image os/output/markos-sd.img \
                 --target sd --write \\.\PhysicalDrive2 --verify
```

Windows specifics learned the hard way:

- Raw-disk writes require an **elevated** shell (UAC). If a FAT volume from
  the card auto-mounts mid-write it can steal the exclusive lock — the
  installer does its best, but `diskpart`'s `select disk N` + `attributes
  disk clear readonly` + `clean` first is the bulletproof sequence (that is
  exactly what `os/output/flash.bat` does on the dev machine).
- From Git Bash, pass the device as `'\\.\PhysicalDrive2'` with
  `MSYS_NO_PATHCONV=1` — otherwise the MSYS layer mangles the path.

### 4. Flash — GUI

```sh
cargo build --release -p markos-installer && ./target/release/markos-installer
```

Same surface as `install.toml`, with a drive picker and progress/verify
display. Writes through the same flasher core.

### 5. First boot

1. Card in the Pi, 27 W supply, Ethernet (or pre-configured WiFi). Power on.
2. The green **ACT LED blinks** during boot, then settles into a slow
   heartbeat once the OS is up — an across-the-room "alive" signal.
3. The **data partition grows to fill the card in the background** (several
   minutes on large cards). Services start immediately; they do not wait.
4. If you configured a `[model_preseed]`, the engine downloads it on first
   boot — on very large cards that download may race the partition grow and
   fail once; retry from the Models tab a few minutes later.
5. Open the web UI over **HTTPS**: `https://<mdns_name>.local`, your static/DHCP
   address, or `https://10.0.0.x:4444`. TLS is on by default with a first-boot
   self-signed certificate (SANs cover the mDNS name, hostname, and static
   IP); browsers will show a warning until you trust it. The inference API
   (`:8080`) stays plain HTTP for LAN clients by design. Sign in with the admin credentials from your
   config. (No credentials are baked in — the provision file is the only
   way an admin comes into existence.)
6. SSH: `ssh root@<address>` with your configured key — key-only, root,
   no passwords, ever.

### 6. Recovery paths (always a way back in)

- **Link-local rescue UI**: plug a laptop straight into the Pi's Ethernet
  port → `https://169.254.9.1:4444`.
- **Factory reset**: hold the GPIO26 button ≥ 3 s during boot — restores
  `/data/state` from the install-time snapshot (models are preserved).
- **Serial console**: UART on GPIO14/15, 115200 8N1.
- Full details: [docs/recovery.md](docs/recovery.md).

### 7. USB SSD / NVMe variant

Build `--variant ssd` and flash `markos-ssd.img`. NVMe/USB boot additionally
needs the board EEPROM `BOOT_ORDER` set once (from Raspberry Pi OS via
`raspi-config`, or `rpi-eeprom-config`) — see the note at the end of
`os/build.sh`.

### 8. Developing the engine without reflashing

The root filesystem is read-only squashfs; OS-level changes need a reflash,
but **engine/llama.cpp changes do not**:

```sh
scripts/engine-build.sh                     # cross-compile engine (WSL)
scripts/deploy-engine.sh                    # scp to the Pi + restart service
```

The deploy lands the binary on `/data/bin` and repoints the service (the
`/etc` overlay is per-boot, so re-run the deploy after each Pi reboot).
Keeper changes go into the source tree and ride along with the next image
build.

## Design invariants

- **Any GGUF, always.** The engine never rejects a model for lack of an
  accelerator path: a matching Axera engine set serves it whole-layer on
  the NPU (24-30 t/s), otherwise per-op NPU matmuls or the CPU reference
  take over — see [docs/axera.md](docs/axera.md).
- **CPU/NEON baseline.** The VideoCore VII GPU is not a compute target;
  no GPU code path exists anywhere. (The Axera M.2 card is an NPU, not a
  Pi GPU path.)
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
| on-target boot (real Pi 5 16 GB, d0 stepping, 2026 production) | **green on hardware**: Pi OS-derived boot env boots the MarkOS kernel (6.18) + squashfs root; first-boot net.conf adoption (static IP + gratuitous ARP), web UI login + tabs, SSH key auth (root), data partition grown 512 M → 936 G with GDT-reserved fs |
| on-target behavior (thermal under sustained load, NVMe EEPROM boot order) | requires physical hardware |
| **Axera AX8850 NPU support** (engine `axcl` feature, ggml-axcl fork with runtime geometry + engine-set manifests; Buildroot driver/runtime packages) | **green on hardware, both tiers measured**: NPU tier (Qwen3-0.6B Q8_0, matching engine set) serves coherent reasoning at **16.1 t/s decode / 135 t/s prefill** with 2.6 GB CMM on card; CPU tier (Qwen2.5-0.5B Q4_K_M, non-matching) serves at **23.8 t/s decode** with **8/8 quality evals passing**. Same engine process, per-model routing: matching models → NPU, any other GGUF → CPU (always coherent). Full bring-up story: [docs/hardware-debug-2026-09-16.md](docs/hardware-debug-2026-09-16.md) |

## Measured performance (on-target, Raspberry Pi 5 16 GB + AX8850)

| model | tier | quant | decode t/s | prefill tok/s | first-request TTFT | warm multi-turn TTFT | quality eval |
|---|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | CPU (non-matching) | Q4_K_M | **23.8** | **94** | **0.3–0.6 s** | **~0.3 s** | 8/8 (arithmetic, factual, reasoning) |
| Qwen3-0.6B | **NPU** (matching engine set) | Q8_0 | **16.1** | **135** | **0.3 s** | **~0.3 s** | coherent reasoning (thinking mode) |

The engine now sits **on top of the raw llama.cpp stack** instead of
underneath it — a CPU-only `llama-bench` built from the same fork on the
same box measures tg64 16.2 / 22.6 / 16.6 t/s at 1/2/4 threads and pp512
84.5 t/s, and the serving layer matches those numbers.

2026-09-17 perf deep dive — the box used to measure **CPU 1.0–1.9 t/s decode
with a ~13 s time-to-first-token on every request, NPU 7.4–9.7 t/s**. Root
causes, all fixed:

1. **Fork pin was one commit behind the opt-in registration fix** — the
   appliance built the axcl backend that registers unconditionally whenever
   the card is up; llama.cpp's scheduler (which can't distinguish it from
   the CPU backend — shared buffer type) routed CPU-tier computation
   through the axcl graph path: single-threaded host fallback for decode
   (2.4 t/s on a 16 t/s-capable stack) and per-request NPU/PCIe probing
   (13 s TTFT). Pin bumped c8d226b → 1bddded.
2. **Decode/batch thread split** (new `threads_decode` model config,
   default = half of `threads`): the Pi 5 is memory-bandwidth bound at
   decode; 2 threads beat 4 (llama-bench tg64: 22.6 vs 16.6 t/s), while
   prefill keeps every core. Lifted the NPU tier 9.7 → 16.1 t/s too (the
   whole-layer path's host-side KV staging contended at 4 threads).
3. **Prompt-prefix KV reuse** (CPU tier): multi-turn chats pre-fill only
   the new suffix instead of the whole conversation (`llama_memory_seq_rm`
   through the shim; whole-layer NPU still clears — no partial removal).
   A repeated 454-token prompt: 5.3 s → 0.4 s.
4. **Context caching + boot warmup**: the inference context is created once
   per model slot; auto-loaded models run a 1-token warmup generation so
   NPU engine loading lands at boot, not on the first request.
5. **Per-request host waste removed**: GGUF metadata cached by (mtime,
   size) and vocab-sized arrays skipped instead of materialized (the engine
   re-walked the full 150k-token GGUF header on every request); stop-string
   scan and output copying are O(1)/token instead of O(output²).

Remaining NPU headroom vs the PoC's 24–30 t/s is template quality, not host
code: the vendor Q8_0 templates run ~1.9 ms/layer vs ~1.17 ms/layer for
Pulsar2-tuned s4 builds. Full detail — methodology, ceilings, per-fix
measurements, reproduction steps — in [docs/performance.md](docs/performance.md);
template headroom in [docs/axera.md](docs/axera.md).

Benchmarks live in the repo now: `scripts/bench_final.py` (authoritative:
first-request, decode, prefill, multi-turn TTFT), `scripts/bench.py`
(classic suite + quality eval), `scripts/e2e_test.py` (31 tests across 8
categories — green on the final build). Raw-stack ceiling cross-check:
`scripts/build-llama-bench.sh`.

### Security audit (2026-09-17)

| category | result |
|---|---|
| Auth: argon2id password hashing | ✓ (no plaintext, no MD5/SHA) |
| Auth: timing-safe API key comparison | ✓ (`constant_time_eq`) |
| Auth: session tokens (crypto-random, cookie-scoped) | ✓ |
| Auth: wrong password → 401, fake session → 401 | ✓ (E2E verified) |
| TLS: self-signed on UI :4444, auto-generated per boot | ✓ |
| TLS: plain HTTP to TLS port rejected | ✓ (E2E verified) |
| XSS: user-supplied model names in error responses | ✓ **fixed** (was reflected, now static) |
| Input validation: header ≤32 KB, body ≤64 MB | ✓ |
| Input validation: null bytes, negative values, oversized input | ✓ (E2E verified) |
| Path traversal: model paths from store, not user input | ✓ |
| SQL/command injection: no SQL, no shell execution | ✓ (N/A) |
| Rate limiting: bounded queue, 429 on overflow | ✓ |
| API port (:8080) authentication | not enforced (by design: LAN-only appliance; admin UI is auth-gated) |

## Non-goals (v1)

No Pi-GPU path (the Axera M.2 NPU card *is* supported — that's an
accelerator, not the VideoCore GPU) · no multi-Pi clustering (noted as
future work in the design doc) · no general server dashboard · no
telemetry, no auto-updates, no runtime dependencies beyond
user-initiated model downloads.
