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
| on-target boot (real Pi 5 16 GB, d0 stepping, 2026 production) | **green on hardware**: Pi OS-derived boot env boots the MarkOS kernel (6.18) + squashfs root; first-boot net.conf adoption (static IP + gratuitous ARP), web UI login + tabs, SSH key auth (root), data partition grown 512 M → 936 G with GDT-reserved fs |
| on-target behavior (thermal under sustained load, NVMe EEPROM boot order) | requires physical hardware |

## Non-goals (v1)

No GPU/accelerator path · no multi-Pi clustering (noted as future work in the
design doc) · no general server dashboard · no telemetry, no auto-updates,
no runtime dependencies beyond user-initiated model downloads.
