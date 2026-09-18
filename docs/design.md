# MarkOS — Design Document

MarkOS is a completely custom, minimal, headless operating system purpose-built for
one job: serving LLM inference on a LAN from a Raspberry Pi 5 (16 GB). No desktop, no
unrelated services, no cloud dependency. This document is the authoritative record of
every architectural decision: what is custom, what is reused, and why.

Companion docs: [`build.md`](build.md) (OS build pipeline walkthrough),
[`recovery.md`](recovery.md) (recovery paths in depth), [`../README.md`](../README.md)
(repo layout and quickstart).

---

## 1. Locked hardware assumptions

| Property | Value | Consequence for the design |
|---|---|---|
| SoC | BCM2712, 4× Cortex-A76 @ 2.4 GHz (Armv8.2-A, DotProd/I8MM, NEON/ASIMD) | Inference is CPU/NEON-only. No GPU-offload code path exists anywhere in the tree. |
| RAM | 16 GB LPDDR4X | Whole-model RAM residency; guardrails budget against ~15.2 GB usable. |
| Accelerator | VideoCore VII — **not usable** for general compute | Explicit non-goal; never appears in code or UI copy. |
| Boot media | SD card, USB SSD, NVMe (PCIe HAT / M.2 HAT+) | Installer asks which target; filesystem strategy differs (§4). |
| Cooling | Active cooler assumed required for sustained load | Installer warns (never blocks) when config implies sustained use. |
| Power | 27 W USB-C PD assumed | Noted in installer output/docs; not enforced in software. |

Prior work note: this repo previously hosted a bare-metal aarch64 MarkOS kernel
(custom SDOT int8 matvec, greedy-parity with llama.cpp on the same GGUF, Pi 5 16 GB).
It remains in git history (`< 5adf58c`) and proved the A76 compute path. This project
pivots to a Linux-based appliance (below) because a maintained kernel + userspace
split gives supervised services, TCP/IP robustness, and model-format breadth (all
GGUF quantizations) far faster than continuing bare-metal development. The pivot does
**not** change the "custom-built" requirement — see the custom/reused boundary in §7.

---

## 2. Implementation language (and why it satisfies "blazing speeds")

**Rust, everywhere**: inference engine, installer, OS userspace tools, and the web UI
is embedded in the engine binary.

- Peak speed: Rust compiles to the same machine code as C/C++ — zero-cost
  abstractions, no GC, no VM, no runtime. The tensor hot path is ggml's NEON kernels
  compiled with `-mcpu=cortex-a76` (identical instruction stream as a C build); the
  serving layer's per-request overhead (allocation-free HTTP/1.1 parsing, bounded
  queues, `Arc`-shared state) is nanoseconds against millisecond-per-token decode.
- Single executables, no heavy runtime: the Windows installer ships as one static
  `.exe` (egui renders via GPU or software fallback, no .NET/WebView2/Electron);
  the engine is one static-ish binary for aarch64 Linux.
- Memory safety in the components that must never crash a headless box.
- Alternatives rejected: C/C++ (unsafe at this scope), Go/C#/Java (GC pauses +/or
  heavyweight runtime bundled), JS/TS (runtime + throughput), Zig (immature
  ecosystem for GUI/HTTP needs).

## 3. OS base choice

**Buildroot** (pinned release, consumed via a `BR2_EXTERNAL=markos` tree in `os/`).

Justification vs. Yocto: MarkOS is a single-board, single-purpose appliance with ~20
packages. Buildroot produces a reproducible image from one `make` invocation with a
defconfig + overlay tree; Yocto's layer/bitbake machinery buys reproducibility we get
for free, at the cost of a much larger build system to maintain. Buildroot is the
KISS-aligned choice. (Not stripping Raspberry Pi OS: we control every byte; rootfs
target is < 40 MB.)

**Init/supervision: BusyBox init for boot stages + s6 (`s6-svscan`/`s6-supervise`) for services.**
The engine and web UI (one supervised service, `markos-engine`), dropbear sshd (if
enabled), and avahi run under `runsv`, which restarts them on crash with exponential
backoff. Buildroot's "path of least resistance" would be systemd; we deliberately
take the simpler runit path — ~1 MB, no D-Bus, no journal, readable per-service
`run` scripts. BusyBox `init` handles single-shot boot stages (mount, provision,
network) and execs runit stage 2.

Kernel: Linux (rpi fork pinned to the version Buildroot's
`raspberrypi5_defconfig` expects) with `bcm2712_defconfig` + a small fragment
(overlayfs, nftables, watchdog, thermal, ext4/squashfs/vfat, cgroup-free minimal).

## 4. Storage / filesystem strategy per boot media

Common partition table (MBR; GPT not needed and less compatible with Pi firmware
tooling):

| # | SD card target | USB SSD / NVMe target | Purpose |
|---|---|---|---|
| 1 | FAT32 `boot` 256 MiB | same | Pi firmware, kernel, DTB, `config.txt`, **provision file injected by installer** |
| 2 | SquashFS RO `rootfsA` | ext4 RW `rootfsA` | active root |
| 3 | SquashFS RO `rootfsB` | ext4 RW `rootfsB` | inactive root (A/B updates, §5) |
| 4 | ext4 `data` (rest of media) | same | models, logs, engine state, config store; **grows to fill media on first boot** |

- **SD card**: rootfs is SquashFS mounted read-only (`/`), so power loss can never
  corrupt the root. Writable state lives on `p4` (`/data`) and is bind-mounted:
  `/data/state` (engine config, user DB, TLS keys) → `/var/lib/markos`,
  `/data/log` → `/var/log/markos`. A small tmpfs overlay covers `/etc` for runtime
  tweaks (lost on reboot by design — persistent system config goes through the
  provision file or `/data/state`). Models never touch the overlay or root: they are
  big, write-once, and SSD-card-unfriendly → `p4`.
- **USB SSD / NVMe**: normal writable ext4 root (wear is a non-issue); models still
  isolated on `p4` so a disk-full from an oversized model cannot take down the root
  filesystem. First boot expands `p4` to fill the disk (`sfdisk` + `resize2fs`) and
  the engine enforces a models-disk high-water mark (default 90%) before accepting
  downloads.
- The installer asks SD vs SSD vs NVMe because it selects which prebuilt image
  variant to write (`markos-sd.img` vs `markos-ssd.img` — squashfs vs ext4 root) and
  adjusts post-flash guidance (EEPROM `BOOT_ORDER` for NVMe/USB boot).

## 5. Update mechanism (rollback-safe, manual, offline-friendly)

**A/B slots with firmware `tryboot`.** The Pi 5 firmware natively supports
`tryboot` (one-shot boot into the alternate `tryboot.txt` → alternate root). Flow:

1. User triggers "check for update" from the web UI (manual only; no phone-home by
   default — the UI fetches the manifest URL the user configured, or the user
   uploads/points at a `.markos-update` image file).
2. `markos-update` (shell + busybox, part of the rootfs) writes the new rootfs into
   the inactive slot (validates checksum), writes its `tryboot.txt`, and reboots
   into tryboot.
3. Early in the new root's boot, a `boot-confirm` supervised service waits for
   "engine healthy for 3 consecutive minutes" (polls `/healthz` locally), then
   commits: copies tryboot.txt → config.txt (making slot B default) and clears the
   try flag.
4. If the box instead dies/watchdogs during the try window, the firmware's next
   normal boot uses the unchanged default `config.txt` → old slot. **A failed update
   can never brick a headless box.**
5. Update channel selection at install time (`stable` | `manual`) only controls
   which manifest URL is pre-filled; nothing auto-installs.

## 6. Recovery path (chosen and implemented)

Three independent ways back in, in order of escalation:

1. **Always-on debug UART** (GPIO 14/15, 115200 8N1) plus a **link-local rescue
   listener**: the engine always binds the web UI additionally on
   `169.254.9.1:4444` (IPv4LL, up whenever the cable is plugged into a laptop).
   A user who misconfigures networking plugs one Ethernet cable directly to the Pi
   and reaches `http://169.254.9.1:4444`. Implemented in the engine (second bind) +
   OS (IPv4LL address assignment at boot).
2. **Factory-reset button (GPIO 26 → GND, held during boot)**: the boot stage checks
   the pin (`pinctrl`/gpio sysfs); if held ≥ 3 s, `markos-recovery` restores
   `/data/state` from the pristine provision snapshot kept on the boot partition,
   reverts any pending A/B try, and reboots. This recovers from a broken web-UI
   config or forgotten credentials with no monitor and no network knowledge.
3. **SSH recovery listener** (if SSH was enabled at install) on the link-local
   address as well.

All three are documented in `docs/recovery.md` and implemented (OS overlay scripts +
engine second bind; the button uses the idle-by-default BCM2712 GPIO 26).

## 7. Inference engine — custom vs. reused boundary (explicit)

**Custom (written for MarkOS, in `engine/`):**

- HTTP/1.1 server (std::net + bounded thread pool, keep-alive, chunked bodies,
  SSE streaming) — no tokio/axum; serving layer is the product.
- OpenAI-compatible API surface: `POST /v1/chat/completions`,
  `POST /v1/completions`, `GET /v1/models` (+ `GET /v1/models/{id}`), streaming via
  SSE (`stream: true`), full sampling-parameter mapping.
- Native control plane for the web UI: `/admin/*` (auth'd) — model registry, load/
  unload/switch, profiles, guardrail estimation, network settings, user management,
  metrics (`GET /admin/metrics/stream`, live SSE), request log, system actions
  (restart engine service, update check, recovery status), health
  (`GET /healthz`, `GET /readyz` for the supervisor/watchdog).
- GGUF metadata reader (pure Rust) — inventory, quantization detection, and the
  memory-guardrail estimator (§8). Reads headers only, never the model body.
- Model manager & scheduler: registry of configured models (disk) vs. resident
  models (RAM); load-on-demand; explicit policy (§8.3); request queue with bounded
  concurrency and queue-depth limits; cooperative cancellation of streamed
  generations.
- Guardrails: RAM estimator + refusal (HTTP 409/422 with reason) before load — the
  backend enforcement point behind the UI's estimates.
- Metrics/logging: in-memory ring + JSONL append (`/data/log/markos.jsonl`) —
  tokens/sec, per-request latency, memory, temperature, errors. No external service.
- Web UI (§9): hand-written vanilla JS/HTML/CSS embedded in the binary
  (`include_bytes!`).
- Auth: session-cookie login (argon2id-hashed credentials), admin/viewer roles,
  optional bearer API key for `/v1`, optional TLS (rustls; self-signed cert
  generated on first boot for LAN use).

**Reused (deliberately, per spec):**

- **llama.cpp / ggml** as the tensor + quantization backend (Q4_K_M, Q5_K_M, Q6_K,
  Q8_0, and everything else GGUF offers), NEON-optimized for Cortex-A76, threaded
  across the 4 A76 cores. Bound through the `llama-cpp-2` crate behind a
  `trait Backend` so the serving layer never touches ggml types directly.
  Codegen: `-Ctarget-cpu=cortex-a76` for both ggml (via its CMake flags) and the
  engine crate.
- **minijinja** to render per-model custom chat templates (Jinja2-style, as the UI
  exposes them). When a model's GGUF carries a chat template and the user has not
  overridden it, template rendering happens in-engine; generation itself is a plain
  prompt completion.
- **argon2** for credential hashing. Everything else in the engine is std-only +
  `serde_json` (serialization), `rustls`/`rcgen` behind the `tls` feature.

The engine is **not** a wrapper around `llama.cpp`'s server binary — `llama-server`
is not used or shipped. What llama.cpp provides is exactly the layer the spec says
may be reused: quantized tensor math.

## 8. Memory model, concurrency, multi-model policy

### 8.1 RAM budget

`usable = MemTotal − reserve(os≈340 MiB incl. page-cache floor) − engine_self(~24 MiB)`.
Read from `/proc/meminfo` at runtime (falls back to 15.2 GiB constant on host).

### 8.2 Guardrail estimate (per model + n_ctx)

```
weights  = gguf file size (mmap'd; counted fully — page cache is reclaimable but
           resident; we refuse configs whose weights exceed usable)
kv_cache = 2 (K+V) × n_layers × n_ctx × n_head_kv × head_dim × bytes(quant; f16=2)
compute  = f(n_ctx, n_batch, n_embd, n_layers) ≈ (12·n_embd·n_batch + 4·n_embd·n_ctx/1024)·n_layers·4B  (documented approximation, +25% safety)
margin   = 15% of (weights + kv + compute)
require  = weights + kv + compute + margin ≤ usable − Σ(resident other models' require)
```

The UI shows the estimate **before** apply and blocks with the reason; the engine
independently recomputes and refuses (HTTP 409 `memory_guardrail`) at load time.
This is the single source of truth in `engine/src/guard.rs`, shared by both.

### 8.3 Concurrency

Single-box, one stream per resident model at appliance-class rates (measured:
~24 t/s CPU decode, ~16 t/s NPU decode — see docs/performance.md): the
scheduler allows **one active generation per resident model** (prefill chunks
between decode steps of the other slot), a configurable bounded FIFO queue
(default depth 4, per-endpoint), and instant `429 Too Many Requests` +
`Retry-After` when the queue is full. No speculative parallelism beyond the
thread split (4 prefill / 2 decode threads). This is honest about the hardware
rather than pretending to be a GPU farm.

### 8.4 Multi-model

Configured ≠ resident. Default policy: **load/unload on switch**; up to 2 models
resident concurrently when the guardrail allows (e.g. a 0.6 B chat model + a small
embeddings/summary model). When a load would exceed the budget, the manager evicts
the least-recently-used idle resident model automatically and reports the eviction;
if nothing can be evicted the load is refused with the computed numbers.

## 9. Web configuration UI (scope fence)

Served by the engine on port 80 (`:4444` TLS/IPv4LL rescue), SPA in vanilla JS
(≈1,300 lines, zero frameworks, zero CDN — the appliance must work airgapped):

- **Models**: installed list w/ disk usage; download from HF repo/URL with
  quantization choice; delete; per-model runtime config (n_ctx, threads, batch,
  chat template editor with live preview, default system prompt).
- **Profiles**: named sampling profiles per model (creative/precise/coding
  presets seeded): temperature, top_p, top_k, min_p, repeat/presence/frequency
  penalty, mirostat (mode/tau/eta), stop sequences, max tokens; one-click switch;
  request-level override via `"profile"` field in OpenAI bodies.
- **Guardrails panel**: estimated RAM for model+context before apply; blocking
  errors with numbers, not silent failures; thermal-throttle visibility
  (temp gauge + throttling flag from `/sys/class/thermal` & `vcgencmd`-equivalent
  sysfs).
- **Network/exposure**: API listen address/port, TLS on/off (self-signed or
  upload), API-key requirement, IP allowlist (optional), UI port.
- **Monitoring**: tok/s (1/10/60 s windows), active+queued requests, RAM, CPU load,
  SoC temperature, recent request log.
- **Users**: admin + read-only accounts (v1 ships both; initial admin set by the
  installer — no default credentials exist anywhere).
- **System** (thin): OS/engine version, update check trigger, recovery-path status
  (button state, IPv4LL listener, last watchdog reboot), restart engine service.

Non-goals enforced by review: no GPU toggles, no clustering, no general server
dashboard, no telemetry.

## 10. Installer (Windows-first)

One Rust crate, two frontends (same binary): GUI (egui) and CLI
(`markos-installer --config install.toml ...` — the automation/reproducibility
path; the GUI generates an equivalent TOML for reuse). Responsibilities:

1. Ask boot-media target (SD / USB SSD / NVMe) → pick image variant + wear strategy
   (§4), warn on sustained-load-without-active-cooler config, note 27 W PD.
2. Collect the full config surface: hostname/tz/locale, network (DHCP/static,
   WiFi WPA2/WPA3-SAE PSK or Ethernet, mDNS name), SSH (keys-only, authorized_keys
   injection, serial console toggle), initial web-UI admin user + password
   (validated: **refuses to proceed without an SSH key or admin password** — a
   headless box must always have a way in), update channel, optional model pre-seed
   (HF repo + quant).
3. Build the image: copy the base image, inject `/boot/markos/provision.env` +
   `authorized_keys` into the FAT32 boot partition with a **custom FAT32 writer**
   (no external tools needed on Windows), expand/patch partition entries when the
   target media type requires it.
4. Validate everything **before** touching any disk; re-validate the target device
   (size match, removable flag, explicit device-path confirmation for
   `\\.\PhysicalDriveN` / `/dev/sdX`), then write with progress + read-back verify
   (optional).

Linux gets the same CLI (and GUI) binary — full feature parity by construction.

## 11. Security baseline

SSH key-only (dropbear; passwords compiled off), no default passwords anywhere
(provision file is required to create the admin), engine refuses first-run without
provisioned admin, nftables deny-by-default inbound (only 22 opt / 80 / 4444 / 8080 /
5353/udp), TLS optional w/ first-boot self-signed, `Journal`-less minimal logging to
`/data/log`, no shell accounts. The web UI and API are the only network-reachable
processes besides optional dropbear.

## 12. Build & validation strategy

- `os/build.sh` (run on a Linux builder/WSL2): fetches pinned Buildroot, applies
  `markos` external tree, emits `markos-sd.img` / `markos-ssd.img` + SHA256SUMS.
  Fully scripted; zero manual steps.
- Engine + installer: `cargo test` (host) covers HTTP/API contract (golden OpenAI
  JSON), guardrail math, FAT32 writer round-trip, provision validation, auth; an
  end-to-end test boots the engine with the mock backend and a synthetic GGUF
  (`scripts/make_gguf_test.py`, ported from the prior kernel work).
- Boot gate: the built image is validated by booting it headless under
  `qemu-system-aarch64 -M virt` (kernel carries a virtio-mmio fragment for
  exactly this; harmless on Pi hardware) — supervised engine, installer
  provisioning, model auto-load and a real OpenAI-compatible completion are
  all exercised on the assembled appliance image. Both media variants
  (`markos-sd.img`, `markos-ssd.img`) are produced by `os/build.sh` and
  content-verified. The remaining untested surface is physical-Pi-only
  behavior (thermal throttling, EEPROM NVMe boot order, the BCM2712 watchdog
  device).
- Incremental milestones per the spec: (1) minimal image boots headless, engine
  serves a small model; (2) web UI layered in; (3) full installer config surface;
  (4) update/recovery last. Milestones 2–4 are code-complete here and validated at
  the unit/e2e level on the host; milestone-1 hardware bring-up runs on the Pi via
  `os/build.sh` artifacts (no Pi required for host-side validation).
