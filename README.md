# MarkOS

A **bare-metal Raspberry Pi appliance for local LLM inference**. No Linux, no
distro, no shell: flash the SD card, power on, and the Pi boots straight into
a minimal inference engine that serves a GGUF model over the network. The
configuration is frozen at install time (the installer bakes it into the
image); at runtime the system is controlled only through its own
authenticated control protocol.

Design rule for every decision: *does this make the model run faster or the
appliance simpler?* If not, it does not exist.

## Why this is different

- **Boots to inference in seconds** — no bootloader chain, no init system.
- **Zero OS overhead** — no scheduler jitter, no syscalls, no page-cache
  surprises; every core does tensor math or nothing.
- **Hard real-time memory policy** — no swap ever; the model either fits or
  the installer refuses to build the image.
- **Remote-first**: TCP control protocol (token-authenticated) for status,
  model load, and inference; install-time config is the only other surface.
- **Optional accelerator**: M5Stack LLM8850 (Axera AX8850, 24 TOPS, 8 GB)
  on the Pi 5's M.2 slot — research + implementation plan in
  [docs/llm8850-research.md](docs/llm8850-research.md) /
  [docs/axera-markos-plan.md](docs/axera-markos-plan.md), building on
  [woolcoxm/Axera-AX8850-GGUF-Support](https://github.com/woolcoxm/Axera-AX8850-GGUF-Support).

## Target hardware

| Board | Status |
|-------|--------|
| QEMU virt (qemu-system-aarch64) | dev/CI loop — all acceptance gates run here |
| Raspberry Pi 5 (BCM2712) | deployment target — A76 + UDOT, M.2 for LLM8850 (`make image-pi5` builds `kernel_2712.img`) |

## What's in the kernel (AArch64, no_std)

| Subsystem | Where | Notes |
|---|---|---|
| Boot / EL normalization / SMP | `main.rs`, `smp.rs`, `psci.rs` | PSCI (virt), spin-table + mailbox (Pi hw) |
| MMU + exception vectors | `mmu.rs`, `vectors.rs` | identity map + ECAM/PCIe windows |
| virtio-blk + read-only FAT32 | `virtio_blk.rs`, `fat.rs` | SD image reads in QEMU; SDHCI on hw is a later phase |
| GGUF v3 parser | `gguf.rs` | tensor table, bounds-checked |
| Execution pool (thread-per-core) | `pool.rs` | no scheduler; cores spin between jobs |
| NEON UDOT int8 matmul | `matmul.rs` | per-function `dotprod` target feature, runtime-checked |
| virtio-net + ARP/IPv4/ICMP + TCP | `net.rs`, `tcp.rs` | one listener, control protocol on top |
| Control protocol | `control.rs` | HELLO/STATUS/LOAD/RUN/PING/ECHO, token auth |
| Install-time config | `config.rs` | MARKOS.CFG (ip/port/token) read once at boot |
| PCIe ECAM enumeration | `pcie.rs` | the LLM8850 on-ramp (Pi-7a) |

## Acceptance gates

Every phase lands with a gate that greps its own serial log / client
verdict. `scripts/regress.sh` runs the full sweep:

```
bash scripts/regress.sh            # all 11 gates
make test-pcie                     # or any single gate
```

| Gate | Proves |
|---|---|
| test-exceptions | brk caught+resumed, data abort caught |
| test-smp | 4 cores online via PSCI, exact shared counter |
| test-block | virtio-blk bring-up + LBA0 read |
| test-fat | FAT32 mount + MODEL.BIN read + pattern check |
| test-pool | parallel sum across all cores |
| test-matmul | UDOT matmul, 256/256 exact vs scalar |
| test-net | MARKOS-PING → MARKOS-PONG over TCP (slirp hostfwd) |
| test-control | HELLO/STATUS/LOAD/RUN — RUN computes matmul on all cores remotely |
| test-install | installer-baked MARKOS.CFG drives port + token; wrong token rejected |
| test-soak | 40 rounds RUN+ECHO on one connection (RX-ring wraparound killer) |
| test-pcie | bare-metal ECAM enumeration (host bridge, NIC, root port) |
| test-model | real 640 MB Qwen3-0.6B q8_0 loaded: full 310-tensor table + payload CRCs byte-match the host reference (`scripts/gguf_ref.py`) |

## Building and running (from Windows, via WSL)

```
wsl bash -lc 'cd /mnt/c/Users/Mark/Desktop/Projects/MarkOS && make run-log'
```

Requires: rustup nightly + `aarch64-unknown-none`, `llvm-objcopy`,
`qemu-system-aarch64`, `sfdisk`/`mkfs.vfat`/`mtools` (test images),
`python3` (host-side clients/installer). For real Pi 5 hardware:
`make image-pi5` produces `kernel_2712.img` (cortex-a76 codegen); copy it
onto the installer-built SD image with the stock Pi firmware blobs.

## Roadmap

- [x] Pi-0 — AArch64 bring-up: boots in QEMU, `kernel alive` on PL011
- [x] Pi-1 — MMU page tables, exception vectors (VBAR), generic timer
- [x] Pi-2 — SMP: all cores online (PSCI on virt; spin-table+mailbox on Pi hw)
- [x] Pi-3 — block storage: virtio-blk + read-only FAT32 + GGUF v3 parse
- [x] Pi-4 — execution pool + NEON UDOT int8 matmul (256/256 exact)
- [x] Pi-5 — virtio-net + ARP/IPv4/ICMP + minimal TCP
- [x] Pi-5b — control protocol: HELLO/STATUS/LOAD/RUN over TCP, token auth
- [x] Pi-6 — installer: MARKOS.CFG (ip/port/token) baked into SD, enforced at boot
- [x] Pi-7a — LLM8850 research + bare-metal PCIe ECAM enumeration
- [ ] Pi-7b — AX8850 transport RE → bare-metal axcl-lite → NPU decode
      ([plan](docs/axera-markos-plan.md))
- [ ] Pi-8 — stats/soak on real hardware; SDHCI for real SD reads;
      real-hardware validation on Pi 5 16GB

An earlier x86_64 exploration (Limine unikernel, phases 0–3) is preserved on
the `x86_64-archive` branch for reference; the Pi is now the only target.

## Scope notes (hard boundaries)

No multi-tenancy, no general scheduler, no POSIX, no shell, no runtime
configuration outside the control protocol. NPU acceleration (LLM8850)
follows the research-gated track above — the NPU compiler (Pulsar2) stays
an offline x86 toolchain; the card is driven by engines, never compiled to.
