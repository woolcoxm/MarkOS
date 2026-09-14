# MarkOS

A **bare-metal Raspberry Pi appliance for local LLM inference**. No Linux, no
distro, no shell: flash the SD card, power on, and the Pi boots straight into
a minimal inference engine that serves a GGUF model over the network. The
configuration is frozen at install time (the `markos-installer` bakes it into
the image); at runtime the system is controlled only through its own
LLM/control protocol.

Design rule for every decision: *does this make the model run faster or the
appliance simpler?* If not, it does not exist.

## Why this is different

- **Boots to inference in seconds** — no bootloader chain, no init system.
- **Zero OS overhead** — no scheduler jitter, no syscalls, no page-cache
  surprises; every core does tensor math or nothing.
- **Hard real-time memory policy** — no swap ever; the model either fits or
  the installer refuses to build the image.
- **Optional accelerator**: M5Stack LLM8850 (Axera AX8850, 24 TOPS, 8 GB) on
  the Pi 5's M.2 slot, driven as a PCIe inference appliance.

## Target hardware

| Board | Status |
|-------|--------|
| QEMU raspi3b | dev/CI loop (automated tests) |
| Raspberry Pi 4 (BCM2711) | bring-up target — best-documented bare-metal Pi |
| Raspberry Pi 5 (BCM2712) | performance target — A76 + UDOT, M.2 for LLM8850 |

## Repository layout

```
kernel/                the kernel crate (no_std, no_main, AArch64)
  src/main.rs          entry stub (park APs, stack, FP/NEON, zero bss) + kmain
  src/uart.rs          PL011 serial console
  aarch64.ld           flat physical layout at 0x80000 (Pi 64-bit load addr)
sd/config.txt          Pi firmware boot config for real hardware
Makefile               build → image → QEMU dev loop
```

## Building and running (from Windows, via WSL)

```
wsl bash -lc 'cd /mnt/c/Users/Mark/Desktop/Projects/MarkOS && make run-log'
```

The automated dev loop runs the image in QEMU (`-M raspi3b`). For real
hardware: copy `kernel8.img` + `sd/config.txt` + the stock Pi firmware blobs
onto a FAT32 SD card and boot; the serial console (GPIO 14/15, 115200) is the
system console.

## Roadmap (Pi-first)

- [x] Pi-0 — AArch64 bring-up: boots in QEMU raspi3b, `kernel alive` on PL011
- [x] Pi-1 — MMU page tables, exception vectors (VBAR), generic timer
- [x] Pi-2 — SMP: all cores online (PSCI on qemu-virt; spin-table+mailbox on Pi hw), shared-counter acceptance
- [ ] Pi-3 — SD card driver (SDHCI) + minimal FAT reader + GGUF load
- [ ] Pi-4 — NEON int8 matmul kernels (UDOT), work-stealing pool, 2 MiB blocks
- [ ] Pi-5 — Network: GENET (Pi 4) / Pi 5 path, minimal TCP, inference+control protocol
- [ ] Pi-6 — Installer: host tool bakes config (IP, port, admin token, model) into SD image
- [ ] Pi-7 — LLM8850 on Pi 5: PCIe/RP1 enumeration + Axera card transport (research-gated)
- [ ] Pi-8 — stats + soak test; real-hardware validation

An earlier x86_64 exploration (Limine unikernel, phases 0–3) is preserved on
the `x86_64-archive` branch for reference; the Pi is now the only target.

## Scope notes (hard boundaries)

No multi-tenancy, no general scheduler, no POSIX, no shell, no runtime
configuration outside the LLM/control protocol. NPU/GPU acceleration (LLM8850)
is a separate, research-gated track — vendor stacks assume Linux, so bare-metal
support means reimplementing their transport from documentation and GPL
drivers, decided explicitly before any work starts.
