# MarkOS

A single-purpose **x86_64 unikernel** that boots directly into an LLM inference
engine. No scheduler, no processes, no filesystem beyond raw block reads for
model weights. Every design decision serves one goal: load a model and run
inference on it as fast as possible.

Built from scratch in `#![no_std]` Rust on the Limine boot protocol, following
a phased plan (Phase 0 … Phase 11) with a QEMU-verified acceptance criterion
per phase.

## Pinned toolchain (reproduce with these)

| Component        | Version                            |
|------------------|------------------------------------|
| Limine bootloader| **v12.9.0** (binary release tarball, `make deps`) |
| `limine` crate   | 0.6.5 (boot protocol interface)    |
| Rust             | nightly (built & tested with 1.100.0-nightly), components: `rust-src`, `llvm-tools-preview` |
| Dev loop         | WSL2 Ubuntu 24.04, QEMU 8.2.2 (TCG), xorriso 1.5.6, GNU make |

## Layout

```
kernel/                 the kernel crate (no_std, no_main)
  src/main.rs           entry point, Limine request table
  src/serial.rs         COM1 serial console
  linker.ld             higher-half linker script (requests get their own PHDR)
target/x86_64-unikernel.json   custom target spec (panic=abort, no red zone,
                               mcmodel=kernel, soft-float for now)
boot/limine.conf        Limine menu config
Makefile                build → image → run, all inside WSL
third_party/limine      vendored by `make deps`, gitignored
```

## Building and running (from Windows, via WSL)

```
wsl bash -lc 'cd /mnt/c/Users/Mark/Desktop/Projects/MarkOS && make run-log'
```

`make deps` first on a fresh checkout. `make run` attaches the serial console
to the terminal; `make debug` starts QEMU paused with a GDB stub on :1234.

## Phase status

- [x] Phase 0 — scaffold: boots in QEMU, prints `kernel alive` on COM1, halts
- [x] Phase 1 — GDT, IDT, TSS, exception handlers, panic register dump
- [x] Phase 2 — physical frame allocator (bitmap)
- [x] Phase 3 — page tables + kernel heap
- [ ] Phase 4 — SMP bring-up (ACPI MADT, INIT-SIPI-SIPI)
- [ ] Phase 5 — work-stealing thread-per-core pool
- [ ] Phase 6 — huge pages, NUMA-aware weight placement
- [ ] Phase 7 — virtio-blk + GGUF parsing
- [ ] Phase 8 — quantized CPU matmul kernels (AVX2/AVX-512)
- [ ] Phase 9 — virtio-net + minimal TCP/IP + token streaming protocol
- [ ] Phase 10 — serial stats (tok/s, memory, per-core util)
- [ ] Phase 11 — 24 h soak test
- [ ] (optional, separate track) GPU compute — explicit decision required first

## Scope notes (hard boundaries)

No multi-tenancy, no general scheduler, no POSIX, no real filesystem, no GUI.
GPU support is an explicitly separate later track (VFIO shim vs. NVK port) and
is *not* part of the base build. See the project brief for the rationale.
