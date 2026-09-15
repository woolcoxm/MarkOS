# Pi 5 16GB — hardware brief and optimization map

Target: Raspberry Pi 5 16GB (BCM2712). Every optimization below is coded
against a documented hardware property. Sources at the bottom.

## Documented hardware

| Unit | Spec | MarkOS use |
|---|---|---|
| CPU | 4× Cortex-A76 @ 2.4 GHz, Armv8.2-A (crypto ext.) | execution pool, one core per role; `-C target-cpu=cortex-a76` codegen (UDOT/FP16 available) |
| L1 | 64 KB I + 64 KB D per core | matvec rows (1088 B q8_0) stream well under 64 KB |
| L2 | 512 KB per core | per-core row slices keep working set local |
| L3 | 2 MB shared | weight-cache reads benefit once resident |
| RAM | 16 GB LPDDR4X, ARM-visible from 0x0 | **whole-model RAM residency** (below) |
| Peripherals | RP1 south bridge @ 0x1F_0000_0000 (ARM view); GIC-400 @ 0x107fff8000/9000 | interrupt-driven serving later; ECAM window mapping already handles >4 GiB MMIO |
| Boot | firmware loads `kernel_2712.img` at 0x80000; DRAM size injected at boot | image-pi5 target |

## The 16 GB advantage: whole-model RAM residency

The 640 MB Qwen3-0.6B q8_0 model is loaded once into a fixed RAM window
(board::WEIGHT_RAM_BASE) and every matvec afterwards reads weights from
RAM — zero storage traffic per token. On the 16 GB board this leaves the
entire model resident with >15 GB of headroom (room for KV cache growth
and future larger models). The window is identity-mapped as Normal
cacheable 2 MiB blocks (Phase 6 huge-page equivalent — no 4 KB TLB
pressure across the model).

## A76 compute path

- **UDOT/SDOT int8 dot product** (Armv8.2 DotProd, present on the A76):
  32 MACs per instruction. The matvec hot loop quantizes activations
  per-32-block (Q8_0 style, matching llama.cpp) and accumulates in i32 —
  the llama.cpp scheme, at A76-native width.
- **FP16 conversions** for the f16 block scales (hardware `fcvtl`).
- Scalar f32 fallback is retained and validated (test-forward worst diff
  1.7e-05 vs numpy).

## Optimization ledger (measured on the TCG dev loop, Qwen3-0.6B q8_0)

| change | decode throughput | mean token latency |
|---|---|---|
| volume streaming (baseline) | 16 milli-tok/s | 60.7 s |
| + RAM weight cache + parallel matvec | **67 milli-tok/s (4.2×)** | **14.8 s** |
| + UDOT int8 dot (this iteration) | pending hw measurement | — |

Under QEMU TCG every guest instruction is interpreted, so absolute
numbers understate the hardware by orders of magnitude; the ledger
proves the accounting and the relative wins. The authoritative
throughput claim is the Pi-8 hardware run: same GGUF, same Pi 5,
llama.cpp vs MarkOS.

## Residual known issues

- NEON dot32 experiment: exact in the serial path (worst diff 1.7e-05),
  deterministic wrong results only when invoked from the parallel pool
  job — reverted, isolated investigation pending.
- Pi 5 16GB rev 1.1 boards need current firmware/DT handling (Linux
  mainline reports); the boot entry point (firmware-loaded kernel) is
  unaffected.

Sources: [Raspberry Pi 5 product page](https://www.raspberrypi.com/products/raspberry-pi-5/),
[Raspberry Pi processors documentation](https://www.raspberrypi.com/documentation/computers/processors.html),
[BCM2712 GIC-400 forum thread](https://forums.raspberrypi.com/viewtopic.php?t=371974),
[bcm2712-rpi-5-b.dts](https://github.com/raspberrypi/linux/blob/rpi-6.18.y/arch/arm64/boot/dts/broadcom/bcm2712-rpi-5-b.dts),
[Circle bcm2712.h MMIO reference](https://circle-rpi.readthedocs.io/en/50.0/basic-system-services/direct-hardware-access.html)
