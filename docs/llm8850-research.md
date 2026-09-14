# M5Stack LLM8850 (Axera AX8850) — Pi-7 research

Status: **research complete, hardware-dependent work deferred to Pi-8** (on
the user's Pi 5 16GB + LLM8850 kit). This document records what is publicly
knowable without the hardware and the resulting architecture decision.

## The hardware

The LLM8850 is an M.2 M-Key (2242) card for the Pi 5 (via the included
PiHat / AI-8850 kit adapter):

- SoC: Axera AX8850 — **it is a complete computer**, not a dumb accelerator:
  8x Cortex-A55 @ 1.7 GHz, 4/8 GB LPDDR4x @ 4266 MT/s, 4 MB QSPI NOR
- NPU: 24 TOPS @ INT8
- Host interface: **PCIe 2.0 x2, operating as a PCIe Endpoint (EP)**
- Card ships running Axera's own SDK; the host never touches the NPU
  directly — everything goes through a request/response transport over PCIe.

## Driver situation (the gating fact)

- There is **no mainline Linux driver** and no public register-level
  documentation for the AX8850's PCIe transport.
- The host-side kernel module (`axclhost`, the "AXCL host driver") ships
  only as a **closed-source DKMS binary package** from Axera/M5Stack apt
  repos. Open parts are user-space only: `AXERA-TECH/axcl-docs` (API docs,
  PCIe-EP product context), `AXERA-TECH/axcl-samples` (inference API
  samples), `AXERA-TECH/ax-llm` (LLM deployment stack that runs ON the
  card's A55 cores).
- A native-driver request for this card was explicitly declined upstream
  (TrueNAS forums, "[Not Accepted] Native Kernel Driver Support for Axera
  AX8850"); community compatibility work (Geerling's
  raspberry-pi-pcie-devices #770/#771, Home Assistant #1225) is all built
  on the closed DKMS driver under Linux.

## Consequence for a bare-metal unikernel

MarkOS cannot reuse the Linux driver, and the AXCL PCIe protocol (BAR
layout, doorbell/command-queue semantics) is undocumented. Supporting the
card means one of:

1. **Reverse-engineer the AXCL transport** from the DKMS `.ko` binary and
   PCIe bus traces (QEMU/virtium or hardware logic analyzer). Feasible but
   a large, fragile effort — and it must be validated against real
   hardware, which we do not have in the dev loop.
2. **Treat the card as an autonomous inference node**: the card runs its
   own SDK with ax-llm on its 8 A55 cores; MarkOS would "only" need PCIe
   link-up plus whatever network/IPC surface the card exposes. Still
   undocumented protocol work, and it makes the card the actual LLM engine
   (not MarkOS).

## Decision

- **MarkOS' primary inference path is the Pi 5 CPU itself** (Cortex-A76
  NEON UDOT/SDOT int8, per-core execution pool) — already implemented and
  acceptance-tested. The 16 GB of Pi LPDDR4X is the model memory. This
  keeps the unikernel's promise: minimal overhead, no OS beneath, full
  hardware control, and it works with or without the card.
- **Pi-7a (landed now): PCIe ECAM enumeration** (`kernel/src/pcie.rs`) —
  generic config-space walk, acceptance-tested against QEMU virt's PCIe
  root port. This is the foundation any AX8850 transport needs (find the
  card, read its BARs and link capabilities).
- **Pi-7b/Pi-8 (hardware-gated):** on the real Pi 5 + card, enumerate the
  AX8850 (vendor/device ID probe), then decide between (1) and (2) using
  bus captures from the working Linux setup (axclhost + axcl-samples
  under Raspberry Pi OS produce the ground-truth protocol trace we can
  reimplement bare-metal).

## Sources

- [M5Stack AI-8850 kit product page](https://shop.m5stack.com/products/ai-8850-llm-acceleration-m-2-kit-4gb-version-ax8850)
- [CNX Software: M5Stack LLM-8850 card analysis](https://www.cnx-software.com/2025/10/03/m5stack-llm-8850-card-an-m-2-m-key-ai-accelerator-module-based-on-axera-ax8850-24-tops-soc/)
- [Geerling raspberry-pi-pcie-devices discussion #770](https://github.com/geerlingguy/raspberry-pi-pcie-devices/discussions/770)
- [AXERA-TECH/axcl-docs](https://github.com/AXERA-TECH/axcl-docs) — AXCL host API docs for AX650N/AX8850 PCIe EP products
- [Home Assistant discussion #1225](https://github.com/orgs/home-assistant/discussions/1225)
