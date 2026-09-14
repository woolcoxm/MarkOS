# Pi-7b-1 — The AX8850 PCIe transport, from the vendor's own source

Status: **transport architecture fully known — it is open source.** The
M5Stack `axclhost` package is DKMS, so it ships the complete kernel-driver
source; the "closed transport" assumption from the earlier research is
obsolete. This document maps what ships where and what MarkOS ports.

## The artifact

- `axclhost_3.6.5-m5stack1_arm64.deb` —
  `https://repo.llm.m5stack.com/m5stack-apt-repo/pool/axclhost/binary-arm64/axclhost_3.6.5-m5stack1_arm64.deb`
  SHA256 `119caaffa453d664ddc0bc76e53bda9cee67e6458dd3dffacb02eedb6bf35837`
  (matched to the card firmware the RE repo verified — `ax650_card.pac`
  ships in the same package).
- Extracted locally to `../Axera-refs/axclhost-root/` (NOT committed to
  this repo: the driver carries an Axera proprietary copyright notice;
  this doc references it by path instead of copying it).
- Contents that matter here:
  - `usr/src/axclhost-1.0/drv/pcie/driver/` — **kernel driver source**
    (14 .c / 9 .h, ~7.6 kLOC)
  - `usr/include/axcl/*.h` — full AXCL API headers (structs, enums,
    prototypes for every axclrt* call)
  - `usr/lib/axcl/libaxcl_*.so` — the closed userspace runtime (the only
    genuinely closed piece left)
  - `usr/bin/axcl/` — `axcl-smi`, `axcl_run_model`, samples (trace targets)

## Driver module map

| Module | Source | Char dev | Role |
|---|---|---|---|
| ax_pcie_host_dev | `host_dev/` | — | EP device abstraction: BAR map, iATU windows, handshake, ops table |
| ax_pcie_msg | `msg/` + `host_dev/ax_pcie_msg_transfer.c` | `/dev/msg_userdev` | control/command channel: kfifo rings + mailbox doorbells |
| ax_pcie_mmb | `mmb/` | `/dev/ax_mmb_dev` | card-memory (CMM) allocator: alloc/cached/flush/invalidate ioctls |
| axcl_host | `axcl_host/axcl_pcie_host.c` | `/dev/axcl_host` | main PCI probe (vendor `0x1f4b`, device `0x0650`), BAR/DMA bring-up |
| ax_pcie_p2p_rc, rc-net | `p2p_rc/`, `net/` | `/dev/p2p` | multi-card + PCIe networking — out of MarkOS scope |

## Transport architecture (constants from `ax_pcie_dev.h` / `ax_pcie_msg_transfer.h`)

- **Card-side shared memory**: `PCIE_SPACE_SHMEM_BASE 0x51000000`,
  768 KiB per device — split 384 KiB send + 384 KiB recv rings
  (`AX_SHARED_{SEND,RECV}MEM_SIZE`), each ring owning 128 KiB of that
  region for IRQ-coupled slots.
- **Mailbox (doorbell) registers** on the card:
  `MAILBOX_REG_MAP_ADDR 0x4520000`, register block `+0xC000`; slot
  request/unlock and int stats/clear regs selected by a "master id" —
  the PCIe host uses master id 5 (`PCIE_MASTERID`). A doorbell message is
  32 bytes: `{src_target_id, customer_data[7]}`.
- **Message framing**: 24-byte header
  `{target_id, slot, port, magic, length, check}` + payload, 32-byte
  aligned; rings are kfifo-shaped (`in/out/mask/data` —
  `struct pcie_kfifo`); up to 128 message ports (`MAX_MSG_PORTS 0x80`);
  normal messages 128 B, max 0x1000 B.
- **Boot handshake** (`ax_pcie_msg_transfer.c` ~line 795-910): host writes
  `DEVICE_CHECKED_FLAG 0x1F4B` to the card SHMEM base → polls until the
  card firmware answers `DEVICE_HANDSHAKE_FLAG 0xA650` → both sides init
  their kfifo rings (`DEVICE_KFIFO_INIT 0xa5a5`) → channels live.
- **Userspace interface**: `/dev/msg_userdev` ioctls
  (`AX_MSG_IOC_CONNECT/CHECK/GET_{LOCAL,REMOTE}_ID/ATTR_INIT/RESET_DEVICE/PCIE_STOP`,
  magic `'M'`, attr `{target_id, port, remote_id[]}`); read/write on the
  device moves message payloads. `/dev/ax_mmb_dev` ioctls (magic `'H'`):
  `ALLOC_MEMORY`, `ALLOC_MEMCACHED`, `FLUSH_CACHED`, `INVALID_CACHED`,
  `SCATTERLIST_ALLOC`, `GET/PUT_MEM_ENTRY` — a card-RAM allocator with
  explicit cache maintenance, exactly the axclrtMalloc/MemFlush/MemInvalidate
  semantics from the AXCL docs.
- The EP's own side runs on an RISC-V MCU (`mmb/rv64_cache.h`); the
  card firmware implements the mailbox + ring protocol and the higher-level
  axcl command dispatch.

## What this changes for MarkOS

The plan's "static binary RE" phase is gone. The remaining unknowns are
now bounded and tractable:

1. **Command payload semantics** (the one closed piece): what
   `libaxcl_rt.so` puts inside msg-channel frames for axclrtMalloc /
   EngineLoadFromMem / Execute etc. Capturable at the open kernel
   boundary — a printk patch on `ax_pcie_msg_transfer.c` (or `/dev/
   msg_userdev` read/write tracing) on the user's Linux rig while running
   `axcl_run_model` dumps every command frame. The AXCL API docs define
   what each command must accomplish.
2. **Pi 5 host side**: the driver's host ops (iATU programming, BAR map,
   doorbell writes) target the Synopsys DWC EP on the card; the MarkOS
   side only needs the BCM2712 root complex to map the card's BARs and
   route its MSI/IRQ — then the handshake + rings are direct MMIO, ported
   from the open source.

## MarkOS port sketch (axcl-lite → bare metal)

| Linux driver piece | MarkOS equivalent |
|---|---|
| `axcl_pcie_host.c` PCI probe | `pcie.rs` ECAM walk (done, Pi-7a) + BAR mapping |
| iATU/window setup in `host_dev/` | mostly EP-side (card-internal); host only maps BARs |
| kfifo rings + `ax_pci_transfer_head` | direct port to `axcl.rs` (no_std, 32-byte aligned frames) |
| mailbox doorbell (`trigger_msg_irq`) | MMIO write to card BAR window per the header constants |
| card IRQ → `host_message_irq_handler` | BCM2712 GIC IRQ/MSI from the PCIe link |
| `/dev/ax_mmb_dev` allocator ioctls | in-kernel calls — no user/kernel boundary exists |
| `/dev/msg_userdev` read/write | direct function calls into `axcl.rs` |

## Next actions (Pi-7b)

- 7b-1c: capture command frames on the rig (instrumented
  `ax_pcie_msg_transfer.c` while `axcl_run_model` executes one engine) →
  command dictionary into this doc.
- 7b-2 (hardware): BCM2712 PCIe RC bring-up in MarkOS; ECAM walk must show
  `1f4b:0650`; map BARs; run the handshake; print card firmware version.
- 7b-3: one malloc + one engine upload + one Execute through bare-metal
  rings; golden-compare against the RE repo's `engine_dump.c` outputs.
