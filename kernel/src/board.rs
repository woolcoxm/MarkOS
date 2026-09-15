//! Board description layer: per-board constants, selected by cargo feature.
//!
//! Two boards are supported:
//!  - `board-pi`  (default): Raspberry Pi (QEMU raspi3b dev machine and real
//!    Pi 4/5 hardware). Legacy peripheral window, spin-table/mailbox release.
//!  - `board-virt`: QEMU `virt` machine — architecturally standard board
//!    (PL011 at 0x09000000, GIC-400, PSCI CPU_ON release). Used for the
//!    automated SMP/GIC acceptance tests.
//!
//! invariants: exactly one board feature must be enabled.

#[cfg(all(feature = "board-virt", feature = "board-pi"))]
compile_error!("select exactly one board feature");

#[cfg(feature = "board-virt")]
pub const UART_BASE: usize = 0x0900_0000;
#[cfg(feature = "board-virt")]
pub const CORE_COUNT: usize = 4;
#[cfg(feature = "board-virt")]
pub const NAME: &str = "qemu-virt";
/// Generic ECAM config space of the virt machine's PCIe host (gpex).
/// QEMU 8.2's highmem virt places it at 256 GiB + 256 MiB (verified via
/// `info mtree`: "pcie-ecam @ 0000004010000000-000000401fffffff").
#[cfg(feature = "board-virt")]
pub const PCIE_ECAM_BASE: usize = 0x0000_0040_1000_0000;
/// RAM weight cache (Phase perf): QEMU virt booted with `-m 2048` has RAM
/// [0x4000_0000, 0xC000_0000); the kernel + statics live in the first
/// ~16 MiB, so a fixed window above 0x4800_0000 is free for the loaded
/// model image. The identity map covers it (Normal, cacheable).
#[cfg(feature = "board-virt")]
pub const WEIGHT_RAM_BASE: usize = 0x4800_0000;
#[cfg(feature = "board-virt")]
pub const WEIGHT_RAM_SIZE: usize = 704 * 1024 * 1024;

#[cfg(feature = "board-pi")]
pub const UART_BASE: usize = 0x3F20_1000;
#[cfg(feature = "board-pi")]
pub const CORE_COUNT: usize = 4;
#[cfg(feature = "board-pi")]
pub const NAME: &str = "raspi (pi3-qemu / pi4-pi5 hw)";
/// Pi 5 (BCM2712): PCIe3 root complex ECAM window. UNVERIFIED placeholder —
/// confirm against the BCM2712 memory map before any Pi 5 PCIe work
/// (Pi-7b, hardware-gated). Kept below 2 GiB-shifted so the generic mmu
/// guard skips building a mapping for it.
#[cfg(feature = "board-pi")]
pub const PCIE_ECAM_BASE: usize = 0x0000_1000_1200_0000;
/// Pi 5 16GB weight cache: fixed DRAM window at 1 GiB, inside the already
/// identity-mapped [0, 2 GiB) region (Normal cacheable 2 MiB blocks).
///
/// Placement rationale:
/// - kernel image + statics end well below 0x1000_0000 (16 MiB)
/// - 0x4000_0000 (1 GiB) is safely above any firmware reservation
/// - 768 MiB covers Qwen3-0.6B q8_0 (640 MB) with 20% headroom
/// - no additional page tables needed (identity map already covers it)
/// - cache fill is blocked on SDHCI (Pi-8) for real SD access
#[cfg(feature = "board-pi")]
pub const WEIGHT_RAM_BASE: usize = 0x4000_0000;
#[cfg(feature = "board-pi")]
pub const WEIGHT_RAM_SIZE: usize = 768 * 1024 * 1024;
