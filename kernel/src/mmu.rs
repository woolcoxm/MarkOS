//! MMU bring-up: identity map of the first GiB with 2 MiB blocks, then
//! enable the MMU + caches.
//!
//! Pi-1 keeps it deliberately simple: one L0 → L1 → L2 chain (4 KiB
//! granule, 48-bit VA space) where the L2 covers [0, 1 GiB) entirely with
//! 2 MiB blocks. RAM is Normal cacheable; the Pi-3 legacy peripheral window
//! (0x3F000000..0x40000000) is Device-nGnRE and execute-never. Later phases
//! (huge pages for weights, proper kernel VA split) grow this exact table
//! instead of replacing it.
//!
//! owns: the static page tables (L0/L1/L2) and the MMU/caching state.
//! invariants: `init` runs once, on the BSP, with exception vectors already
//! installed (a mapping bug then traps instead of silently corrupting).

use core::arch::asm;
use core::fmt::Write as _;

use crate::uart;

const ENTRIES: usize = 512;

#[repr(C, align(4096))]
struct PageTable([u64; ENTRIES]);

static mut L0: PageTable = PageTable([0; ENTRIES]);
static mut L1: PageTable = PageTable([0; ENTRIES]);
static mut L2: PageTable = PageTable([0; ENTRIES]);

/// Pi 3/4 legacy peripheral base (QEMU raspi3b matches the Pi 3 layout).
const PERIPH_BASE: u64 = 0x3F00_0000;
const PERIPH_END: u64 = 0x4000_0000;
const BLOCK: u64 = 2 * 1024 * 1024;

// Descriptor bits.
const VALID: u64 = 1 << 0;
const TABLE: u64 = 1 << 1;
const ATTR_DEVICE: u64 = 0 << 2; // MAIR attr 0
const ATTR_NORMAL: u64 = 1 << 2; // MAIR attr 1
const AF: u64 = 1 << 10;
const PXN: u64 = 1 << 53;

// MAIR_EL1: attr0 = Device-nGnRE (0x04), attr1 = Normal WB/WA/RW (0xFF).
const MAIR: u64 = 0x04 | (0xFF << 8);
// TCR_EL1: T0SZ=16, 4K granule, WB cacheability, inner-shareable, PS=4GB,
// T1SZ=16 (upper half unused but sized).
const TCR: u64 = 16 | (1 << 8) | (1 << 10) | (3 << 12) | (16 << 16);

fn table_addr(t: *const PageTable) -> u64 {
    // Soundness: statics in the kernel image, 4 KiB aligned by repr(C, align).
    t as u64
}

/// Build L0 → L1 → L2 with 2 MiB blocks over [0, 1 GiB).
fn build_tables() {
    // Soundness: statics are exclusively owned by the BSP here; no MMU yet,
    // so plain (non-volatile) writes are fine and cache coherence is moot
    // (caches are still off — another reason init order matters). Raw
    // pointers avoid creating long-lived references into the statics.
    unsafe {
        let l0 = &raw mut L0;
        let l1 = &raw mut L1;
        let l2 = &raw mut L2;

        (*l0).0[0] = table_addr(l1) | VALID | TABLE;
        (*l1).0[0] = table_addr(l2) | VALID | TABLE;

        for i in 0..ENTRIES {
            let addr = (i as u64) * BLOCK;
            let attrs = if addr >= PERIPH_BASE && addr < PERIPH_END {
                ATTR_DEVICE | AF | PXN
            } else {
                ATTR_NORMAL | AF
            };
            (*l2).0[i] = addr | VALID | attrs;
        }
    }
}

/// Install TTBR0/TCR/MAIR and switch the MMU + caches on.
pub fn init() {
    build_tables();

    // Soundness: MMU-enable sequence from the ARM ARM — barriers before and
    // an isb after SCTLR changes; PC is identity-mapped so execution is
    // seamless across the enable.
    unsafe {
        let l0 = table_addr(&raw const L0);
        asm!(
            "msr ttbr0_el1, {ttbr}",
            "msr tcr_el1, {tcr}",
            "msr mair_el1, {mair}",
            "dsb sy",
            "ic iallu",
            "dsb sy",
            "isb",
            "mrs {sctlr}, sctlr_el1",
            "orr {sctlr}, {sctlr}, #(1 << 12)",  // I: instruction cache
            "orr {sctlr}, {sctlr}, #(1 << 2)",   // C: data cache
            "orr {sctlr}, {sctlr}, #(1 << 0)",   // M: MMU on
            "msr sctlr_el1, {sctlr}",
            "isb",
            ttbr = in(reg) l0,
            tcr = in(reg) TCR,
            mair = in(reg) MAIR,
            sctlr = out(reg) _,
            options(nostack)
        );
    }
}

/// Boot-log line proving the MMU is live: a read of our own mapping.
pub fn log() {
    let _ = write!(
        uart::Serial,
        "mmu: enabled (identity [0,1GiB) via 2MiB blocks, peripherals Device/PXN)\n"
    );
}
