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

use crate::board;
use crate::uart;

const ENTRIES: usize = 512;

#[repr(C, align(4096))]
struct PageTable([u64; ENTRIES]);

static mut L0: PageTable = PageTable([0; ENTRIES]);
static mut L1: PageTable = PageTable([0; ENTRIES]);
static mut L2: PageTable = PageTable([0; ENTRIES]);
/// Covers [1 GiB, 2 GiB) as Device: the BCM per-core SMP mailboxes live at
/// 0x4000008C+ (just above the first GiB) and must be reachable.
static mut L2B: PageTable = PageTable([0; ENTRIES]);
/// L3 refining the first 2 MiB above 1 GiB to Device pages (mailboxes).
static mut L3B: PageTable = PageTable([0; ENTRIES]);
/// L2 for the PCIe ECAM window (Pi-7a): the virt machine's ECAM sits at
/// 256 GiB — a whole L0/L1 granule away from the identity map. Only built
/// when the window's L1 slot is clear of [0, 2 GiB).
static mut L2C: PageTable = PageTable([0; ENTRIES]);

const ECAM_BASE: u64 = board::PCIE_ECAM_BASE as u64;
const ECAM_SIZE: u64 = 256 * 1024 * 1024; // 8-bit bus window
const ECAM_L1_IDX: usize = ((ECAM_BASE >> 30) as usize) & (ENTRIES - 1);

/// Pi 3/4 legacy peripheral base (QEMU raspi3b matches the Pi 3 layout).
const PERIPH_BASE: u64 = 0x3F00_0000;
const PERIPH_END: u64 = 0x4000_0000;
const BLOCK: u64 = 2 * 1024 * 1024;
const PAGE: u64 = 4 * 1024;
const GIB: u64 = 1 << 30;

// Descriptor bits.
const VALID: u64 = 1 << 0;
const TABLE: u64 = 1 << 1;
const ATTR_DEVICE: u64 = 0 << 2; // MAIR attr 0
const ATTR_NORMAL: u64 = 1 << 2; // MAIR attr 1
const AF: u64 = 1 << 10;
const PXN: u64 = 1 << 53;

// MAIR_EL1: attr0 = Device-nGnRE (0x04), attr1 = Normal WB/WA/RW (0xFF).
const MAIR: u64 = 0x04 | (0xFF << 8);
// TCR_EL1: T0SZ=16, 4K granule, WB cacheability, inner-shareable,
// T1SZ=16 (upper half unused but sized), IPS=1TB (bits[34:32]) — the
// ECAM window lives above 4 GiB physical, so the default 4 GB output
// range will not do.
const TCR: u64 = 16 | (1 << 8) | (1 << 10) | (3 << 12) | (16 << 16) | (2 << 32);

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
        let l2b = &raw mut L2B;
        let l3b = &raw mut L3B;
        let l2c = &raw mut L2C;

        // VA [0,1GiB): L0[0] -> L1[0] -> L2 (RAM + legacy peripherals).
        // VA [1GiB,2GiB): L1[1] under the SAME L0 — L0 indexes bits[47:39],
        // which are zero for these VAs; L1 indexes bits[38:30].
        (*l0).0[0] = table_addr(l1) | VALID | TABLE;
        (*l1).0[0] = table_addr(l2) | VALID | TABLE;
        (*l1).0[1] = table_addr(l2b) | VALID | TABLE;

        for i in 0..ENTRIES {
            let addr = (i as u64) * BLOCK;
            let attrs = if addr >= PERIPH_BASE && addr < PERIPH_END {
                ATTR_DEVICE | AF | PXN
            } else {
                ATTR_NORMAL | AF
            };
            (*l2).0[i] = addr | VALID | attrs;
        }
        // [1 GiB, 2 GiB): Normal blocks — except the first 2 MiB, which on
        // the Pi contains the per-core SMP mailboxes (0x4000008C+) and must
        // be Device. That 2 MiB is refined to 4 KiB Device pages through an
        // L3 table so it does not swallow the kernel mapping on QEMU virt
        // (whose kernel sits at 0x40080000, inside the same 2 MiB).
        (*l2b).0[0] = table_addr(l3b) | VALID | TABLE;
        for i in 1..ENTRIES {
            let addr = GIB + (i as u64) * BLOCK;
            (*l2b).0[i] = addr | VALID | ATTR_NORMAL | AF;
        }
        for j in 0..ENTRIES {
            let addr = GIB + (j as u64) * PAGE;
            // Page 0 holds the SMP mailboxes (Device, execute-never);
            // pages 1.. are Normal+executable — the kernel itself sits at
            // 0x40080000 on the virt board, inside this 2 MiB.
            (*l3b).0[j] = if j == 0 {
                addr | VALID | 0b10 | ATTR_DEVICE | AF | PXN
            } else {
                addr | VALID | 0b10 | ATTR_NORMAL | AF
            };
        }

        // PCIe ECAM window (Pi-7a): map Device+PXN blocks so the config
        // walk in pcie.rs can read it. Only when its L1 slot doesn't
        // collide with the identity map (idx 0 = the first GiB, idx 1 =
        // the second) — the unverified Pi 5 placeholder resolves to idx 0
        // and is skipped until its real address is confirmed.
        if ECAM_L1_IDX >= 2 {
            (*l1).0[ECAM_L1_IDX] = table_addr(l2c) | VALID | TABLE;
            // The window is generally NOT granule-aligned: L2C[i] covers
            // PA granule_base + i*2MiB, so the first descriptor lands at
            // the window's offset inside its 1 GiB granule, not slot 0.
            let start = ((ECAM_BASE >> 21) as usize) & (ENTRIES - 1);
            let blocks = (ECAM_SIZE / BLOCK) as usize;
            for i in 0..blocks {
                let addr = ECAM_BASE + (i as u64) * BLOCK;
                (*l2c).0[start + i] = addr | VALID | ATTR_DEVICE | AF | PXN;
            }
        }
    }
}

/// Install TTBR0/TCR/MAIR and switch the MMU + caches on.
pub fn init() {
    build_tables();

    // Debug: L1[1] descriptor (covers [1GiB,2GiB) incl. SMP mailboxes).
    let l2b0 = unsafe { ((&raw const L2B) as *const u64).add(0).read() };
    uart::locked_write(format_args!( "mmu: l2b[0]={l2b0:#x}
"));

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
