//! D-cache maintenance for cross-core data sharing.
//!
//! The APs run with their MMU off (uncached accesses) while the BSP uses
//! the cacheable identity map. Any buffer handed from the BSP to the APs
//! must be CLEANED (written back) first, and any buffer written by the APs
//! must be INVALIDATED on the BSP before its reads — otherwise the stale
//! cache lines mask the fresh RAM contents.
//!
//! owns: nothing; operates on caller-provided ranges.
//! invariants: cache line size 64 bytes (ARMv8 architected minimum; read
//! CTR_EL1.DminLine for exact size on real hardware).

use core::arch::asm;

const CACHE_LINE: usize = 64;

/// Clean (write back) dirty D-cache lines covering [start, start+len).
pub fn clean_range(start: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut addr = start & !(CACHE_LINE - 1);
    let end = start + len;
    // Soundness: dc cvac writes back one cache line at VA; the range is
    // identity-mapped RAM owned by the caller at this point.
    unsafe {
        while addr < end {
            asm!("dc cvac, {a}", a = in(reg) addr, options(nostack));
            addr += CACHE_LINE;
        }
        asm!("dsb sy", options(nostack, preserves_flags));
    }
}

/// Invalidate D-cache lines covering [start, start+len): subsequent reads
/// fetch fresh RAM contents (used after another agent wrote the RAM).
pub fn invalidate_range(start: usize, len: usize) {
    if len == 0 {
        return;
    }
    let mut addr = start & !(CACHE_LINE - 1);
    let end = start + len;
    // Soundness: dc ivac discards the local cached copy; the RAM holds the
    // up-to-date data written by the device/other cores.
    unsafe {
        while addr < end {
            asm!("dc ivac, {a}", a = in(reg) addr, options(nostack));
            addr += CACHE_LINE;
        }
        asm!("dsb sy", options(nostack, preserves_flags));
    }
}
