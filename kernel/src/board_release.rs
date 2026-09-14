//! Per-board core release implementations.
//!
//! Selected by the same board features as board.rs. Each `release_core`
//! wakes exactly one parked core and makes it enter `secondary_entry`
//! (physical address) with MMU off.

#[cfg(feature = "board-virt")]
mod imp {
    
    /// PSCI CPU_ON — QEMU virt implements PSCI 0.2 over HVC.
    pub fn release_core(core: u64, entry: u64) {
        let status = crate::psci::cpu_on(core, entry, 0);
        if status != 0 {
            // Soundness: none — a failed CPU_ON is logged, never ignored.
            crate::uart::locked_write(format_args!("board: psci cpu_on(core {core}) status {status}\n"));
        }
    }
}

#[cfg(feature = "board-pi")]
mod imp {
    use core::arch::asm;

    /// Pi firmware release: publish the entry address into the ARM64
    /// spin-table slot for this core (Pi 4/5 armstub + QEMU raspi machines)
    /// and the legacy BCM2837 mailbox (Pi 3-era). Writing both covers every
    /// known Pi boot path; a core responds to whichever mechanism parked it.
    pub fn release_core(core: u64, entry: u64) {
        // 1. ARM64 spin-table slot at 0xD8 + 0x10*core (Pi 4/5 armstub).
        let slot = (0xD8 + 0x10 * core) as *mut u64;
        // 2. BCM2837 core mailboxes: several layouts exist across firmware
        //    revisions and QEMU's AArch32 stub (which reads 0x400000CC +
        //    0x10*core). Write every candidate — a core that doesn't use a
        //    candidate simply ignores it, and all targets are reserved
        //    MMIO/RAM in the identity map (Device-mapped, 32-bit regs).
        let candidates = [
            0x4000_008C + 0x10 * core, // BCM2836/7 write-to-set (classic)
            0x4000_00CC + 0x10 * core, // QEMU raspi3b stub read address
            0x4000_008C + 0x04 * core, // 4-byte-stride variant
        ];
        // Soundness: all targets are reserved platform MMIO in the identity
        // map; volatile writes are the defined access pattern.
        unsafe {
            core::ptr::write_volatile(slot, entry);
            for c in candidates {
                (c as *mut u32).write_volatile(entry as u32);
            }
            asm!("dsb sy", options(nostack, preserves_flags));
            asm!("sev", options(nomem, nostack, preserves_flags));
        }
    }
}

pub use imp::release_core;
