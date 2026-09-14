//! PSCI (Power State Coordination Interface) — ARM's standard boot/release
//! protocol, implemented by QEMU's `virt` machine and by Pi UEFI firmware.
//!
//! owns: nothing; PSCI is firmware-provided.
//! invariants: conduit is HVC (QEMU virt default); call only after EL
//! normalization (EL1+ required for hvc to reach the right firmware).

use core::arch::asm;

/// PSCI 0.2 SMC32 function ID: CPU_ON.
const FID_CPU_ON: u64 = 0x8400_0003;
/// PSCI 0.2 SMC32 function ID: SYSTEM_OFF.
const FID_SYSTEM_OFF: u64 = 0x8400_0008;

/// Turn a core on: it enters at `entry` (physical) in AArch64 with MMU off,
/// caches off, x0 = context. Returns the PSCI status (0 = success).
pub fn cpu_on(target_mpidr: u64, entry: u64, context: u64) -> i64 {
    let status: i64;
    // Soundness: HVC #0 traps to the firmware's PSCI dispatcher; arguments
    // and return value follow the PSCI calling convention.
    unsafe {
        asm!(
            "hvc #0",
            in("x0") FID_CPU_ON,
            in("x1") target_mpidr,
            in("x2") entry,
            in("x3") context,
            lateout("x0") status,
            options(nostack)
        );
    }
    status
}

/// Power the system off (QEMU virt exits). Acceptance gates call this after
/// their PASS marker so the run ends immediately instead of waiting out the
/// timeout while the kernel parks.
pub fn system_off() -> ! {
    // Soundness: HVC #0 with the SYSTEM_OFF FID per the PSCI convention;
    // never returns on success.
    unsafe {
        asm!(
            "hvc #0",
            in("x0") FID_SYSTEM_OFF,
            options(noreturn, nostack)
        );
    }
}
