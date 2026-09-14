//! CPU feature detection (AArch64 architectural ID registers).
//!
//! The deployment board is the Pi 5 (Cortex-A76, ARMv8.2): its UDOT/SDOT
//! int8 dot-product instructions are the core of the quantized matmul
//! kernels, and its FP16 arithmetic accelerates half-precision paths.
//! Detection is feature-based (ID_AA64ISAR0_EL1 fields), not model-based,
//! so any future board with the right features is handled correctly.

use core::arch::asm;
use core::fmt::Write as _;

use crate::uart;

/// Main ID register: implementer, variant, part number, revision.
pub fn midr() -> u64 {
    let v: u64;
    // Soundness: read-only system register, readable at EL1.
    unsafe { asm!("mrs {}, midr_el1", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// Instruction set attribute register 0.
pub fn isar0() -> u64 {
    let v: u64;
    // Soundness: read-only system register.
    unsafe { asm!("mrs {}, id_aa64isar0_el1", out(reg) v, options(nomem, nostack, preserves_flags)) };
    v
}

/// CPU part number (MIDR bits 15:4).
pub fn part_number() -> u16 {
    ((midr() >> 4) & 0xFFF) as u16
}

/// Human-readable core name for known parts.
pub fn core_name() -> &'static str {
    match part_number() {
        0xD03 => "Cortex-A53",
        0xD04 => "Cortex-A35",
        0xD07 => "Cortex-A57",
        0xD08 => "Cortex-A72",
        0xD09 => "Cortex-A73",
        0xD0A => "Cortex-A75",
        0xD0B => "Cortex-A76",
        0xD40 => "Neoverse-V1",
        0xD49 => "Neoverse-N2",
        _ => "unknown",
    }
}

/// ARMv8.2 int8 dot-product (SDOT/UDOT) available — required by the
/// quantized matmul kernels' fast path.
pub fn has_dotprod() -> bool {
    (isar0() >> 44) & 0xF != 0
}

/// ARMv8.2 FP16 arithmetic available.
pub fn has_fp16() -> bool {
    (isar0() >> 20) & 0xF != 0
}

/// Boot-log line: what silicon are we optimizing for?
pub fn log() {
    let _ = write!(
        uart::Serial,
        "cpu: {} (part {:#03x}) dotprod={} fp16={}\n",
        core_name(),
        part_number(),
        if has_dotprod() { "yes" } else { "no" },
        if has_fp16() { "yes" } else { "no" }
    );
    if !has_dotprod() {
        uart::write_str(
            "cpu: NOTE int8 UDOT fast path unavailable on this core; inference kernels\n",
        );
        uart::write_str(
            "cpu: will need the scalar fallback (performance target is Pi 5 / A76).\n",
        );
    }
}
