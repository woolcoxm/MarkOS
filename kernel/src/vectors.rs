//! AArch64 exception handling: VBAR_EL1 vector table + shared trap handler.
//!
//! ARM has 16 vector slots (4 exception classes × 4 origins), each 128 bytes.
//! All slots branch to one shared entry that saves the full register file,
//! reads ESR_EL1/FAR_EL1/ELR_EL1/SPSR_EL1, and calls `exception_common`.
//!
//! owns: the 2 KiB-aligned vector table and the exception stack frame layout.
//! invariants: `init` runs once per core with the MMU configuration that
//! identity-maps this image; the handler is allowed to adjust ELR (e.g. to
//! skip a `brk`) and returns the new ELR; anything unrecoverable parks.

use core::arch::global_asm;
use core::fmt::Write as _;

use crate::uart;

global_asm!(
    ".align 11",                 // 2 KiB: 16 vectors × 128 bytes
    ".globl __vectors_start",
    "__vectors_start:",
    // Each of the 16 slots must be exactly 128 bytes: the CPU indexes
    // VBAR + exception_class*0x80 + origin*0x80, so missing stride padding
    // makes it land mid-code (this bit us once — see git history).
    ".rept 16",
    "    b vector_common",
    "    .space 124",
    ".endr",
    "",
    "vector_common:",
    // Frame: 288 bytes — x0..x29 (pairs), x30, esr, far, elr, spsr.
    "    sub sp, sp, #288",
    "    stp x0, x1, [sp, #0]",
    "    stp x2, x3, [sp, #16]",
    "    stp x4, x5, [sp, #32]",
    "    stp x6, x7, [sp, #48]",
    "    stp x8, x9, [sp, #64]",
    "    stp x10, x11, [sp, #80]",
    "    stp x12, x13, [sp, #96]",
    "    stp x14, x15, [sp, #112]",
    "    stp x16, x17, [sp, #128]",
    "    stp x18, x19, [sp, #144]",
    "    stp x20, x21, [sp, #160]",
    "    stp x22, x23, [sp, #176]",
    "    stp x24, x25, [sp, #192]",
    "    stp x26, x27, [sp, #208]",
    "    stp x28, x29, [sp, #224]",
    "    str x30, [sp, #240]",
    "    mrs x0, esr_el1",
    "    str x0, [sp, #256]",
    "    mrs x1, far_el1",
    "    str x1, [sp, #264]",
    "    mrs x2, elr_el1",
    "    str x2, [sp, #272]",
    "    mrs x3, spsr_el1",
    "    str x3, [sp, #280]",
    "    mov x0, sp",
    "    bl exception_common",
    // Handler returns the (possibly adjusted) ELR in x0.
    "    msr elr_el1, x0",
    "    ldp x0, x1, [sp, #0]",
    "    ldp x2, x3, [sp, #16]",
    "    ldp x4, x5, [sp, #32]",
    "    ldp x6, x7, [sp, #48]",
    "    ldp x8, x9, [sp, #64]",
    "    ldp x10, x11, [sp, #80]",
    "    ldp x12, x13, [sp, #96]",
    "    ldp x14, x15, [sp, #112]",
    "    ldp x16, x17, [sp, #128]",
    "    ldp x18, x19, [sp, #144]",
    "    ldp x20, x21, [sp, #160]",
    "    ldp x22, x23, [sp, #176]",
    "    ldp x24, x25, [sp, #192]",
    "    ldp x26, x27, [sp, #208]",
    "    ldp x28, x29, [sp, #224]",
    "    ldr x30, [sp, #240]",
    "    add sp, sp, #288",
    "    eret",
);

/// Offsets into the exception frame built by `vector_common`.
const OFF_ESR: usize = 256;
const OFF_FAR: usize = 264;
const OFF_ELR: usize = 272;
const OFF_SPSR: usize = 280;

/// ESR_EL1.Exception Class (bits 31:26) values we distinguish.
const EC_BRK: u64 = 0x3C;
const EC_DATA_ABORT_SAME_EL: u64 = 0x25;
const EC_INSTR_ABORT_SAME_EL: u64 = 0x21;

/// Install the vector table and mask all interrupt sources (DAIF).
pub fn init() {
    // Soundness: extern static holding its own address; only the address is read.
    let table = &raw const __vectors_start as usize as u64;
    // Soundness: the table is a static, 2 KiB-aligned array in the image;
    // VBAR_EL1 simply points the CPU at it.
    unsafe {
        core::arch::asm!(
            "msr vbar_el1, {table}",
            "isb",
            "msr daifset, #0xF",
            table = in(reg) table,
            options(nomem, nostack)
        );
    }
}

unsafe extern "C" {
    static __vectors_start: u8;
}

/// Shared trap entry: classify, log, and either resume (returns new ELR) or
/// park forever for unrecoverable faults.
#[unsafe(no_mangle)]
extern "C" fn exception_common(frame: *mut u64) -> u64 {
    // Soundness: `frame` is the stack area the asm stub prepared; it stays
    // valid for the duration of this call.
    let (esr, far, elr, _spsr) = unsafe {
        (
            frame.add(OFF_ESR / 8).read_volatile(),
            frame.add(OFF_FAR / 8).read_volatile(),
            frame.add(OFF_ELR / 8).read_volatile(),
            frame.add(OFF_SPSR / 8).read_volatile(),
        )
    };
    let ec = (esr >> 26) & 0x3F;

    match ec {
        EC_BRK => {
            let _ = write!(
                uart::Serial,
                "CAUGHT exception: brk elr={:#x}\n",
                elr
            );
            elr + 4 // resume past the brk instruction
        }
        EC_DATA_ABORT_SAME_EL => {
            let _ = write!(
                uart::Serial,
                "CAUGHT exception: data_abort far={:#x} elr={:#x} iss={:#x}\n",
                far,
                elr,
                esr & 0xFFFF
            );
            crate::park()
        }
        EC_INSTR_ABORT_SAME_EL => {
            let _ = write!(
                uart::Serial,
                "CAUGHT exception: instr_abort far={:#x} elr={:#x}\n",
                far,
                elr
            );
            crate::park()
        }
        _ => {
            let _ = write!(
                uart::Serial,
                "CAUGHT exception: unknown ec={ec:#x} far={far:#x} elr={elr:#x}\n"
            );
            crate::park()
        }
    }
}
