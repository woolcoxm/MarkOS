//! General-purpose register capture and hex dump for panic/exception logs.
//!
//! Invariant / honest caveat: a register dump taken from Rust code reflects
//! the values at the point of capture (the compiler may already have spilled
//! and reloaded registers), not the exact silicon state at the fault. The
//! authoritative fault context is the CPU-pushed `InterruptStackFrame` printed
//! by the IDT handlers; this dump supplements Rust-level panics.

use core::arch::asm;
use core::fmt::Write as _;

use crate::serial;

#[derive(Clone, Copy)]
pub struct Regs {
    pub rax: u64,
    pub rbx: u64,
    pub rcx: u64,
    pub rdx: u64,
    pub rsi: u64,
    pub rdi: u64,
    pub rbp: u64,
    pub r8: u64,
    pub r9: u64,
    pub r10: u64,
    pub r11: u64,
    pub r12: u64,
    pub r13: u64,
    pub r14: u64,
    pub r15: u64,
}

/// Capture the current values of the GP registers on this core.
#[inline(never)]
pub fn capture() -> Regs {
    let (rax, rcx, rdx, rsi, rdi);
    let (r8, r9, r10, r11, r12, r13, r14, r15);
    let rbx: u64;
    let rbp: u64;
    // Soundness: side-effect-free register reads. rbx/rbp cannot be used as
    // operands directly (LLVM-internal / frame pointer), so they are copied
    // into scratch registers by explicit movs inside the block.
    unsafe {
        asm!(
            "mov {rbx_out}, rbx",
            "mov {rbp_out}, rbp",
            rbx_out = out(reg) rbx,
            rbp_out = out(reg) rbp,
            out("rax") rax, out("rcx") rcx, out("rdx") rdx,
            out("rsi") rsi, out("rdi") rdi,
            out("r8") r8, out("r9") r9, out("r10") r10, out("r11") r11,
            out("r12") r12, out("r13") r13, out("r14") r14, out("r15") r15,
            options(nomem, nostack, preserves_flags)
        );
    }
    Regs { rax, rbx, rcx, rdx, rsi, rdi, rbp, r8, r9, r10, r11, r12, r13, r14, r15 }
}

/// Print `title` plus a 3-per-row register table over serial.
pub fn dump(title: &str) {
    let r = capture();
    let _ = write!(serial::Serial, "registers ({title}):\n");
    let _ = write!(
        serial::Serial,
        "  rax={:#018x} rbx={:#018x} rcx={:#018x}\n  rdx={:#018x} rsi={:#018x} rdi={:#018x}\n",
        r.rax, r.rbx, r.rcx, r.rdx, r.rsi, r.rdi
    );
    let _ = write!(
        serial::Serial,
        "  r08={:#018x} r09={:#018x} r10={:#018x}\n  r11={:#018x} r12={:#018x} r13={:#018x}\n  r14={:#018x} r15={:#018x} rbp={:#018x}\n",
        r.r8, r.r9, r.r10, r.r11, r.r12, r.r13, r.r14, r.r15, r.rbp
    );
}
