//! MarkOS — a bare-metal Raspberry Pi unikernel that boots straight into an
//! LLM inference engine.
//!
//! Pi-1 (this commit): exception level normalization (EL3/EL2 → EL1), the
//! VBAR_EL1 vector table with a full-register shared trap handler, MMU
//! enable over an identity map of the first GiB (2 MiB blocks, caches on),
//! and the generic timer (counter reads + counter-based delays).
//!
//! owns: the kernel entry point and the boot stack.
//! invariants: non-BSP cores stay parked until the SMP phase; exceptions
//! are terminal except the deliberately-resumable `brk` selftest.

#![no_std]
#![no_main]

mod mmu;
mod timer;
mod uart;
mod vectors;

use core::{arch::global_asm, panic::PanicInfo};

global_asm!(
    ".section .text.boot",
    ".globl _start",
    "_start:",
    // NOTE: boot-critical asm must not use `ldr =symbol` literal pools —
    // LLVM places pools in ways the linker script can silently corrupt once
    // more .text blobs exist (seen live: pool landed inside the vector
    // table's NOP padding). PC-relative adr/adrp and encodable movs only.
    "    ldr x4, =0x3F201000",     // (debug) PL011 DR, pre-init write works
    "    mov w5, #83",             // 'S'
    "    str w5, [x4]",
    "    mrs x0, mpidr_el1",
    "    and x0, x0, #3",          // affinity level 0 = core id on the Pi
    "    cbnz x0, parked",
    "    mov w5, #99",             // 'c' passed core check
    "    str w5, [x4]",
    "",
    // Normalize the exception level to EL1. Boards differ: QEMU's raspi
    // machines, the Pi's armstub, and bare boot ROM entries hand the kernel
    // EL1, EL2, or EL3 — everything below assumes EL1, so drop explicitly.
    "    mrs x0, CurrentEL",
    "    lsr x0, x0, #2",
    "    add x0, x0, #48",         // digit of the starting EL
    "    str w0, [x4]",
    "    mrs x0, CurrentEL",
    "    cmp x0, #0xC",            // EL3
    "    b.eq from_el3",
    "    cmp x0, #0x8",            // EL2
    "    b.eq from_el2",
    "    b el_ready",
    "",
    "from_el3:",
    "    mov w5, #51",             // '3'
    "    str w5, [x4]",
    "    mov x1, #0x400",
    "    orr x1, x1, #1",          // SCR_EL3: NS=1, RW=1 (lower EL is AArch64)
    "    msr scr_el3, x1",
    "    msr cptr_el3, xzr",       // no FP/SIMD traps from EL3
    "    adr x1, el_ready",
    "    msr elr_el3, x1",
    "    mov x1, #0x3C5",          // SPSR: DAIF masked, EL1h
    "    msr spsr_el3, x1",
    "    eret",
    "",
    "from_el2:",
    "    mov w5, #50",             // '2'
    "    str w5, [x4]",
    "    mov x1, #0x80000000",     // HCR_EL2.RW=1 (EL1 is AArch64)
    "    msr hcr_el2, x1",
    "    mov w5, #104",            // 'h' — hcr written
    "    str w5, [x4]",
    "    adr x1, el_ready",
    "    msr elr_el2, x1",
    "    mov x1, #0x3C5",          // SPSR: DAIF masked, EL1h
    "    msr spsr_el2, x1",
    "    eret",
    "",
    "el_ready:",
    "    mov w5, #76",             // 'L' reached el_ready
    "    str w5, [x4]",
    "    msr spsel, #1",           // use SP_EL1 as the kernel stack
    "    mov w5, #115",            // 's' stack selected
    "    str w5, [x4]",
    "    adrp x0, __stack_top",
    "    add x0, x0, :lo12:__stack_top",
    "    mov sp, x0",
    "    mov x1, #(3 << 20)",      // CPACR_EL1.FPEN: allow FP/NEON
    "    msr cpacr_el1, x1",
    // Zero .bss — the loader makes no guarantees about it.
    "    adrp x0, __bss_start",
    "    add x0, x0, :lo12:__bss_start",
    "    adrp x1, __bss_end",
    "    add x1, x1, :lo12:__bss_end",
    "1:",
    "    cmp x0, x1",
    "    b.hs 2f",
    "    stp xzr, xzr, [x0], #16",
    "    b 1b",
    "2:",
    "    mov w5, #75",             // 'K' calling kmain
    "    str w5, [x4]",
    "    bl kmain",
    // Should never return; if it does, park here too.
    "parked:",
    "    wfe",
    "    b parked",
);

fn current_el() -> u8 {
    let el: u64;
    // Soundness: read-only system register.
    unsafe { core::arch::asm!("mrs {}, CurrentEL", out(reg) el, options(nomem, nostack, preserves_flags)) };
    ((el >> 2) & 0x3) as u8
}

/// Kernel entry, called by the boot stub on core 0 at EL1.
#[unsafe(no_mangle)]
extern "C" fn kmain() -> ! {
    uart::init();
    uart::write_str("kernel alive\n");
    let _ = core::fmt::write(&mut uart::Serial, format_args!("boot: running at EL{}\n", current_el()));

    vectors::init();
    uart::write_str("vectors: VBAR_EL1 installed, DAIF masked\n");

    mmu::init();
    mmu::log();

    timer::log();

    uart::write_str("cpu bring-up complete\n");

    #[cfg(feature = "selftest-exceptions")]
    selftest_exceptions();

    // The selftests above are diverging, so this loop is unreachable when a
    // selftest feature is enabled — that is expected, not a bug.
    #[allow(unreachable_code)]
    loop {
        // Soundness: `wfe` parks the core on a no-op event wait; nothing
        // else in this phase ever sends the event, so this is a clean idle.
        unsafe { core::arch::asm!("wfe", options(nomem, nostack, preserves_flags)) }
    }
}

/// Park this core forever (interrupts stay masked until the GIC phase).
pub fn park() -> ! {
    loop {
        // Soundness: terminal wait; the core never resumes by design.
        unsafe { core::arch::asm!("wfe", options(nomem, nostack, preserves_flags)) }
    }
}

/// Acceptance test (Pi-1): a `brk` is caught and execution resumes past it;
/// a read of an unmapped virtual address takes a data abort that is caught
/// and logged with the fault address (then parks — it is fatal by design).
#[cfg(feature = "selftest-exceptions")]
fn selftest_exceptions() -> ! {
    uart::write_str("selftest: triggering brk\n");
    // Soundness: `brk #0` is the deliberate fault; the brk handler advances
    // ELR past the 4-byte instruction so this code resumes.
    unsafe { core::arch::asm!("brk #0", options(nomem, nostack, preserves_flags)) };
    uart::write_str("PASS: brk caught and execution resumed\n");

    uart::write_str("selftest: triggering data abort (read of unmapped VA)\n");
    let addr: u64 = 0x0000_4000_0000; // 16 GiB — outside the 1 GiB identity map
    // Soundness: the deliberate fault is the whole point; raw asm so the
    // dereference address is exactly `addr`.
    unsafe {
        core::arch::asm!(
            "ldr x1, [x0]",
            in("x0") addr,
            lateout("x1") _,
            options(nostack)
        );
    }
    uart::write_str("FAIL: data abort did not trigger\n");
    park()
}

/// Panic path: print and park. Interrupts are masked at EL1.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    uart::write_str("KERNEL PANIC: ");
    let _ = core::fmt::write(&mut uart::Serial, format_args!("{}\n", info));
    park()
}
