//! MarkOS — a bare-metal Raspberry Pi unikernel that boots straight into an
//! LLM inference engine.
//!
//! Pi-0 (this commit): AArch64 bring-up. The Pi firmware (or QEMU's raspi
//! machine) loads the image at physical 0x80000 and starts every core at
//! `_start`; the entry stub parks all non-BSP cores, sets the boot stack,
//! enables FP/NEON, zeroes .bss, then calls `kmain`. Prints "kernel alive"
//! on the PL011 UART and idles with `wfe`.
//!
//! owns: the kernel entry point and the boot stack.
//! invariants: runs identity-mapped with the MMU off (Phase Pi-0); every
//! non-BSP core stays parked until the SMP phase gives them real work.

#![no_std]
#![no_main]

mod uart;

use core::{arch::global_asm, panic::PanicInfo};

global_asm!(
    ".section .text.boot",
    ".globl _start",
    "_start:",
    // The firmware/QEMU starts every core here; park all but core 0.
    "    mrs x0, mpidr_el1",
    "    and x0, x0, #3",          // affinity level 0 = core id on the Pi
    "    cbnz x0, parked",
    "    ldr x0, =__stack_top",
    "    mov sp, x0",
    "    mov x1, #(3 << 20)",      // CPACR_EL1.FPEN: allow FP/NEON (kernels need it later)
    "    msr cpacr_el1, x1",
    // Zero .bss — the loader makes no guarantees about it.
    "    ldr x0, =__bss_start",
    "    ldr x1, =__bss_end",
    "1:",
    "    cmp x0, x1",
    "    b.hs 2f",
    "    stp xzr, xzr, [x0], #16",
    "    b 1b",
    "2:",
    "    bl kmain",
    // Should never return; if it does, park here too.
    "parked:",
    "    wfe",
    "    b parked",
);

/// Kernel entry, called by the boot stub on core 0.
#[unsafe(no_mangle)]
extern "C" fn kmain() -> ! {
    uart::init();
    uart::write_str("kernel alive\n");

    loop {
        // Soundness: `wfe` parks the core on a no-op event wait; nothing
        // else in this phase ever sends the event, so this is a clean idle.
        unsafe { core::arch::asm!("wfe", options(nomem, nostack, preserves_flags)) }
    }
}

/// Panic path: print and park. Interrupts are never enabled in Pi-0.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    uart::write_str("KERNEL PANIC: ");
    let _ = core::fmt::write(&mut uart::Serial, format_args!("{}\n", info));
    loop {
        unsafe { core::arch::asm!("wfe", options(nomem, nostack, preserves_flags)) }
    }
}
