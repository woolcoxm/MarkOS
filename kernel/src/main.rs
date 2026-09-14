//! MarkOS — a bare-metal Raspberry Pi unikernel that boots straight into an
//! LLM inference engine.
//!
//! Pi-2 (this commit): SMP — every core the firmware starts (or PSCI
//! releases) is brought online through `secondary_main`, contributing to an
//! exact shared-counter acceptance test. Release paths: ARM64 spin-table
//! (Pi firmware/QEMU raspi), PSCI CPU_ON (QEMU virt), legacy CORE_RELEASE
//! (cores that entered our own stub).
//!
//! owns: the kernel entry point and the boot stack.
//! invariants: non-BSP cores park in the stub (or firmware parking) until
//! smp::start_aps releases them; interrupts stay masked until the GIC phase.

#![no_std]
#![no_main]

mod board;
mod board_release;
mod cpu;
mod mmu;
mod psci;
mod smp;
mod timer;
mod uart;
mod vectors;

use core::{arch::global_asm, panic::PanicInfo};

global_asm!(
    ".section .text.boot",
    ".globl _start",
    "_start:",
    // The firmware/QEMU starts every core here; park all but core 0.
    "    mrs x0, mpidr_el1",
    "    and x0, x0, #3",          // affinity level 0 = core id on the Pi
    "    cbnz x0, secondary_wait",
    "",
    // Normalize the exception level to EL1. Boards differ: QEMU's raspi
    // machines, the Pi's armstub, and bare boot ROM entries hand the kernel
    // EL1, EL2, or EL3 — everything below assumes EL1, so drop explicitly.
    "    mrs x0, CurrentEL",
    "    cmp x0, #0xC",            // EL3
    "    b.eq from_el3",
    "    cmp x0, #0x8",            // EL2
    "    b.eq from_el2",
    "    b el_ready",
    "",
    "from_el3:",
    "    mov x1, #0x401",          // SCR_EL3: NS=1, RW=1 (lower EL is AArch64)
    "    msr scr_el3, x1",
    "    msr cptr_el3, xzr",       // no FP/SIMD traps from EL3
    "    adr x1, el_ready",
    "    msr elr_el3, x1",
    "    mov x1, #0x3C5",          // SPSR: DAIF masked, EL1h
    "    msr spsr_el3, x1",
    "    eret",
    "",
    "from_el2:",
    "    mov x1, #0x80000000",     // HCR_EL2.RW=1 (EL1 is AArch64)
    "    msr hcr_el2, x1",
    // NOTE: no `msr cptr_el2` here — CPTR_EL2 is ARMv8.1+; on the v8.0
    // Cortex-A53 (QEMU raspi3b) it is unallocated and would fault.
    "    adr x1, el_ready",
    "    msr elr_el2, x1",
    "    mov x1, #0x3C5",          // SPSR: DAIF masked, EL1h
    "    msr spsr_el2, x1",
    "    eret",
    "",
    "el_ready:",
    "    msr spsel, #1",           // use SP_EL1 as the kernel stack
    "    adrp x0, __stack_top",
    "    add x0, x0, :lo12:__stack_top",
    "    mov sp, x0",
    "    mov x1, #(3 << 20)",      // CPACR_EL1.FPEN: allow FP/NEON (inference kernels)
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
    "    bl kmain",
    // Should never return; if it does, park here too.
    "parked:",
    "    wfe",
    "    b parked",
    "",
    // Cores 1..3 spin here from the moment they enter the kernel. The BSP
    // publishes each core's private stack top into CORE_RELEASE[core] and
    // wakes everyone with sev; the released core adopts that stack and
    // calls secondary_main. x0 keeps the core id for secondary_main.
    "secondary_wait:",
    "    adrp x1, CORE_RELEASE",
    "    add x1, x1, :lo12:CORE_RELEASE",
    "    add x1, x1, x0, lsl #3",
    "1:",
    "    wfe",
    "    ldr x2, [x1]",
    "    cbz x2, 1b",
    "    mov sp, x2",
    "    bl secondary_main",
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
    uart::locked_write(format_args!("boot: board={} running at EL{}\n", board::NAME, current_el()),
    );

    vectors::init();
    uart::write_str("vectors: VBAR_EL1 installed, DAIF masked\n");

    mmu::init();
    mmu::log();

    timer::log();
    cpu::log();

    uart::write_str("cpu bring-up complete\n");

    // SMP: release the parked cores and verify all of them are live.
    match smp::start_aps(board::CORE_COUNT) {
        Ok((aps, work_expected)) => {
            smp::bsp_do_work();
            let work = smp::work_total();
            let aps_expected = board::CORE_COUNT - 1;
            let ok = aps as usize == aps_expected && work == work_expected;
            uart::locked_write(format_args!(
                    "smp: {aps}/{aps_expected} APs online, shared counter {work}/{work_expected} — {}\n",
                    if ok { "PASS" } else { "FAIL" }
                ));
        }
        Err(e) => {
            uart::locked_write(format_args!("smp: {e}\n"));
        }
    }

    #[cfg(feature = "selftest-exceptions")]
    selftest_exceptions();

    // The selftests below are diverging, so this loop is unreachable when a
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
    // 4 TiB: canonical, beyond the [0,2GiB) identity map -> level-0
    // translation fault. (0x40000000 itself is now mapped!)
    let addr: u64 = 0x0000_0400_0000_0000;
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
    uart::locked_write(format_args!("{}\n", info));
    park()
}
