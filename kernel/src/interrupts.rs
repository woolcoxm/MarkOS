//! Interrupt Descriptor Table: handlers for every CPU exception vector.
//!
//! owns: the static IDT.
//! invariants: filled and `load()`ed once, on the BSP, with interrupts still
//! disabled and before SMP exists — hence plain static mutation is race-free.
//! Every handler is terminal (`halt_loop`) except `breakpoint`, which returns
//! to prove non-faulting delivery. The double-fault handler runs on the IST
//! stack from `gdt.rs` so it survives a corrupt/broken stack.

use x86_64::structures::idt::{
    InterruptDescriptorTable, InterruptStackFrame, PageFaultErrorCode,
};

use crate::serial;

static mut IDT: InterruptDescriptorTable = InterruptDescriptorTable::new();

pub fn init() {
    // Soundness: single-core, interrupts masked, no other references to IDT —
    // filling it before load() is data-race free; it stays valid forever.
    let idt = unsafe { &mut *(&raw mut IDT) };

    idt.divide_error.set_handler_fn(divide_error);
    idt.debug.set_handler_fn(debug);
    idt.non_maskable_interrupt.set_handler_fn(nmi);
    idt.breakpoint.set_handler_fn(breakpoint);
    idt.overflow.set_handler_fn(overflow);
    idt.bound_range_exceeded.set_handler_fn(bound_range_exceeded);
    idt.invalid_opcode.set_handler_fn(invalid_opcode);
    idt.device_not_available.set_handler_fn(device_not_available);
    // Soundness: index 0 of the IST, configured in gdt.rs with a valid
    // 64 KiB stack; `set_stack_index` is unsafe because a wrong index would
    // make the CPU switch to a garbage stack on double fault.
    unsafe {
        idt.double_fault
            .set_handler_fn(double_fault)
            .set_stack_index(crate::gdt::DOUBLE_FAULT_IST as u16);
    }
    idt.invalid_tss.set_handler_fn(invalid_tss);
    idt.segment_not_present.set_handler_fn(segment_not_present);
    idt.stack_segment_fault.set_handler_fn(stack_segment_fault);
    idt.general_protection_fault.set_handler_fn(general_protection_fault);
    idt.page_fault.set_handler_fn(page_fault);
    idt.x87_floating_point.set_handler_fn(x87_floating_point);
    idt.alignment_check.set_handler_fn(alignment_check);
    idt.machine_check.set_handler_fn(machine_check);
    idt.simd_floating_point.set_handler_fn(simd_floating_point);
    idt.virtualization.set_handler_fn(virtualization);
    idt.security_exception.set_handler_fn(security_exception);

    // IDT is a `static` — it lives forever at a fixed address, as the CPU
    // requires, which is exactly what the safe `load(&'static self)` demands.
    // Interrupts remain disabled; enabling them is a later phase that will
    // also add the PIC/APIC and timer.
    idt.load();
}

fn frame_line(name: &str, frame: &InterruptStackFrame, err: Option<u64>) {
    // The CPU-pushed frame is the authoritative fault context. `format_args!`
    // only — no heap exists yet.
    match err {
        Some(c) => {
            let _ = core::fmt::write(
                &mut serial::Serial,
                format_args!(
                    "CAUGHT exception: {name} rip={:#x} cs={:#x} rflags={:#x} rsp={:#x} ss={:#x} err={c:#x}\n",
                    frame.instruction_pointer.as_u64(),
                    frame.code_segment.0,
                    frame.cpu_flags.bits(),
                    frame.stack_pointer.as_u64(),
                    frame.stack_segment.0,
                ),
            );
        }
        None => {
            let _ = core::fmt::write(
                &mut serial::Serial,
                format_args!(
                    "CAUGHT exception: {name} rip={:#x} cs={:#x} rflags={:#x} rsp={:#x} ss={:#x} err=none\n",
                    frame.instruction_pointer.as_u64(),
                    frame.code_segment.0,
                    frame.cpu_flags.bits(),
                    frame.stack_pointer.as_u64(),
                    frame.stack_segment.0,
                ),
            );
        }
    }
}

fn fatal(name: &'static str, frame: &InterruptStackFrame, err: Option<u64>) -> ! {
    frame_line(name, frame, err);
    crate::regs::dump(name);
    crate::halt_loop();
}

extern "x86-interrupt" fn divide_error(frame: InterruptStackFrame) {
    fatal("divide_error", &frame, None);
}

extern "x86-interrupt" fn debug(frame: InterruptStackFrame) {
    fatal("debug", &frame, None);
}

extern "x86-interrupt" fn nmi(frame: InterruptStackFrame) {
    fatal("non_maskable_interrupt", &frame, None);
}

extern "x86-interrupt" fn breakpoint(frame: InterruptStackFrame) {
    frame_line("breakpoint", &frame, None);
    // int3 is fully restartable; returning proves the IDT is live without
    // ending the boot — used by the selftest path.
}

extern "x86-interrupt" fn overflow(frame: InterruptStackFrame) {
    fatal("overflow", &frame, None);
}

extern "x86-interrupt" fn bound_range_exceeded(frame: InterruptStackFrame) {
    fatal("bound_range_exceeded", &frame, None);
}

extern "x86-interrupt" fn invalid_opcode(frame: InterruptStackFrame) {
    fatal("invalid_opcode", &frame, None);
}

extern "x86-interrupt" fn device_not_available(frame: InterruptStackFrame) {
    fatal("device_not_available", &frame, None);
}

extern "x86-interrupt" fn double_fault(frame: InterruptStackFrame, err: u64) -> ! {
    fatal("double_fault", &frame, Some(err));
}

extern "x86-interrupt" fn invalid_tss(frame: InterruptStackFrame, err: u64) {
    fatal("invalid_tss", &frame, Some(err));
}

extern "x86-interrupt" fn segment_not_present(frame: InterruptStackFrame, err: u64) {
    fatal("segment_not_present", &frame, Some(err));
}

extern "x86-interrupt" fn stack_segment_fault(frame: InterruptStackFrame, err: u64) {
    fatal("stack_segment_fault", &frame, Some(err));
}

extern "x86-interrupt" fn general_protection_fault(frame: InterruptStackFrame, err: u64) {
    fatal("general_protection_fault", &frame, Some(err));
}

extern "x86-interrupt" fn page_fault(frame: InterruptStackFrame, err: PageFaultErrorCode) {
    frame_line("page_fault", &frame, None);
    let _ = core::fmt::write(
        &mut serial::Serial,
        format_args!("page_fault error code: {err:?}\n"),
    );
    crate::regs::dump("page_fault");
    crate::halt_loop();
}

extern "x86-interrupt" fn x87_floating_point(frame: InterruptStackFrame) {
    fatal("x87_floating_point", &frame, None);
}

extern "x86-interrupt" fn alignment_check(frame: InterruptStackFrame, err: u64) {
    fatal("alignment_check", &frame, Some(err));
}

extern "x86-interrupt" fn machine_check(frame: InterruptStackFrame) -> ! {
    fatal("machine_check", &frame, None);
}

extern "x86-interrupt" fn simd_floating_point(frame: InterruptStackFrame) {
    fatal("simd_floating_point", &frame, None);
}

extern "x86-interrupt" fn virtualization(frame: InterruptStackFrame) {
    fatal("virtualization", &frame, None);
}

extern "x86-interrupt" fn security_exception(frame: InterruptStackFrame, err: u64) {
    fatal("security_exception", &frame, Some(err));
}
