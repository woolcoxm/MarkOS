//! Kernel GDT: flat kernel code/data segments plus a TSS whose Interrupt
//! Stack Table entry 0 backs the double-fault handler.
//!
//! owns: the (static) GDT memory and the double-fault IST stack.
//! invariants: `init()` runs once, on the BSP, before any exception can be
//! taken (CPU faults are possible before that, but we accept that boot risk —
//! a fault before GDT load triple-faults by definition).

use spin::Once;
use x86_64::VirtAddr;
use x86_64::instructions::tables::load_tss;
use x86_64::registers::segmentation::{CS, Segment};
use x86_64::structures::gdt::{Descriptor, GlobalDescriptorTable};
use x86_64::structures::tss::TaskStateSegment;

/// IST slot backing the double-fault handler (see `interrupts.rs`).
pub const DOUBLE_FAULT_IST: usize = 0;

/// 64 KiB is far more than a double fault needs to print and park; it also
/// gives later phases headroom if they route more vectors through the IST.
const DF_STACK_QWORDS: usize = 64 * 1024 / 8;

/// Double-fault IST stack. `static mut` because the CPU writes to it on an
/// IST switch; our code only ever takes its address. Zero-initialized data
/// lands in .bss (writable), never .rodata.
static mut DF_STACK: [u64; DF_STACK_QWORDS] = [0; DF_STACK_QWORDS];

static GDT: Once<GlobalDescriptorTable> = Once::new();
static TSS: Once<TaskStateSegment> = Once::new();

/// Build, load, and switch to the kernel GDT + TSS.
pub fn init() {
    // IST stacks grow downward on switch, so the entry points one byte past
    // the end of the backing memory.
    let tss = TSS.call_once(|| {
        // Soundness: only the address of DF_STACK is taken; no aliasing with
        // any reference exists at this point (kernel is single-core, IF off).
        let top = VirtAddr::from_ptr(&raw const DF_STACK) + (DF_STACK_QWORDS * 8) as u64;
        let mut t = TaskStateSegment::new();
        t.interrupt_stack_table[DOUBLE_FAULT_IST] = top;
        t
    });

    let mut gdt = GlobalDescriptorTable::new();
    let code = gdt.append(Descriptor::kernel_code_segment());
    let _data = gdt.append(Descriptor::kernel_data_segment());
    let tss_selector = gdt.append(Descriptor::tss_segment(tss));

    let gdt = GDT.call_once(move || gdt);
    gdt.load();

    // Soundness: the selectors were just produced from the GDT we loaded
    // above; `CS::set_reg` performs the required far return, `load_tss` ltr.
    unsafe {
        CS::set_reg(code);
        load_tss(tss_selector);
    }
}
