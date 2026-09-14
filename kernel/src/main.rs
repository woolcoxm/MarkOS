//! MarkOS — a single-purpose x86_64 unikernel that boots straight into an LLM
//! inference engine.
//!
//! Phase 0 (this commit): boot via the Limine boot protocol, prove we are
//! alive over the COM1 serial port, then spin in a `hlt` loop. No interrupts,
//! no memory management yet.
//!
//! owns: the kernel entry point and the Limine request table.
//! invariants: all Limine requests must stay in this module, declared between
//! the `_REQUESTS_START` and `_REQUESTS_END` markers (see linker.ld).

#![no_std]
#![no_main]

mod serial;

use core::{arch::asm, panic::PanicInfo};

use limine::{
    BaseRevision, RequestsEndMarker, RequestsStartMarker,
    request::EntryPointRequest,
};

#[used]
#[unsafe(link_section = ".limine_requests_start")]
static _REQUESTS_START: RequestsStartMarker = RequestsStartMarker::new();

// Negotiate the newest base revision both sides understand; the boot is
// aborted below if the bootloader cannot give us at least revision 0.
#[used]
#[unsafe(link_section = ".limine_requests")]
static BASE_REVISION: BaseRevision = BaseRevision::new();

// Limine jumps straight into the kernel via this function pointer; there is
// no architecture entry stub because the bootloader hands us a working
// higher-half virtual address space and a stack.
#[used]
#[unsafe(link_section = ".limine_requests")]
static ENTRY_POINT: EntryPointRequest = EntryPointRequest::new(kernel_main);

#[used]
#[unsafe(link_section = ".limine_requests_end")]
static _REQUESTS_END: RequestsEndMarker = RequestsEndMarker::new();

/// Kernel entry point, called by the bootloader on the BSP with a valid stack
/// and the kernel mapped at its link address.
#[unsafe(no_mangle)]
unsafe extern "C" fn kernel_main() -> ! {
    if !BASE_REVISION.is_supported() {
        serial::write_str("limine: base revision unsupported, halting\n");
        halt();
    }

    serial::init();
    serial::write_str("kernel alive\n");

    loop {
        // Soundness: `hlt` with interrupts still masked by the bootloader;
        // it merely parks the core until the next NMI/SMI, which we ignore.
        unsafe { asm!("hlt", options(nomem, nostack, preserves_flags)); }
    }
}

/// Panic path: this runs before Phase 1's register-dump machinery exists, so
/// print what we have and halt. Interrupts are never enabled in Phase 0.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial::write_str("KERNEL PANIC: ");
    let _ = core::fmt::write(&mut serial::Serial, format_args!("{}\n", info));
    halt()
}

/// Park this core forever with interrupts masked.
fn halt() -> ! {
    loop {
        // Soundness: masking interrupts before parking; this core never wakes
        // except on NMI/SMI, which is exactly the "stop doing work" semantics.
        unsafe { asm!("cli; hlt", options(nomem, nostack, preserves_flags)); }
    }
}
