//! MarkOS — a single-purpose x86_64 unikernel that boots straight into an LLM
//! inference engine.
//!
//! Phase 1 (this commit): CPU bring-up — GDT with TSS (double-fault IST), a
//! full IDT with handlers for every CPU exception, a panic handler that logs
//! message + register dump, and selftest builds that deliberately take a
//! divide-by-zero / page fault to prove the machinery works.
//!
//! owns: the kernel entry point and the Limine request table.
//! invariants: all Limine requests must stay in this module, declared between
//! the `_REQUESTS_START` and `_REQUESTS_END` markers (see linker.ld).

#![no_std]
#![no_main]
// Nightly: interrupt handlers use the x86-interrupt calling convention.
#![feature(abi_x86_interrupt)]

mod gdt;
mod heap;
mod interrupts;
mod mem;
mod paging;
mod physmem;
mod regs;
mod serial;

// Heap-backed collections (Vec, Box) — available after `heap::init`.
extern crate alloc;

use core::{arch::asm, arch::global_asm, panic::PanicInfo};

use limine::{
    request::{EntryPointRequest, HhdmRequest, MemmapRequest},
    BaseRevision, RequestsEndMarker, RequestsStartMarker,
};

// Limine leaves x87/SSE disabled at kernel entry (CR0.EM set), but the
// x86_64 baseline ABI this target compiles for freely uses SSE instructions
// (e.g. `xorps` zeroing). This stub runs before ANY Rust code: clear CR0.EM,
// set CR0.MP/NE, set CR4.OSFXSR|OSXMMEXCPT, then call the Rust entry.
// AVX/OSXSAVE enablement is deliberately deferred to the compute-kernel
// phase, which owns the FP feature policy.
global_asm!(
    ".globl kernel_entry",
    "kernel_entry:",
    "    mov %cr0, %rax",
    "    and $~(1 << 2), %ax",      // clear EM: no #UD on SSE
    "    or  $((1 << 1) | (1 << 5)), %ax", // set MP (SSE present), NE (native #MF)
    "    mov %rax, %cr0",
    "    mov %cr4, %rax",
    "    or  $((1 << 9) | (1 << 10)), %ax", // OSFXSR: save SSE state; OSXMMEXCPT
    "    mov %rax, %cr4",
    "    and $-16, %rsp",  // Limine's initial stack is not 16-byte aligned;
                           // `movaps`-style spills in Rust code require it.
    "    xor %ebp, %ebp",  // terminate frame-pointer chains cleanly
    "    call {kernel_main}",
    kernel_main = sym kernel_main,
    options(att_syntax)
);

unsafe extern "C" {
    fn kernel_entry() -> !;
}

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
static ENTRY_POINT: EntryPointRequest = EntryPointRequest::new(kernel_entry);

// Boot-time information the kernel builds on: the physical memory map and
// the base of the higher-half direct map (phys + offset = virt).
#[used]
#[unsafe(link_section = ".limine_requests")]
pub static MEMMAP: MemmapRequest = MemmapRequest::new();
#[used]
#[unsafe(link_section = ".limine_requests")]
pub static HHDM: HhdmRequest = HhdmRequest::new();

#[used]
#[unsafe(link_section = ".limine_requests_end")]
static _REQUESTS_END: RequestsEndMarker = RequestsEndMarker::new();

/// Kernel entry point, called by the bootloader on the BSP with a valid stack
/// and the kernel mapped at its link address.
#[unsafe(no_mangle)]
unsafe extern "C" fn kernel_main() -> ! {
    if !BASE_REVISION.is_supported() {
        serial::write_str("limine: base revision unsupported, halting\n");
        halt_loop();
    }

    serial::init();
    serial::write_str("MarkOS: CPU bring-up\n");

    gdt::init();
    serial::write_str("gdt: kernel code/data + TSS loaded, IST0 reserved for double fault\n");

    interrupts::init();
    serial::write_str("idt: all CPU exception vectors handled\n");

    physmem::dump_memory_map();
    let (managed_mib, free) = physmem::init();
    let _ = core::fmt::write(
        &mut serial::Serial,
        format_args!("physmem: managing largest usable region ({managed_mib} MiB), {free} free frames\n"),
    );

    paging::init();
    heap::init().expect("heap init failed");
    heap::log_stats();

    serial::write_str("cpu bring-up complete\n");

    #[cfg(feature = "selftest-div")]
    selftest_divide();
    #[cfg(feature = "selftest-pf")]
    selftest_page_fault();
    #[cfg(feature = "selftest-frames")]
    selftest_frames();
    #[cfg(feature = "selftest-heap")]
    selftest_heap();

    // The selftests above are diverging, so this loop is unreachable when a
    // selftest feature is enabled — that is expected, not a bug.
    #[allow(unreachable_code)]
    loop {
        // Soundness: `hlt` parks the core until the next (never-enabled,
        // Phase 1) interrupt or NMI/SMI, which we deliberately ignore.
        unsafe { asm!("hlt", options(nomem, nostack, preserves_flags)); }
    }
}

/// Park this core forever with interrupts masked. Terminal outcome for every
/// unrecoverable fault until later phases gain recovery paths.
pub fn halt_loop() -> ! {
    loop {
        // Soundness: masking interrupts before parking; this core never wakes
        // except on NMI/SMI, which is exactly the "stop doing work" semantics.
        unsafe { asm!("cli; hlt", options(nomem, nostack, preserves_flags)); }
    }
}

/// Acceptance test (Phase 1): deliberately execute `div` by zero and expect
/// the `divide_error` handler to catch and log it (then halt).
#[cfg(feature = "selftest-div")]
fn selftest_divide() -> ! {
    serial::write_str("selftest: triggering divide-by-zero\n");
    let zero: u64 = core::hint::black_box(0);
    // Soundness: the deliberate fault is the whole point. Safe `/` would hit
    // Rust's mandatory div-by-zero check (proven by the panic handler), so we
    // issue the raw `div` to raise #DE in hardware.
    unsafe {
        asm!(
            "div rcx", // dividend RDX:RAX, divisor 0 => #DE
            in("rcx") zero,
            lateout("rax") _,
            lateout("rdx") _,
            options(nostack)
        );
    }
    unreachable!("divide by zero did not fault");
}

/// Acceptance test (Phase 1): deliberately read an unmapped canonical
/// address and expect the `page_fault` handler to catch and log it.
#[cfg(feature = "selftest-pf")]
fn selftest_page_fault() -> ! {
    serial::write_str("selftest: triggering page fault\n");
    // 64 GiB: unmapped in QEMU (2 GiB RAM). NOTE: the address must be
    // canonical — e.g. 0xdeadbeef0000 has bit 47 set with zeros above, which
    // is non-canonical and correctly raises #GP instead of #PF.
    let addr: u64 = core::hint::black_box(0x0000_0010_0000_0000);
    // Soundness: the deliberate fault is the whole point. Raw asm so the
    // dereference address is exactly `addr`, unaffected by codegen.
    // (This target's inline asm dialect is Intel syntax.)
    unsafe {
        asm!(
            "mov rcx, qword ptr [rax]", // load from unmapped VA => #PF
            in("rax") addr,
            lateout("rcx") _,
            options(nostack)
        );
    }
    unreachable!("bad read did not fault");
}

/// Acceptance test (Phase 2): allocate/free thousands of frames in a stress
/// loop; the free-frame count must return to its starting value every round
/// (no leaks) and no allocation may fail while frames remain.
#[cfg(feature = "selftest-frames")]
fn selftest_frames() -> ! {
    use x86_64::structures::paging::{PhysFrame, Size4KiB};

    serial::write_str("selftest: frame stress (10,240 alloc/dealloc pairs)\n");
    const BATCH: usize = 256;
    const ROUNDS: usize = 40;

    let before = physmem::free_frames();
    let mut slots: [Option<PhysFrame<Size4KiB>>; BATCH] = [None; BATCH];

    for round in 0..ROUNDS {
        for slot in slots.iter_mut() {
            *slot = physmem::alloc_frame();
        }
        if slots.iter().any(|s| s.is_none()) {
            serial::write_str("FAIL: frame alloc returned None mid-test\n");
            halt_loop();
        }
        if physmem::free_frames() != before - BATCH {
            serial::write_str("FAIL: free count wrong after batch alloc\n");
            halt_loop();
        }
        for slot in slots.iter_mut() {
            let f = slot.take().expect("slot held a frame");
            if let Err(e) = physmem::dealloc_frame(f) {
                let _ = core::fmt::write(
                    &mut serial::Serial,
                    format_args!("FAIL: dealloc rejected a live frame: {e}\n"),
                );
                halt_loop();
            }
        }
        if physmem::free_frames() != before {
            let _ = core::fmt::write(
                &mut serial::Serial,
                format_args!(
                    "FAIL: leak — free {} != {} after round {round}\n",
                    physmem::free_frames(),
                    before
                ),
            );
            halt_loop();
        }
    }

    let _ = core::fmt::write(
        &mut serial::Serial,
        format_args!(
            "PASS: frame stress — {n} alloc/dealloc pairs, free count stable at {before}\n",
            n = BATCH * ROUNDS
        ),
    );
    halt_loop()
}

/// Acceptance test (Phase 3): heap churn with varying sizes (forcing
/// fragmentation), Vec-heavy verification against a reference, and a
/// used/free invariant check at the end (no leaks).
#[cfg(feature = "selftest-heap")]
fn selftest_heap() -> ! {
    use alloc::vec::Vec;
    use x86_64::structures::paging::page_table::PageTableFlags;
    use x86_64::structures::paging::Page;
    use x86_64::VirtAddr;

    serial::write_str("selftest: heap churn + fragmentation\n");

    // Phase 0 — direct paging roundtrip on a scratch page just below the
    // heap window: map, write through the mapping, translate, unmap, and
    // confirm frame accounting. Run twice: the first trip may allocate
    // intermediate page tables (a permanent, one-time cost); the second
    // must net zero frames — that is the leak check.
    for trip in 0..2u32 {
        let scratch = Page::containing_address(VirtAddr::new(paging::HEAP_BASE - 0x1000));
        let ff0 = physmem::free_frames();
        let Some(frame) = physmem::alloc_frame() else {
            serial::write_str("FAIL: no frame for paging roundtrip\n");
            halt_loop();
        };
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE;
        if let Err(e) = paging::map(scratch, frame, flags) {
            let _ = core::fmt::write(&mut serial::Serial, format_args!("FAIL: map: {e}\n"));
            halt_loop();
        }
        // Soundness: the page was just mapped writable to a frame we own.
        let vptr = scratch.start_address().as_u64() as *mut u64;
        unsafe { vptr.write_volatile(0x0A11CED0C5_1234) };
        if paging::translate(scratch.start_address()) != Some(frame.start_address()) {
            serial::write_str("FAIL: translate disagrees with the mapping\n");
            halt_loop();
        }
        let Ok(back) = paging::unmap(scratch) else {
            serial::write_str("FAIL: unmap refused a live mapping\n");
            halt_loop();
        };
        if back != frame || paging::translate(scratch.start_address()).is_some() {
            serial::write_str("FAIL: unmap did not restore the previous state\n");
            halt_loop();
        }
        if physmem::dealloc_frame(back).is_err() {
            serial::write_str("FAIL: dealloc rejected the unmapped frame\n");
            halt_loop();
        }
        let ff1 = physmem::free_frames();
        if trip == 1 && ff1 != ff0 {
            let _ = core::fmt::write(
                &mut serial::Serial,
                format_args!("FAIL: roundtrip leaked frames: {ff0} -> {ff1}\n"),
            );
            halt_loop();
        }
    }
    serial::write_str("paging roundtrip: map/write/translate/unmap ok\n");

    let (used0, free0) = heap::stats();

    // Phase A — 128 vectors of varying (LCG-chosen) sizes, filled with a
    // deterministic pattern, verified immediately.
    let mut blobs: Vec<(u64, usize, Vec<u8>)> = Vec::new();
    let mut seed: u64 = 0x1234_5678_9abc_def0;
    for i in 0..128u64 {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        let size = 1 + (seed >> 33) as usize % (32 * 1024);
        let mut v: Vec<u8> = Vec::with_capacity(size);
        for j in 0..size {
            v.push((i ^ j as u64 ^ seed) as u8);
        }
        blobs.push((seed, size, v));
    }
    for (i, (s, size, v)) in blobs.iter().enumerate() {
        for j in 0..*size {
            let want = (i as u64 ^ j as u64 ^ *s) as u8;
            if v[j] != want {
                let _ = core::fmt::write(
                    &mut serial::Serial,
                    format_args!("FAIL: heap corruption at blob {i} byte {j}\n"),
                );
                halt_loop();
            }
        }
    }

    // Phase B — free every other blob (leaves holes), then allocate 64
    // fresh 32 KiB vectors which must fit into the freed holes.
    let mut i = 0;
    while i < blobs.len() {
        if i % 2 == 0 {
            blobs.remove(i);
        } else {
            i += 1;
        }
    }
    let mut holes: Vec<Vec<u8>> = Vec::new();
    for k in 0..64u64 {
        let mut v: Vec<u8> = Vec::with_capacity(32 * 1024);
        for j in 0..32 * 1024 {
            v.push((k ^ j as u64) as u8);
        }
        holes.push(v);
    }
    // Survivors must still hold their patterns (nothing aliased anything).
    for (i, (s, size, v)) in blobs.iter().enumerate() {
        for j in (0..*size).step_by(1024) {
            let want = (i as u64 ^ j as u64 ^ *s) as u8;
            if v[j] != want {
                serial::write_str("FAIL: heap corruption after fragmentation\n");
                halt_loop();
            }
        }
    }

    // Phase C — release everything; used bytes must return to (near) zero.
    // Near, not exactly: this allocator stores per-hole bookkeeping headers
    // inside freed blocks, so churn leaves a small constant residue. A real
    // leak of allocation-scale size would blow far past this bound.
    blobs.clear();
    holes.clear();
    const BOOKKEEPING_SLACK: usize = 64 * 1024;
    let (used1, free1) = heap::stats();
    if used1 > used0 + BOOKKEEPING_SLACK || free1 < free0.saturating_sub(BOOKKEEPING_SLACK) {
        let _ = core::fmt::write(
            &mut serial::Serial,
            format_args!(
                "FAIL: heap leak — used {used1} (was {used0}), free {free1} (was {free0})\n"
            ),
        );
        halt_loop();
    }

    serial::write_str("PASS: heap churn, fragmentation recovery, no leaks\n");
    halt_loop()
}

/// Panic path: print the panic message, a register snapshot, and halt.
/// Interrupts are never enabled in Phase 1; no re-entrancy is possible.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    serial::write_str("KERNEL PANIC: ");
    let _ = core::fmt::write(&mut serial::Serial, format_args!("{}\n", info));
    regs::dump("panic");
    halt_loop()
}
