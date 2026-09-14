//! MarkOS — a bare-metal Raspberry Pi unikernel that boots straight into an
//! LLM inference engine.
//!
//! Pi-4 (this commit): the execution model — a fixed thread-per-core
//! execution pool (no scheduler, cores spin between jobs) plus acceptance
//! tests for every subsystem so far.
//!
//! owns: the kernel entry point and the boot stack.
//! invariants: non-BSP cores park in the stub (or firmware parking) until
//! smp::start_aps releases them; interrupts stay masked until the GIC phase.

#![no_std]
#![no_main]

mod board;
mod board_release;
mod cache;
mod control;
mod cpu;
mod fat;
mod gguf;
mod matmul;
mod mmu;
mod net;
mod pool;
mod psci;
mod smp;
mod tcp;
mod timer;
mod uart;
mod vectors;
mod virtio_blk;

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

/// Park this core forever (interrupts stay masked until the GIC phase).
pub fn park() -> ! {
    loop {
        // Soundness: terminal wait; the core never resumes by design.
        unsafe { core::arch::asm!("wfe", options(nomem, nostack, preserves_flags)) }
    }
}

/// Shared model scratch: the loaded MODEL.BIN bytes (test models are small;
/// GB-scale weights get dedicated regions in a later phase).
static mut MODEL_BUF: [u8; 65536] = [0u8; 65536];

/// Helper: mount the FAT volume, open MODEL.BIN, read it fully into `buf`.
fn vol_open_and_read(buf: &mut [u8]) -> Result<usize, &'static str> {
    let vol = fat::mount()?;
    let file = vol.open_model()?;
    vol.read_file(&file, buf)
}

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
    uart::locked_write(format_args!(
        "boot: board={} running at EL{}\n",
        board::NAME,
        current_el()
    ));

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
    #[cfg(feature = "selftest-block")]
    selftest_block();
    #[cfg(feature = "selftest-fat")]
    selftest_fat();
    #[cfg(feature = "selftest-pool")]
    selftest_pool();
    #[cfg(feature = "selftest-matmul")]
    selftest_matmul();
    #[cfg(feature = "selftest-net")]
    selftest_net();

    // The loop below is unreachable when a diverging selftest ran.
    #[allow(unreachable_code)]
    loop {
        // Soundness: `wfe` parks the core on a no-op event wait; nothing
        // else in this phase ever sends the event, so this is a clean idle.
        unsafe { core::arch::asm!("wfe", options(nomem, nostack, preserves_flags)) }
    }
}

#[cfg(feature = "selftest-matmul")]
fn selftest_matmul() -> ! {
    uart::write_str("selftest: NEON UDOT int8 matmul over the loaded GGUF model
");
    // Micro-test: 16 bytes of 1s x 16 bytes of 2s = 32 via one UDOT.
    unsafe {
        let mut a16 = [1u8; 16];
        let mut b16 = [2u8; 16];
        let mut tmp = [0u32; 4];
        core::arch::asm!(
            "ld1 {{v0.16b}}, [{a}]",
            "ld1 {{v1.16b}}, [{b}]",
            "movi v2.4s, #0",
            "udot v2.4s, v0.16b, v1.16b",
            "st1 {{v2.4s}}, [{o}]",
            a = in(reg) &a16,
            b = in(reg) &b16,
            o = in(reg) tmp.as_mut_ptr(),
            options(nostack)
        );
        let sum: u32 = tmp.iter().sum();
        uart::locked_write(format_args!("udot micro: {sum} (expect 32)
"));
    }
    if let Err(e) = virtio_blk::init() {
        uart::locked_write(format_args!("FAIL: virtio init: {e}
"));
        crate::park()
    }
    if let Err(e) = fat::mount() {
        uart::locked_write(format_args!("FAIL: FAT mount: {e}
"));
        crate::park()
    }
    static mut MODEL_LEN: usize = 0usize;
    unsafe {
        let buf = core::slice::from_raw_parts_mut((&raw mut MODEL_BUF) as *mut u8, 65536);
        match vol_open_and_read(buf) {
            Ok(n) => MODEL_LEN = n,
            Err(e) => {
                uart::locked_write(format_args!("FAIL: model read: {e}
"));
                crate::park()
            }
        }
    }
    // Soundness: single-core BSP access; APs are parked.
    let model_len = unsafe { MODEL_LEN };


    // Parse header + collect the tensor table.
    let model = unsafe { core::slice::from_raw_parts((&raw const MODEL_BUF) as *const u8, model_len) };
    let info = match gguf::parse_and_dump(model) {
        Ok(i) => i,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: gguf: {e}
"));
            crate::park()
        }
    };


    // Locate mm.a / mm.b offsets within the data section.
    let Some((_na, _da, _ta, a_off)) = gguf::find_tensor(b"mm.a") else {
        uart::write_str("FAIL: mm.a not found
");
        crate::park()
    };
    let Some((_nb, _db, _tb, b_off)) = gguf::find_tensor(b"mm.b") else {
        uart::write_str("FAIL: mm.b not found
");
        crate::park()
    };
    let a_len = matmul::M * matmul::K;
    let b_len = matmul::N * matmul::K;


    // Copy tensor bytes into the matmul buffers (data_start + offset).
    // Soundness: model bytes are identity-mapped RAM loaded by the FAT read;
    // both tensors lie fully inside the loaded file length.
    unsafe {
        core::ptr::copy_nonoverlapping(
            model.as_ptr().add(info.data_start as usize + a_off as usize),
            &raw mut matmul::A_BUF as *mut u8, a_len);
        core::ptr::copy_nonoverlapping(
            model.as_ptr().add(info.data_start as usize + b_off as usize),
            &raw mut matmul::B_BUF as *mut u8, b_len);
    }


    // Scalar reference.
    let mut c_ref = [0i32; matmul::M * matmul::N];
    matmul::matmul_scalar(
        unsafe { core::slice::from_raw_parts((&raw const matmul::A_BUF) as *const u8, a_len) },
        unsafe { core::slice::from_raw_parts((&raw const matmul::B_BUF) as *const u8, b_len) },
        &mut c_ref,
    );


    // Debug: the first bytes of each tensor as the kernel sees them.
    let a_view = unsafe { core::slice::from_raw_parts((&raw const matmul::A_BUF) as *const u8, 16) };
    let b_view = unsafe { core::slice::from_raw_parts((&raw const matmul::B_BUF) as *const u8, 16) };
    uart::locked_write(format_args!("mm: a[0..8]={:02x?} b[0..8]={:02x?}
", &a_view[..8], &b_view[..8]));

    // Clean A/B so the APs' uncached reads see the loaded data.
    cache::clean_range(
        (&raw const matmul::A_BUF) as usize,
        core::mem::size_of::<[u8; matmul::M * matmul::K]>(),
    );
    cache::clean_range(
        (&raw const matmul::B_BUF) as usize,
        core::mem::size_of::<[u8; matmul::N * matmul::K]>(),
    );

    // NEON UDOT path across all cores.
    pool::run_on_all(matmul::pool_matmul_job, 0, board::CORE_COUNT);
    // The APs wrote C_BUF with their MMU off (uncached); invalidate the
    // BSP's stale cached lines so the comparison reads fresh RAM.
    cache::invalidate_range(
        (&raw const matmul::C_BUF) as usize,
        core::mem::size_of::<[i32; matmul::M * matmul::N]>(),
    );


    // Compare C buffers exactly.
    let c_buf = unsafe { core::slice::from_raw_parts((&raw const matmul::C_BUF) as *const i32, matmul::M * matmul::N) };
    let mut mismatches = 0usize;
    for (i, (r, g)) in c_ref.iter().zip(c_buf.iter()).enumerate() {
        if r != g {
            if mismatches == 0 {
                uart::locked_write(format_args!("FAIL: first mismatch at C[{i}] ref={r} got={g}
"));
            }
            mismatches += 1;
        }
    }
    // Debug: first outputs from both paths.
    for i in 0..8 {
        uart::locked_write(format_args!(
            "matmul: C[{i}] udot={} scalar={}
", c_buf[i], c_ref[i]
        ));
    }
    if mismatches == 0 {
        uart::locked_write(format_args!(
            "PASS: matmul {}/{} elements exact (C00={} C1515={} SUM={})
",
            c_ref.len(), c_ref.len(),
            c_buf[0], c_buf[matmul::M * matmul::N - 1],
            c_buf.iter().fold(0i64, |a, &v| a + v as i64)
        ));
    } else {
        uart::locked_write(format_args!("FAIL: matmul {mismatches} mismatches
"));
    }
    park()
}


/// Appliance network service (Pi-5): bring up virtio-net and run the
/// poll/serve loop forever — TCP 8080 answers MARKOS-PING with
/// MARKOS-PONG for the transport acceptance gate.
#[cfg(feature = "selftest-net")]
fn selftest_net() -> ! {
    match net::init() {
        Ok(()) => {
            let mut mac = [0u8; 18];
            let m = net::mac_string(&mut mac);
            uart::locked_write(format_args!(
                "net: virtio-net up mac={:?} ip=10.0.2.15:{}
",
                core::str::from_utf8(&mac[..m]).unwrap_or("?"),
                net::LISTEN_PORT
            ));
        }
        Err(e) => {
            uart::locked_write(format_args!("FAIL: net init: {e}
"));
            crate::park()
        }
    }
    uart::write_str("net: serving
");
    net::serve_loop()
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
    // translation fault.
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

/// Acceptance test (Pi-3a): virtio-blk bring-up + first read (LBA0 MBR
/// signature check) against the QEMU virtio-mmio device.
#[cfg(feature = "selftest-block")]
fn selftest_block() -> ! {
    uart::write_str("selftest: virtio-blk bring-up + LBA0 read\n");
    match virtio_blk::bring_up_and_verify() {
        Ok(sectors) => {
            uart::locked_write(format_args!(
                "PASS: block device verified ({sectors} sectors)\n"
            ));
        }
        Err(e) => {
            uart::locked_write(format_args!("FAIL: block: {e}\n"));
        }
    }
    park()
}

/// Acceptance test (Pi-3b/c): FAT32 mount over the block device, MODEL.BIN
/// lookup in the root directory, full cluster-chain read, then a GGUF v3
/// parse of the model file.
#[cfg(feature = "selftest-fat")]
fn selftest_fat() -> ! {
    uart::write_str("selftest: FAT32 mount + file read\n");
    if let Err(e) = virtio_blk::init() {
        uart::locked_write(format_args!("FAIL: virtio init: {e}\n"));
        crate::park()
    }
    static mut FAT_BUF: [u8; 65536] = [0u8; 65536];

    match fat::mount() {
        Ok(vol) => match vol.open_model() {
            Ok(file) => {
                let mut bytes = 0usize;
                // Soundness: FAT_BUF is boot-stage scratch owned by this
                // selftest; the device DMAs into it while nothing else runs.
                unsafe {
                    let buf = core::slice::from_raw_parts_mut(
                        (&raw mut FAT_BUF) as *mut u8,
                        65536,
                    );
                    match vol.read_file(&file, buf) {
                        Ok(n) => {
                            bytes = n;
                        }
                        Err(e) => {
                            uart::locked_write(format_args!("FAIL: FAT read_file: {e}\n"));
                            crate::park()
                        }
                    }
                    // GGUF parse: the model file must be well-formed GGUF v3.
                    match gguf::parse_and_dump(&buf[..file.size as usize]) {
                        Ok(info) => {
                            uart::locked_write(format_args!(
                                "PASS: gguf parsed v{} tensors={}\n",
                                info.version,
                                info.tensor_count
                            ));
                        }
                        Err(e) => {
                            uart::locked_write(format_args!("FAIL: gguf: {e}\n"));
                        }
                    }
                }
                uart::locked_write(format_args!(
                    "PASS: FAT32 file read, {bytes} bytes\n"
                ));
            }
            Err(e) => {
                uart::locked_write(format_args!("FAIL: FAT open: {e}\n"));
            }
        },
        Err(e) => {
            uart::locked_write(format_args!("FAIL: FAT mount: {e}\n"));
        }
    }
    park()
}

/// Acceptance test (Pi-4): parallel sum over a 65536-element array across
/// every core via the execution pool, matching the single-threaded
/// reference exactly.
#[cfg(feature = "selftest-pool")]
#[repr(C)]
struct PoolSumDesc {
    data: *const u64,
    len: usize,
    partials: *mut u64,
}

#[cfg(feature = "selftest-pool")]
fn selftest_pool() -> ! {
    uart::write_str("selftest: parallel sum over 65536 elements\n");
    static mut POOL_DATA: [u64; 65536] = [0; 65536];
    static mut POOL_PARTIALS: [u64; 8] = [0; 8];

    unsafe {
        for i in 0..65536usize {
            POOL_DATA[i] = (i % 251) as u64;
        }
    }

    // Single-threaded reference.
    let mut reference = 0u64;
    // Soundness: POOL_DATA is exclusively owned by this selftest.
    unsafe {
        for i in 0..65536usize {
            reference = reference.wrapping_add(POOL_DATA[i]);
        }
    }

    // Parallel: split the array into per-core contiguous chunks.
    let desc = PoolSumDesc {
        data: &raw const POOL_DATA as *const u64,
        len: 65536,
        partials: &raw mut POOL_PARTIALS as *mut u64,
    };
    pool::run_on_all(pool_sum_job, &desc as *const PoolSumDesc as u64, board::CORE_COUNT);

    let mut total = 0u64;
    for p in 0..board::CORE_COUNT {
        // Soundness: per-core slots, written before the pool barrier.
        total = total.wrapping_add(unsafe { POOL_PARTIALS[p] });
    }

    if total == reference {
        uart::locked_write(format_args!(
            "pool: PASS sum={total} across {}/{} cores\n",
            board::CORE_COUNT,
            board::CORE_COUNT
        ));
    } else {
        uart::locked_write(format_args!(
            "FAIL: parallel sum {total} != reference {reference}\n"
        ));
    }
    park()
}

#[cfg(feature = "selftest-pool")]
fn pool_sum_job(core_id: usize, arg: u64) {
    // Soundness: `arg` points at the PoolSumDesc published by the BSP for
    // the duration of the job; chunks are disjoint per core.
    let d = unsafe { &*(arg as *const PoolSumDesc) };
    let chunk = d.len / board::CORE_COUNT;
    let start = core_id * chunk;
    let end = if core_id + 1 == board::CORE_COUNT { d.len } else { start + chunk };
    let mut acc = 0u64;
    for i in start..end {
        acc = acc.wrapping_add(unsafe { d.data.add(i).read() });
    }
    unsafe { d.partials.add(core_id).write_volatile(acc) };
}

/// Panic path: print and park. Interrupts are masked at EL1.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    uart::write_str("KERNEL PANIC: ");
    uart::locked_write(format_args!("{}\n", info));
    park()
}
