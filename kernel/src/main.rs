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
#![feature(core_intrinsics)]
#![no_main]

mod board;
mod board_release;
mod cache;
mod config;
mod control;
mod cpu;
mod engine;
mod fat;
mod gguf;
mod matmul;
mod mmu;
mod net;
mod pcie;
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
    #[cfg(feature = "selftest-model")]
    selftest_model();
    #[cfg(feature = "selftest-forward")]
    selftest_forward();
    #[cfg(feature = "selftest-gen")]
    selftest_gen();
    #[cfg(feature = "selftest-pcie")]
    selftest_pcie();

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
    // Installer-baked config first: it sets the control IP/port/token.
    control::boot();
    match net::init() {
        Ok(()) => {
            let mut mac = [0u8; 18];
            let m = net::mac_string(&mut mac);
            let ip = config::ip();
            uart::locked_write(format_args!(
                "net: virtio-net up mac={:?} ip={}.{}.{}.{}:{}
",
                core::str::from_utf8(&mac[..m]).unwrap_or("?"),
                ip[0], ip[1], ip[2], ip[3],
                config::port()
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

/// Pi-7a acceptance: PCIe ECAM walk on the virt machine. The QEMU command
/// line attaches a pcie-root-port (Red Hat vendor 1b36) — finding it proves
/// the config-space walk; the same code targets the BCM2712 root complex
/// and the LLM8850 endpoint on real hardware.
#[cfg(feature = "selftest-pcie")]
fn selftest_pcie() -> ! {
    uart::write_str("selftest: PCIe ECAM enumeration\n");
    let n = pcie::scan();
    let mut out = [pcie::Device::ZERO; 8];
    let found = pcie::devices(&mut out);
    let has_root_port = out[..found].iter().any(|d| d.vendor_id == 0x1B36);
    if found > 0 && has_root_port {
        uart::locked_write(format_args!(
            "PASS: pcie enumerated {found} device(s), root port present\n"
        ));
    } else {
        uart::locked_write(format_args!(
            "FAIL: pcie scan found {n} device(s), root port={}\n",
            has_root_port
        ));
    }
    park()
}

/// Phase 7 acceptance: load a REAL GGUF (Qwen3-0.6B q8_0, 640 MB) from the
/// QEMU disk image. Only the metadata section is buffered — the full tensor
/// table (310 tensors) is parsed, and chosen tensor payloads are CRC-checked
/// by reading them directly from the FAT volume at their data-section
/// offsets (read_at). The gate diffs this output against the host
/// reference produced by scripts/gguf_ref.py on the same file.
#[cfg(feature = "selftest-model")]
fn selftest_model() -> ! {
    uart::write_str("selftest: real GGUF load\n");
    if let Err(e) = virtio_blk::init() {
        uart::locked_write(format_args!("FAIL: virtio init: {e}\n"));
        park()
    }

    // Metadata section of the GGUF (kv pairs incl. tokenizer arrays + the
    // full tensor table). Qwen3-0.6B's metadata ends at ~5.95 MB.
    const META_BUF_LEN: usize = 8 * 1024 * 1024;
    static mut META_BUF: [u8; META_BUF_LEN] = [0u8; META_BUF_LEN];

    let vol = match fat::mount() {
        Ok(v) => v,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: fat mount: {e}\n"));
            park()
        }
    };
    let file = match vol.open_model() {
        Ok(f) => f,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: open model: {e}\n"));
            park()
        }
    };
    let n = {
        // Soundness: META_BUF is selftest-stage scratch on the BSP; the
        // device DMAs into it while nothing else runs.
        let buf = unsafe {
            core::slice::from_raw_parts_mut(
                (&raw mut META_BUF) as *mut u8,
                META_BUF_LEN,
            )
        };
        match vol.read_at(&file, 0, buf) {
            Ok(n) => n,
            Err(e) => {
                uart::locked_write(format_args!("FAIL: metadata read: {e}\n"));
                park()
            }
        }
    };

    let info = match gguf::parse_and_dump(unsafe {
        // Soundness: META_BUF is selftest scratch, single-core owned.
        core::slice::from_raw_parts((&raw const META_BUF) as *const u8, n)
    }) {
        Ok(i) => i,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: gguf parse: {e}\n"));
            park()
        }
    };
    // Byte-matches scripts/gguf_ref.py output.
    uart::locked_write(format_args!(
        "model: version={} tensors={} kv={} data_start={:#x} align={}\n",
        info.version, info.tensor_count, info.kv_count, info.data_start, info.alignment
    ));

    // First 8 tensors of the table, in table order.
    let mut snap = [gguf::TensorEntry {
        name: [0; gguf::MAX_NAME],
        name_len: 0,
        dims: [0; 4],
        ttype: 0,
        offset: 0,
    }; 8];
    let n_snap = gguf::snapshot(&mut snap);
    for (i, t) in snap.iter_mut().enumerate().take(n_snap) {
        uart::locked_write(format_args!("MT {i} ",));
        for &b in t.name() {
            uart::write_byte(if (0x20..0x7F).contains(&b) { b } else { b'?' });
        }
        uart::locked_write(format_args!(
            " {}x{}x{}x{} type={} off={}\n",
            t.dims[0], t.dims[1], t.dims[2], t.dims[3], t.ttype, t.offset
        ));
    }

    // Payload checks: CRC-32 + first bytes of chosen tensors, read straight
    // from the volume at their data-section offsets. Missing names are
    // skipped (matches the reference: Qwen3-0.6B has no output.weight).
    const CHECK: [&[u8]; 3] = [
        b"token_embd.weight",
        b"blk.0.attn_q.weight",
        b"output.weight",
    ];
    const VAL_BUF_LEN: usize = 256;
    static mut VAL_BUF: [u8; VAL_BUF_LEN] = [0u8; VAL_BUF_LEN];
    for want in CHECK {
        let Some((_name, dims, ttype, off)) = gguf::find_tensor(want) else {
            continue;
        };
        let abs = info.data_start + off;
        let val = unsafe {
            core::slice::from_raw_parts_mut((&raw mut VAL_BUF) as *mut u8, VAL_BUF_LEN)
        };
        let got = match vol.read_at(&file, abs, val) {
            Ok(g) => g,
            Err(e) => {
                uart::locked_write(format_args!(
                    "FAIL: tensor read at {abs:#x}: {e}\n"
                ));
                park()
            }
        };
        if got < val.len() {
            uart::locked_write(format_args!(
                "FAIL: short tensor read at {abs:#x}: {got}\n"
            ));
            park()
        }
        let crc = gguf::crc32(val);
        uart::locked_write(format_args!("MV ",));
        for &b in want {
            uart::write_byte(b);
        }
        uart::locked_write(format_args!(
            " crc={crc:08x} first4={:02x}{:02x}{:02x}{:02x} dims={}x{}x{}x{} type={ttype}\n",
            val[0], val[1], val[2], val[3],
            dims[0], dims[1], dims[2], dims[3]
        ));
    }

    uart::locked_write(format_args!("PASS: model load\n"));
    psci::system_off()
}

/// Phase 8a acceptance: forward pass through decoder layer 0 of the REAL
/// Qwen3-0.6B GGUF — BPE tokenizer, embedding, RMSNorm, q8_0 matmuls,
/// per-head q/k norm, RoPE, GQA attention, SwiGLU FFN — for a two-token
/// prompt. Output lines are tolerance-compared against the numpy reference
/// (scripts/forward_ref.py) by the test-forward gate.
#[cfg(feature = "selftest-forward")]
fn selftest_forward() -> ! {
    const META_LEN: usize = 8 * 1024 * 1024;
    static mut META_BUF: [u8; META_LEN] = [0u8; META_LEN];
    const QW_LEN: usize = 128;
    static mut QW: [f32; QW_LEN] = [0f32; QW_LEN];
    static mut KW: [f32; QW_LEN] = [0f32; QW_LEN];
    static mut PROBS: [f32; 8] = [0f32; 8];
    const PROMPT: &[u8] = b"hello world";

    uart::write_str("selftest: forward pass (layer 0)\n");
    if let Err(e) = virtio_blk::init() {
        uart::locked_write(format_args!("FAIL: virtio init: {e}\n"));
        park()
    }

    let vol = match fat::mount() {
        Ok(v) => v,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: fat mount: {e}\n"));
            park()
        }
    };
    let file = match vol.open_model() {
        Ok(f) => f,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: open model: {e}\n"));
            park()
        }
    };

    let n = {
        let buf = unsafe {
            core::slice::from_raw_parts_mut((&raw mut META_BUF) as *mut u8, META_LEN)
        };
        match vol.read_at(&file, 0, buf) {
            Ok(n) => n,
            Err(e) => {
                uart::locked_write(format_args!("FAIL: metadata read: {e}\n"));
                park()
            }
        }
    };
    let meta = unsafe {
        core::slice::from_raw_parts((&raw const META_BUF) as *const u8, n)
    };
    let info = match gguf::parse_and_dump(meta) {
        Ok(i) => i,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: gguf parse: {e}\n"));
            park()
        }
    };
    let geo = match engine::geometry() {
        Ok(g) => g,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: geometry: {e}\n"));
            park()
        }
    };
    uart::locked_write(format_args!(
        "geo: layers={} embd={} heads={} kv={} head_dim={} ffn={}\n",
        geo.n_layers, geo.n_embd, geo.n_heads, geo.n_kv, geo.head_dim, geo.n_ff
    ));

    let mut ids = [0u32; engine::MAX_TOKENS];
    let n_tok = match engine::tokenize(meta, PROMPT, &mut ids) {
        Ok(n) => n,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: tokenize: {e}\n"));
            park()
        }
    };
    uart::locked_write(format_args!(
        "TOKS prompt={} ids=",
        core::str::from_utf8(PROMPT).unwrap_or("?")
    ));
    for (i, id) in ids.iter().take(n_tok).enumerate() {
        if i > 0 {
            uart::write_byte(b',');
        }
        uart::locked_write(format_args!("{id}"));
    }
    uart::locked_write(format_args!(" n_tokens={n_tok}\n"));

    // Absolute tensor base = data section start + table offset.
    let ds = info.data_start;
    let base = |name: &[u8]| -> Option<(u64, [u64; 4], u32)> {
        gguf::find_tensor(name).map(|(_, dims, ttype, off)| (ds + off, *dims, ttype))
    };
    let Some((emb_abs, emb_dims, _)) = base(b"token_embd.weight") else {
        uart::write_str("FAIL: token_embd.weight\n");
        park()
    };
    let row_elems = emb_dims[0] as usize;

    let mut qw = [0f32; QW_LEN];
    let mut kw = [0f32; QW_LEN];
    let mut nw = [0f32; QW_LEN];
    let hd = geo.head_dim;

    let act = engine::activations_mut();
    let mut probs = [0f32; 8];

    macro_rules! need {
        ($name:expr) => {
            match base($name) {
                Some(v) => v,
                None => {
                    uart::locked_write(format_args!("FAIL: tensor {}\n", core::str::from_utf8($name).unwrap_or("?")));
                    park()
                }
            }
        };
    }

    for t in 0..n_tok {
        uart::write_str("dbg: t loop
");
        // Embedding row for this token.
        let row = emb_abs + (ids[t] as u64) * (row_elems / 32 * 34) as u64;
        if engine::dequant_q8_0_row(&vol, &file, row, row_elems, &mut act.x[..geo.n_embd]).is_err()
        {
            uart::write_str("FAIL: embed\n");
            park()
        }
        uart::write_str("dbg: dequant ok
");
        let rms = engine::fsqrt32(
            act.x[..geo.n_embd].iter().map(|v| v * v).sum::<f32>() / geo.n_embd as f32,
        );
        uart::locked_write(format_args!(
            "EMB{t} id={} rms={rms:.6e} v={:.6e},{:.6e},{:.6e},{:.6e}\n",
            ids[t], act.x[0], act.x[1], act.x[2], act.x[3]
        ));

        // Input RMSNorm.
        let (an_abs, _, _) = need!(b"blk.0.attn_norm.weight");
        let _ = engine::rmsnorm_with_weight(
            &vol, &file, an_abs, &act.x[..geo.n_embd], &mut act.n1[..geo.n_embd], geo.eps,
        );
        uart::locked_write(format_args!(
            "NRM{t} v={:.6e},{:.6e},{:.6e},{:.6e}\n",
            act.n1[0], act.n1[1], act.n1[2], act.n1[3]
        ));

        // Q/K/V projections.
        let (q_abs, q_dims, _) = need!(b"blk.0.attn_q.weight");
        let (k_abs, k_dims, _) = need!(b"blk.0.attn_k.weight");
        let (v_abs, v_dims, _) = need!(b"blk.0.attn_v.weight");
        let _ = engine::matvec_q8_0(
            &vol, &file, q_abs, q_dims[0] as usize, q_dims[1] as usize,
            &act.n1[..geo.n_embd], &mut act.q,
        );
        let _ = engine::matvec_q8_0(
            &vol, &file, k_abs, k_dims[0] as usize, k_dims[1] as usize,
            &act.n1[..geo.n_embd], &mut act.k,
        );
        let _ = engine::matvec_q8_0(
            &vol, &file, v_abs, v_dims[0] as usize, v_dims[1] as usize,
            &act.n1[..geo.n_embd], &mut act.v,
        );

        // Per-head q/k norm + RoPE at position t.
        let (qn_abs, _, _) = need!(b"blk.0.attn_q_norm.weight");
        let (kn_abs, _, _) = need!(b"blk.0.attn_k_norm.weight");
        let _ = engine::read_f32_vec(&vol, &file, qn_abs, hd, &mut qw[..hd]);
        let _ = engine::read_f32_vec(&vol, &file, kn_abs, hd, &mut kw[..hd]);
        engine::head_norm_rope(
            &mut act.q[..geo.n_heads * hd], geo.n_heads, hd, &qw[..hd], geo.eps, t, geo.theta,
        );
        engine::head_norm_rope(
            &mut act.k[..geo.n_kv * hd], geo.n_kv, hd, &kw[..hd], geo.eps, t, geo.theta,
        );
        uart::locked_write(format_args!(
            "QK{t} q={:.6e},{:.6e},{:.6e},{:.6e} k={:.6e},{:.6e},{:.6e},{:.6e}\n",
            act.q[0], act.q[1], act.q[2], act.q[3],
            act.k[0], act.k[1], act.k[2], act.k[3]
        ));

        // Cache K/V for this position, then causal attention.
        act.kc[t][..geo.n_kv * hd].copy_from_slice(&act.k[..geo.n_kv * hd]);
        act.vc[t][..geo.n_kv * hd].copy_from_slice(&act.v[..geo.n_kv * hd]);
        act.n_pos = t + 1;
        engine::attend(act, t, &geo, &mut probs);
        if t == 0 {
            uart::locked_write(format_args!(
                "ATT{t} p0={:.6e} o={:.6e},{:.6e},{:.6e},{:.6e}\n",
                probs[0], act.attn[0], act.attn[1], act.attn[2], act.attn[3]
            ));
        } else {
            uart::locked_write(format_args!(
                "ATT{t} p0={:.6e} p1={:.6e} o={:.6e},{:.6e},{:.6e},{:.6e}\n",
                probs[0], probs[1], act.attn[0], act.attn[1], act.attn[2], act.attn[3]
            ));
        }

        // Output projection + residual.
        let (o_abs, o_dims, _) = need!(b"blk.0.attn_output.weight");
        let _ = engine::matvec_q8_0(
            &vol, &file, o_abs, o_dims[0] as usize, o_dims[1] as usize,
            &act.attn[..o_dims[0] as usize], &mut act.mid[..o_dims[1] as usize],
        );
        for i in 0..geo.n_embd {
            act.mid[i] += act.x[i];
        }
        uart::locked_write(format_args!(
            "MID{t} v={:.6e},{:.6e},{:.6e},{:.6e}\n",
            act.mid[0], act.mid[1], act.mid[2], act.mid[3]
        ));

        // FFN: norm -> gate/up -> SiLU gate -> down -> residual.
        let (fn_abs, _, _) = need!(b"blk.0.ffn_norm.weight");
        let _ = engine::rmsnorm_with_weight(
            &vol, &file, fn_abs, &act.mid[..geo.n_embd], &mut act.n1[..geo.n_embd], geo.eps,
        );
        let (g_abs, g_dims, _) = need!(b"blk.0.ffn_gate.weight");
        let (u_abs, u_dims, _) = need!(b"blk.0.ffn_up.weight");
        let (d_abs, d_dims, _) = need!(b"blk.0.ffn_down.weight");
        let n_ff = g_dims[1] as usize;
        let _ = engine::matvec_q8_0(
            &vol, &file, g_abs, g_dims[0] as usize, n_ff,
            &act.n1[..geo.n_embd], &mut act.gate[..n_ff],
        );
        let _ = engine::matvec_q8_0(
            &vol, &file, u_abs, u_dims[0] as usize, u_dims[1] as usize,
            &act.n1[..geo.n_embd], &mut act.up[..u_dims[1] as usize],
        );
        engine::silu(&mut act.gate[..n_ff]);
        for i in 0..n_ff {
            act.gate[i] *= act.up[i];
        }
        let _ = engine::matvec_q8_0(
            &vol, &file, d_abs, d_dims[0] as usize, d_dims[1] as usize,
            &act.gate[..d_dims[0] as usize], &mut act.hid[..d_dims[1] as usize],
        );
        for i in 0..d_dims[1] as usize {
            act.hid[i] += act.mid[i];
        }
        let mut sum = 0f64;
        for i in 0..d_dims[1] as usize {
            sum += act.hid[i] as f64;
        }
        uart::locked_write(format_args!(
            "HID{t} v={:.6e},{:.6e},{:.6e},{:.6e} sum={sum:.6e}\n",
            act.hid[0], act.hid[1], act.hid[2], act.hid[3]
        ));
    }
    uart::locked_write(format_args!("PASS: forward\n"));
    psci::system_off()
}

/// Phase 8b acceptance: FULL model decode — 28-layer prefill of the
/// tokenized prompt, final norm, tied-embedding lm_head argmax, and 4
/// greedy steps — validated token-for-token against the numpy reference
/// (scripts/forward_ref.py --gen).
#[cfg(feature = "selftest-gen")]
fn selftest_gen() -> ! {
    const META_LEN: usize = 8 * 1024 * 1024;
    static mut META_BUF: [u8; META_LEN] = [0u8; META_LEN];
    const GEN_STEPS: usize = 2;
    const PROMPT: &[u8] = b"hello world";

    uart::write_str("selftest: full decode\n");
    if let Err(e) = virtio_blk::init() {
        uart::locked_write(format_args!("FAIL: virtio init: {e}\n"));
        park()
    }
    let vol = match fat::mount() {
        Ok(v) => v,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: fat mount: {e}\n"));
            park()
        }
    };
    let file = match vol.open_model() {
        Ok(f) => f,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: open model: {e}\n"));
            park()
        }
    };
    let n = {
        let buf = unsafe {
            core::slice::from_raw_parts_mut((&raw mut META_BUF) as *mut u8, META_LEN)
        };
        match vol.read_at(&file, 0, buf) {
            Ok(n) => n,
            Err(e) => {
                uart::locked_write(format_args!("FAIL: metadata read: {e}\n"));
                park()
            }
        }
    };
    let meta = unsafe {
        core::slice::from_raw_parts((&raw const META_BUF) as *const u8, n)
    };
    let info = match gguf::parse_and_dump(meta) {
        Ok(i) => i,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: gguf parse: {e}\n"));
            park()
        }
    };
    let geo = match engine::geometry() {
        Ok(g) => g,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: geometry: {e}\n"));
            park()
        }
    };

    let mut ids = [0u32; engine::MAX_TOKENS];
    let n_tok = match engine::tokenize(meta, PROMPT, &mut ids) {
        Ok(n) => n,
        Err(e) => {
            uart::locked_write(format_args!("FAIL: tokenize: {e}\n"));
            park()
        }
    };
    uart::locked_write(format_args!(
        "TOKS prompt={} ids=",
        core::str::from_utf8(PROMPT).unwrap_or("?")
    ));
    for (i, id) in ids.iter().take(n_tok).enumerate() {
        if i > 0 {
            uart::write_byte(b',');
        }
        uart::locked_write(format_args!("{id}"));
    }
    uart::locked_write(format_args!(" n_tokens={n_tok}\n"));

    let ds = info.data_start;
    let Some((_, emb_dims, _, emb_off)) = gguf::find_tensor(b"token_embd.weight") else {
        uart::write_str("FAIL: token_embd.weight\n");
        park()
    };
    let row_elems = emb_dims[0] as usize;
    let emb_abs = ds + emb_off;
    let vocab = emb_dims[1] as u64;
    let Some((_, _, _, fin_off)) = gguf::find_tensor(b"output_norm.weight") else {
        uart::write_str("FAIL: output_norm.weight\n");
        park()
    };
    let fin_abs = ds + fin_off;

    let act = engine::activations_mut();
    let mut gen_ids = [0u32; GEN_STEPS];

    // Prefill: prompt positions 0..n_tok.
    for pos in 0..n_tok {
        let row = emb_abs + (ids[pos] as u64) * (row_elems / 32 * 34) as u64;
        if engine::dequant_q8_0_row(&vol, &file, row, row_elems, &mut act.x[..geo.n_embd])
            .is_err()
        {
            uart::write_str("FAIL: embed\n");
            park()
        }
        for l in 0..geo.n_layers as usize {
            if engine::layer_forward(&vol, &file, ds, l, &geo, pos, act).is_err() {
                uart::write_str("FAIL: layer\n");
                park()
            }
        }
    }

    let mut lsum = 0f64;
    for i in 0..geo.n_embd {
        lsum += act.x[i] as f64;
    }
    uart::locked_write(format_args!(
        "LFIN v={:.6e},{:.6e},{:.6e},{:.6e} sum={lsum:.6e}
",
        act.x[0], act.x[1], act.x[2], act.x[3]
    ));

    // Greedy generation from the last prompt position.
    for g in 0..GEN_STEPS {
        let _ = engine::rmsnorm_with_weight(
            &vol, &file, fin_abs, &act.x[..geo.n_embd], &mut act.n1[..geo.n_embd], geo.eps,
        );
        let Ok((tok, logit)) =
            engine::argmax_q8_0(&vol, &file, emb_abs, row_elems, vocab, &act.n1[..geo.n_embd])
        else {
            uart::write_str("FAIL: lm_head\n");
            park()
        };
        gen_ids[g] = tok;
        uart::locked_write(format_args!("STEP{g} tok={tok} logit={logit:.6e}\n"));
        if g + 1 < GEN_STEPS {
            let row = emb_abs + (tok as u64) * (row_elems / 32 * 34) as u64;
            if engine::dequant_q8_0_row(&vol, &file, row, row_elems, &mut act.x[..geo.n_embd])
                .is_err()
            {
                uart::write_str("FAIL: embed\n");
                park()
            }
            let pos = n_tok + g;
            for l in 0..geo.n_layers as usize {
                if engine::layer_forward(&vol, &file, ds, l, &geo, pos, act).is_err() {
                    uart::write_str("FAIL: layer\n");
                    park()
                }
            }
        }
    }

    uart::locked_write(format_args!("GEN ids="));
    for (i, g) in gen_ids.iter().enumerate() {
        if i > 0 {
            uart::write_byte(b',');
        }
        uart::locked_write(format_args!("{g}"));
    }
    uart::locked_write(format_args!(" n={GEN_STEPS}\n"));
    uart::locked_write(format_args!("PASS: gen\n"));
    psci::system_off()
}

/// Panic path: print and park. Interrupts are masked at EL1.
#[panic_handler]
fn panic(info: &PanicInfo) -> ! {
    uart::write_str("KERNEL PANIC: ");
    uart::locked_write(format_args!("{}\n", info));
    park()
}
