//! Fixed thread-per-core execution pool — no scheduler, no sleeps.
//!
//! Model: the BSP publishes a job (function + argument) and bumps a
//! generation counter; every released core spins on the generation, runs
//! the function with its own core id, and reports completion. Cores never
//! sleep between jobs — spinning eliminates scheduler and wake-up latency,
//! which matters more for inference throughput than power draw on a
//! wall-powered appliance.
//!
//! owns: the job mailbox (fn ptr, arg, generation) and the completion
//! counter.
//! invariants:
//! - Jobs run sequentially: the BSP waits for all APs to finish job N
//!   before publishing job N+1 (guaranteed by `run_on_all` waiting for
//!   JOBS_DONE to return to zero relative to the previous dispatch).
//! - `JOB_FN` must point at a `fn(usize, u64)` — BSP publishes only
//!   function items from this image (single address space, no ASLR).
//! - The job function must not block, sleep, or park — it runs to
//!   completion and returns.

use core::arch::asm;
use core::hint::spin_loop;
use core::sync::atomic::{AtomicU64, Ordering};

use crate::uart;

static JOB_FN: AtomicU64 = AtomicU64::new(0);
static JOB_ARG: AtomicU64 = AtomicU64::new(0);
static JOB_GEN: AtomicU64 = AtomicU64::new(0);
static JOBS_DONE: AtomicU64 = AtomicU64::new(0);

/// AP-side pool loop: takes a core id, runs published jobs forever.
pub fn ap_pool_loop(core_id: usize) -> ! {
    uart::locked_write(format_args!("pool: ap {core_id} entering pool loop
"));
    let mut seen: u64 = 0;
    loop {
        let generation = JOB_GEN.load(Ordering::Acquire);
        if generation > seen {
            seen = generation;
            // Soundness: JOB_FN is only ever published by `run_on_all` from
            // function items in this image; single address space means the
            // pointer is valid on every core.
            let func = unsafe {
                core::mem::transmute::<usize, fn(usize, u64)>(JOB_FN.load(Ordering::Acquire) as usize)
            };
            let arg = JOB_ARG.load(Ordering::Acquire);
            func(core_id, arg);
            JOBS_DONE.fetch_add(1, Ordering::Release);
        } else {
            spin_loop();
        }
    }
}

/// Run `func(core_id, arg)` on every core (BSP included) and return after
/// all cores finished. Sequential: call again only after this returns.
pub fn run_on_all(func: fn(usize, u64), arg: u64, core_count: usize) {
    uart::write_str("pool: publishing job
");
    JOBS_DONE.store(0, Ordering::Release);
    JOB_FN.store(func as usize as u64, Ordering::Release);
    JOB_ARG.store(arg, Ordering::Release);
    // Release: publish generation last — APs acquire it before reading the
    // job fields.
    JOB_GEN.fetch_add(1, Ordering::AcqRel);

    func(0, arg); // the BSP is core 0 and runs the job inline
    uart::write_str("pool: bsp share done, waiting for APs
");

    let mut bsp_probe = 0u64;
    while JOBS_DONE.load(Ordering::Acquire) < (core_count - 1) as u64 {
        spin_loop();
        bsp_probe += 1;
        if bsp_probe >= 0x0040_0000 {
            bsp_probe = 0;
            uart::locked_write(format_args!(
                "pool: bsp probe gen={} jobs_done={}
",
                JOB_GEN.load(Ordering::Acquire),
                JOBS_DONE.load(Ordering::Acquire)
            ));
        }
    }
    uart::write_str("pool: all cores done
");
}

/// Park helper kept for symmetry (unused by the pool itself).
#[allow(dead_code)]
pub fn noop_job(_: usize, _: u64) {
    // Soundness: memory fence for benchmark loops; no side effects.
    unsafe { asm!("dmb sy", options(nomem, nostack, preserves_flags)) }
}
