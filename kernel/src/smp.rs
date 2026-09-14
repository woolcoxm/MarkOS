//! SMP: per-core release and the Phase acceptance counters.
//!
//! Release mechanisms, by board:
//!  - qemu-virt: PSCI CPU_ON (architected firmware protocol).
//!  - Raspberry Pi: the firmware's armstub parks secondary cores spinning
//!    on the ARM64 spin-table slots at 0xD8 + 0x10*core — the BSP publishes
//!    the entry address there and wakes them with `sev`. (Legacy: cores that
//!    DID enter the kernel stub wait on CORE_RELEASE; both paths converge.)
//!
//! owns: the per-core AP stacks and the ONLINE/WORK counters.
//! invariants:
//! - A released core does its WORK increments BEFORE setting ONLINE; the
//!   BSP reads ONLINE with SeqCst, so an online core's work is visible.
//! - Secondaries run with their own MMU off (identity addresses, uncached)
//!   — sufficient for bring-up; per-core MMU setup is a later phase.
//! - Real-Pi caveat: BCM2837 (Pi 3) cores are not cache-coherent with each
//!   other; the deployment boards (Pi 4/5, A72/A76) are coherent.

use core::arch::global_asm;
use core::hint::spin_loop;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::uart;

static AP_ONLINE: AtomicU32 = AtomicU32::new(0);
static AP_WORK: AtomicU64 = AtomicU64::new(0);
const WORK_ITERATIONS: u64 = 1000;

/// 64 KiB private stack per core (index = core id; index 0 unused: the BSP
/// keeps the firmware-provided boot stack). no_mangle: referenced by name
/// from the secondary entry stub.
#[unsafe(no_mangle)]
static mut AP_STACKS: [[u64; 0x10000 / 8]; 4] = [[0; 0x10000 / 8]; 4];

/// Legacy release slots for cores that DID enter our kernel stub and wait
/// there (belt-and-suspenders; the spin-table/PSCI paths are the norm).
#[unsafe(no_mangle)] // referenced by name from the boot stub's adrp
#[unsafe(link_section = ".data")]
static CORE_RELEASE: [AtomicU64; 4] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

global_asm!(
    ".globl secondary_entry",
    "secondary_entry:",
    // Entered with MMU off, no stack, x0 = context (ignored: core id is
    // re-read from MPIDR so every release path works).
    "    mrs x0, mpidr_el1",
    "    and x0, x0, #3",          // core id
    "    adrp x1, AP_STACKS",
    "    add x1, x1, :lo12:AP_STACKS",
    "    mov x2, #0x10000",        // stack size per core
    "    madd x1, x0, x2, x1",     // AP_STACKS + core_id * 0x10000
    "    add x1, x1, x2",          // top of this core's stack
    "    mov sp, x1",
    "    bl secondary_main",
    "    b secondary_park",
    "secondary_park:",
    "    wfe",
    "    b secondary_park",
);

unsafe extern "C" {
    static secondary_entry: u8;
}

/// Entry for released cores; `core_id` arrives in x0 from the stub.
/// Every core contributes WORK_ITERATIONS increments, then reports online
/// and parks. MMU note: secondaries run with their own MMU off (identity
/// addresses, uncached) — sufficient for bring-up; per-core MMU setup is a
/// later phase if/when a core needs it.
#[unsafe(no_mangle)]
extern "C" fn secondary_main(core_id: u64) -> ! {
    for _ in 0..WORK_ITERATIONS {
        AP_WORK.fetch_add(1, Ordering::SeqCst);
    }
    uart::locked_write(format_args!(
        "ap: core {core_id} online, work contributed\n"
    ));
    // Report online LAST: the BSP treats ONLINE as "this core is completely
    // done", so its follow-up output cannot race the AP's log line.
    AP_ONLINE.fetch_add(1, Ordering::SeqCst);
    crate::park()
}

fn ap_stack_top(core: usize) -> u64 {
    // Soundness: AP_STACKS is a static per-core array; slot ownership is
    // exclusive (BSP releases at most one core per slot, once).
    let addr = unsafe {
        (&raw const AP_STACKS[core]) as usize + core::mem::size_of::<[u64; 0x10000 / 8]>()
    };
    addr as u64
}

/// Release every parked core and wait for all of them to report online.
/// Returns (cores_online, work_expected) for the caller's check.
pub fn start_aps(core_count: usize) -> Result<(u32, u64), &'static str> {
    // Soundness: the release targets are this kernel image itself and
    // identity-mapped RAM — valid on every supported board.
    let entry = unsafe { &raw const secondary_entry } as usize as u64;

    for core in 1..core_count {
        let before = AP_ONLINE.load(Ordering::SeqCst);

        // Legacy stub-parked path (a core that entered the kernel stub).
        CORE_RELEASE[core].store(ap_stack_top(core), Ordering::SeqCst);

        // Board-specific release.
        crate::board_release::release_core(core as u64, entry);

        let mut tries = 0u64;
        while AP_ONLINE.load(Ordering::SeqCst) == before {
            spin_loop();
            tries += 1;
            // Fast fail: on boards whose release path doesn't apply (e.g.
            // QEMU raspi3b's AArch32-parked secondaries) the boot continues
            // single-core instead of hanging the acceptance run.
            if tries > 20_000_000 {
                return Err("core failed to come online");
            }
        }
    }
    Ok((AP_ONLINE.load(Ordering::SeqCst), (core_count as u64) * WORK_ITERATIONS))
}

/// BSP participation in the shared-counter acceptance test.
pub fn bsp_do_work() {
    for _ in 0..WORK_ITERATIONS {
        AP_WORK.fetch_add(1, Ordering::SeqCst);
    }
}

pub fn online() -> u32 {
    AP_ONLINE.load(Ordering::SeqCst)
}

pub fn work_total() -> u64 {
    AP_WORK.load(Ordering::SeqCst)
}
