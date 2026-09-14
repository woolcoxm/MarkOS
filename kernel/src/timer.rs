//! Generic timer: system counter reads and counter-based delays.
//!
//! owns: nothing; CNTPCT/CNTFRQ are architectural registers.
//! invariants: safe to call once the kernel is running at EL1 (always true
//! after the boot stub).

use core::arch::asm;
use core::fmt::Write as _;

use crate::uart;

/// Counter frequency in Hz (firmware/QEMU constant for the platform).
pub fn frequency() -> u64 {
    let freq: u64;
    // Soundness: read-only system register.
    unsafe { asm!("mrs {}, cntfrq_el0", out(reg) freq, options(nomem, nostack, preserves_flags)) };
    freq
}

/// Raw system counter value (ticks since boot).
pub fn counter() -> u64 {
    let ticks: u64;
    // Soundness: read-only system register.
    unsafe { asm!("mrs {}, cntpct_el0", out(reg) ticks, options(nomem, nostack, preserves_flags)) };
    ticks
}

/// Milliseconds since boot (used by observability/selftest phases).
#[allow(dead_code)]
pub fn uptime_ms() -> u64 {
    counter() / (frequency() / 1000)
}

/// Busy-poll delay measured against the counter (no interrupts needed).
pub fn delay_ms(ms: u64) {
    let target = counter() + ms * (frequency() / 1000);
    while counter() < target {
        core::hint::spin_loop();
    }
}

/// Boot-log line: frequency and a measured delay sanity sample.
pub fn log() {
    let freq = frequency();
    let t0 = counter();
    delay_ms(20);
    let ticks = counter() - t0;
    let _ = write!(
        uart::Serial,
        "timer: cntfrq={freq}Hz, measured 20ms delay = {} ticks ({:.1}ms)\n",
        ticks,
        (ticks as f64 / freq as f64) * 1000.0
    );
}
