//! PL011 UART — the serial debug console.
//!
//! owns: nothing; the UART is platform hardware. The base address comes
//! from the board layer (see board.rs).
//! invariants: single-writer at a time during bring-up; later phases route
//! all output through a lock.

use core::fmt;
use core::sync::atomic::{AtomicBool, Ordering};

const UART0_BASE: usize = crate::board::UART_BASE;

// PL011 register offsets.
const DR: usize = 0x00;
const FR: usize = 0x18;
const IBRD: usize = 0x24;
const FBRD: usize = 0x28;
const LCRH: usize = 0x2C;
const CR: usize = 0x30;
const IMSC: usize = 0x38;

const FR_TXFF: u32 = 1 << 5;

fn reg(offset: usize) -> *mut u32 {
    (UART0_BASE + offset) as *mut u32
}

/// Bring the PL011 up at 115200 8N1, no interrupts.
pub fn init() {
    // Divisor for a 48 MHz UART clock: 48e6 / (16 * 115200) = 26.041.
    // QEMU ignores the divisor; real Pi firmware honours it.
    unsafe {
        reg(IMSC).write_volatile(0); // no UART interrupts yet
        reg(CR).write_volatile(0); // disable while configuring
        reg(IBRD).write_volatile(26);
        reg(FBRD).write_volatile(3);
        reg(LCRH).write_volatile(0x70); // 8 data bits, FIFO on, 1 stop, no parity
        reg(CR).write_volatile(0x301); // UARTEN | TXE | RXE
    }
}

pub fn write_byte(b: u8) {
    // Soundness: device register access; volatile is the defined pattern.
    unsafe {
        while reg(FR).read_volatile() & FR_TXFF != 0 {
            core::hint::spin_loop();
        }
        reg(DR).write_volatile(b as u32);
    }
}

/// Serial transmit lock: cores print concurrently once SMP is up; without
/// this, multi-core log lines interleave mid-write (observed live).
static TX_LOCK: AtomicBool = AtomicBool::new(false);

/// Lock-protected formatted write: the whole line lands as one unit, so
/// multi-core output does not interleave mid-line.
pub fn locked_write(args: fmt::Arguments) {
    while TX_LOCK
        .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
        .is_err()
    {
        core::hint::spin_loop();
    }
    let _ = fmt::write(&mut Serial, args);
    TX_LOCK.store(false, Ordering::Release);
}

pub fn write_str(s: &str) {
    for b in s.bytes() {
        if b == b'\n' {
            write_byte(b'\r');
        }
        write_byte(b);
    }
}

/// `core::fmt::Write` adapter so `write!` works without alloc.
pub struct Serial;

impl fmt::Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        write_str(s);
        Ok(())
    }
}
