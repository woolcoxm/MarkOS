//! COM1 (0x3F8) serial port output — the console of this OS for its whole life.
//!
//! owns: nothing global; all functions are reentrant per call but interleaving
//! between cores may interleave characters (fine for a diagnostic console).
//! invariant: `init()` must have run before `write_*` for reliable line
//! settings, though QEMU's default state also accepts raw writes.

use core::arch::asm;
use core::fmt;

const COM1: u16 = 0x3f8;

/// Program 8250 UART: 115200 baud, 8N1, FIFO on, no interrupts.
pub fn init() {
    outb(COM1 + 1, 0x00); // disable UART interrupts
    outb(COM1 + 3, 0x80); // DLAB on
    outb(COM1 + 0, 0x01); // divisor low byte: 1 => 115200 baud
    outb(COM1 + 1, 0x00); // divisor high byte
    outb(COM1 + 3, 0x03); // DLAB off, 8 data bits, no parity, 1 stop
    outb(COM1 + 2, 0xc7); // enable + clear FIFOs, 14-byte threshold
    outb(COM1 + 4, 0x0b); // DTR, RTS, OUT2
}

pub fn write_byte(b: u8) {
    // Wait for the transmit holding register to be empty (bit 5 of LSR).
    while (inb(COM1 + 5) & 0x20) == 0 {
        core::hint::spin_loop();
    }
    outb(COM1, b);
}

pub fn write_str(s: &str) {
    for b in s.bytes() {
        if b == b'\n' {
            write_byte(b'\r');
        }
        write_byte(b);
    }
}

/// `core::fmt::Write` adapter so `write!` formatting works without alloc.
pub struct Serial;

impl fmt::Write for Serial {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        write_str(s);
        Ok(())
    }
}

#[inline]
fn outb(port: u16, value: u8) {
    // Soundness: writing one byte to an x86 I/O port; `nomem` is safe because
    // the port side effects we rely on are ordering-sensitive, not memory.
    unsafe { asm!("out dx, al", in("dx") port, in("al") value, options(nomem, nostack, preserves_flags)) }
}

#[inline]
fn inb(port: u16) -> u8 {
    let value: u8;
    // Soundness: reading one byte from an x86 I/O port; the value comes from
    // the device, so `nomem` must NOT be used here.
    unsafe { asm!("in al, dx", out("al") value, in("dx") port, options(nostack, preserves_flags)) };
    value
}
