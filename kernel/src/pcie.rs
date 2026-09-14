//! PCIe ECAM enumeration (Pi-7a): walk the config space and log devices.
//!
//! The LLM8850 (Axera AX8850) sits on the Pi 5's PCIe bus as an endpoint;
//! before any transport can be built we need to find it and read its
//! configuration (vendor/device, BARs, link caps). This module implements
//! the generic ECAM walk — the same code that will run against the BCM2712
//! root complex on real hardware (Pi-7b).
//!
//! ECAM addressing: byte offset = (bus << 20) | (device << 15) |
//! (function << 12) + register offset. We scan bus 0..2, 32 devices, 1
//! function each, which covers the root complex, any root ports, and the
//! endpoint behind them on this class of topology.
//!
//! owns: nothing — pure reads against the ECAM window.
//! invariants: called after MMU init with the ECAM window inside the
//! identity map; config space reads to absent devices return all-1s.

use crate::board;

pub const VENDOR_INVALID: u16 = 0xFFFF;

#[derive(Clone, Copy)]
pub struct Device {
    pub bus: u8,
    pub dev: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub class_code: u32, // 24-bit: base | sub | prog-if
}

impl Device {
    pub const ZERO: Device = Device {
        bus: 0,
        dev: 0,
        vendor_id: 0,
        device_id: 0,
        class_code: 0,
    };
}

const MAX_DEVICES: usize = 8;

static mut DEVICES: [Device; MAX_DEVICES] = [Device::ZERO; MAX_DEVICES];
static mut DEVICE_COUNT: usize = 0;

fn cfg_addr(bus: u8, dev: u8, fn_: u8, off: u16) -> *const u32 {
    (board::PCIE_ECAM_BASE
        + ((bus as usize) << 20)
        + ((dev as usize) << 15)
        + ((fn_ as usize) << 12)
        + (off as usize)) as *const u32
}

fn cfg_read32(bus: u8, dev: u8, fn_: u8, off: u16) -> u32 {
    // Soundness: ECAM config reads are volatile device MMIO; the window is
    // identity-mapped by the board's MMU setup.
    unsafe { cfg_addr(bus, dev, fn_, off).read_volatile() }
}

/// Enumerate the bus and stash the found devices. Returns the count.
pub fn scan() -> usize {
    unsafe { DEVICE_COUNT = 0 };
    for bus in 0..=1u8 {
        for dev in 0..32u8 {
            let word = cfg_read32(bus, dev, 0, 0);
            let vendor = (word & 0xFFFF) as u16;
            if vendor == VENDOR_INVALID {
                continue;
            }
            let d = Device {
                bus,
                dev,
                vendor_id: vendor,
                device_id: (word >> 16) as u16,
                class_code: cfg_read32(bus, dev, 0, 8) >> 8,
            };
            unsafe {
                if DEVICE_COUNT < MAX_DEVICES {
                    DEVICES[DEVICE_COUNT] = d;
                    DEVICE_COUNT += 1;
                }
            }
            uart_log_device(&d);
        }
    }
    unsafe { DEVICE_COUNT }
}

/// Snapshot of the last scan into `out`; returns the number copied.
pub fn devices(out: &mut [Device]) -> usize {
    let n = unsafe { DEVICE_COUNT }.min(out.len());
    for (i, slot) in out[..n].iter_mut().enumerate() {
        *slot = unsafe { DEVICES[i] };
    }
    n
}

fn uart_log_device(d: &Device) {
    crate::uart::locked_write(format_args!(
        "pcie: {:02x}:{:02x}.0 vendor={:04x} device={:04x} class={:06x}\n",
        d.bus, d.dev, d.vendor_id, d.device_id, d.class_code
    ));
}
