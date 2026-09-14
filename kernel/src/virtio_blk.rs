//! virtio-mmio transport + virtio-blk device (QEMU `virt` test board).
//!
//! Implements just enough of virtio v2 (modern, split virtqueue) to run
//! read requests against a single block device: transport probe, feature
//! negotiation, queue setup, one-outstanding-request polling I/O.
//!
//! owns: the virtio-mmio device registers (via the identity map), the
//! virtqueue memory (static, 4 KiB-aligned), and the request header/status
//! scratch buffers.
//! invariants:
//! - `init` runs once on the BSP before any block access.
//! - One request in flight at a time (descriptors 0..2 are the whole
//!   chain); the data buffer must be identity-mapped RAM (VA == PA) so the
//!   device DMA sees what we see.
//! - Real-hardware note: queue memory is Normal-cacheable; the Pi 4/5 DMA
//!   path will need cache maintenance (QEMU TCG does not model caches).

use core::fmt::Write as _;
use core::hint::spin_loop;
use core::ptr;

use crate::uart;

/// QEMU `virt` places virtio-mmio devices at 0x0A000000, one slot per
/// 0x200 bytes, 32 slots.
const MMIO_BASE: u64 = 0x0A00_0000;
const MMIO_STRIDE: u64 = 0x200;
const MMIO_SLOTS: u64 = 32;
const MAGIC: u32 = 0x7472_6976;
const VERSION_MODERN: u32 = 2;
const DEVICE_ID_BLOCK: u32 = 2;

// Register offsets (standard-headers/linux/virtio_mmio.h).
const REG_MAGIC: u64 = 0x000;
const REG_VERSION: u64 = 0x004;
const REG_DEVICE_ID: u64 = 0x008;
const REG_DEVICE_FEATURES: u64 = 0x010;
const REG_DEVICE_FEATURES_SEL: u64 = 0x014;
const REG_DRIVER_FEATURES: u64 = 0x020;
const REG_DRIVER_FEATURES_SEL: u64 = 0x024;
const REG_QUEUE_SEL: u64 = 0x030;
const REG_QUEUE_NUM_MAX: u64 = 0x034;
const REG_QUEUE_NUM: u64 = 0x038;
const REG_QUEUE_READY: u64 = 0x044;
const REG_QUEUE_NOTIFY: u64 = 0x050;
const REG_STATUS: u64 = 0x070;
const REG_QUEUE_DESC_LOW: u64 = 0x080;
const REG_QUEUE_DESC_HIGH: u64 = 0x084;
const REG_QUEUE_AVAIL_LOW: u64 = 0x090;
const REG_QUEUE_AVAIL_HIGH: u64 = 0x094;
const REG_QUEUE_USED_LOW: u64 = 0x0A0;
const REG_QUEUE_USED_HIGH: u64 = 0x0A4;
const REG_CONFIG: u64 = 0x100;

// Status bits.
const ST_ACK: u32 = 1;
const ST_DRIVER: u32 = 2;
const ST_DRIVER_OK: u32 = 4;
const ST_FEATURES_OK: u32 = 8;

// Split-virtqueue descriptor flags.
const DESC_NEXT: u16 = 1;
const DESC_WRITE: u16 = 2;

const QUEUE_SIZE: usize = 16;
const SECTOR: usize = 512;

#[repr(C)]
#[derive(Clone, Copy)]
struct Desc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

#[repr(C)]
struct AvailRing {
    flags: u16,
    idx: u16,
    ring: [u16; QUEUE_SIZE],
    used_event: u16,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct UsedElem {
    id: u32,
    len: u32,
}

#[repr(C)]
struct UsedRing {
    flags: u16,
    idx: u16,
    ring: [UsedElem; QUEUE_SIZE],
}

/// All queue memory in one 4 KiB-aligned static: desc | avail | used.
#[repr(C, align(4096))]
struct VirtQueue {
    desc: [Desc; QUEUE_SIZE],
    avail: AvailRing,
    used: UsedRing,
}

static mut VQ: VirtQueue = VirtQueue {
    desc: [Desc { addr: 0, len: 0, flags: 0, next: 0 }; QUEUE_SIZE],
    avail: AvailRing { flags: 0, idx: 0, ring: [0; QUEUE_SIZE], used_event: 0 },
    used: UsedRing { flags: 0, idx: 0, ring: [UsedElem { id: 0, len: 0 }; QUEUE_SIZE] },
};

/// Request scratch: 16-byte virtio-blk request header + 1-byte status.
#[repr(C, align(8))]
struct BlkReq {
    hdr_type: u32,
    hdr_reserved: u32,
    hdr_sector: u64,
    status: u8,
}

static mut REQ: BlkReq = BlkReq {
    hdr_type: 0,
    hdr_reserved: 0,
    hdr_sector: 0,
    status: 0,
};

static mut DEV_BASE: u64 = 0;
/// Used-ring index already consumed by the driver (poll completion).
static mut LAST_USED_IDX: u16 = 0;

fn rd32(base: u64, offset: u64) -> u32 {
    // Soundness: virtio-mmio device registers, identity-mapped; volatile is
    // the defined access pattern for device memory.
    unsafe { ((base + offset) as *const u32).read_volatile() }
}

fn wr32(base: u64, offset: u64, value: u32) {
    // Soundness: as above.
    unsafe { ((base + offset) as *mut u32).write_volatile(value) }
}

fn rd64_split(base: u64, low_off: u64) -> u64 {
    (rd32(base, low_off) as u64) | ((rd32(base, low_off + 4) as u64) << 32)
}

fn wr64_split(base: u64, low_off: u64, value: u64) {
    wr32(base, low_off, value as u32);
    wr32(base, low_off + 4, (value >> 32) as u32);
}

/// Scan the virtio-mmio slots for a block device; reset it, negotiate
/// features, and set up virtqueue 0.
pub fn init() -> Result<(), &'static str> {
    for slot in 0..MMIO_SLOTS {
        let base = MMIO_BASE + slot * MMIO_STRIDE;
        if rd32(base, REG_MAGIC) != MAGIC || rd32(base, REG_VERSION) != VERSION_MODERN {
            continue;
        }
        if rd32(base, REG_DEVICE_ID) != DEVICE_ID_BLOCK {
            continue;
        }
        return init_block(base);
    }
    Err("no virtio-blk device found on the virtio-mmio bus")
}

fn init_block(base: u64) -> Result<(), &'static str> {
    // Reset, then acknowledge + claim driver role.
    wr32(base, REG_STATUS, 0);
    wr32(base, REG_STATUS, rd32(base, REG_STATUS) | ST_ACK);
    wr32(base, REG_STATUS, rd32(base, REG_STATUS) | ST_DRIVER);

    // Feature negotiation: accept the device's features and require
    // VIRTIO_F_VERSION_1 (feature bit 32) for the modern interface.
    wr32(base, REG_DEVICE_FEATURES_SEL, 0);
    let dev_lo = rd32(base, REG_DEVICE_FEATURES);
    wr32(base, REG_DRIVER_FEATURES_SEL, 0);
    wr32(base, REG_DRIVER_FEATURES, dev_lo);
    wr32(base, REG_DEVICE_FEATURES_SEL, 1);
    let dev_hi = rd32(base, REG_DEVICE_FEATURES);
    wr32(base, REG_DRIVER_FEATURES_SEL, 1);
    wr32(base, REG_DRIVER_FEATURES, dev_hi | 1);

    wr32(base, REG_STATUS, rd32(base, REG_STATUS) | ST_FEATURES_OK);
    if rd32(base, REG_STATUS) & ST_FEATURES_OK == 0 {
        return Err("device rejected feature set (FEATURES_OK clear)");
    }

    // Virtqueue 0: publish desc/avail/used and mark ready.
    wr32(base, REG_QUEUE_SEL, 0);
    let max = rd32(base, REG_QUEUE_NUM_MAX);
    if max < QUEUE_SIZE as u32 {
        return Err("virtqueue too small");
    }
    wr32(base, REG_QUEUE_NUM, QUEUE_SIZE as u32);

    // Soundness: the queue static is 4 KiB-aligned identity-mapped RAM owned
    // exclusively by this module; the device reads/writes it over DMA.
    let vq_addr = &raw const VQ as u64;
    let avail_addr = vq_addr + core::mem::offset_of!(VirtQueue, avail) as u64;
    let used_addr = vq_addr + core::mem::offset_of!(VirtQueue, used) as u64;
    wr64_split(base, REG_QUEUE_DESC_LOW, vq_addr);
    wr64_split(base, REG_QUEUE_AVAIL_LOW, avail_addr);
    wr64_split(base, REG_QUEUE_USED_LOW, used_addr);
    wr32(base, REG_QUEUE_READY, 1);

    wr32(base, REG_STATUS, rd32(base, REG_STATUS) | ST_DRIVER_OK);
    // Publish the device base only after full setup succeeds.
    unsafe { DEV_BASE = base; }
    Ok(())
}

/// Device capacity in 512-byte sectors (virtio-blk config offset 0).
pub fn capacity_sectors() -> u64 {
    let base = unsafe { DEV_BASE };
    if base == 0 {
        return 0;
    }
    rd64_split(base, REG_CONFIG)
}

/// Submit a read of `count` 512-byte sectors starting at `lba` into `buf`,
/// then poll the used ring for completion. Single outstanding request.
pub fn read_sectors(lba: u64, count: usize, buf: &mut [u8]) -> Result<(), &'static str> {
    if buf.len() < count * SECTOR {
        return Err("buffer smaller than read size");
    }
    if count == 0 {
        return Ok(());
    }
    let base = unsafe { DEV_BASE };
    if base == 0 {
        return Err("block device not initialized");
    }

    // Soundness: REQ/desc/avail are driver-owned queue memory; `buf` is
    // identity-mapped RAM the device DMAs directly; a single in-flight
    // request owns descriptor indices 0..2.
    unsafe {
        let vq = &raw mut VQ;
        let req_addr = (&raw const REQ) as u64;

        // Request header: type 0 = IN (read sectors into the data buffer).
        let req = &raw mut REQ;
        (*req).hdr_type = 0;
        (*req).hdr_reserved = 0;
        (*req).hdr_sector = lba;
        (*vq).desc[0] = Desc { addr: req_addr, len: 16, flags: DESC_NEXT, next: 1 };
        (*vq).desc[1] = Desc {
            addr: buf.as_mut_ptr() as u64,
            len: (count * SECTOR) as u32,
            flags: DESC_NEXT | DESC_WRITE,
            next: 2,
        };
        // Status byte lives in REQ right after the header.
        (*vq).desc[2] = Desc { addr: req_addr + 16, len: 1, flags: DESC_WRITE, next: 0 };
        ((req_addr + 16) as *mut u8).write_volatile(0);

        let avail_idx = (*vq).avail.idx;
        (*vq).avail.ring[(avail_idx as usize) % QUEUE_SIZE] = 0; // head desc
        (*vq).avail.idx = avail_idx.wrapping_add(1);
    }

    wr32(base, REG_QUEUE_NOTIFY, 0); // virtio-blk uses virtqueue 0

    // Poll the used ring until the device consumes our request.
    unsafe {
        let vq = &raw mut VQ;
        let mut polls = 0u64;
        while (*vq).used.idx == LAST_USED_IDX {
            spin_loop();
            polls += 1;
            if polls > 50_000_000 {
                uart::locked_write(format_args!(
                    "blk: FAIL — read timed out (lba {lba}, {count} sectors)\n"
                ));
                return Err("block read timed out");
            }
        }
        LAST_USED_IDX = (*vq).used.idx;
    }

    // The device wrote the status byte into REQ.status.
    let ok = unsafe { ((&raw const REQ) as *const u8).add(16).read_volatile() } == 0;
    if ok {
        Ok(())
    } else {
        Err("virtio-blk reported request failure")
    }
}

/// Block device bring-up + first-read acceptance helper (used by kmain).
pub fn bring_up_and_verify() -> Result<u64, &'static str> {
    init()?;
    let sectors = capacity_sectors();
    if sectors == 0 {
        return Err("device reports zero capacity");
    }

    // Read LBA 0 into a scratch buffer and look for the MBR signature the
    // test image carries at offset 510.
    let mut buf = [0u8; SECTOR];
    read_sectors(0, 1, &mut buf)?;
    // Debug: first/last bytes of the sector.
    let _ = write!(
        uart::Serial,
        "blk: lba0[0..8]={:02x?} [508..512]={:02x?}
",
        &buf[0..8],
        &buf[508..512]
    );
    let sig_ok = buf[510] == 0x55 && buf[511] == 0xAA;
    if !sig_ok {
        return Err("LBA0 missing MBR signature 55 AA");
    }
    Ok(sectors)
}
