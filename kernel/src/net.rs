//! Networking: virtio-net driver + minimal ARP/IPv4/ICMP/TCP.
//!
//! Appliance scope: static IPv4 (QEMU slirp default 10.0.2.15), one TCP
//! listener, one connection at a time, no retransmission yet (documented
//! Pi-8 hardening). Polling model: the BSP drains the NIC in its main loop;
//! the APs stay in the execution pool for tensor work.
//!
//! owns: the virtio-net device registers, RX/TX buffer memory, the TCP
//! connection state, and the MARKOS protocol handler.
//! invariants:
//! - Single-threaded: only the BSP runs net::poll (the APs never touch the
//!   NIC queues or connection state).
//! - DMA coherence: RX buffers are invalidated before CPU reads and the TX
//!   buffers cleaned before the device reads them (cache.rs).

use core::fmt::Write as _;

use crate::{cache, timer, uart};

const NET_MMIO_BASE: u64 = 0x0A00_0000;
const NET_MMIO_STRIDE: u64 = 0x200;
const NET_MMIO_SLOTS: u64 = 32;
const MAGIC: u32 = 0x7472_6976;
const VERSION_MODERN: u32 = 2;
const DEVICE_ID_NET: u32 = 1;

// Register offsets (same map as virtio_blk).
const REG_MAGIC: u64 = 0x000;
const REG_VERSION: u64 = 0x004;
const REG_DEVICE_ID: u64 = 0x008;
const REG_STATUS: u64 = 0x070;
const REG_QUEUE_SEL: u64 = 0x030;
const REG_QUEUE_NUM_MAX: u64 = 0x034;
const REG_QUEUE_NUM: u64 = 0x038;
const REG_QUEUE_READY: u64 = 0x044;
const REG_QUEUE_NOTIFY: u64 = 0x050;
const REG_DEVICE_FEATURES: u64 = 0x010;
const REG_DEVICE_FEATURES_SEL: u64 = 0x014;
const REG_DRIVER_FEATURES: u64 = 0x020;
const REG_DRIVER_FEATURES_SEL: u64 = 0x024;
const REG_QUEUE_DESC_LOW: u64 = 0x080;
const REG_QUEUE_DESC_HIGH: u64 = 0x084;
const REG_QUEUE_AVAIL_LOW: u64 = 0x090;
const REG_QUEUE_AVAIL_HIGH: u64 = 0x094;
const REG_QUEUE_USED_LOW: u64 = 0x0A0;
const REG_QUEUE_USED_HIGH: u64 = 0x0A4;
const REG_CONFIG: u64 = 0x100;

const ST_ACK: u32 = 1;
const ST_DRIVER: u32 = 2;
const ST_DRIVER_OK: u32 = 4;
const ST_FEATURES_OK: u32 = 8;

const DESC_NEXT: u16 = 1;
const DESC_WRITE: u16 = 2;

const QRX: u32 = 0;
const QTX: u32 = 1;
const QSIZE: usize = 32;

const VNET_HDR: usize = 12;
const FRAME_MAX: usize = 1536;

const ETH_DST: usize = 0;
const ETH_SRC: usize = 6;
const ETH_TYPE: usize = 12;
const ETH_HDR: usize = 14;
const ETYPE_ARP: u16 = 0x0806;
const ETYPE_IP: u16 = 0x0800;

const PROTO_ICMP: u8 = 1;
const PROTO_TCP: u8 = 6;

/// Appliance static addresses (QEMU slirp defaults; the installer bakes
/// these for real deployments later).
const OUR_IP: [u8; 4] = [10, 0, 2, 15];
pub const LISTEN_PORT: u16 = 8080;

static mut NET_BASE: u64 = 0;
static mut OUR_MAC: [u8; 6] = [0x52, 0x54, 0x00, 0x12, 0x34, 0x56];

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
    ring: [u16; QSIZE],
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
    ring: [UsedElem; QSIZE],
}

#[repr(C, align(4096))]
struct Queue {
    desc: [Desc; QSIZE],
    avail: AvailRing,
    used: UsedRing,
}

static mut QRX_Q: Queue = Queue {
    desc: [Desc { addr: 0, len: 0, flags: 0, next: 0 }; QSIZE],
    avail: AvailRing { flags: 0, idx: 0, ring: [0; QSIZE], used_event: 0 },
    used: UsedRing { flags: 0, idx: 0, ring: [UsedElem { id: 0, len: 0 }; QSIZE] },
};
static mut QTX_Q: Queue = Queue {
    desc: [Desc { addr: 0, len: 0, flags: 0, next: 0 }; QSIZE],
    avail: AvailRing { flags: 0, idx: 0, ring: [0; QSIZE], used_event: 0 },
    used: UsedRing { flags: 0, idx: 0, ring: [UsedElem { id: 0, len: 0 }; QSIZE] },
};

/// RX buffer sets: vnet header buffer + frame buffer per chain.
static mut RX_HDR: [[u8; VNET_HDR]; 8] = [[0; VNET_HDR]; 8];
static mut RX_FRAME: [[u8; FRAME_MAX]; 8] = [[0; FRAME_MAX]; 8];
/// TX buffer sets (round-robin, fire-and-forget at this scale).
static mut TX_HDR: [[u8; VNET_HDR]; 4] = [[0; VNET_HDR]; 4];
static mut TX_FRAME: [[u8; FRAME_MAX]; 4] = [[0; FRAME_MAX]; 4];
static mut TX_NEXT: usize = 0;

static mut RX_LAST_SEEN: u16 = 0;

// ===== virtio-mmio helpers =====

fn rd32(base: u64, off: u64) -> u32 {
    // Soundness: device registers, identity-mapped; volatile is the defined
    // access pattern.
    unsafe { ((base + off) as *const u32).read_volatile() }
}
fn wr32(base: u64, off: u64, v: u32) {
    unsafe { ((base + off) as *mut u32).write_volatile(v) }
}
fn wr64_split(base: u64, off: u64, v: u64) {
    wr32(base, off, v as u32);
    wr32(base, off + 4, (v >> 32) as u32);
}

fn setup_queue(base: u64, sel: u32, q: &'static mut Queue) -> Result<(), &'static str> {
    wr32(base, REG_QUEUE_SEL, sel);
    let max = rd32(base, REG_QUEUE_NUM_MAX);
    if max < QSIZE as u32 {
        return Err("virtqueue too small");
    }
    wr32(base, REG_QUEUE_NUM, QSIZE as u32);
    let q_addr = (&raw const *q) as *const Queue as usize as u64;
    wr64_split(base, 0x080, q_addr); // desc low/high
    wr64_split(base, 0x090, q_addr + core::mem::offset_of!(Queue, avail) as u64);
    wr64_split(base, 0x0A0, q_addr + core::mem::offset_of!(Queue, used) as u64);
    wr32(base, REG_QUEUE_READY, 1);
    Ok(())
}

/// Bring up the virtio-net device (probe slot scan like the block device).
pub fn init() -> Result<(), &'static str> {
    let mut base = 0u64;
    for slot in 0..NET_MMIO_SLOTS {
        let b = NET_MMIO_BASE + slot * NET_MMIO_STRIDE;
        if rd32(b, REG_MAGIC) == MAGIC
            && rd32(b, REG_VERSION) == VERSION_MODERN
            && rd32(b, REG_DEVICE_ID) == DEVICE_ID_NET
        {
            base = b;
            break;
        }
    }
    if base == 0 {
        return Err("no virtio-net device found");
    }

    wr32(base, REG_STATUS, 0);
    wr32(base, REG_STATUS, rd32(base, REG_STATUS) | ST_ACK);
    wr32(base, REG_STATUS, rd32(base, REG_STATUS) | ST_DRIVER);

    // Accept device features (MAC included).
    wr32(base, REG_DEVICE_FEATURES_SEL, 0);
    let dev_lo = rd32(base, REG_DEVICE_FEATURES);
    wr32(base, REG_DRIVER_FEATURES_SEL, 0);
    wr32(base, REG_DRIVER_FEATURES, dev_lo);
    wr32(base, REG_DEVICE_FEATURES_SEL, 1);
    let dev_hi = rd32(base, REG_DEVICE_FEATURES);
    wr32(base, REG_DRIVER_FEATURES_SEL, 1);
    wr32(base, REG_DRIVER_FEATURES, dev_hi | 1); // VIRTIO_F_VERSION_1

    wr32(base, REG_STATUS, rd32(base, REG_STATUS) | ST_FEATURES_OK);
    if rd32(base, REG_STATUS) & ST_FEATURES_OK == 0 {
        return Err("net device rejected features");
    }

    // Read our MAC from the device config (offset 0).
    unsafe {
        for i in 0..6 {
            OUR_MAC[i] = ((base + REG_CONFIG + i as u64) as *const u8).read_volatile();
        }
    }

    unsafe {
        setup_queue(base, QRX, &mut *(&raw mut QRX_Q))?;
        setup_queue(base, QTX, &mut *(&raw mut QTX_Q))?;
    }

    wr32(base, REG_STATUS, rd32(base, REG_STATUS) | ST_DRIVER_OK);

    // Post the RX chains only AFTER DRIVER_OK: the spec forbids buffer
    // notifications before it, and QEMU would ignore the early ones.
    unsafe {
        for i in 0..8usize {
            cache::invalidate_range((&raw const RX_FRAME[i]) as usize, FRAME_MAX);
            let hdr = (&raw const RX_HDR[i]) as usize as u64;
            let frame = (&raw const RX_FRAME[i]) as usize as u64;
            let q = &raw mut QRX_Q;
            (*q).desc[2 * i] = Desc { addr: hdr, len: VNET_HDR as u32, flags: DESC_WRITE | DESC_NEXT, next: (2 * i + 1) as u16 };
            (*q).desc[2 * i + 1] = Desc { addr: frame, len: FRAME_MAX as u32, flags: DESC_WRITE, next: 0 };
            let idx = (*q).avail.idx;
            (*q).avail.ring[(idx as usize) % QSIZE] = (2 * i) as u16;
            (*q).avail.idx = idx.wrapping_add(1);
        }
    }
    wr32(base, REG_QUEUE_NOTIFY, QRX);

    // Debug: verify the RX post landed.
    unsafe {
        let qaddr = (&raw const QRX_Q) as usize;
        let avail_idx = ((qaddr + core::mem::offset_of!(Queue, avail) + 2) as *const u16).read_volatile();
        uart::locked_write(format_args!(
            "net: rx posted qaddr={qaddr:#x} avail_idx={avail_idx}
"
        ));
    }

    unsafe { NET_BASE = base };
    Ok(())
}

/// Our appliance IPv4 address.
pub fn our_ip() -> [u8; 4] {
    OUR_IP
}

pub fn mac_string(buf: &mut [u8]) -> usize {
    // Soundness: OUR_MAC is 6 bytes; formats "xx:xx:xx:xx:xx:xx".
    let mac = unsafe { &*(&raw const OUR_MAC) };
    let mut i = 0;
    while i < 6 {
        let hex = b"0123456789abcdef";
        buf[i * 3] = hex[(mac[i] >> 4) as usize];
        buf[i * 3 + 1] = hex[(mac[i] & 0xF) as usize];
        if i < 5 {
            buf[i * 3 + 2] = b':';
        }
        i += 1;
    }
    17
}

pub fn mac_bytes() -> [u8; 6] {
    unsafe { *(&raw const OUR_MAC) }
}

// ===== frame dispatch =====

/// Drain completed RX chains and dispatch each frame.
pub fn poll() {
    unsafe {
        let q = &raw mut QRX_Q;
        let used_idx = (*q).used.idx;
        while RX_LAST_SEEN != used_idx {
            let used_slot = (RX_LAST_SEEN as usize) % QSIZE;
            RX_LAST_SEEN = RX_LAST_SEEN.wrapping_add(1);
            // Which chain came back: the used element carries the head
            // descriptor id (2*buffer). The used-ring POSITION must not be
            // reused as the buffer index — with 8 posted chains in a
            // 32-entry used ring they diverge after the first wrap.
            let slot = ((*q).used.ring[used_slot].id as usize / 2) % 8;
            // The frame lives in the RX_FRAME[slot] buffer (after the 12-byte
            // vnet header the device wrote into RX_HDR[slot]).
            let len = {
                let elem_len =
                    ((*q).used.ring[used_slot].len as usize).saturating_sub(VNET_HDR);
                elem_len.min(FRAME_MAX)
            };
            // Invalidate before the CPU reads the DMA-written frame.
            cache::invalidate_range((&raw const RX_FRAME[slot]) as usize, FRAME_MAX);
            let frame = &RX_FRAME[slot][..len];
            handle_frame(frame);
            // Re-post the chain for the next packet.
            let hdr = (&raw const RX_HDR[slot]) as usize as u64;
            let f = (&raw const RX_FRAME[slot]) as usize as u64;
            (*q).desc[2 * slot] = Desc { addr: hdr, len: VNET_HDR as u32, flags: DESC_WRITE | DESC_NEXT, next: (2 * slot + 1) as u16 };
            (*q).desc[2 * slot + 1] = Desc { addr: f, len: FRAME_MAX as u32, flags: DESC_WRITE, next: 0 };
            let idx = (*q).avail.idx;
            (*q).avail.ring[(idx as usize) % QSIZE] = (2 * slot) as u16;
            (*q).avail.idx = idx.wrapping_add(1);
            wr32(unsafe { NET_BASE }, REG_QUEUE_NOTIFY, QRX);
        }
    }
}

fn handle_frame(frame: &[u8]) {
    if frame.len() < ETH_HDR + 1 {
        return;
    }
    uart::locked_write(format_args!(
        "net: frame {} bytes type={:02x}{:02x}
",
        frame.len(),
        frame[ETH_TYPE],
        frame[ETH_TYPE + 1]
    ));
    let etype = u16::from_be_bytes([frame[ETH_TYPE], frame[ETH_TYPE + 1]]);
    let body = &frame[ETH_HDR..];
    let src_mac = {
        let mut m = [0u8; 6];
        m.copy_from_slice(&frame[ETH_SRC..ETH_SRC + 6]);
        m
    };
    match etype {
        ETYPE_ARP => handle_arp(frame, body, src_mac),
        ETYPE_IP => handle_ip(body, src_mac),
        _ => {}
    }
}

// ===== ARP =====

fn handle_arp(frame: &[u8], body: &[u8], src_mac: [u8; 6]) {
    if body.len() < 28 {
        return;
    }
    let op = u16::from_be_bytes([body[6], body[7]]);
    if op != 1 {
        return; // only requests
    }
    let target_ip = [body[24], body[25], body[26], body[27]];
    if target_ip != OUR_IP {
        return;
    }
    // Reply: eth header + 28-byte ARP reply.
    let mut out = [0u8; ETH_HDR + 28];
    out[ETH_DST..ETH_DST + 6].copy_from_slice(&frame[ETH_SRC..ETH_SRC + 6]);
    out[ETH_SRC..ETH_SRC + 6].copy_from_slice(&our_mac());
    out[ETH_TYPE..ETH_TYPE + 2].copy_from_slice(&ETYPE_ARP.to_be_bytes());
    let arp = &mut out[ETH_HDR..];
    arp[0..2].copy_from_slice(&1u16.to_be_bytes());      // hardware: ethernet
    arp[2..4].copy_from_slice(&0x0800u16.to_be_bytes()); // protocol: IPv4
    arp[4] = 6;
    arp[5] = 4;
    arp[6..8].copy_from_slice(&2u16.to_be_bytes());      // op: reply
    arp[8..14].copy_from_slice(&our_mac());
    arp[14..18].copy_from_slice(&OUR_IP);
    arp[18..24].copy_from_slice(&body[18..24]);          // sender (their) MAC/IP
    arp[24..28].copy_from_slice(&body[14..18]);          // target: requester IP
    net_send(&out);
}

// ===== IPv4 =====

fn handle_ip(body: &[u8], src_mac: [u8; 6]) {
    if body.len() < 20 {
        return;
    }
    if body[0] >> 4 != 4 {
        return;
    }
    let proto = body[9];
    let src_ip = [body[12], body[13], body[14], body[15]];
    let dst_ip = [body[16], body[17], body[18], body[19]];
    if dst_ip != OUR_IP {
        return;
    }
    let ihl = (body[0] & 0xF) as usize * 4;
    let payload = &body[ihl..];
    match proto {
        PROTO_ICMP => handle_icmp(body, payload, src_mac),
        PROTO_TCP => crate::tcp::input(payload, src_ip, src_mac),
        _ => {}
    }
}

fn handle_icmp(ip_hdr: &[u8], payload: &[u8], src_mac: [u8; 6]) {
    if payload.is_empty() || payload[0] != 8 || payload.len() > 1400 {
        return; // echo request only (type 8), size-capped
    }
    // Echo reply: swap addresses, type 0, recompute the checksum.
    let mut out = [0u8; 1600];
    out[ETH_DST..ETH_DST + 6].copy_from_slice(&src_mac);
    out[ETH_SRC..ETH_SRC + 6].copy_from_slice(&our_mac());
    out[ETH_TYPE..ETH_TYPE + 2].copy_from_slice(&ETYPE_IP.to_be_bytes());
    let ip = &mut out[ETH_HDR..ETH_HDR + 20];
    ip[0] = 0x45;
    let total = (20 + payload.len()) as u16;
    ip[2..4].copy_from_slice(&total.to_be_bytes());
    ip[8] = 64;
    ip[9] = PROTO_ICMP;
    ip[12..16].copy_from_slice(&OUR_IP);
    ip[16..20].copy_from_slice(&ip_hdr[12..16]);
    let csum = ip_checksum(&ip[..20]);
    ip[10..12].copy_from_slice(&csum);
    out[ETH_HDR + 20..].copy_from_slice(payload);
    out[ETH_HDR + 20] = 0; // echo reply
    net_send(&out);
}

// ===== frame TX =====

fn our_mac() -> [u8; 6] {
    unsafe { *(&raw const OUR_MAC) }
}

/// Transmit one pre-built Ethernet frame.
fn net_send(frame: &[u8]) {
    if frame.len() > FRAME_MAX {
        return;
    }
    unsafe {
        let slot = TX_NEXT % 4;
        TX_NEXT = TX_NEXT.wrapping_add(1);
        TX_FRAME[slot][..frame.len()].copy_from_slice(frame);
        cache::clean_range((&raw const TX_FRAME[slot]) as usize, frame.len());
        let hdr = (&raw const TX_HDR[slot]) as usize as u64;
        let f = (&raw const TX_FRAME[slot]) as usize as u64;
        let q = &raw mut QTX_Q;
        let idx = (*q).avail.idx;
        let d0 = ((idx as usize) * 2) % QSIZE;
        let d1 = (d0 + 1) % QSIZE;
        (*q).desc[d0] = Desc { addr: (&raw const TX_HDR[slot]) as usize as u64, len: VNET_HDR as u32, flags: DESC_NEXT, next: d1 as u16 };
        (*q).desc[d1] = Desc { addr: f, len: frame.len() as u32, flags: 0, next: 0 };
        (*q).avail.ring[idx as usize % QSIZE] = d0 as u16;
        (*q).avail.idx = idx.wrapping_add(1);
        wr32(unsafe { NET_BASE }, REG_QUEUE_NOTIFY, QTX);
    }
}

/// Build + send an IPv4 frame.
pub fn ip_send(dst_mac: &[u8; 6], dst_ip: &[u8; 4], proto: u8, payload: &[u8]) {
    if payload.len() > 1400 {
        return; // oversized for the static TX frame buffer
    }
    let mut out = [0u8; 1600];
    out[ETH_DST..ETH_DST + 6].copy_from_slice(dst_mac);
    out[ETH_SRC..ETH_SRC + 6].copy_from_slice(&our_mac());
    out[ETH_TYPE..ETH_TYPE + 2].copy_from_slice(&ETYPE_IP.to_be_bytes());
    let ip = &mut out[ETH_HDR..ETH_HDR + 20];
    ip[0] = 0x45;
    let total = (20 + payload.len()) as u16;
    ip[2..4].copy_from_slice(&total.to_be_bytes());
    ip[8] = 64;
    ip[9] = proto;
    ip[12..16].copy_from_slice(&OUR_IP);
    ip[16..20].copy_from_slice(dst_ip);
    let csum = ip_checksum(&ip[..20]);
    ip[10..12].copy_from_slice(&csum);
    out[ETH_HDR + 20..ETH_HDR + 20 + payload.len()].copy_from_slice(payload);
    net_send(&out[..ETH_HDR + 20 + payload.len()]);
}

fn ip_checksum(data: &[u8]) -> [u8; 2] {
    let mut sum = sum_ones_complement(data);
    let s = !(sum as u16);
    s.to_be_bytes()
}

fn sum_ones_complement(data: &[u8]) -> u32 {
    let mut sum = 0u32;
    let mut i = 0;
    while i + 1 < data.len() {
        sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        i += 2;
    }
    if i < data.len() {
        sum += (data[i] as u32) << 8;
    }
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    sum
}

pub fn serve_loop() -> ! {
    let mut beat: u64 = 0;
    loop {
        poll();
        beat += 1;
        if beat % 4_000_000 == 0 {
            uart::write_str("net: alive
");
        }
    }
}
