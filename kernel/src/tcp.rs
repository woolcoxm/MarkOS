//! Minimal TCP: one listener (port 8080), one connection at a time.
//!
//! Scope: SYN/SYN-ACK/ACK handshake, in-order data with ACKs, FIN close.
//! No retransmission, no out-of-order handling, no window probing — the
//! appliance serves one reliable LAN client at a time (documented Pi-8
//! hardening item).
//!
//! owns: the TCP connection state and sequence numbers.
//! invariants: BSP-only execution (the net poll loop); state transitions
//! are driven by received segments only. Single connection: a new SYN mid-
//! connection is ignored until it closes.

use crate::net;
use crate::uart;

const LISTEN_PORT: u16 = 8080;

const FLAG_FIN: u8 = 1;
const FLAG_SYN: u8 = 2;
const FLAG_PSH: u8 = 8;
const FLAG_ACK: u8 = 16;

// Connection states.
const ST_LISTEN: u8 = 0;
const ST_SYN_RCVD: u8 = 1;
const ST_ESTAB: u8 = 2;

static mut STATE: u8 = ST_LISTEN;
static mut PEER_MAC: [u8; 6] = [0; 6];
static mut PEER_IP: [u8; 4] = [0; 4];
static mut PEER_PORT: u16 = 0;
static mut SND_NXT: u32 = 0;
static mut RCV_NXT: u32 = 0;
static ISS: u32 = 0x4D41_524B; // "MARK"

const PONG: &[u8] = b"MARKOS-PONG";

/// Entry: a TCP segment arrived (IPv4 payload addressed to us).
pub fn input(seg: &[u8], src_ip: [u8; 4], src_mac: [u8; 6]) {
    if seg.len() < 20 {
        return;
    }
    let src_port = u16::from_be_bytes([seg[0], seg[1]]);
    let dst_port = u16::from_be_bytes([seg[2], seg[3]]);
    let seq = u32::from_be_bytes([seg[4], seg[5], seg[6], seg[7]]);
    let ack = u32::from_be_bytes([seg[8], seg[9], seg[10], seg[11]]);
    let doff = ((seg[12] >> 4) & 0xF) as usize * 4;
    let flags = seg[13];
    let payload_len = seg.len().saturating_sub(doff);
    if dst_port != LISTEN_PORT || doff < 20 || doff > seg.len() {
        return;
    }
    let payload = &seg[doff..];

    let state = unsafe { STATE };
    match state {
        ST_LISTEN => {
            if flags & FLAG_SYN != 0 {
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        src_mac.as_ptr(),
                        &raw mut PEER_MAC as *mut u8,
                        6,
                    );
                    core::ptr::copy_nonoverlapping(
                        src_ip.as_ptr(),
                        &raw mut PEER_IP as *mut u8,
                        4,
                    );
                    PEER_PORT = src_port;
                    SND_NXT = ISS.wrapping_add(1);
                    RCV_NXT = seq.wrapping_add(1);
                }
                unsafe { STATE = ST_SYN_RCVD; }
                // SYN-ACK: seq = ISS, ack = their seq + 1.
                tcp_send_seg(ISS, seq.wrapping_add(1), FLAG_SYN | FLAG_ACK, &[]);
            }
        }
        ST_SYN_RCVD => {
            if flags & FLAG_ACK != 0 && ack == snd_nxt_load() {
                unsafe { STATE = ST_ESTAB; }
                uart::write_str("tcp: established\n");
            }
        }
        ST_ESTAB => {
            if payload_len > 0 && seq == rcv_nxt_load() {
                // In-order data: accept and run the protocol handler.
                handle_payload(payload);
                rcv_nxt_store(seq.wrapping_add(payload_len as u32));
            } else if payload_len > 0 {
                // Duplicate/out-of-order: re-acknowledge what we expect.
                send_ack(rcv_nxt_load());
            }
            if flags & FLAG_FIN != 0 {
                // Acknowledge their FIN; return to listening.
                send_ack(rcv_nxt_load().wrapping_add(1));
                unsafe {
                    unsafe { STATE = ST_LISTEN; }
                }
                uart::write_str("tcp: closed, listening\n");
            }
        }
        _ => {}
    }
}

/// Protocol handler for received payloads: the MARKOS-PING transport probe,
/// then the control protocol; anything else echoes (transport bring-up aid).
fn handle_payload(payload: &[u8]) {
    if payload == b"MARKOS-PING" {
        reply(b"MARKOS-PONG");
        uart::write_str("net: PASS ping/pong round trip\n");
        return;
    }
    let mut resp = [0u8; 512];
    let n = crate::control::dispatch(payload, &mut resp);
    if n > 0 {
        // Responses are lines: guarantee the trailing newline the client
        // reads to (dispatch content may omit it).
        let n = if resp[n - 1] != b'\n' {
            resp[n] = b'\n';
            n + 1
        } else {
            n
        };
        uart::locked_write(format_args!("control: replied {} bytes\n", n));
        reply(&resp[..n]);
    } else {
        reply(payload);
    }
}

/// Send payload to the peer and advance SND_NXT.
fn reply(payload: &[u8]) {
    tcp_send_seg(snd_nxt_load(), rcv_nxt_load(), FLAG_PSH | FLAG_ACK, payload);
    snd_nxt_store(snd_nxt_load().wrapping_add(payload.len() as u32));
}

fn snd_nxt_load() -> u32 {
    unsafe { SND_NXT }
}
fn snd_nxt_store(v: u32) {
    unsafe { SND_NXT = v; }
}
fn rcv_nxt_load() -> u32 {
    unsafe { RCV_NXT }
}
fn rcv_nxt_store(v: u32) {
    unsafe { RCV_NXT = v; }
}

/// Send a bare ACK for `ack`.
fn send_ack(ack: u32) {
    tcp_send_seg(snd_nxt_load(), ack, FLAG_ACK, &[]);
}

/// Build and send a TCP segment to the stored peer.
fn tcp_send_seg(seq: u32, ack: u32, flags: u8, payload: &[u8]) {
    let (peer_mac, peer_ip) = unsafe { (PEER_MAC, PEER_IP) };
    let our_ip = net::our_ip();

    // TCP header (20 bytes) + payload, checksum computed over the
    // pseudo-header + segment with the checksum field zero.
    let mut seg = [0u8; 20 + 1460];
    seg[0..2].copy_from_slice(&LISTEN_PORT.to_be_bytes());
    seg[2..4].copy_from_slice(&peer_port().to_be_bytes());
    seg[4..8].copy_from_slice(&seq.to_be_bytes());
    seg[8..12].copy_from_slice(&ack.to_be_bytes());
    seg[12] = 0x50; // data offset: 5 words
    seg[13] = flags;
    seg[14..16].copy_from_slice(&0xFFFFu16.to_be_bytes()); // window
    // checksum stays zero during computation
    seg[20..20 + payload.len()].copy_from_slice(payload);

    let mut sum = 0u32;
    checksum_add(&mut sum, &our_ip);
    checksum_add(&mut sum, &peer_ip);
    // zero(1) + proto 6(1) + len(2) as big-endian words
    sum += 6;
    sum += (20 + payload.len()) as u32;
    checksum_add(&mut sum, &seg[..20 + payload.len()]);
    let csum = !fold(sum);
    seg[16..18].copy_from_slice(&csum.to_be_bytes());

    let total = 20 + payload.len();
    net::ip_send(&peer_mac, &peer_ip, 6, &seg[..total]);
}

fn checksum_add(sum: &mut u32, data: &[u8]) {
    let mut i = 0;
    while i + 1 < data.len() {
        *sum += u16::from_be_bytes([data[i], data[i + 1]]) as u32;
        i += 2;
    }
    if i < data.len() {
        *sum += (data[i] as u32) << 8;
    }
}

fn fold(mut sum: u32) -> u16 {
    while sum >> 16 != 0 {
        sum = (sum & 0xFFFF) + (sum >> 16);
    }
    sum as u16
}

fn peer_port() -> u16 {
    unsafe { PEER_PORT }
}
