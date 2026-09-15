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

/// Control port — set at boot from MARKOS.CFG (installer-baked).
fn listen_port() -> u16 {
    crate::config::port()
}

const FLAG_FIN: u8 = 1;
const FLAG_SYN: u8 = 2;
const FLAG_RST: u8 = 4;
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
    if dst_port != listen_port() || doff < 20 || doff > seg.len() {
        return;
    }
    let payload = &seg[doff..];

    let state = unsafe { STATE };
    match state {
        ST_LISTEN => {
            if flags & FLAG_SYN != 0 {
                accept_syn(src_ip, src_mac, src_port, seq);
            }
        }
        ST_SYN_RCVD => {
            if flags & FLAG_ACK != 0 && ack == snd_nxt_load() {
                unsafe { STATE = ST_ESTAB; }
                uart::write_str("tcp: established\n");
            }
        }
        ST_ESTAB => {
            if flags & FLAG_RST != 0 {
                // Peer abandoned the connection: free the slot.
                unsafe { STATE = ST_LISTEN; }
                uart::write_str("tcp: reset, listening\n");
                return;
            }
            if flags & FLAG_SYN != 0 && payload_len == 0 {
                // A fresh SYN while established means the peer dropped
                // the old session and its FIN/RST never reached us (the
                // appliance serves one client at a time, so the newest
                // handshake wins): reset to it.
                uart::write_str("tcp: peer re-handshake\n");
                accept_syn(src_ip, src_mac, src_port, seq);
                return;
            }
            if payload_len > 0 && seq == rcv_nxt_load() {
                // Advance RCV_NXT BEFORE running the handler: the reply
                // must ACK the request bytes it answers, and a peer
                // retransmission of the same request during a long handler
                // (LOAD) is then seen as duplicate, not re-executed.
                // Stream commands (GEN) need this most: they run for
                // minutes while the decode loop services the NIC between
                // tokens, and a poll sent mid-generation must validate as
                // in-order inside that drain — a frame rejected here is
                // discarded, which would strand the byte stream.
                rcv_nxt_store(seq.wrapping_add(payload_len as u32));
                handle_payload(payload);
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

/// Accept a handshake: lock the peer, initialize sequence numbers, reply
/// SYN-ACK, and move to SYN-RCVD.
fn accept_syn(src_ip: [u8; 4], src_mac: [u8; 6], src_port: u16, seq: u32) {
    unsafe {
        core::ptr::copy_nonoverlapping(src_mac.as_ptr(), &raw mut PEER_MAC as *mut u8, 6);
        core::ptr::copy_nonoverlapping(src_ip.as_ptr(), &raw mut PEER_IP as *mut u8, 4);
        PEER_PORT = src_port;
        SND_NXT = ISS.wrapping_add(1);
        RCV_NXT = seq.wrapping_add(1);
        STATE = ST_SYN_RCVD;
    }
    // SYN-ACK: seq = ISS, ack = their seq + 1.
    tcp_send_seg(ISS, seq.wrapping_add(1), FLAG_SYN | FLAG_ACK, &[]);
}

/// Protocol handler for received payloads: the MARKOS-PING transport probe,
/// then the control protocol; anything else echoes (transport bring-up aid).
fn handle_payload(payload: &[u8]) {
    if payload == b"MARKOS-PING" {
        reply(b"MARKOS-PONG");
        uart::write_str("net: PASS ping/pong round trip\n");
        return;
    }
    // GEN streams multiple replies as tokens are decoded; it talks to the
    // peer directly through stream().
    if crate::control::is_stream_command(payload) {
        uart::locked_write(format_args!("control: stream command\n"));
        crate::control::gen_stream(payload);
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

/// Stream one payload to the peer and advance SND_NXT. Used by the control
/// protocol's streaming commands (GEN emits a line per generated token).
pub fn stream(payload: &[u8]) {
    reply(payload);
    // Drain the NIC immediately after every streamed line: the peer reacts
    // to a TOK within milliseconds (a STATS poll), and a frame that lands
    // while the previous one is still being answered must be consumed as
    // soon as its sequence number becomes expected — the single-buffer
    // drain discards a frame it rejects, so leaving it pending past the
    // next handler run would strand the stream. Bounded reentrancy: a
    // command answered inside this drain replies through reply(), which
    // does not re-enter poll().
    net::poll();
}

/// Drain frames that arrived while GEN was decoding and answer any control
/// command they carry — a STATS poll sent on the connection mid-generation
/// is replied to here (Phase 10: stats update live during a sustained run).
/// Called from the GEN decode loop on the BSP between tokens: net::poll is
/// a serialized used-ring drain, and a reentrant GEN is refused by the
/// control layer's in-progress guard.
pub fn service_between_tokens() {
    net::poll();
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
    seg[0..2].copy_from_slice(&listen_port().to_be_bytes());
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
