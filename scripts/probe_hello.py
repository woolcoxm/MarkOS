#!/usr/bin/env python3
"""Dump TCP-level detail from a QEMU filter-dump pcap (guest=52:54:00..)."""
import struct
import sys

data = open(sys.argv[1] if len(sys.argv) > 1 else "net.pcap", "rb").read()
off = 24
n = 0
GUEST = "525400123456"
while off + 16 <= len(data):
    ts, tu, incl, orig = struct.unpack("<IIII", data[off:off + 16])
    off += 16
    pkt = data[off:off + incl]
    off += incl
    n += 1
    if len(pkt) < 34 or int.from_bytes(pkt[12:14], "big") != 0x0800:
        print(f"pkt {n}: len={incl} non-IP")
        continue
    src_mac = pkt[6:12].hex()
    direction = "TX" if src_mac == GUEST else "RX"
    ip = pkt[14:]
    ihl = (ip[0] & 0xF) * 4
    proto = ip[9]
    src_ip = ".".join(str(b) for b in ip[12:16])
    dst_ip = ".".join(str(b) for b in ip[16:20])
    if proto != 6:
        print(f"pkt {n}: {direction} {src_ip}->{dst_ip} proto={proto} len={incl}")
        continue
    tcp = ip[ihl:]
    sport, dport = int.from_bytes(tcp[0:2], "big"), int.from_bytes(tcp[2:4], "big")
    seq, ack = int.from_bytes(tcp[4:8], "big"), int.from_bytes(tcp[8:12], "big")
    doff = ((tcp[12] >> 4) & 0xF) * 4
    flags = tcp[13]
    payload = tcp[doff:]
    fl = "".join(c for c, b in zip("FSRPAU", [1, 2, 4, 8, 16, 32]) if flags & b)
    preview = payload[:24].decode("latin1").replace("\n", "\\n")
    print(f"pkt {n}: {direction} {src_ip}:{sport}->{dst_ip}:{dport} "
          f"seq={seq} ack={ack} fl={fl} paylen={len(payload)} {preview!r}")
print("total packets:", n)
