#!/usr/bin/env python3
"""Compact TCP table from a QEMU filter-dump pcap: the last N TCP segments,
one line each: t=reltime src>sport dst.dport flags seq..seqend ack len."""
import struct
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "net.pcap"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 80
sport_f = int(sys.argv[3]) if len(sys.argv) > 3 else 0  # filter: client port

rows = []
with open(path, "rb") as f:
    magic = f.read(4)
    if magic == b"\xd4\xc3\xb2\xa1":
        endian = "<"
    elif magic == b"\xa1\xb2\xc3\xd4":
        endian = ">"
    else:
        sys.exit("not a pcap")
    f.read(20)
    t0 = None
    while True:
        hdr = f.read(16)
        if len(hdr) < 16:
            break
        ts, tus, caplen, _ = struct.unpack(endian + "IIII", hdr)
        data = f.read(caplen)
        if t0 is None:
            t0 = ts + tus / 1e6
        if len(data) < 34:
            continue
        etype = struct.unpack(">H", data[12:14])[0]
        if etype != 0x0800:
            continue
        ip = data[14:]
        ihl = (ip[0] & 0xF) * 4
        proto = ip[9]
        if proto != 6:
            continue
        src = ".".join(str(b) for b in ip[12:16])
        dst = ".".join(str(b) for b in ip[16:20])
        tcp = ip[ihl:]
        sp, dp = struct.unpack(">HH", tcp[0:4])
        if sport_f and sp != sport_f and dp != sport_f:
            continue
        seq, ack = struct.unpack(">II", tcp[4:12])
        doff = ((tcp[12] >> 4) & 0xF) * 4
        fl = tcp[13]
        flags = "".join(nm for bit, nm in
                        [(2, "S"), (16, "A"), (8, "P"), (1, "F"), (4, "R")]
                        if fl & bit)
        plen = len(ip) - ihl - doff
        rows.append((ts + tus / 1e6 - t0, f"{src}:{sp}>{dst}.{dp} {flags:<4} "
                     f"seq={seq}..{seq + plen} ack={ack} len={plen}"))

for r in rows[-n:]:
    print(f"t={r[0]:9.3f} {r[1]}")
print(f"total tcp rows: {len(rows)}")
