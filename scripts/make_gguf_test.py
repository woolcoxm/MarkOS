#!/usr/bin/env python3
"""Generate a synthetic GGUF v3 test model for MarkOS matmul acceptance.

Contents:
- kv 'general.name'      = string 'markos-test'
- kv 'general.alignment' = u32 32
- tensor 'mm.a': dims [64, 16] (K x M, row-major), 16*64 bytes, u8 pattern
- tensor 'mm.b': dims [64, 16] (K x N, row-major), 16*64 bytes, u8 pattern

Byte patterns (non-negative so unsigned UDOT == signed reference math):
  mm.a[i] = (i*7)    % 121
  mm.b[i] = (i*11)   % 101

Reference matmul (computed by --check):
  C[m][n] = sum_k A[m*64+k] * B[n*64+k]   for m,n in 0..16
Prints C00, C1515 and the full checksum for the Makefile gate.
"""
import struct
import sys

M = 16
K = 64
N = 16
ALIGN = 32


def main(path: str, check: bool) -> None:
    a = bytes(((i * 7) % 121) for i in range(M * K))
    b = bytes(((i * 11) % 101) for i in range(N * K))

    if check:
        c = [[sum(a[m * K + k] * b[n * K + k] for k in range(K)) for n in range(N)] for m in range(M)]
        total = sum(sum(row) for row in c)
        print(f"REF C00={c[0][0]} C1515={c[15][15]} SUM={total}")
        return

    out = bytearray()
    out += struct.pack("<I", 0x46554747)      # magic "GGUF"
    out += struct.pack("<I", 3)               # version 3
    out += struct.pack("<Q", 2)               # tensor_count
    out += struct.pack("<Q", 2)               # metadata_kv_count

    def kv_str(key: str, val: str) -> bytes:
        r = struct.pack("<Q", len(key)) + key.encode()
        r += struct.pack("<I", 8) + struct.pack("<Q", len(val)) + val.encode()
        return r

    def kv_u32(key: str, val: int) -> bytes:
        r = struct.pack("<Q", len(key)) + key.encode()
        r += struct.pack("<I", 4) + struct.pack("<I", val)
        return r

    out += kv_str("general.name", "markos-test")
    out += kv_u32("general.alignment", ALIGN)

    def tensor(name: str, dims, offset: int) -> bytes:
        r = struct.pack("<Q", len(name)) + name.encode()
        r += struct.pack("<I", len(dims))
        for d in dims:
            r += struct.pack("<Q", d)
        r += struct.pack("<I", 0)             # type 0: F32 (raw bytes reused)
        r += struct.pack("<Q", offset)
        return r

    out += tensor("mm.a", [K, M], 0)
    out += tensor("mm.b", [K, N], len(a))
    while len(out) % ALIGN != 0:
        out += b"\0"
    out += a
    out += b

    with open(path, "wb") as f:
        f.write(out)
    print(f"wrote {path}: {len(out)} bytes (expect header+tensors ~ {ALIGN}+{len(a)+len(b)})")


if __name__ == "__main__":
    main(sys.argv[1], check="--check" in sys.argv)
