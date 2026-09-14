#!/usr/bin/env python3
"""Generate a minimal valid GGUF v3 test file (MODEL.BIN) for MarkOS.

Layout (self-consistent reference for the kernel's GGUF parser):
- header: magic GGUF, version 3, 1 tensor, 2 metadata KVs
- kv 'general.name' = string 'markos-test'
- kv 'general.alignment' = u32 32
- tensor 'test.weight': dims [4,4], type F32(0), offset 0
- data: 16 x f32 = 1.0 .. 16.0, padded to 32-byte alignment
"""
import struct
import sys

def main(path: str) -> None:
    out = bytearray()
    out += struct.pack("<I", 0x46554747)          # magic "GGUF"
    out += struct.pack("<I", 3)                   # version 3
    out += struct.pack("<Q", 1)                   # tensor_count
    out += struct.pack("<Q", 2)                   # metadata_kv_count

    def kv_str(key: str, val: str) -> bytes:
        b = struct.pack("<Q", len(key)) + key.encode()
        b += struct.pack("<I", 8)                 # value type: string
        b += struct.pack("<Q", len(val)) + val.encode()
        return b

    def kv_u32(key: str, val: int) -> bytes:
        b = struct.pack("<Q", len(key)) + key.encode()
        b += struct.pack("<I", 4)                 # value type: u32
        b += struct.pack("<I", val)
        return b

    out += kv_str("general.name", "markos-test")
    out += kv_u32("general.alignment", 32)

    # tensor info: test.weight, dims [4,4], F32, offset 0
    name = b"test.weight"
    out += struct.pack("<Q", len(name)) + name
    out += struct.pack("<I", 2)                   # n_dims
    out += struct.pack("<Q", 4)
    out += struct.pack("<Q", 4)
    out += struct.pack("<I", 0)                   # type: F32
    out += struct.pack("<Q", 0)                   # offset in data section

    # pad to 32-byte alignment
    while len(out) % 32 != 0:
        out += b"\0"

    # data: 16 f32 values 1.0..16.0
    for i in range(1, 17):
        out += struct.pack("<f", float(i))

    with open(path, "wb") as f:
        f.write(out)
    print(f"wrote {path}: {len(out)} bytes")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tests/MODEL.BIN")
