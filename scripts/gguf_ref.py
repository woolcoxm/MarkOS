#!/usr/bin/env python3
"""Host reference for the real-GGUF load gate (brief Phase 7).

Parses a GGUF file with the same minimal rules as kernel/src/gguf.rs and
emits the exact lines the kernel's selftest-model prints. The Makefile
gate diffs the kernel's serial output against this file.

Usage: gguf_ref.py <model.gguf> <out.txt>"""
import struct
import sys
import zlib

MAGIC = 0x46554747  # "GGUF" little-endian
CHECK_NAMES = (b"token_embd.weight", b"blk.0.attn_q.weight", b"output.weight")
CHECK_LEN = 256
FIRST_N = 8


class Reader:
    def __init__(self, data):
        self.data = data
        self.pos = 0

    def u32(self):
        (v,) = struct.unpack_from("<I", self.data, self.pos)
        self.pos += 4
        return v

    def u64(self):
        (v,) = struct.unpack_from("<Q", self.data, self.pos)
        self.pos += 8
        return v

    def string(self):
        n = self.u64()
        s = self.data[self.pos:self.pos + n]
        self.pos += n
        return s

    def value(self, vt):
        if vt in (0, 1, 7):
            self.pos += 1
        elif vt in (2, 3):
            self.pos += 2
        elif vt in (4, 5, 6):
            self.pos += 4
        elif vt in (10, 11, 12):
            self.pos += 8
        elif vt == 8:
            self.string()
        elif vt == 9:
            elem = self.u32()
            count = self.u64()
            for _ in range(count):
                self.value(elem)
        else:
            raise ValueError(f"unknown gguf value type {vt}")


def main():
    path, outp = sys.argv[1], sys.argv[2]
    data = open(path, "rb").read()
    r = Reader(data)

    magic = r.u32()
    if magic != MAGIC:
        sys.exit(f"bad magic {magic:#x}")
    version = r.u32()
    tensor_count = r.u64()
    kv_count = r.u64()

    alignment = 32
    for _ in range(kv_count):
        key = r.string()
        vt = r.u32()
        vpos = r.pos
        r.value(vt)
        if key == b"general.alignment" and vt == 4:
            (alignment,) = struct.unpack_from("<I", data, vpos)

    tensors = []
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.u32()
        dims = [0, 0, 0, 0]
        for i in range(n_dims):
            dims[i] = r.u64()
        ttype = r.u32()
        offset = r.u64()
        tensors.append((name, dims, ttype, offset))

    data_start = -(-r.pos // alignment) * alignment

    lines = [
        f"model: version={version} tensors={tensor_count} kv={kv_count} "
        f"data_start={hex(data_start)} align={alignment}"
    ]
    for i, (name, dims, ttype, offset) in enumerate(tensors[:FIRST_N]):
        lines.append(
            f"MT {i} {name.decode()} {dims[0]}x{dims[1]}x{dims[2]}x{dims[3]} "
            f"type={ttype} off={offset}"
        )
    for want in CHECK_NAMES:
        for name, dims, ttype, offset in tensors:
            if name == want:
                at = data_start + offset
                blob = data[at:at + CHECK_LEN]
                crc = zlib.crc32(blob) & 0xFFFFFFFF
                lines.append(
                    f"MV {name.decode()} crc={crc:08x} "
                    f"first4={blob[:4].hex()} dims="
                    f"{dims[0]}x{dims[1]}x{dims[2]}x{dims[3]} type={ttype}"
                )
                break

    open(outp, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
