#!/usr/bin/env python3
"""Generate a synthetic GGUF v3 test model for MarkOS engine tests/e2e.

Ported from the prior bare-metal MarkOS kernel test suite. Produces a tiny
GGUF with real llama-style metadata (arch=qwen2 shape facts) so the engine's
GGUF reader, guardrail estimator and (with mock backend) the whole serving
path can be exercised without a multi-GB download.

Usage: make_gguf_test.py OUT.gguf
"""
import struct
import sys

# qwen2-ish tiny shape
ARCH = "qwen2"
N_LAYERS = 2
N_EMBD = 64
N_HEAD = 4
N_HEAD_KV = 2
CTX = 4096
VOCAB = 256
ALIGN = 32


def kv_str(key: str, val: str) -> bytes:
    return (struct.pack("<Q", len(key)) + key.encode()
            + struct.pack("<I", 8) + struct.pack("<Q", len(val)) + val.encode())


def kv_u32(key: str, val: int) -> bytes:
    return (struct.pack("<Q", len(key)) + key.encode()
            + struct.pack("<I", 4) + struct.pack("<I", val))


def tensor(name: str, dims, dtype: int, offset: int) -> bytes:
    r = struct.pack("<Q", len(name)) + name.encode()
    r += struct.pack("<I", len(dims))
    for d in dims:
        r += struct.pack("<Q", d)
    r += struct.pack("<I", dtype)
    return r + struct.pack("<Q", offset)


def build() -> bytes:
    all_tensors = [("token_embd.weight", [N_EMBD, VOCAB])]
    for b in range(N_LAYERS):
        hd = N_EMBD // N_HEAD
        all_tensors += [
            (f"blk.{b}.attn_q.weight", [N_EMBD, N_EMBD]),
            (f"blk.{b}.attn_k.weight", [N_EMBD, N_HEAD_KV * hd]),
            (f"blk.{b}.attn_v.weight", [N_EMBD, N_HEAD_KV * hd]),
            (f"blk.{b}.attn_output.weight", [N_EMBD, N_EMBD]),
            (f"blk.{b}.ffn_down.weight", [N_EMBD * 2, N_EMBD]),
            (f"blk.{b}.ffn_gate.weight", [N_EMBD, N_EMBD * 2]),
            (f"blk.{b}.ffn_up.weight", [N_EMBD, N_EMBD * 2]),
        ]
    all_tensors.append(("output.weight", [N_EMBD, VOCAB]))

    head = bytearray()
    head += struct.pack("<I", 0x46554747)     # magic "GGUF"
    head += struct.pack("<I", 3)              # version 3
    head += struct.pack("<Q", len(all_tensors))
    head += struct.pack("<Q", 12)             # kv count below
    head += kv_str("general.architecture", ARCH)
    head += kv_str("general.name", "markos-test")
    head += kv_u32("general.alignment", ALIGN)
    head += kv_u32(f"{ARCH}.block_count", N_LAYERS)
    head += kv_u32(f"{ARCH}.embedding_length", N_EMBD)
    head += kv_u32(f"{ARCH}.attention.head_count", N_HEAD)
    head += kv_u32(f"{ARCH}.attention.head_count_kv", N_HEAD_KV)
    head += kv_u32(f"{ARCH}.context_length", CTX)
    head += kv_u32(f"{ARCH}.vocab_size", VOCAB)
    head += kv_str("tokenizer.chat_template",
                   "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}[assistant] ")
    head += kv_str("tokenizer.ggml.model", "llama")
    head += kv_u32("tokenizer.ggml.tokens_size", VOCAB)  # informational

    body = bytearray()
    offset = 0
    for name, dims in all_tensors:
        ne = 1
        for d in dims:
            ne *= d
        nbytes = ne * 2  # F16
        head += tensor(name, dims, 1, offset)
        body += b"\x00" * nbytes
        offset += nbytes
    while len(head) % ALIGN != 0:
        head += b"\x00"
    return bytes(head) + bytes(body)


def main(path: str) -> None:
    data = build()
    with open(path, "wb") as f:
        f.write(data)
    print(f"wrote {path}: {len(data)} bytes")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
