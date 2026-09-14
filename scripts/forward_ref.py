#!/usr/bin/env python3
"""Host reference for the Phase 8a forward-pass gate.

Tokenizes a fixed prompt with the GGUF's own Qwen3 (GPT-2-style) BPE, runs
one decoder layer of the same GGUF in numpy (embedding -> RMSNorm -> q8_0
matmuls -> per-head q/k RMSNorm -> RoPE -> GQA attention -> FFN SwiGLU),
and emits the lines the kernel must reproduce (within f32 tolerance).

Usage: forward_ref.py <model.gguf> <out.txt>"""
import math
import re
import struct
import sys

import numpy as np

MAGIC = 0x46554747
PROMPT = b"hello world"
VT_STR, VT_ARR = 8, 9


def parse_header(f):
    """Header walk: kvs of interest, vocab/merges array locations, tensors."""

    def u32():
        return struct.unpack("<I", f.read(4))[0]

    def u64():
        return struct.unpack("<Q", f.read(8))[0]

    def string():
        n = u64()
        return f.read(n)

    def skip_value(vt):
        if vt in (0, 1, 7):
            f.read(1)
        elif vt in (2, 3):
            f.read(2)
        elif vt in (4, 5, 6):
            f.read(4)
        elif vt in (10, 11, 12):
            f.read(8)
        elif vt == VT_STR:
            string()
        elif vt == VT_ARR:
            e, c = u32(), u64()
            for _ in range(c):
                skip_value(e)

    if u32() != MAGIC:
        sys.exit("bad magic")
    _version = u32()  # u32 version between magic and the u64 counters
    tensor_count = u64()
    kv_count = u64()

    kvs = {}
    vocab = merges = None
    for _ in range(kv_count):
        key = string().decode()
        vt = u32()
        if vt in (4, 5, 6):
            raw = f.read(4)
            fmt = {4: "<i", 5: "<i", 6: "<f"}[vt]
            kvs[key] = struct.unpack(fmt, raw)[0]
        elif vt == VT_ARR and key in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges"):
            e, c = u32(), u64()
            off = f.tell()
            if e != VT_STR:
                sys.exit(f"{key}: not a string array")
            for _ in range(c):
                string()
            if key.endswith("tokens"):
                vocab = (off, c)
            else:
                merges = (off, c)
        else:
            skip_value(vt)

    tensors = {}
    for _ in range(tensor_count):
        name = string().decode()
        n_dims = u32()
        dims = [u64() for _ in range(n_dims)]
        ttype = u32()
        offset = u64()
        tensors[name] = (dims, ttype, offset)

    # Tensor offsets are relative to the aligned data section.
    alignment = kvs.get("general.alignment", 32)
    data_start = -(-f.tell() // alignment) * alignment
    return kvs, vocab, merges, tensors, data_start


def dequant_q8_0(f, at, n_elems):
    # GGUF QK8_0 block: f16 scale first, then 32 int8 quants (34 B / 32 elems).
    nb = n_elems // 32
    f.seek(at)
    raw = f.read(nb * 34)
    a = np.frombuffer(raw, dtype=np.uint8).reshape(nb, 34)
    scales = a[:, 0:2].copy().view(np.float16).astype(np.float32).reshape(nb, 1)
    qs = a[:, 2:34].copy().view(np.int8).astype(np.float32)
    return (qs * scales).reshape(-1)


def vec_f32(f, at, n):
    f.seek(at)
    return np.frombuffer(f.read(n * 4), dtype="<f4").astype(np.float32)


def rmsnorm(x, w, eps):
    return x / math.sqrt(float(np.mean(x * x)) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def bytes_to_unicode():
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(0xA1, 0xAC + 1))
        + list(range(0xAE, 0xFF + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


PRETOKEN = re.compile(
    r"'(?:[sdmt]|ll|ve|re)| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+(?!\S)|\s+"
)


def tokenize(f, vocab_off, vocab_n, merges_off, merges_n, text):
    def strings_at(off, count):
        f.seek(off)
        for _ in range(count):
            n = struct.unpack("<Q", f.read(8))[0]
            yield f.read(n)

    vocab = {}
    for i, s in enumerate(strings_at(vocab_off, vocab_n)):
        vocab[s] = i
    ranks = {}
    for i, s in enumerate(strings_at(merges_off, merges_n)):
        a, _, b = s.partition(b" ")
        ranks[(a, b)] = i

    be = bytes_to_unicode()
    ids = []
    for piece in PRETOKEN.findall(text.decode()):
        # Byte-unicode encode the piece; each mapped character (1-2 bytes in
        # UTF-8) is one starting symbol — merge/vocab strings are exactly
        # these characters' UTF-8 bytes concatenated.
        sym = "".join(be[b] for b in piece.encode("utf-8"))
        word = [ch.encode("utf-8") for ch in sym]
        while len(word) > 1:
            best, best_rank = None, None
            for a, b in zip(word, word[1:]):
                r = ranks.get((a, b))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = (a, b), r
            if best is None:
                break
            merged, i = [], 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best:
                    merged.append(word[i] + word[i + 1])
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        for w in word:
            if w not in vocab:
                sys.exit(f"piece not in vocab: {w!r}")
            ids.append(vocab[w])
    return ids


def main():
    path, outp = sys.argv[1], sys.argv[2]
    f = open(path, "rb")
    kvs, vocab, merges, tensors, data_start = parse_header(f)
    if vocab is None or merges is None:
        sys.exit("vocab/merges arrays not found")

    def at(name):
        return data_start + tensors[name][2]

    n_embd = kvs["qwen3.embedding_length"]
    n_heads = kvs["qwen3.attention.head_count"]
    n_kv = kvs["qwen3.attention.head_count_kv"]
    head_dim = kvs["qwen3.attention.key_length"]
    theta = kvs["qwen3.rope.freq_base"]
    eps = kvs["qwen3.attention.layer_norm_rms_epsilon"]

    ids = tokenize(f, vocab[0], vocab[1], merges[0], merges[1], PROMPT)
    lines = [f"TOKS prompt={PROMPT.decode()} ids={','.join(map(str, ids))} n_tokens={len(ids)}"]

    k_cache, v_cache = [], []
    for t, tok in enumerate(ids):
        dims, ttype, offset = tensors["token_embd.weight"]
        row_elems = dims[0]
        x = dequant_q8_0(f, at("token_embd.weight") + tok * (row_elems // 32) * 34, row_elems)
        emb_rms = math.sqrt(float(np.mean(x * x)))
        lines.append(
            f"EMB{t} id={tok} rms={emb_rms:.6e} v={x[0]:.6e},{x[1]:.6e},{x[2]:.6e},{x[3]:.6e}"
        )

        w = vec_f32(f, at("blk.0.attn_norm.weight"), n_embd)
        n1 = rmsnorm(x, w, eps)
        lines.append(f"NRM{t} v={n1[0]:.6e},{n1[1]:.6e},{n1[2]:.6e},{n1[3]:.6e}")

        def q8_matvec(name, vec):
            dims, ttype, offset = tensors[name]
            n_out, n_in = int(dims[1]), int(dims[0])
            w = dequant_q8_0(f, at(name), n_out * n_in).reshape(n_out, n_in)
            return w @ vec

        q = q8_matvec("blk.0.attn_q.weight", n1)
        k = q8_matvec("blk.0.attn_k.weight", n1)
        v = q8_matvec("blk.0.attn_v.weight", n1)

        qw = vec_f32(f, at("blk.0.attn_q_norm.weight"), head_dim)
        kw = vec_f32(f, at("blk.0.attn_k_norm.weight"), head_dim)
        q = q.reshape(n_heads, head_dim)
        k = k.reshape(n_kv, head_dim)
        v = v.reshape(n_kv, head_dim)
        inv = theta ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
        ang = t * inv
        cos, sin = np.cos(ang), np.sin(ang)

        def rope(h):
            x1, x2 = h[: head_dim // 2], h[head_dim // 2:]
            return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin])

        # Qwen3 order: per-head RMSNorm FIRST, then RoPE.
        q = np.stack([rope(rmsnorm(row, qw, eps)) for row in q])
        k = np.stack([rope(rmsnorm(row, kw, eps)) for row in k])
        lines.append(
            f"QK{t} q={q[0][0]:.6e},{q[0][1]:.6e},{q[0][2]:.6e},{q[0][3]:.6e}"
            f" k={k[0][0]:.6e},{k[0][1]:.6e},{k[0][2]:.6e},{k[0][3]:.6e}"
        )

        k_cache.append(k)
        v_cache.append(v)

        attn_out = np.zeros(n_heads * head_dim, dtype=np.float32)
        probs0 = None
        for h in range(n_heads):
            kvh = h // (n_heads // n_kv)
            scores = np.array(
                [float(np.dot(q[h], kc[kvh]) / math.sqrt(head_dim)) for kc in k_cache]
            )
            p = np.exp(scores - scores.max())
            p = p / p.sum()
            if h == 0:
                probs0 = p.copy()
            o = sum(p[i] * v_cache[i][kvh] for i in range(len(k_cache)))
            attn_out[h * head_dim:(h + 1) * head_dim] = o
        lines.append(
            f"ATT{t} p0={probs0[0]:.6e}"
            + (f" p1={probs0[1]:.6e}" if len(probs0) > 1 else "")
            + f" o={attn_out[0]:.6e},{attn_out[1]:.6e},{attn_out[2]:.6e},{attn_out[3]:.6e}"
        )

        wo = q8_matvec("blk.0.attn_output.weight", attn_out)
        mid = x + wo
        lines.append(f"MID{t} v={mid[0]:.6e},{mid[1]:.6e},{mid[2]:.6e},{mid[3]:.6e}")

        w2 = vec_f32(f, at("blk.0.ffn_norm.weight"), n_embd)
        n2 = rmsnorm(mid, w2, eps)
        gate = q8_matvec("blk.0.ffn_gate.weight", n2)
        up = q8_matvec("blk.0.ffn_up.weight", n2)
        act = silu(gate) * up
        down = q8_matvec("blk.0.ffn_down.weight", act)
        hid = mid + down
        lines.append(
            f"HID{t} v={hid[0]:.6e},{hid[1]:.6e},{hid[2]:.6e},{hid[3]:.6e}"
            f" sum={float(np.sum(hid.astype(np.float64))):.6e}"
        )

    lines.append("PASS: forward")
    open(outp, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
