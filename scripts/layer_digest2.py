#!/usr/bin/env python3
"""Debug: per-layer residual checksums for BOTH prompt positions."""
import importlib.util
import math
import numpy as np

spec = importlib.util.spec_from_file_location("fr", "scripts/forward_ref.py")
fr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fr)

f = open("/home/mark/markos-models/Qwen3-0.6B-Q8_0.gguf", "rb")
kvs, vocab, merges, tensors, ds = fr.parse_header(f)
n_embd = kvs["qwen3.embedding_length"]
n_heads = kvs["qwen3.attention.head_count"]
n_kv = kvs["qwen3.attention.head_count_kv"]
head_dim = kvs["qwen3.attention.key_length"]
n_layers = kvs["qwen3.block_count"]
theta = kvs["qwen3.rope.freq_base"]
eps = kvs["qwen3.attention.layer_norm_rms_epsilon"]


def at(name):
    return ds + tensors[name][2]


def deq(name):
    dims, ttype, off = tensors[name]
    n_out, n_in = int(dims[1]), int(dims[0])
    return fr.dequant_q8_0(f, at(name), n_out * n_in).reshape(n_out, n_in)


inv = theta ** (-np.arange(0, head_dim, 2, dtype=np.float64) / head_dim)
kc = [[] for _ in range(n_layers)]
vc = [[] for _ in range(n_layers)]

for pos, tok in enumerate([14990, 1879]):
    x = fr.dequant_q8_0(f, at("token_embd.weight") + tok * 1088, n_embd)
    for L in range(n_layers):
        n1 = fr.rmsnorm(x, fr.vec_f32(f, at(f"blk.{L}.attn_norm.weight"), n_embd), eps)
        q = (deq(f"blk.{L}.attn_q.weight") @ n1).reshape(n_heads, head_dim)
        k = (deq(f"blk.{L}.attn_k.weight") @ n1).reshape(n_kv, head_dim)
        v = (deq(f"blk.{L}.attn_v.weight") @ n1).reshape(n_kv, head_dim)
        qw = fr.vec_f32(f, at(f"blk.{L}.attn_q_norm.weight"), head_dim)
        kw = fr.vec_f32(f, at(f"blk.{L}.attn_k_norm.weight"), head_dim)
        ang = pos * inv
        cos, sin = np.cos(ang), np.sin(ang)

        def rope(h):
            x1, x2 = h[: head_dim // 2], h[head_dim // 2:]
            return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin])

        q = np.stack([rope(fr.rmsnorm(row, qw, eps)) for row in q])
        k = np.stack([rope(fr.rmsnorm(row, kw, eps)) for row in k])
        kc[L].append(k)
        vc[L].append(v)
        attn_out = np.zeros(n_heads * head_dim, dtype=np.float32)
        for h in range(n_heads):
            kvh = h // (n_heads // n_kv)
            scores = np.array(
                [float(np.dot(q[h], kc[L][p][kvh]) / math.sqrt(head_dim))
                 for p in range(len(kc[L]))]
            )
            pr = np.exp(scores - scores.max())
            pr = pr / pr.sum()
            attn_out[h * head_dim:(h + 1) * head_dim] = sum(
                pr[i] * vc[L][i][kvh] for i in range(len(kc[L]))
            )
        if L == 27 and pos == 1:
            qsum = float(np.sum(q.astype(np.float64)))
            ksum = float(np.sum(k.astype(np.float64)))
            vsum = float(np.sum(v.astype(np.float64)))
            print(f"DBG27 qsum={qsum:.6e} ksum={ksum:.6e} vsum={vsum:.6e}", flush=True)
        mid = x + deq(f"blk.{L}.attn_output.weight") @ attn_out
        if L == 27 and pos == 1:
            asum = float(np.sum(attn_out.astype(np.float64)))
            msum = float(np.sum(mid.astype(np.float64)))
            print(f"DBG27 asum={asum:.6e} msum={msum:.6e}", flush=True)
        n2 = fr.rmsnorm(mid, fr.vec_f32(f, at(f"blk.{L}.ffn_norm.weight"), n_embd), eps)
        gate = deq(f"blk.{L}.ffn_gate.weight") @ n2
        up = deq(f"blk.{L}.ffn_up.weight") @ n2
        gated = fr.silu(gate) * up
        if L == 27 and pos == 1:
            n2s = float(np.sum(n2.astype(np.float64)))
            gs = float(np.sum(gate.astype(np.float64)))
            us = float(np.sum(up.astype(np.float64)))
            ps = float(np.sum(gated.astype(np.float64)))
            print(f"DBGF n2sum={n2s:.6e} gsum={gs:.6e} usum={us:.6e}", flush=True)
            print(f"DBGF prodsum={ps:.6e}", flush=True)
            ds27 = deq(f"blk.{L}.ffn_down.weight")
            downsum = float(np.sum((ds27 @ gated).astype(np.float64)))
            print(f"DBGF downsum={downsum:.6e}", flush=True)
        x = mid + deq(f"blk.{L}.ffn_down.weight") @ gated
        s = float(np.sum(x.astype(np.float64)))
        print(f"LD p{pos} L{L} xsum={s:.6e}", flush=True)
