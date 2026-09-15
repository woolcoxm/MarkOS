#!/usr/bin/env python3
"""Debug: per-layer residual checksums for position 0 of the gen prompt.
At position 0 attention is the identity over V (single position, prob 1),
so each layer reduces to: x += o_proj @ v, x += down @ (silu(gate)*up @ up)."""
import importlib.util
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
eps = kvs["qwen3.attention.layer_norm_rms_epsilon"]


def at(name):
    return ds + tensors[name][2]


def deq(name):
    dims, ttype, off = tensors[name]
    n_out, n_in = int(dims[1]), int(dims[0])
    w = fr.dequant_q8_0(f, at(name), n_out * n_in).reshape(n_out, n_in)
    return w


x = fr.dequant_q8_0(f, at("token_embd.weight") + 14990 * 1088, n_embd)
for L in range(n_layers):
    n1 = fr.rmsnorm(x, fr.vec_f32(f, at(f"blk.{L}.attn_norm.weight"), n_embd), eps)
    if L == 0:
        s = float(np.sum(n1.astype(np.float64)))
        print(
            f"DBG n1 v={n1[0]:.6e},{n1[1]:.6e},{n1[2]:.6e},{n1[3]:.6e} sum={s:.6e}",
            flush=True,
        )
    v = (deq(f"blk.{L}.attn_v.weight") @ n1).reshape(n_kv, head_dim)
    attn_out = np.zeros(n_heads * head_dim, dtype=np.float32)
    for h in range(n_heads):
        kvh = h // (n_heads // n_kv)
        attn_out[h * head_dim:(h + 1) * head_dim] = v[kvh]
    if L == 0:
        s = float(np.sum(attn_out.astype(np.float64)))
        print(
            f"DBG attn v={attn_out[0]:.6e},{attn_out[1]:.6e},{attn_out[2]:.6e},{attn_out[3]:.6e} asum={s:.6e}",
            flush=True,
        )
        s2 = float(np.sum(mid0 := (x + deq(f"blk.{L}.attn_output.weight") @ attn_out).astype(np.float64)))
        print(
            f"DBG mid v={mid0[0]:.6e},{mid0[1]:.6e},{mid0[2]:.6e},{mid0[3]:.6e} msum={s2:.6e}",
            flush=True,
        )
    mid = x + deq(f"blk.{L}.attn_output.weight") @ attn_out
    n2 = fr.rmsnorm(mid, fr.vec_f32(f, at(f"blk.{L}.ffn_norm.weight"), n_embd), eps)
    gate = deq(f"blk.{L}.ffn_gate.weight") @ n2
    up = deq(f"blk.{L}.ffn_up.weight") @ n2
    gated = fr.silu(gate) * up
    if L == 0:
        print(
            f"DBG gate v={gate[0]:.6e},{gate[1]:.6e},{gate[2]:.6e},{gate[3]:.6e} gsum={float(np.sum(gate.astype(np.float64))):.6e}",
            flush=True,
        )
        print(f"DBG prod gsum={float(np.sum(gated.astype(np.float64))):.6e}", flush=True)
    down = deq(f"blk.{L}.ffn_down.weight") @ gated
    if L == 0:
        print(f"DBG down dsum={float(np.sum(down.astype(np.float64))):.6e}", flush=True)
    x = mid + down
    print(f"LD {L} xsum={float(np.sum(x.astype(np.float64))):.6e}", flush=True)
