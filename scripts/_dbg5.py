#!/usr/bin/env python3
"""One-shot: stage sums for L27 p1 in layer_forward; host digest L27 p1."""

e = open("kernel/src/engine.rs").read()
old = """    let row = layer * MAX_POS + pos;
    // Soundness: static KV cache, exclusive BSP access during the pass.
    unsafe {
        KCACHE[row][..kv_dim].copy_from_slice(&act.k[..kv_dim]);
        VCACHE[row][..kv_dim].copy_from_slice(&act.v[..kv_dim]);
    }"""
new = """    let row = layer * MAX_POS + pos;
    // Soundness: static KV cache, exclusive BSP access during the pass.
    unsafe {
        KCACHE[row][..kv_dim].copy_from_slice(&act.k[..kv_dim]);
        VCACHE[row][..kv_dim].copy_from_slice(&act.v[..kv_dim]);
    }
    if layer == 27 && pos == 1 {
        let mut s = 0f64;
        for v in act.q[..geo.n_heads * hd].iter() {
            s += *v as f64;
        }
        let mut s2 = 0f64;
        for v in act.k[..kv_dim].iter() {
            s2 += *v as f64;
        }
        let mut s3 = 0f64;
        for v in act.v[..kv_dim].iter() {
            s3 += *v as f64;
        }
        crate::uart::locked_write(format_args!(
            "DBG27 qsum={s:.6e} ksum={s2:.6e} vsum={s3:.6e}\\n"
        ));
    }"""
assert old in e
e = e.replace(old, new)

old2 = """    for i in 0..n_embd {
        act.mid[i] += act.x[i];
    }

    let fn_abs = ds"""
new2 = """    for i in 0..n_embd {
        act.mid[i] += act.x[i];
    }
    if layer == 27 && pos == 1 {
        let mut s = 0f64;
        for v in act.attn[..o_dims[0] as usize].iter() {
            s += *v as f64;
        }
        let mut s2 = 0f64;
        for v in act.mid[..n_embd].iter() {
            s2 += *v as f64;
        }
        crate::uart::locked_write(format_args!(
            "DBG27 asum={s:.6e} msum={s2:.6e}\\n"
        ));
    }

    let fn_abs = ds"""
assert old2 in e
e = e.replace(old2, new2)
open("kernel/src/engine.rs", "w").write(e)

r = open("scripts/layer_digest2.py").read()
old3 = """        mid = x + deq(f"blk.{L}.attn_output.weight") @ attn_out"""
new3 = """        if L == 27 and pos == 1:
            qsum = float(np.sum(q.astype(np.float64)))
            ksum = float(np.sum(k.astype(np.float64)))
            vsum = float(np.sum(v.astype(np.float64)))
            print(f"DBG27 qsum={qsum:.6e} ksum={ksum:.6e} vsum={vsum:.6e}", flush=True)
            asum = float(np.sum(attn_out.astype(np.float64)))
        mid = x + deq(f"blk.{L}.attn_output.weight") @ attn_out
        if L == 27 and pos == 1:
            msum = float(np.sum(mid.astype(np.float64)))
            print(f"DBG27 asum={asum:.6e} msum={msum:.6e}", flush=True)"""
assert old3 in r
r = r.replace(old3, new3)
open("scripts/layer_digest2.py", "w").write(r)
print("L27 instrumentation added")
