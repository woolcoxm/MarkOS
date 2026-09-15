#!/usr/bin/env python3
"""One-shot: L27 p1 FFN stage sums in layer_forward + host digest."""

e = open("kernel/src/engine.rs").read()
old = """    silu(&mut act.gate[..n_ff]);
    for i in 0..n_ff {
        act.gate[i] *= act.up[i];
    }"""
new = """    if layer == 27 && pos == 1 {
        let mut s = 0f64;
        for v in act.gate[..n_ff].iter() {
            s += *v as f64;
        }
        let mut s2 = 0f64;
        for v in act.up[..n_ff].iter() {
            s2 += *v as f64;
        }
        let mut s3 = 0f64;
        for v in act.n1[..n_embd].iter() {
            s3 += *v as f64;
        }
        crate::uart::locked_write(format_args!(
            "DBGF n2sum={s3:.6e} gsum={s:.6e} usum={s2:.6e}\\n"
        ));
    }
    silu(&mut act.gate[..n_ff]);
    for i in 0..n_ff {
        act.gate[i] *= act.up[i];
    }
    if layer == 27 && pos == 1 {
        let mut s = 0f64;
        for v in act.gate[..n_ff].iter() {
            s += *v as f64;
        }
        crate::uart::locked_write(format_args!("DBGF prodsum={s:.6e}\\n"));
    }"""
assert old in e
e = e.replace(old, new)
open("kernel/src/engine.rs", "w").write(e)

r = open("scripts/layer_digest2.py").read()
old3 = """        x = mid + deq(f"blk.{L}.ffn_down.weight") @ (fr.silu(gate) * up)
        s = float(np.sum(x.astype(np.float64)))
        print(f"LD p{pos} L{L} xsum={s:.6e}", flush=True)"""
new3 = """        gated = fr.silu(gate) * up
        if L == 27 and pos == 1:
            n2s = float(np.sum(n2.astype(np.float64)))
            gs = float(np.sum(gate.astype(np.float64)))
            us = float(np.sum(up.astype(np.float64)))
            ps = float(np.sum(gated.astype(np.float64)))
            print(f"DBGF n2sum={n2s:.6e} gsum={gs:.6e} usum={us:.6e}", flush=True)
            print(f"DBGF prodsum={ps:.6e}", flush=True)
        x = mid + deq(f"blk.{L}.ffn_down.weight") @ gated
        s = float(np.sum(x.astype(np.float64)))
        print(f"LD p{pos} L{L} xsum={s:.6e}", flush=True)"""
assert old3 in r
r = r.replace(old3, new3)
open("scripts/layer_digest2.py", "w").write(r)
print("ffn sums added")
