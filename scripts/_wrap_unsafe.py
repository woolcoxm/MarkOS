#!/usr/bin/env python3
"""One-shot: wrap matvec_udot mutable-static accesses in unsafe blocks."""

src = open("kernel/src/engine.rs").read()

# 1. Wrap the per-block quantization in unsafe
old1 = """    let n_blk = n_in / 32;
    for b in 0..n_blk {
        let mut bmax = 0f32;
        for j in 0..32 {
            let v = x[b * 32 + j];
            let a = if v < 0.0 { -v } else { v };
            if a > bmax {
                bmax = a;
            }
        }
        let bs = if bmax > 0.0 { bmax / 127.0 } else { 1.0 };
        XS[b] = bs;
        for j in 0..32 {
            let q = round_i32(x[b * 32 + j] / bs).clamp(-127, 127);
            XQ_BUF[b * 32 + j] = (q as i8) as u8;
        }
    }

    // Pool-parallel: each core computes its share of rows."""
new1 = """    let n_blk = n_in / 32;
    unsafe {
    for b in 0..n_blk {
        let mut bmax = 0f32;
        for j in 0..32 {
            let v = x[b * 32 + j];
            let a = if v < 0.0 { -v } else { v };
            if a > bmax {
                bmax = a;
            }
        }
        let bs = if bmax > 0.0 { bmax / 127.0 } else { 1.0 };
        XS_BUF[b] = bs;
        for j in 0..32 {
            let q = round_i32(x[b * 32 + j] / bs).clamp(-127, 127);
            XQ_BUF[b * 32 + j] = (q as i8) as u8;
        }
    }
    }

    // Pool-parallel: each core computes its share of rows."""
assert old1 in src, "quantization block not found"
src = src.replace(old1, new1, 1)

# 2. Fix XQ_BUF/XS_BUF references in the pool job
old2 = """    let sx = PJ_SX;
    let dotprod = PJ_DOTPROD;
    let xq = PJ_XQ_PTR as *const u8;"""
new2 = """    let dotprod = PJ_DOTPROD;
    let xq = PJ_XQ_PTR as *const u8;
    let xs = PJ_XS_PTR as *const f32;"""
assert old2 in src, "pool job head not found"
src = src.replace(old2, new2, 1)

old3 = "            acc_f += dot * sb * sx;"
new3 = """            let sxb = core::ptr::read((xs as usize + b * 4) as *const f32);
            acc_f += dot * sb * sxb;"""
# Only replace inside psdot_pool_job (search from psdot_pool_job onwards)
pi = src.index("fn psdot_pool_job")
pe = src.index("fn matvec_udot", pi)
body = src[pi:pe]
n_replaced = body.count(old3)
body = body.replace(old3, new3)
src = src[:pi] + body + src[pe:]
print(f"replaced {n_replaced} sx references in pool job")

open("kernel/src/engine.rs", "w").write(src)
print("done")
