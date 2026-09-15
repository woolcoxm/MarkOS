#!/usr/bin/env python3
"""One-shot: pool-parallel SDOT matvec in matvec_udot (clean rewrite)."""

src = open("kernel/src/engine.rs").read()

# Find and replace the entire matvec_udot function
old_start = src.index("/// Serving-path fast matvec: quantizes x once (one symmetric scale), then")
old_end = src.index("pub fn matvec_q8_0(", old_start)

new_fn = r'''/// Pool job descriptor (BSP writes, all cores read after cache clean).
static mut PJ_ROW_BASE: u64 = 0;
static mut PJ_ROW_STRIDE: usize = 0;
static mut PJ_XQ_PTR: usize = 0;
static mut PJ_N_BLK: usize = 0;
static mut PJ_N_OUT: usize = 0;
static mut PJ_SX: f32 = 0.0;
static mut PJ_Y_PTR: usize = 0;

/// Pool job: compute rows [core_id*rp, (core_id+1)*rp) of y.
/// Soundness: pure NEON math on in-bounds rows of the RAM cache; y writes
/// are disjoint per core; XQ is read-only shared. Runs on MMU-off APs.
#[target_feature(enable = "dotprod")]
unsafe fn sdot_pool_job(core_id: usize, _arg: u64) {
    let n_cores = board::CORE_COUNT;
    let rows_per = PJ_N_OUT / n_cores;
    let start = core_id * rows_per;
    let n_blk = PJ_N_BLK;
    let sx = PJ_SX;
    let xq = PJ_XQ_PTR as *const u8;
    let wbase = PJ_ROW_BASE;
    let stride = PJ_ROW_STRIDE;
    let y = PJ_Y_PTR as *mut f32;

    for r in start..(start + rows_per) {
        let row_base = wbase + (r as u64) * stride as u64;
        let mut acc_f = 0f32;
        for b in 0..n_blk {
            let boff = b * 34;
            let sb = f16_to_f32(u16::from_le_bytes(
                core::ptr::read((row_base as usize + boff) as *const [u8; 2]),
            ));
            let mut dot = 0f32;
            unsafe {
                sdot_block(
                    (row_base + (boff + 2)) as *const u8,
                    xq.un_add((b * 32)),
                    &mut dot,
                );
            }
            acc_f += dot * sb * sx;
        }
        y.add(r).write(acc_f);
    }
}

/// Serving-path fast matvec: quantizes x once (one symmetric scale), then
/// pool-parallel SDOT row dots from the RAM weight cache. Requires
/// ram_weights + has_dotprod (checked by the caller).
fn matvec_udot(
    vol: &FatVolume,
    file: &File,
    abs: u64,
    n_in: usize,
    n_out: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), &'static str> {
    static mut XQ: [u8; MAX_DIM] = [0; MAX_DIM];

    let sx = quantize_x_static(x, unsafe {
        core::slice::from_raw_parts_mut((&raw mut XQ) as *mut u8, MAX_DIM)
    });

    unsafe {
        PJ_ROW_BASE = board::WEIGHT_RAM_BASE as u64 + abs;
        PJ_ROW_STRIDE = n_in / 32 * 34;
        PJ_XQ_PTR = (&raw const XQ) as usize;
        PJ_N_BLK = n_in / 32;
        PJ_N_OUT = n_out;
        PJ_SX = sx;
        PJ_Y_PTR = y.as_mut_ptr() as usize;
    }

    // Clean descriptor + XQ so the MMU-off cores read current values.
    cache::clean_range((&raw const PJ_ROW_BASE) as usize, 48);
    cache::clean_range((&raw const XQ) as usize, n_in);

    pool::run_on_all(sdot_pool_job, 0, board::CORE_COUNT);

    // APs wrote y with MMU off (uncached, straight to RAM): invalidate the
    // BSP's cached copies so it reads the combined results.
    cache::invalidate_range(y.as_mut_ptr() as usize, n_out * 4);

    Ok(())
}

'''

# Find the old matvec_udot and replace it
old_start = src.index("/// Serving-path fast matvec: quantizes x once")
old_end = src.index("pub fn matvec_q8_0(", old_start)
src = src[:old_start] + new_fn + src[old_end:]

# Now rewrite sdot_pool_job with proper raw pointer arithmetic
# (the tmp version above uses .un_add which doesn't exist)
src = src.replace(
    """            let mut dot = 0f32;
            unsafe {
                sdot_block(
                    (row_base + (boff + 2)) as *const u8,
                    xq.un_add((b * 32)) as *const u8,
                    &mut dot,
                );
            }
            acc_f += dot * sb * sx;""",
    """            let mut dot = 0f32;
            let qp = (row_base + (boff + 2) as usize) as *const u8;
            let xp = xq.add(b * 32);
            // Soundness: sdot_block is pure NEON math on in-bounds blocks.
            unsafe {
                dot = sdot_block(qp, xp) as f32;
            }
            acc_f += dot * sb * sx;""",
)

open("kernel/src/engine.rs", "w").write(src)
print("pool-parallel SDOT matvec installed")
