//! NEON int8 matmul kernels, dispatched through the execution pool.
//!
//! Compute C[M][N] = sum_k A[m][k] * B[n][k] — the linear-layer shape used
//! by transformer inference (B is the weight matrix of an output row).
//!
//! Two paths:
//!  - `matmul_scalar`: straightforward i32 triple loop, the reference.
//!  - `matmul_udot_rows`: NEON UDOT (ARMv8.2 int8 dot product) — 16 int8
//!    MACs per instruction; the Pi 5's Cortex-A76 implements it (runtime
//!    checked via cpu::has_dotprod before dispatch).
//!
//! UDOT lane semantics: Vd.4S lane j accumulates the dot product of the 4
//! consecutive bytes of Vn and Vm belonging to k-group j — one UDOT over
//! 16-byte chunks advances 4 k-groups at once; the 4 lanes are summed at
//! the end.
//!
//! owns: the A/B/C scratch buffers for the acceptance test.
//! invariants: K must be a multiple of 16 (UDOT chunk size); jobs run
//! through pool::run_on_all (sequential, barrier at end).

use core::arch::asm;

use crate::board;

pub const M: usize = 16;
pub const K: usize = 64;
pub const N: usize = 16;

pub static mut A_BUF: [u8; M * K] = [0; M * K];
pub static mut B_BUF: [u8; N * K] = [0; N * K];
pub static mut C_BUF: [i32; M * N] = [0; M * N];

/// Scalar reference path: exact i32 accumulation.
pub fn matmul_scalar(a: &[u8], b: &[u8], c: &mut [i32]) {
    for m in 0..M {
        for n in 0..N {
            let mut acc: i64 = 0;
            for k in 0..K {
                acc += a[m * K + k] as i64 * b[n * K + k] as i64;
            }
            c[m * N + n] = acc as i32;
        }
    }
}

/// Compute rows [m_start, m_end) of C with the NEON UDOT path.
///
/// Soundness: a/b/c are the module-owned buffers; each core writes only its
/// own row range; NEON temps v0/v1/v2 are scratch within each asm block.
pub unsafe fn matmul_udot_rows(a: &[u8], b: &[u8], c: &mut [i32], m_start: usize, m_end: usize) {
    for m in m_start..m_end {
        for n in 0..N {
            let mut acc = [0u32; 4];
            acc = [0; 4];   // explicit re-zero: defeats LICM hoisting of the init
            let mut tmp = [0u32; 4];
            let mut kk = 0usize;
            while kk < K {
                // Soundness: chunks lie inside a/b; NEON temps v0-v2 are
                // scratch within this asm block.
                asm!(
                    "ld1 {{v0.16b}}, [{a}]",
                    "ld1 {{v1.16b}}, [{b}]",
                    "movi v2.4s, #0",
                    "udot v2.4s, v0.16b, v1.16b",
                    "st1 {{v2.4s}}, [{tmp}]",
                    a = in(reg) &a[m * K + kk],
                    b = in(reg) &b[n * K + kk],
                    tmp = in(reg) tmp.as_mut_ptr(),
                    lateout("v0") _, lateout("v1") _, lateout("v2") _,
                    options(nostack)
                );
                for lane in 0..4 {
                    acc[lane] = acc[lane].wrapping_add(tmp[lane]);
                }
                kk += 16;
            }
            c[m * N + n] = (acc[0]
                .wrapping_add(acc[1]))
                .wrapping_add(acc[2].wrapping_add(acc[3])) as i32;
        }
    }
}

/// Pool job: each core computes a contiguous row range of C with the NEON
/// UDOT path. `arg`/context unused — the descriptors are module statics.
pub fn pool_matmul_job(core_id: usize, _arg: u64) {
    // Soundness: A_BUF/B_BUF/C_BUF are module-owned scratch; row ranges are
    // disjoint per core and written before the pool barrier.
    let a = &raw const A_BUF;
    let b = &raw const B_BUF;
    let c = &raw mut C_BUF;
    unsafe {
        let rows_per = M / board::CORE_COUNT;
        let m_start = core_id * rows_per;
        let m_end = if core_id + 1 == board::CORE_COUNT { M } else { m_start + rows_per };
        matmul_udot_rows(&*a, &*b, &mut *c, m_start, m_end);
    }
}
