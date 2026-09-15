//! Phase 8a: Qwen3 inference core — BPE tokenizer, q8_0 streaming matvec,
//! RMSNorm, RoPE, GQA attention, SwiGLU FFN.
//!
//! Everything reads weights straight from the FAT volume at the tensor
//! offsets gguf:: reported (read_at) — the 640 MB model never sits in RAM.
//! Compute is correctness-first scalar f32; NEON tiling lands in the perf
//! phase.
//!
//! owns: the scratch buffers below (BSP-only, selftest/serve context —
//! the pool cores are parked during a forward pass in this phase).
//! invariants: no panics on model data — malformed input yields Err.

use core::intrinsics::{roundf32, roundf64, sqrtf32, sqrtf64};

use crate::board;
use crate::cache;
use crate::cpu;
use crate::fat::{FatVolume, File};
use crate::pool;

// Soundness: the intrinsics below are the hardware round/sqrt operations —
// pure math, no memory effects; core does not expose them on this target.
fn fround32(x: f32) -> f32 {
    unsafe { roundf32(x) }
}
fn fround64(x: f64) -> f64 {
    unsafe { roundf64(x) }
}
pub fn fsqrt32(x: f32) -> f32 {
    unsafe { sqrtf32(x) }
}
fn fsqrt64(x: f64) -> f64 {
    unsafe { sqrtf64(x) }
}
use crate::gguf;

// ===== math helpers =====

/// f16 (IEEE half) -> f32, exact.
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    if exp == 0 {
        if man == 0 {
            return f32::from_bits(sign << 31);
        }
        let v = (man as f32) * 5.960_464_5e-8; // 2^-24
        return if sign == 1 { -v } else { v };
    }
    let bits = if exp == 31 {
        (sign << 31) | 0x7F80_0000 | (man << 13)
    } else {
        (sign << 31) | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// expf for moderate arguments (|x| < 80): 2^k * exp(r) with a degree-6
/// Taylor on |r| <= ln2/2 — ~1 ulp against the host's math.exp in f32.
fn expf(x: f32) -> f32 {
    // Range guards: beyond +-88 the 2^k bit construction over/underflows.
    // silu relies on exp(-large) -> 0 and exp(+large) -> inf saturating.
    if x > 88.0 {
        return f32::INFINITY;
    }
    if x < -100.0 {
        return 0.0;
    }
    const LN2: f32 = 0.693_147_2;
    const INV_LN2: f32 = 1.442_695;
    let k = fround32(x * INV_LN2);
    let r = x - k * LN2;
    let r2 = r * r;
    let e = 1.0
        + r
        + r2 * 0.5
        + r2 * r * (1.0 / 6.0)
        + r2 * r2 * (1.0 / 24.0)
        + r2 * r2 * r * (1.0 / 120.0)
        + r2 * r2 * r2 * (1.0 / 720.0);
    let ki = k as i32;
    let scale = f32::from_bits(((ki + 127) as u32) << 23);
    scale * e
}

/// cos/sin for moderate f64 angles (|x| < 16): quadrant reduction by pi/2
/// + Taylor. Used for RoPE at small positions.
fn sincos(x: f64) -> (f64, f64) {
    const FRAC_PI_2: f64 = 1.5707963267948966;
    let k = fround64(x / FRAC_PI_2) as i64;
    let r = x - k as f64 * FRAC_PI_2;
    let r2 = r * r;
    // cos(r) = 1 - r2/2! + r2^2/4! - r2^3/6! + r2^4/8!
    let c = 1.0 + r2 * (-0.5 + r2 * (1.0 / 24.0 + r2 * (-1.0 / 720.0 + r2 / 40320.0)));
    // sin(r) = r - r^3/3! + r^5/5! - r^7/7!
    let r3 = r2 * r;
    let s = r + r3 * (-1.0 / 6.0 + r2 * (1.0 / 120.0 + r2 * (-1.0 / 5040.0)));
    match k & 3 {
        0 => (c, s),
        1 => (-s, c),
        2 => (-c, -s),
        _ => (s, -c),
    }
}

// ===== geometry =====

pub struct Geometry {
    pub n_layers: u32,
    pub n_embd: usize,
    pub n_heads: usize,
    pub n_kv: usize,
    pub head_dim: usize,
    pub n_ff: usize,
    pub theta: f64,
    pub eps: f32,
}

pub fn geometry() -> Result<Geometry, &'static str> {
    let g = |name: &[u8]| {
        gguf::kv_u32(name).ok_or("gguf: missing geometry kv")
    };
    Ok(Geometry {
        n_layers: g(b"qwen3.block_count")?,
        n_embd: g(b"qwen3.embedding_length")? as usize,
        n_heads: g(b"qwen3.attention.head_count")? as usize,
        n_kv: g(b"qwen3.attention.head_count_kv")? as usize,
        head_dim: g(b"qwen3.attention.key_length")? as usize,
        n_ff: g(b"qwen3.feed_forward_length")? as usize,
        theta: gguf::kv_f32(b"qwen3.rope.freq_base").unwrap_or(1_000_000.0) as f64,
        eps: gguf::kv_f32(b"qwen3.attention.layer_norm_rms_epsilon").unwrap_or(1e-6),
    })
}

// ===== scratch (BSP-only; pool cores are parked during forward passes) =====

const W_CHUNK_LEN: usize = 256 * 1024;
static mut W_CHUNK: [u8; W_CHUNK_LEN] = [0; W_CHUNK_LEN];

/// Per-token activations, sized for Qwen3-class small models.
const MAX_DIM: usize = 3072;
pub struct Activations {
    pub x: [f32; MAX_DIM],   // embedding / residual stream (n_embd)
    pub n1: [f32; MAX_DIM],  // normed input
    pub q: [f32; MAX_DIM],   // n_heads * head_dim
    pub k: [f32; MAX_DIM],   // n_kv * head_dim
    pub v: [f32; MAX_DIM],
    pub attn: [f32; MAX_DIM],
    pub mid: [f32; MAX_DIM],
    pub gate: [f32; MAX_DIM],
    pub up: [f32; MAX_DIM],
    pub hid: [f32; MAX_DIM],
    /// KV cache: positions 0..N x n_kv x head_dim.
    pub kc: [[f32; MAX_DIM]; 8],
    pub vc: [[f32; MAX_DIM]; 8],
    pub n_pos: usize,
}

/// ~310 KB — lives in .bss, never on a stack.
static mut ACT: Activations = Activations {
    x: [0.0; MAX_DIM],
    n1: [0.0; MAX_DIM],
    q: [0.0; MAX_DIM],
    k: [0.0; MAX_DIM],
    v: [0.0; MAX_DIM],
    attn: [0.0; MAX_DIM],
    mid: [0.0; MAX_DIM],
    gate: [0.0; MAX_DIM],
    up: [0.0; MAX_DIM],
    hid: [0.0; MAX_DIM],
    kc: [[0.0; MAX_DIM]; 8],
    vc: [[0.0; MAX_DIM]; 8],
    n_pos: 0,
};

/// Exclusive engine scratch.
/// Soundness: BSP-only — the pool cores are parked during a forward pass,
/// and the serve loop is single-threaded by design.
pub fn activations_mut() -> &'static mut Activations {
    unsafe { &mut *(&raw mut ACT) }
}

// ===== streaming matvec =====

/// Dequantize one q8_0 row (embedding row, n elements) into `out`.
pub fn dequant_q8_0_row(
    vol: &FatVolume,
    file: &File,
    abs: u64,
    n: usize,
    out: &mut [f32],
) -> Result<(), &'static str> {
    let row_bytes = n / 32 * 34;
    if row_bytes > W_CHUNK_LEN {
        return Err("dequant: row too large");
    }
    // Soundness: W_CHUNK is engine scratch on the BSP; the device DMAs
    // into it while the forward pass is the only runner.
    let chunk = unsafe {
        core::slice::from_raw_parts_mut((&raw mut W_CHUNK) as *mut u8, W_CHUNK_LEN)
    };
    wread(vol, file, abs, row_bytes, &mut chunk[..row_bytes])?;
    for b in 0..n / 32 {
        let s = f16_to_f32(u16::from_le_bytes([chunk[b * 34], chunk[b * 34 + 1]]));
        for j in 0..32 {
            out[b * 32 + j] = s * ((chunk[b * 34 + 2 + j] as i8) as f32);
        }
    }
    Ok(())
}

/// Read an F32 vector (norm weights) from the volume.
pub fn read_f32_vec(
    vol: &FatVolume,
    file: &File,
    abs: u64,
    n: usize,
    out: &mut [f32],
) -> Result<(), &'static str> {
    let row_bytes = n * 4;
    if row_bytes > W_CHUNK_LEN {
        return Err("f32vec: too large");
    }
    let chunk = unsafe {
        core::slice::from_raw_parts_mut((&raw mut W_CHUNK) as *mut u8, W_CHUNK_LEN)
    };
    wread(vol, file, abs, row_bytes, &mut chunk[..row_bytes])?;
    for i in 0..n {
        out[i] = f32::from_le_bytes([
            chunk[i * 4],
            chunk[i * 4 + 1],
            chunk[i * 4 + 2],
            chunk[i * 4 + 3],
        ]);
    }
    Ok(())
}

/// y[n_out] = W @ x, W q8_0 [n_out rows x n_in] read from the volume in
/// row chunks. GGUF q8_0 block: f16 scale + 32 int8 quants (34 B).

pub fn matvec_q8_0(
    vol: &FatVolume,
    file: &File,
    abs: u64,
    n_in: usize,
    n_out: usize,
    x: &[f32],
    y: &mut [f32],
) -> Result<(), &'static str> {
    // A76 fast path: quantized activations + SDOT int8 row dots, from the
    // RAM weight cache when installed. Runtime-gated by cpu::has_dotprod.
    if cpu::has_dotprod() && ram_weights() {
        return matvec_udot(vol, file, abs, n_in, n_out, x, y);
    }
    // RAM cache live: rows split across the pool cores.
    if ram_weights() && n_out % board::CORE_COUNT == 0 && n_out <= MAX_DIM {
        return matvec_q8_0_par(x, y, abs, n_in, n_out);
    }
    let row_bytes = n_in / 32 * 34;
    if row_bytes == 0 || row_bytes > W_CHUNK_LEN {
        return Err("matvec: row too large");
    }
    let chunk_rows = W_CHUNK_LEN / row_bytes;
    // Soundness: W_CHUNK is engine scratch on the BSP; the device DMAs
    // into it while the forward pass is the only runner.
    let chunk = unsafe {
        core::slice::from_raw_parts_mut((&raw mut W_CHUNK) as *mut u8, W_CHUNK_LEN)
    };
    let n_blk = n_in / 32;
    let mut done = 0usize;
    while done < n_out {
        let rows = chunk_rows.min(n_out - done);
        wread(
            vol,
            file,
            abs + (done as u64) * row_bytes as u64,
            rows * row_bytes,
            &mut chunk[..rows * row_bytes],
        )?;
        for r in 0..rows {
            let rb = &chunk[r * row_bytes..(r + 1) * row_bytes];
            let mut acc = 0f32;
            for b in 0..n_blk {
                let blk = &rb[b * 34..b * 34 + 34];
                let s = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for j in 0..32 {
                    acc += x[b * 32 + j] * s * ((blk[2 + j] as i8) as f32);
                }
            }
            y[done + r] = acc;
        }
        done += rows;
    }
    Ok(())
}

/// RMSNorm: x/sqrt(mean(x^2)+eps) * w, w F32 read from the volume.
pub fn rmsnorm_with_weight(
    vol: &FatVolume,
    file: &File,
    w_abs: u64,
    x: &[f32],
    out: &mut [f32],
    eps: f32,
) -> Result<(), &'static str> {
    let n = x.len();
    // Soundness: scratch as above.
    let chunk = unsafe {
        core::slice::from_raw_parts_mut((&raw mut W_CHUNK) as *mut u8, W_CHUNK_LEN)
    };
    if n * 4 > W_CHUNK_LEN {
        return Err("rmsnorm: weight too large");
    }
    vol.read_at(file, w_abs, &mut chunk[..n * 4])?;
    let mut sum = 0f32;
    for i in 0..n {
        sum += x[i] * x[i];
    }
    let inv = 1.0 / fsqrt32(sum / n as f32 + eps);
    for i in 0..n {
        let w = f32::from_le_bytes([chunk[i * 4], chunk[i * 4 + 1], chunk[i * 4 + 2], chunk[i * 4 + 3]]);
        out[i] = x[i] * inv * w;
    }
    Ok(())
}

/// In-place per-head RMSNorm (Qwen3 q_norm/k_norm), then RoPE.
pub fn head_norm_rope(
    h: &mut [f32],
    n_heads: usize,
    head_dim: usize,
    w: &[f32],
    eps: f32,
    pos: usize,
    theta: f64,
) {
    let half = head_dim / 2;
    // theta^(-i/half) recurrence: inv_step = theta^(-1/half) — `half` is a
    // power of two, so the root is `log2(half)` square roots.
    let inv_step = theta_inv_step(theta, half as f64);
    for head in 0..n_heads {
        let base = head * head_dim;
        let hslice = &mut h[base..base + head_dim];
        let mut sum = 0f32;
        for v in hslice.iter() {
            sum += v * v;
        }
        let inv = 1.0 / fsqrt32(sum / head_dim as f32 + eps);
        for (i, v) in hslice.iter_mut().enumerate() {
            *v *= inv * w[i];
        }
        let mut freq = 1.0f64;
        for i in 0..half {
            let ang = (pos as f64) * freq;
            let (c, s) = sincos(ang);
            let x1 = hslice[i] as f64;
            let x2 = hslice[i + half] as f64;
            hslice[i] = (x1 * c - x2 * s) as f32;
            hslice[i + half] = (x2 * c + x1 * s) as f32;
            freq *= inv_step;
        }
    }
}

/// theta^(-1/n): n must be a power of two; log2(n) square roots.
fn theta_inv_step(theta: f64, n: f64) -> f64 {
    let mut v = theta;
    let mut k = n;
    while k > 1.0 {
        v = fsqrt64(v);
        k *= 0.5;
    }
    1.0 / v
}

/// Dot + softmax attention for one token over the cached positions.
pub fn attend(
    act: &mut Activations,
    t: usize,
    geo: &Geometry,
    probs_first_head: &mut [f32],
) {
    let hd = geo.head_dim;
    let q_per_kv = geo.n_heads / geo.n_kv;
    let scale = 1.0 / fsqrt32(hd as f32);
    for h in 0..geo.n_heads {
        let kvh = h / q_per_kv;
        // scores over positions 0..=t
        let mut max = f32::MIN;
        let mut scores = [0f32; 8];
        for p in 0..=t {
            let mut dot = 0f32;
            for i in 0..hd {
                dot += act.q[h * hd + i] * act.kc[p][kvh * hd + i];
            }
            let s = dot * scale;
            scores[p] = s;
            if s > max {
                max = s;
            }
        }
        let mut sum = 0f32;
        for p in 0..=t {
            scores[p] = expf(scores[p] - max);
            sum += scores[p];
        }
        let inv = 1.0 / sum;
        if h == 0 {
            for p in 0..=t {
                probs_first_head[p] = scores[p] * inv;
            }
        }
        for i in 0..hd {
            let mut acc = 0f32;
            for p in 0..=t {
                acc += scores[p] * inv * act.vc[p][kvh * hd + i];
            }
            act.attn[h * hd + i] = acc;
        }
    }
}

/// SiLU in place: v * sigmoid(v).
pub fn silu(v: &mut [f32]) {
    for e in v.iter_mut() {
        let s = expf(-*e);
        *e = *e / (1.0 + s);
    }
}

// ===== tokenizer: Qwen3 byte-level BPE from the GGUF vocab =====
//
// The vocab/merges string arrays live inside the metadata buffer; queries
// walk them sequentially (the prompt is tiny, so a handful of linear
// passes over ~150k strings is cheap next to the matmuls).

pub const MAX_TOKENS: usize = 16;
const MAX_SYMS: usize = 64;
const MAX_SYM_BYTES: usize = 256;
const MAX_PIECES: usize = 16;

struct StrIter<'a> {
    data: &'a [u8],
    pos: usize,
    left: u32,
}

impl<'a> StrIter<'a> {
    fn new(data: &'a [u8], off: usize, count: u32) -> Self {
        Self { data, pos: off, left: count }
    }
    fn next(&mut self) -> Option<&'a [u8]> {
        if self.left == 0 {
            return None;
        }
        let hdr = self.data.get(self.pos..self.pos + 8)?;
        let len = u64::from_le_bytes([hdr[0], hdr[1], hdr[2], hdr[3], hdr[4], hdr[5], hdr[6], hdr[7]])
            as usize;
        self.pos += 8;
        let s = self.data.get(self.pos..self.pos + len)?;
        self.pos += len;
        self.left -= 1;
        Some(s)
    }
}

/// GPT-2 byte->unicode codepoint mapping (vocab strings are its UTF-8).
fn byte_unicode_cp(b: u8) -> u32 {
    let b = b as u32;
    if (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b) {
        b
    } else if b < 33 {
        256 + b
    } else if (127..=160).contains(&b) {
        256 + 33 + (b - 127)
    } else {
        256 + 67 // byte 173
    }
}

/// Byte-unicode encode one prompt byte into `out` (1-2 bytes); returns len.
fn byte_encode(b: u8, out: &mut [u8; 2]) -> usize {
    let cp = byte_unicode_cp(b);
    if cp < 0x80 {
        out[0] = cp as u8;
        1
    } else {
        out[0] = 0xC0 | (cp >> 6) as u8;
        out[1] = 0x80 | (cp & 0x3F) as u8;
        2
    }
}

fn is_space(b: u8) -> bool {
    b == b' ' || b == b'\n' || b == b'\t' || b == b'\r'
}

/// GPT-2-style pre-tokenization, ASCII subset: `' contractions` are not
/// special-cased (they tokenize through the "other" class). Pieces are
/// (start, len) ranges over the prompt.
fn pretokenize(prompt: &[u8], pieces: &mut [(usize, usize)]) -> Result<usize, &'static str> {
    let len = prompt.len();
    let mut i = 0usize;
    let mut n = 0usize;
    while i < len {
        let mut start = i;
        if is_space(prompt[i]) {
            let ws = i;
            while i < len && is_space(prompt[i]) {
                i += 1;
            }
            if i == len {
                pieces[n] = (ws, i - ws);
                n += 1;
                break;
            }
            // Trailing run (minus its last space) is its own piece; the
            // last space leads the next piece.
            if i - ws > 1 {
                pieces[n] = (ws, i - ws - 1);
                n += 1;
            }
            start = i - 1;
        }
        let b = prompt[i];
        if b.is_ascii_alphabetic() {
            while i < len && prompt[i].is_ascii_alphabetic() {
                i += 1;
            }
        } else if b.is_ascii_digit() {
            while i < len && prompt[i].is_ascii_digit() {
                i += 1;
            }
        } else if !is_space(b) {
            while i < len && !prompt[i].is_ascii_alphanumeric() && !is_space(prompt[i]) {
                i += 1;
            }
        }
        pieces[n] = (start, i - start);
        n += 1;
        if n >= MAX_PIECES {
            return Err("tokenize: too many pieces");
        }
    }
    Ok(n)
}

/// Tokenize `prompt` with the GGUF's BPE; returns the token count.
pub fn tokenize(
    data: &[u8],
    prompt: &[u8],
    ids: &mut [u32; MAX_TOKENS],
) -> Result<usize, &'static str> {
    let (voff, vcount) = gguf::string_array(b"tokenizer.ggml.tokens").ok_or("no vocab array")?;
    let (moff, mcount) = gguf::string_array(b"tokenizer.ggml.merges").ok_or("no merges array")?;

    let mut pieces = [(0usize, 0usize); MAX_PIECES];
    let np = pretokenize(prompt, &mut pieces)?;

    let mut sym = [0u8; MAX_SYM_BYTES];
    let mut starts = [0usize; MAX_SYMS];
    let mut lens = [0usize; MAX_SYMS];
    let mut n_ids = 0usize;

    for &(ps, pl) in &pieces[..np] {
        // Byte-unicode encode the piece into individual symbols.
        let mut sym_len = 0usize;
        let mut ns = 0usize;
        for &b in &prompt[ps..ps + pl] {
            let mut e = [0u8; 2];
            let n = byte_encode(b, &mut e);
            if sym_len + n > MAX_SYM_BYTES || ns >= MAX_SYMS {
                return Err("tokenize: piece too long");
            }
            sym[sym_len..sym_len + n].copy_from_slice(&e[..n]);
            starts[ns] = sym_len;
            lens[ns] = n;
            sym_len += n;
            ns += 1;
        }

        // Greedy lowest-rank merging: one pass over the merges array finds
        // the best adjacent pair for this round.
        loop {
            if ns < 2 {
                break;
            }
            let mut best_rank = u32::MAX;
            let mut best_i = usize::MAX;
            let mut it = StrIter::new(data, moff, mcount);
            let mut rank = 0u32;
            while let Some(m) = it.next() {
                let mut i = 0usize;
                while i + 1 < ns {
                    let (sa, la) = (starts[i], lens[i]);
                    let (sb, lb) = (starts[i + 1], lens[i + 1]);
                    if m.len() == la + 1 + lb
                        && &m[..la] == &sym[sa..sa + la]
                        && m[la] == b' '
                        && &m[la + 1..] == &sym[sb..sb + lb]
                    {
                        if rank < best_rank {
                            best_rank = rank;
                            best_i = i;
                        }
                        break;
                    }
                    i += 1;
                }
                rank += 1;
            }
            if best_i == usize::MAX {
                break;
            }
            // Merge symbols best_i and best_i+1 (bytes are adjacent).
            lens[best_i] += lens[best_i + 1];
            for j in best_i + 1..ns - 1 {
                starts[j] = starts[j + 1];
                lens[j] = lens[j + 1];
            }
            ns -= 1;
        }

        // Final symbols -> ids by vocab lookup.
        for s in 0..ns {
            let a = starts[s];
            let l = lens[s];
            let mut it = StrIter::new(data, voff, vcount);
            let mut found = None;
            let mut id = 0u32;
            while let Some(v) = it.next() {
                if v.len() == l && v == &sym[a..a + l] {
                    found = Some(id);
                    break;
                }
                id += 1;
            }
            let Some(id) = found else {
                return Err("tokenize: symbol not in vocab");
            };
            if n_ids >= MAX_TOKENS {
                return Err("tokenize: too many tokens");
            }
            ids[n_ids] = id;
            n_ids += 1;
        }
    }
    Ok(n_ids)
}

// ===== full-model decode (Phase 8b) =====

pub const MAX_POS: usize = 8; // cached positions for the gate
const MAX_KV_DIM: usize = 1024; // n_kv * head_dim (8 * 128)
const MAX_LAYERS: usize = 32;

/// Per-layer KV cache: row = layer * MAX_POS + pos.
/// Soundness: BSP-only scratch, as the rest of the engine buffers.
static mut KCACHE: [[f32; MAX_KV_DIM]; MAX_LAYERS * MAX_POS] =
    [[0.0; MAX_KV_DIM]; MAX_LAYERS * MAX_POS];
static mut VCACHE: [[f32; MAX_KV_DIM]; MAX_LAYERS * MAX_POS] =
    [[0.0; MAX_KV_DIM]; MAX_LAYERS * MAX_POS];

/// Write "blk.<layer><suffix>" into `buf`, return the name slice.
fn blk_name<'a>(buf: &'a mut [u8], layer: usize, suffix: &[u8]) -> &'a [u8] {
    const P: &[u8] = b"blk.";
    buf[..4].copy_from_slice(P);
    let mut i = 4;
    let mut v = layer;
    if v == 0 {
        buf[i] = b'0';
        i += 1;
    } else {
        let mut d = [0u8; 8];
        let mut n = 0usize;
        while v > 0 {
            d[n] = b'0' + (v % 10) as u8;
            n += 1;
            v /= 10;
        }
        while n > 0 {
            n -= 1;
            buf[i] = d[n];
            i += 1;
        }
    }
    buf[i..i + suffix.len()].copy_from_slice(suffix);
    i += suffix.len();
    &buf[..i]
}

/// One transformer layer over the residual stream `act.x` at `pos`.
/// K/V for this (layer, pos) land in the static cache.
pub fn layer_forward(
    vol: &FatVolume,
    file: &File,
    ds: u64,
    layer: usize,
    geo: &Geometry,
    pos: usize,
    act: &mut Activations,
) -> Result<(), &'static str> {
    let n_embd = geo.n_embd;
    let hd = geo.head_dim;
    let kv_dim = geo.n_kv * hd;
    if kv_dim > MAX_KV_DIM || layer >= MAX_LAYERS || pos >= MAX_POS {
        return Err("layer: geometry exceeds cache");
    }

    let mut nm = [0u8; 48];
    let norm_abs = ds
        + gguf::find_tensor(blk_name(&mut nm, layer, b".attn_norm.weight"))
            .ok_or("layer: missing attn_norm")?
            .3;
    rmsnorm_with_weight(vol, file, norm_abs, &act.x[..n_embd], &mut act.n1[..n_embd], geo.eps)?;

    for (suffix, out) in [
        (&b".attn_q.weight"[..], &mut act.q[..]),
        (&b".attn_k.weight"[..], &mut act.k[..]),
        (&b".attn_v.weight"[..], &mut act.v[..]),
    ] {
        let nm2 = blk_name(&mut nm, layer, suffix);
        let Some((_, dims, _, off)) = gguf::find_tensor(nm2) else {
            return Err("layer: missing projection");
        };
        matvec_q8_0(
            vol,
            file,
            ds + off,
            dims[0] as usize,
            dims[1] as usize,
            &act.n1[..n_embd],
            out,
        )?;
    }

    let qw_abs = ds
        + gguf::find_tensor(blk_name(&mut nm, layer, b".attn_q_norm.weight"))
            .ok_or("layer: missing q_norm")?
            .3;
    let kw_abs = ds
        + gguf::find_tensor(blk_name(&mut nm, layer, b".attn_k_norm.weight"))
            .ok_or("layer: missing k_norm")?
            .3;
    let mut qw = [0f32; MAX_KV_DIM];
    let mut kw = [0f32; MAX_KV_DIM];
    read_f32_vec(vol, file, qw_abs, hd, &mut qw[..hd])?;
    read_f32_vec(vol, file, kw_abs, hd, &mut kw[..hd])?;
    head_norm_rope(
        &mut act.q[..geo.n_heads * hd],
        geo.n_heads,
        hd,
        &qw[..hd],
        geo.eps,
        pos,
        geo.theta,
    );
    head_norm_rope(
        &mut act.k[..kv_dim],
        geo.n_kv,
        hd,
        &kw[..hd],
        geo.eps,
        pos,
        geo.theta,
    );

    let row = layer * MAX_POS + pos;
    // Soundness: static KV cache, exclusive BSP access during the pass.
    unsafe {
        KCACHE[row][..kv_dim].copy_from_slice(&act.k[..kv_dim]);
        VCACHE[row][..kv_dim].copy_from_slice(&act.v[..kv_dim]);
    }

    // Causal attention over cached positions 0..=pos.
    let scale = 1.0 / fsqrt32(hd as f32);
    let q_per_kv = geo.n_heads / geo.n_kv;
    for h in 0..geo.n_heads {
        let kvh = h / q_per_kv;
        let mut max = f32::MIN;
        let mut scores = [0f32; MAX_POS];
        for p in 0..=pos {
            let krow = layer * MAX_POS + p;
            let mut dot = 0f32;
            let krow_slice = unsafe {
                core::slice::from_raw_parts(
                    (&raw const KCACHE[krow]) as *const f32,
                    MAX_KV_DIM,
                )
            };
            for i in 0..hd {
                dot += act.q[h * hd + i] * krow_slice[kvh * hd + i];
            }
            let s = dot * scale;
            scores[p] = s;
            if s > max {
                max = s;
            }
        }
        let mut sum = 0f32;
        for p in 0..=pos {
            scores[p] = expf(scores[p] - max);
            sum += scores[p];
        }
        let inv = 1.0 / sum;
        for i in 0..hd {
            let mut acc = 0f32;
            for p in 0..=pos {
                let vrow = layer * MAX_POS + p;
                let vrow_slice = unsafe {
                    core::slice::from_raw_parts(
                        (&raw const VCACHE[vrow]) as *const f32,
                        MAX_KV_DIM,
                    )
                };
                acc += scores[p] * inv * vrow_slice[kvh * hd + i];
            }
            act.attn[h * hd + i] = acc;
        }
    }

    let (o_abs, o_dims, _) = {
        let nm2 = blk_name(&mut nm, layer, b".attn_output.weight");
        let Some((_, dims, _, off)) = gguf::find_tensor(nm2) else {
            return Err("layer: missing attn_output");
        };
        (ds + off, dims, ())
    };
    matvec_q8_0(
        vol,
        file,
        o_abs,
        o_dims[0] as usize,
        o_dims[1] as usize,
        &act.attn[..o_dims[0] as usize],
        &mut act.mid[..o_dims[1] as usize],
    )?;
    for i in 0..n_embd {
        act.mid[i] += act.x[i];
    }

    let fn_abs = ds
        + gguf::find_tensor(blk_name(&mut nm, layer, b".ffn_norm.weight"))
            .ok_or("layer: missing ffn_norm")?
            .3;
    rmsnorm_with_weight(vol, file, fn_abs, &act.mid[..n_embd], &mut act.n1[..n_embd], geo.eps)?;

    let (g_abs, g_dims) = {
        let nm2 = blk_name(&mut nm, layer, b".ffn_gate.weight");
        let Some((_, dims, _, off)) = gguf::find_tensor(nm2) else {
            return Err("layer: missing ffn_gate");
        };
        (ds + off, dims)
    };
    let u_abs = ds
        + gguf::find_tensor(blk_name(&mut nm, layer, b".ffn_up.weight"))
            .ok_or("layer: missing ffn_up")?
            .3;
    let (d_abs, d_dims) = {
        let nm2 = blk_name(&mut nm, layer, b".ffn_down.weight");
        let Some((_, dims, _, off)) = gguf::find_tensor(nm2) else {
            return Err("layer: missing ffn_down");
        };
        (ds + off, dims)
    };
    let n_ff = g_dims[1] as usize;
    matvec_q8_0(
        vol,
        file,
        g_abs,
        g_dims[0] as usize,
        n_ff,
        &act.n1[..n_embd],
        &mut act.gate[..n_ff],
    )?;
    matvec_q8_0(
        vol,
        file,
        u_abs,
        g_dims[0] as usize,
        n_ff,
        &act.n1[..n_embd],
        &mut act.up[..n_ff],
    )?;
    silu(&mut act.gate[..n_ff]);
    for i in 0..n_ff {
        act.gate[i] *= act.up[i];
    }
    matvec_q8_0(
        vol,
        file,
        d_abs,
        d_dims[0] as usize,
        d_dims[1] as usize,
        &act.gate[..d_dims[0] as usize],
        &mut act.hid[..d_dims[1] as usize],
    )?;
    // Residual: x_out = mid (x + attn) + down. Assignment, not +=: act.x
    // still holds the layer input here, and mid already includes it.
    for i in 0..n_embd {
        act.x[i] = act.mid[i] + act.hid[i];
    }
    Ok(())
}

/// Greedy argmax over a q8_0 matrix's rows (tied lm_head): returns
/// (row index, logit). Ties resolve to the first row.
pub fn argmax_q8_0(
    vol: &FatVolume,
    file: &File,
    abs: u64,
    n_in: usize,
    n_rows: u64,
    x: &[f32],
) -> Result<(u32, f32), &'static str> {
    let row_bytes = n_in / 32 * 34;
    if row_bytes == 0 || row_bytes > W_CHUNK_LEN {
        return Err("argmax: row too large");
    }
    let chunk_rows = W_CHUNK_LEN / row_bytes;
    let n_blk = n_in / 32;
    let chunk = unsafe {
        core::slice::from_raw_parts_mut((&raw mut W_CHUNK) as *mut u8, W_CHUNK_LEN)
    };
    let mut best_i = 0u32;
    let mut best_v = f32::MIN;
    let mut done = 0u64;
    while done < n_rows {
        let rows = (chunk_rows as u64).min(n_rows - done) as usize;
        vol.read_at(
            file,
            abs + done * row_bytes as u64,
            &mut chunk[..rows * row_bytes],
        )?;
        for r in 0..rows {
            let rb = &chunk[r * row_bytes..(r + 1) * row_bytes];
            let mut acc = 0f32;
            for b in 0..n_blk {
                let blk = &rb[b * 34..b * 34 + 34];
                let s = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
                for j in 0..32 {
                    acc += x[b * 32 + j] * s * ((blk[2 + j] as i8) as f32);
                }
            }
            if acc > best_v {
                best_v = acc;
                best_i = done as u32 + r as u32;
            }
        }
        done += rows as u64;
    }
    Ok((best_i, best_v))
}

// ===== detokenization (Phase 9) =====

/// Reverse map a byte-unicode codepoint back to its byte.
fn cp_to_byte(cp: u32) -> Option<u8> {
    match cp {
        0..=126 => Some(cp as u8),
        161..=172 | 174..=255 => Some(cp as u8),
        256..=288 => Some((cp - 256) as u8),
        289..=322 => Some((cp - 162) as u8),
        323 => Some(173),
        _ => None,
    }
}

/// Decode vocab entry uid=1000(mark) gid=1000(mark) groups=1000(mark),27(sudo) back to raw bytes (reverse of byte-unicode).
/// Walks the vocab array sequentially — id lookups are rare (one per
/// generated token) so the linear scan is acceptable.
pub fn detok(data: &[u8], id: u32, out: &mut [u8]) -> Option<usize> {
    let (voff, vcount) = gguf::string_array(b"tokenizer.ggml.tokens")?;
    if id >= vcount {
        return None;
    }
    let mut it = StrIter::new(data, voff, vcount);
    let mut i = 0u32;
    let s = loop {
        let s = it.next()?;
        if i == id {
            break s;
        }
        i += 1;
    };
    let mut n = 0usize;
    let mut j = 0usize;
    while j < s.len() {
        let b = s[j];
        let (cp, used) = if b < 0x80 {
            (b as u32, 1usize)
        } else if b & 0xE0 == 0xC0 && j + 1 < s.len() {
            (
                (((b & 0x1F) as u32) << 6) | ((s[j + 1] & 0x3F) as u32),
                2usize,
            )
        } else {
            return None;
        };
        let byte = cp_to_byte(cp)?;
        if n >= out.len() {
            return None;
        }
        out[n] = byte;
        n += 1;
        j += used;
    }
    Some(n)
}

// ===== RAM weight cache + parallel matvec (perf phase) =====
//
// Weights stream from the volume once into a fixed RAM window at LOAD;
// every matvec afterwards reads from RAM, and the row range is split
// across the pool cores. Requires board::WEIGHT_RAM_SIZE > 0.

static mut RAM_WEIGHTS: bool = false;

pub fn set_ram_weights(on: bool) {
    unsafe {
        RAM_WEIGHTS = on;
    }
}

pub fn ram_weights() -> bool {
    unsafe { RAM_WEIGHTS }
}

/// Weight read: from the RAM cache when installed, else from the volume.
/// Soundness: the cache is a fixed window sized >= the model file at
/// LOAD; reads here are within [0, file_size).
pub fn wread(
    vol: &FatVolume,
    file: &File,
    abs: u64,
    len: usize,
    dst: &mut [u8],
) -> Result<(), &'static str> {
    if unsafe { RAM_WEIGHTS } && board::WEIGHT_RAM_SIZE > 0 {
        if abs as usize + len > board::WEIGHT_RAM_SIZE {
            return Err("wread: outside cache");
        }
        let src = unsafe {
            core::slice::from_raw_parts(
                (board::WEIGHT_RAM_BASE + abs as usize) as *const u8,
                len,
            )
        };
        dst[..len].copy_from_slice(src);
        Ok(())
    } else {
        vol.read_at(file, abs, dst).map(|_| ())
    }
}

/// Parallel matvec descriptor: one contiguous q8_0 matrix in the RAM
/// cache times one shared x vector, rows split across the pool cores.
static mut MV_ABS: u64 = 0;
static mut MV_N_IN: usize = 0;
static mut MV_ROWS_PER: usize = 0;
static mut MV_ROWS_LAST: usize = 0;
static mut MV_X: *const f32 = core::ptr::null();
static mut MV_Y: *const f32 = core::ptr::null();
static mut MV_BARRIER: u32 = 0;

fn mv_job(core_id: usize, _arg: u64) {
    let n_out = unsafe { MV_ROWS_PER };
    let abs = unsafe { MV_ABS } as usize;
    let n_in = unsafe { MV_N_IN };
    let row_bytes = n_in / 32 * 34;
    let n_blk = n_in / 32;
    // Soundness: disjoint row ranges per core; x/y point at engine
    // statics; the cache-maintenance in matvec_par makes BSP writes
    // visible to MMU-off cores and flushes core writes back for the BSP.
    for r in 0..n_out {
        let row = core_id * n_out + r;
        let base = board::WEIGHT_RAM_BASE + abs + row * row_bytes;
        let mut acc = 0f32;
        for b in 0..n_blk {
            let blk = unsafe {
                core::slice::from_raw_parts((base + b * 34) as *const u8, 34)
            };
            let s = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
            let xoff = b * 32;
            // Soundness: in-bounds block of the cached row/activation.
            unsafe {
                for j in 0..32 {
                    acc += *MV_X.add(xoff + j) * s * ((blk[2 + j] as i8) as f32);
                }
            }
        }
        unsafe {
            let yp = MV_Y as *mut f32;
            *yp.add(row) = acc;
        }
    }
    unsafe {
        MV_BARRIER += 1;
    }
}

/// RAM-cache matvec across all pool cores. Requires set_ram_weights(true).
pub fn matvec_q8_0_par(x: &[f32], y: &mut [f32], abs: u64, n_in: usize, n_out: usize) -> Result<(), &'static str> {
    if !ram_weights() || board::WEIGHT_RAM_SIZE == 0 {
        return Err("par matvec: no ram cache");
    }
    if n_out % board::CORE_COUNT != 0 {
        return Err("par matvec: rows not divisible by cores");
    }
    unsafe {
        MV_ABS = abs;
        MV_N_IN = n_in;
        MV_ROWS_PER = n_out / board::CORE_COUNT;
        MV_ROWS_LAST = 0;
        MV_X = x.as_ptr();
        MV_Y = y.as_ptr() as *const f32;
        MV_BARRIER = 0;
    }
    // BSP writes x (cached): make it visible to the MMU-off cores.
    cache::clean_range(x.as_ptr() as usize, n_in * 4);
    // The job descriptor statics were written by the BSP through its cache
    // moments ago — the MMU-off cores read them from RAM, so clean them.
    cache::clean_range(
        (&raw const MV_ABS) as usize,
        (&raw const MV_BARRIER) as usize - (&raw const MV_ABS) as usize + 8,
    );
    pool::run_on_all(mv_job, 0, board::CORE_COUNT);
    // Results: the BSP wrote its own rows through the cache (dirty lines),
    // the MMU-off cores wrote theirs straight to RAM. Clean the BSP's rows
    // out to RAM, then drop ALL cached copies so the BSP reads every
    // core's result from memory.
    cache::clean_range(y.as_ptr() as usize, n_out * 4);
    cache::invalidate_range(y.as_ptr() as usize, n_out * 4);
    Ok(())
}

/// Round-half-away f32 -> i32 without libm.
fn round_i32(v: f32) -> i32 {
    let a = if v >= 0.0 { v + 0.5 } else { v - 0.5 };
    a as i32
}

/// SDOT: signed int8 dot of one 32-byte block (two 16B halves) into 4
/// i32 lanes, pairwise-reduced to a scalar.
/// Soundness: pure NEON math on in-bounds block pointers; dotprod is
/// per-function and runtime-gated by cpu::has_dotprod.
#[target_feature(enable = "dotprod")]
unsafe fn sdot_block(qp: *const u8, xp: *const u8) -> i32 {
    let mut lanes = [0i32; 4];
    core::arch::asm!(
        "movi v17.4s, #0",
        "ld1 {{v18.16b}}, [{qp}], #16",
        "ld1 {{v19.16b}}, [{xp}], #16",
        "sdot v17.4s, v18.16b, v19.16b",
        "ld1 {{v18.16b}}, [{qp}], #16",
        "ld1 {{v19.16b}}, [{xp}], #16",
        "sdot v17.4s, v18.16b, v19.16b",
        "addp v17.4s, v17.4s, v17.4s",
        "addp v17.4s, v17.4s, v17.4s",
        "st1 {{v17.4s}}, [{outp}]",
        qp = inout(reg) qp => _,
        xp = inout(reg) xp => _,
        outp = in(reg) lanes.as_mut_ptr(),
        lateout("v17") _, lateout("v18") _, lateout("v19") _,
        options(nostack),
    );
    lanes[0]
}

/// Serving-path fast matvec: quantizes x once (one symmetric scale), then
/// SDOT row dots from the weight source, applying per-block q8_0 scales.
/// Requires ram_weights + has_dotprod (checked by the caller).


// ===== Pool job statics for parallel SDOT matvec =====
// Soundness: written by the BSP before run_on_all and cleaned; read by
// all cores via the identity map (MMU-off APs access physical directly).

static mut PJ_ROW_BASE: u64 = 0;
static mut PJ_ROW_STRIDE: usize = 0;
static mut PJ_N_BLK: usize = 0;
static mut PJ_N_OUT: usize = 0;
static mut PJ_SX: f32 = 0.0;
static mut PJ_DOTPROD: bool = false;
static mut PJ_Y_PTR: usize = 0;
static mut PJ_XQ_PTR: usize = 0;

/// Pool job: compute rows [core_id*rp, (core_id+1)*rp) of the output.
/// Reads weight rows from the RAM cache window, does SDOT per-block dots
/// with per-block f16 scales, writes results to y.
/// Soundness: pure NEON math on in-bounds rows of the RAM cache; y writes
/// are disjoint per core; XQ is read-only shared (cleaned by the BSP).
fn psdot_pool_job(core_id: usize, _arg: u64) {
    // Soundness: the descriptor statics and shared buffers are written by
    // the BSP before run_on_all and cleaned; the y writes are disjoint
    // per core. All static accesses are wrapped in unsafe blocks below.
    unsafe {
    let n_cores = board::CORE_COUNT;
    let rows_per = PJ_N_OUT / n_cores;
    let start = core_id * rows_per;
    let n_blk = PJ_N_BLK;
    let sx = PJ_SX;
    let dotprod = PJ_DOTPROD;
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
            if dotprod {
                // Soundness: sdot_block is pure NEON math on in-bounds
                // blocks of the cached row and the quantized activations.
                unsafe {
                    dot = sdot_block(
                        (row_base as usize + boff + 2) as *const u8,
                        (xq as usize + b * 32) as *const u8,
                    ) as f32;
                }
            } else {
                for j in 0..32 {
                    let w = core::ptr::read((row_base as usize + boff + 2 + j) as *const u8);
                    let xv = core::ptr::read((xq as usize + b * 32 + j) as *const u8);
                    dot += (w as i8) as f32 * (xv as f32);
                }
            }
            acc_f += dot * sb * sx;
        }
        y.add(r).write(acc_f);
    }
    } // unsafe: end of psdot_pool_job body
}
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

    let mut max = 0f32;
    for v in x.iter() {
        let a = if *v < 0.0 { -*v } else { *v };
        if a > max {
            max = a;
        }
    }
    let sx = if max > 0.0 { max / 127.0 } else { 1.0 };
    // Soundness: XQ is engine scratch on the BSP; activations quantize
    // once and are read by every row dot.
    let xq = unsafe {
        core::slice::from_raw_parts_mut((&raw mut XQ) as *mut u8, MAX_DIM)
    };
    for (i, v) in x.iter().enumerate() {
        let q = round_i32(v / sx).clamp(-127, 127);
        xq[i] = (q as i8) as u8;
    }

    // Pool-parallel: each core computes its share of rows. The quantized
    // activations (XQ) are cleaned so the MMU-off APs can read them; y is
    // invalidated after so the BSP reads the APs' uncached writes.
    let row_bytes = n_in / 32 * 34;
    let n_blk = n_in / 32;
    let dotprod = cpu::has_dotprod();

    // Clean XQ so the MMU-off AP cores see the BSP's quantized writes.
    cache::clean_range(xq.as_ptr() as usize, n_in);

    // Set up the pool job descriptor.
    unsafe {
        PJ_ROW_BASE = board::WEIGHT_RAM_BASE as u64 + abs;
        PJ_ROW_STRIDE = row_bytes;
        PJ_N_BLK = n_blk;
        PJ_N_OUT = n_out;
        PJ_SX = sx;
        PJ_DOTPROD = dotprod;
        PJ_Y_PTR = y.as_mut_ptr() as usize;
        PJ_XQ_PTR = xq.as_ptr() as usize;
    }

    pool::run_on_all(psdot_pool_job, 0, board::CORE_COUNT);

    // The APs wrote y with MMU off (uncached, straight to RAM): drop the
    // BSP's cached copies so it reads the combined results.
    cache::invalidate_range(y.as_mut_ptr() as usize, n_out * 4);

    Ok(())
}
