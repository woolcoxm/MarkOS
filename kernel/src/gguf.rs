//! Minimal GGUF v3 parser — header, metadata walk, tensor table.
//!
//! Reimplements only the parsing subset of the GGUF spec needed to locate
//! tensors in a loaded model file. No heap: the caller provides the buffer
//! (small headers fit RAM; GB-scale model DATA is streamed later using the
//! tensor offsets this parser reports).
//!
//! owns: nothing; parses a caller-provided byte slice.
//! invariants: all reads are bounds-checked — malformed input yields Err,
//! never a panic (no-panic policy for steady-state paths).

#![allow(dead_code)] // VT_* constants document the on-disk format

use crate::uart;

const MAGIC_GGUF: u32 = 0x4655_4747; // "GGUF" little-endian

// GGUF metadata value types (format documentation; not all are matched).
const VT_U8: u32 = 0;
const VT_I8: u32 = 1;
const VT_U16: u32 = 2;
const VT_I16: u32 = 3;
const VT_U32: u32 = 4;
const VT_I32: u32 = 5;
const VT_F32: u32 = 6;
const VT_BOOL: u32 = 7;
const VT_STRING: u32 = 8;
const VT_ARRAY: u32 = 9;
const VT_U64: u32 = 10;
const VT_I64: u32 = 11;
const VT_F64: u32 = 12;

fn vt_fixed_size(vt: u32) -> Option<u64> {
    Some(match vt {
        VT_U8 | VT_I8 | VT_BOOL => 1,
        VT_U16 | VT_I16 => 2,
        VT_U32 | VT_I32 | VT_F32 => 4,
        VT_U64 | VT_I64 | VT_F64 => 8,
        _ => return None,
    })
}

/// Bounds-checked byte cursor.
struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn u32(&mut self) -> Result<u32, &'static str> {
        let b = self.data.get(self.pos..self.pos + 4).ok_or("gguf: truncated (u32)")?;
        self.pos += 4;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64(&mut self) -> Result<u64, &'static str> {
        let b = self.data.get(self.pos..self.pos + 8).ok_or("gguf: truncated (u64)")?;
        self.pos += 8;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(b);
        Ok(u64::from_le_bytes(arr))
    }
    fn string(&mut self) -> Result<&'a [u8], &'static str> {
        let len = self.u64()? as usize;
        let b = self.data.get(self.pos..self.pos + len).ok_or("gguf: truncated (string)")?;
        self.pos += len;
        Ok(b)
    }
    fn skip(&mut self, n: usize) -> Result<(), &'static str> {
        self.pos = self.pos.checked_add(n).ok_or("gguf: overflow")?;
        if self.pos > self.data.len() {
            return Err("gguf: truncated (skip)");
        }
        Ok(())
    }
}

fn skip_value(cur: &mut Cursor, vt: u32, depth: u32) -> Result<(), &'static str> {
    if depth > 8 {
        return Err("gguf: array nesting too deep");
    }
    if let Some(sz) = vt_fixed_size(vt) {
        return cur.skip(sz as usize);
    }
    match vt {
        VT_STRING => {
            let len = cur.u64()? as usize;
            cur.skip(len)
        }
        VT_ARRAY => {
            let elem_vt = cur.u32()?;
            let count = cur.u64()?;
            for _ in 0..count {
                skip_value(cur, elem_vt, depth + 1)?;
            }
            Ok(())
        }
        _ => Err("gguf: unknown metadata value type"),
    }
}

/// Write a name/byte-slice, filtering non-printables (no panic).
fn write_name(name: &[u8]) {
    for &b in name {
        if (0x20..0x7F).contains(&b) {
            uart::write_byte(b);
        } else {
            uart::write_byte(b'?');
        }
    }
}

/// Parsed summary: everything later phases need to locate tensor data.
pub struct GgufInfo {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    /// Byte offset of the tensor data section (alignment applied).
    pub data_start: u64,
    /// Alignment reported by the file (general.alignment, default 32).
    pub alignment: u64,
}

/// Collected tensor table (filled by parse_and_dump, no heap).
/// 512 entries covers every small-model GGUF (Qwen3-0.6B ≈ 200-300).
pub const MAX_TENSORS: usize = 512;
pub const MAX_NAME: usize = 48;

static mut T_NAMES: [[u8; MAX_NAME]; MAX_TENSORS] = [[0; MAX_NAME]; MAX_TENSORS];
static mut T_NAME_LEN: [usize; MAX_TENSORS] = [0; MAX_TENSORS];
static mut T_DIMS: [[u64; 4]; MAX_TENSORS] = [[0; 4]; MAX_TENSORS];
static mut T_TYPE: [u32; MAX_TENSORS] = [0; MAX_TENSORS];
static mut T_OFFSET: [u64; MAX_TENSORS] = [0; MAX_TENSORS];
static mut T_COUNT: usize = 0;

/// Snapshot row: one tensor's identity, copied out of the statics.
#[derive(Clone, Copy)]
pub struct TensorEntry {
    pub name: [u8; MAX_NAME],
    pub name_len: usize,
    pub dims: [u64; 4],
    pub ttype: u32,
    pub offset: u64,
}

impl TensorEntry {
    pub fn name(&self) -> &[u8] {
        &self.name[..self.name_len]
    }
}

/// Copy the parsed table into `out`; returns the number of entries.
pub fn snapshot(out: &mut [TensorEntry]) -> usize {
    let n = unsafe { T_COUNT }.min(out.len());
    // Soundness: table is filled once during single-core boot, read-only after.
    for i in 0..n {
        unsafe {
            out[i] = TensorEntry {
                name: T_NAMES[i],
                name_len: T_NAME_LEN[i],
                dims: T_DIMS[i],
                ttype: T_TYPE[i],
                offset: T_OFFSET[i],
            };
        }
    }
    n
}

/// IEEE CRC-32 (matches zlib.crc32), table-free bit loop.
pub fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xEDB8_8320 & mask);
        }
    }
    !crc
}

// ===== metadata capture: scalar kvs + string-array locations =====
//
// The inference engine needs scalar kv values (geometry: head counts,
// eps, rope base) and the byte ranges of the tokenizer's string arrays
// (vocab, merges) inside the caller's metadata buffer. Captured during
// parse_and_dump; read-only afterwards.

const MAX_KV: usize = 48;
const MAX_KV_KEY: usize = 48;
const MAX_STR_ARRAYS: usize = 4;
const MAX_SA_KEY: usize = 32;

#[derive(Clone, Copy)]
struct KvScalar {
    key: [u8; MAX_KV_KEY],
    key_len: usize,
    /// Raw 4 bytes; interpret via `is_float` (u32/i32 share `u`).
    u: u32,
    f: f32,
    is_float: bool,
}

static mut KV_SCALARS: [KvScalar; MAX_KV] = [KvScalar {
    key: [0; MAX_KV_KEY],
    key_len: 0,
    u: 0,
    f: 0.0,
    is_float: false,
}; MAX_KV];
static mut KV_N: usize = 0;

static mut SA_KEYS: [[u8; MAX_SA_KEY]; MAX_STR_ARRAYS] = [[0; MAX_SA_KEY]; MAX_STR_ARRAYS];
static mut SA_KEY_LEN: [usize; MAX_STR_ARRAYS] = [0; MAX_STR_ARRAYS];
static mut SA_OFF: [usize; MAX_STR_ARRAYS] = [0; MAX_STR_ARRAYS];
static mut SA_ELEMS: [u32; MAX_STR_ARRAYS] = [0; MAX_STR_ARRAYS];
static mut SA_N: usize = 0;

fn capture_kv_scalar(key: &[u8], u: u32, f: f32, is_float: bool) {
    unsafe {
        let n = KV_N;
        if n >= MAX_KV {
            return;
        }
        let len = key.len().min(MAX_KV_KEY);
        KV_SCALARS[n].key[..len].copy_from_slice(&key[..len]);
        KV_SCALARS[n].key_len = len;
        KV_SCALARS[n].u = u;
        KV_SCALARS[n].f = f;
        KV_SCALARS[n].is_float = is_float;
        KV_N = n + 1;
    }
}

fn capture_string_array(key: &[u8], off: usize, elems: u32) {
    unsafe {
        let n = SA_N;
        if n >= MAX_STR_ARRAYS {
            return;
        }
        let len = key.len().min(MAX_SA_KEY);
        SA_KEYS[n][..len].copy_from_slice(&key[..len]);
        SA_KEY_LEN[n] = len;
        SA_OFF[n] = off;
        SA_ELEMS[n] = elems;
        SA_N = n + 1;
    }
}

fn keys_eq(a: &[u8], b: &[u8]) -> bool {
    a == b
}

/// Scalar kv value (u32/i32 flavors).
pub fn kv_u32(name: &[u8]) -> Option<u32> {
    unsafe {
        for i in 0..KV_N {
            if keys_eq(&KV_SCALARS[i].key[..KV_SCALARS[i].key_len], name)
                && !KV_SCALARS[i].is_float
            {
                return Some(KV_SCALARS[i].u);
            }
        }
    }
    None
}

/// Scalar kv value (f32 flavors).
pub fn kv_f32(name: &[u8]) -> Option<f32> {
    unsafe {
        for i in 0..KV_N {
            if keys_eq(&KV_SCALARS[i].key[..KV_SCALARS[i].key_len], name) && KV_SCALARS[i].is_float
            {
                return Some(KV_SCALARS[i].f);
            }
        }
    }
    None
}

/// Location of a string-array kv inside the parsed buffer:
/// (byte offset of the first element's length field, element count).
pub fn string_array(name: &[u8]) -> Option<(usize, u32)> {
    unsafe {
        for i in 0..SA_N {
            if keys_eq(&SA_KEYS[i][..SA_KEY_LEN[i]], name) {
                return Some((SA_OFF[i], SA_ELEMS[i]));
            }
        }
    }
    None
}

/// Cursor read accessors used by the capture below.
fn cur_u32_at(data: &[u8], pos: usize) -> Option<u32> {
    let b = data.get(pos..pos + 4)?;
    Some(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Look up a tensor by name (exact byte match).
pub fn find_tensor(name: &[u8]) -> Option<(&'static [u8], &'static [u64; 4], u32, u64)> {
    // Soundness: table is filled once during single-core boot, read-only after.
    unsafe {
        for i in 0..T_COUNT {
            let n = &T_NAMES[i][..T_NAME_LEN[i]];
            if n == name {
                return Some((n, &T_DIMS[i], T_TYPE[i], T_OFFSET[i]));
            }
        }
    }
    None
}

/// Parse (and log) the GGUF header of an in-memory model file.
pub fn parse_and_dump(data: &[u8]) -> Result<GgufInfo, &'static str> {
    let mut cur = Cursor { data, pos: 0 };

    let magic = cur.u32()?;
    if magic != MAGIC_GGUF {
        return Err("GGUF magic mismatch");
    }
    let version = cur.u32()?;
    let tensor_count = cur.u64()?;
    let kv_count = cur.u64()?;

    // Metadata: string and u32 values are printed; everything else is
    // skipped by walking its structure (vocab arrays can be huge).
    let mut alignment: u64 = 32;
    for _ in 0..kv_count {
        let key = cur.string()?;
        let vt = cur.u32()?;

        // Capture engine-relevant metadata before/while consuming.
        if vt == VT_U32 || vt == VT_I32 || vt == VT_F32 {
            let raw = cur_u32_at(data, cur.pos).ok_or("gguf: truncated (kv scalar)")?;
            let f = f32::from_bits(raw);
            let _ = cur.u32();
            capture_kv_scalar(key, raw, f, vt == VT_F32);
            continue;
        }
        if vt == VT_ARRAY && (key == b"tokenizer.ggml.tokens" || key == b"tokenizer.ggml.merges")
        {
            let elem_vt = cur.u32()?;
            let count = cur.u64()?;
            if elem_vt != VT_STRING {
                return Err("gguf: tokenizer array is not strings");
            }
            capture_string_array(key, cur.pos, count as u32);
            // Skip the elements (u64 length + payload each).
            for _ in 0..count {
                let _ = cur.string()?;
            }
            continue;
        }

        uart::write_str("gguf: kv '");
        write_name(key);
        uart::write_str("' ");
        match vt {
            VT_STRING => {
                let val = cur.string()?;
                uart::write_str("= '");
                write_name(val);
                uart::write_str("'\n");
            }
            VT_U32 => {
                let b = data.get(cur.pos..cur.pos + 4).ok_or("gguf: truncated (u32 kv)")?;
                let val = u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                cur.pos += 4;
                uart::locked_write(format_args!("= {val}\n"));
                if key == b"general.alignment" {
                    alignment = val as u64;
                }
            }
            _ => {
                uart::locked_write(format_args!("(type {vt}, value skipped)\n"));
                skip_value(&mut cur, vt, 1)?;
            }
        }
    }

    // Tensor table: name, dims, quant type, offset within the data section.
    for _ in 0..tensor_count {
        let name = cur.string()?;
        let n_dims = cur.u32()?;
        if n_dims == 0 || n_dims > 4 {
            return Err("gguf: tensor dim count out of range");
        }
        let mut dims = [0u64; 4];
        for d in dims.iter_mut().take(n_dims as usize) {
            *d = cur.u64()?;
        }
        let ttype = cur.u32()?;
        let offset = cur.u64()?;

        // Collect into the static table (first MAX_TENSORS entries).
        unsafe {
            let idx = T_COUNT;
            if idx < MAX_TENSORS {
                let nl = name.len().min(MAX_NAME);
                T_NAMES[idx][..nl].copy_from_slice(&name[..nl]);
                T_NAME_LEN[idx] = nl;
                T_DIMS[idx] = dims;
                T_TYPE[idx] = ttype;
                T_OFFSET[idx] = offset;
                T_COUNT = idx + 1;
            }
        }

        uart::write_str("gguf: tensor '");
        write_name(name);
        uart::locked_write(format_args!(
            "' dims=[{}, {}, {}, {}] type={ttype} offset={offset}\n",
            dims[0], dims[1], dims[2], dims[3]
        ));
    }

    // Data section begins at the next multiple of `alignment`.
    let data_start = (cur.pos as u64).div_ceil(alignment) * alignment;

    uart::locked_write(format_args!(
        "gguf: version={version} tensors={tensor_count} kvs={kv_count} data@{data_start:#x} align={alignment}\n"
    ));

    Ok(GgufInfo { version, tensor_count, kv_count, data_start, alignment })
}
