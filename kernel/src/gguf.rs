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
pub const MAX_TENSORS: usize = 8;
pub const MAX_NAME: usize = 32;

static mut T_NAMES: [[u8; MAX_NAME]; MAX_TENSORS] = [[0; MAX_NAME]; MAX_TENSORS];
static mut T_NAME_LEN: [usize; MAX_TENSORS] = [0; MAX_TENSORS];
static mut T_DIMS: [[u64; 4]; MAX_TENSORS] = [[0; 4]; MAX_TENSORS];
static mut T_TYPE: [u32; MAX_TENSORS] = [0; MAX_TENSORS];
static mut T_OFFSET: [u64; MAX_TENSORS] = [0; MAX_TENSORS];
static mut T_COUNT: usize = 0;

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
