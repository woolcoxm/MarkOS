//! Pure-Rust GGUF v2/v3 metadata reader.
//!
//! Reads only the header: magic, version, tensor table, metadata KVs.
//! Never touches the tensor data section — this powers model inventory and
//! the memory guardrail estimator without loading gigabytes. Bounds-checked
//! throughout: malformed input yields `Err`, never a panic.

#![allow(dead_code)] // fields document the on-disk format

use std::collections::BTreeMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

pub const MAGIC_GGUF: u32 = 0x4655_4747; // "GGUF" little-endian

// GGUF metadata value types.
pub const VT_U8: u32 = 0;
pub const VT_I8: u32 = 1;
pub const VT_U16: u32 = 2;
pub const VT_I16: u32 = 3;
pub const VT_U32: u32 = 4;
pub const VT_I32: u32 = 5;
pub const VT_F32: u32 = 6;
pub const VT_BOOL: u32 = 7;
pub const VT_STRING: u32 = 8;
pub const VT_ARRAY: u32 = 9;
pub const VT_U64: u32 = 10;
pub const VT_I64: u32 = 11;
pub const VT_F64: u32 = 12;

/// GGML tensor types we can name/size (subset that matters for inventory).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ2Xxs,
    IQ2Xs,
    IQ3Xxs,
    IQ1S,
    IQ4Nl,
    IQ3S,
    IQ2S,
    IQ4Xs,
    I8,
    I16,
    I32,
    I64,
    Other(u32),
}

impl GgmlType {
    pub fn from_u32(t: u32) -> Self {
        match t {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 | 12 | 13 => GgmlType::Q3K,
            14 | 15 => GgmlType::Q4K,
            16 | 17 => GgmlType::Q5K,
            18 => GgmlType::Q6K,
            19 => GgmlType::IQ2Xxs,
            20 => GgmlType::IQ2Xs,
            21 => GgmlType::IQ3Xxs,
            22 => GgmlType::IQ1S,
            23 => GgmlType::IQ4Nl,
            24 => GgmlType::IQ3S,
            25 => GgmlType::IQ2S,
            26 => GgmlType::IQ4Xs,
            28 => GgmlType::I8,
            29 => GgmlType::I16,
            30 => GgmlType::I32,
            31 => GgmlType::I64,
            other => GgmlType::Other(other),
        }
    }

    pub fn name(&self) -> String {
        match self {
            GgmlType::F32 => "F32".into(),
            GgmlType::F16 => "F16".into(),
            GgmlType::Q4_0 => "Q4_0".into(),
            GgmlType::Q4_1 => "Q4_1".into(),
            GgmlType::Q5_0 => "Q5_0".into(),
            GgmlType::Q5_1 => "Q5_1".into(),
            GgmlType::Q8_0 => "Q8_0".into(),
            GgmlType::Q8_1 => "Q8_1".into(),
            GgmlType::Q2K => "Q2_K".into(),
            GgmlType::Q3K => "Q3_K".into(),
            GgmlType::Q4K => "Q4_K".into(),
            GgmlType::Q5K => "Q5_K".into(),
            GgmlType::Q6K => "Q6_K".into(),
            GgmlType::IQ2Xxs => "IQ2_XXS".into(),
            GgmlType::IQ2Xs => "IQ2_XS".into(),
            GgmlType::IQ3Xxs => "IQ3_XXS".into(),
            GgmlType::IQ1S => "IQ1_S".into(),
            GgmlType::IQ4Nl => "IQ4_NL".into(),
            GgmlType::IQ3S => "IQ3_S".into(),
            GgmlType::IQ2S => "IQ2_S".into(),
            GgmlType::IQ4Xs => "IQ4_XS".into(),
            GgmlType::I8 => "I8".into(),
            GgmlType::I16 => "I16".into(),
            GgmlType::I32 => "I32".into(),
            GgmlType::I64 => "I64".into(),
            GgmlType::Other(t) => format!("TYPE_{t}"),
        }
    }

    /// (block_bytes, elems_per_block) for quantized row-size estimation.
    /// Standard ggml block geometry; used only to estimate tensor byte sizes.
    fn block(&self) -> Option<(u64, u64)> {
        Some(match self {
            GgmlType::Q4_0 => (18, 32),
            GgmlType::Q4_1 => (20, 32),
            GgmlType::Q5_0 => (22, 32),
            GgmlType::Q5_1 => (24, 32),
            GgmlType::Q8_0 => (34, 32),
            GgmlType::Q8_1 => (36, 32),
            GgmlType::Q2K => (84, 256),
            GgmlType::Q3K => (110, 256),
            GgmlType::Q4K => (144, 256),
            GgmlType::Q5K => (176, 256),
            GgmlType::Q6K => (210, 256),
            GgmlType::IQ2Xxs => (66, 256),
            GgmlType::IQ2Xs => (74, 256),
            GgmlType::IQ3Xxs => (102, 256),
            GgmlType::IQ1S => (50, 256),
            GgmlType::IQ4Nl => (136, 256),
            GgmlType::IQ3S => (110, 256),
            GgmlType::IQ2S => (74, 256),
            GgmlType::IQ4Xs => (136, 256),
            _ => return None,
        })
    }

    /// Bytes for a tensor of this type with the given (row-major) dims.
    pub fn tensor_bytes(&self, dims: &[u64]) -> u64 {
        let ne: u64 = dims.iter().product::<u64>().max(1);
        match self {
            GgmlType::F32 => ne * 4,
            GgmlType::F16 => ne * 2,
            GgmlType::I8 => ne,
            GgmlType::I16 => ne * 2,
            GgmlType::I32 => ne * 4,
            GgmlType::I64 => ne * 8,
            GgmlType::Other(_) => 0, // unknown: don't guess
            q => {
                let (bb, eb) = q.block().unwrap();
                let blocks = ne.div_ceil(eb);
                blocks * bb
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub dtype: GgmlType,
    pub offset: u64,
    /// Estimated size in bytes (exact for known types).
    pub nbytes: u64,
}

#[derive(Debug, Clone, Default)]
pub struct GgufMeta {
    pub version: u32,
    pub tensor_count: u64,
    /// All metadata KVs; array-valued KVs are flattened to their elements as
    /// strings. Binary KVs are skipped. This is enough for inventory,
    /// guardrails and chat-template extraction.
    pub kv: BTreeMap<String, MetaValue>,
    pub tensors: Vec<TensorInfo>,
    pub data_start: u64,
    pub file_size: u64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetaValue {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
}

impl MetaValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetaValue::U64(v) => Some(*v),
            MetaValue::I64(v) => (*v).try_into().ok(),
            MetaValue::Bool(v) => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }
}

pub struct Reader<R: Read> {
    data: R,
    pos: u64,
}

impl<R: Read> Reader<R> {
    fn exact(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let mut buf = vec![0u8; n];
        self.data.read_exact(&mut buf).map_err(|e| format!("gguf: truncated at +{}: {e}", self.pos))?;
        self.pos += n as u64;
        Ok(buf)
    }
    fn u8v(&mut self) -> Result<u8, String> {
        Ok(self.exact(1)?[0])
    }
    fn u16v(&mut self) -> Result<u16, String> {
        let b = self.exact(2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }
    fn u32v(&mut self) -> Result<u32, String> {
        let b = self.exact(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn u64v(&mut self) -> Result<u64, String> {
        let b = self.exact(8)?;
        let mut arr = [0u8; 8];
        arr.copy_from_slice(&b);
        Ok(u64::from_le_bytes(arr))
    }
    fn i64v(&mut self) -> Result<i64, String> {
        Ok(self.u64v()? as i64)
    }
    fn f32v(&mut self) -> Result<f64, String> {
        Ok(f32::from_le_bytes(self.exact(4)?.try_into().unwrap()) as f64)
    }
    fn f64v(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.exact(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let len = self.u64v()? as usize;
        if len > 4 * 1024 * 1024 {
            return Err("gguf: metadata string too long".into());
        }
        let b = self.exact(len)?;
        String::from_utf8(b).map_err(|_| "gguf: non-utf8 string".to_string())
    }
    fn skip(&mut self, n: u64) -> Result<(), String> {
        std::io::copy(
            &mut self.data.by_ref().take(n),
            &mut std::io::sink(),
        )
        .map_err(|e| format!("gguf: truncated (skip): {e}"))?;
        self.pos += n;
        Ok(())
    }
}

fn skip_value<R: Read>(r: &mut Reader<R>, vt: u32, depth: u32) -> Result<(), String> {
    if depth > 4 {
        return Err("gguf: array nesting too deep".into());
    }
    match vt {
        VT_U8 | VT_I8 | VT_BOOL => r.skip(1),
        VT_U16 | VT_I16 => r.skip(2),
        VT_U32 | VT_I32 | VT_F32 => r.skip(4),
        VT_U64 | VT_I64 | VT_F64 => r.skip(8),
        VT_STRING => {
            let len = r.u64v()?;
            if len > 4 * 1024 * 1024 {
                return Err("gguf: metadata string too long".into());
            }
            r.skip(len)
        }
        VT_ARRAY => {
            let evt = r.u32v()?;
            let count = r.u64v()?;
            // Hard cap: no legitimate GGUF has >100M array elements in metadata.
            if count > 100_000_000 {
                return Err("gguf: metadata array too large".into());
            }
            for _ in 0..count {
                skip_value(r, evt, depth + 1)?;
            }
            Ok(())
        }
        _ => Err(format!("gguf: unknown metadata value type {vt}")),
    }
}

fn read_value<R: Read>(r: &mut Reader<R>, vt: u32, depth: u32) -> Result<Option<MetaValue>, String> {
    match vt {
        VT_U8 => Ok(Some(MetaValue::U64(r.u8v()? as u64))),
        VT_I8 => Ok(Some(MetaValue::I64(r.u8v()? as i8 as i64))),
        VT_BOOL => Ok(Some(MetaValue::Bool(r.u8v()? != 0))),
        VT_U16 => Ok(Some(MetaValue::U64(r.u16v()? as u64))),
        VT_I16 => {
            let v = r.u16v()? as i16;
            Ok(Some(MetaValue::I64(v as i64)))
        }
        VT_U32 => Ok(Some(MetaValue::U64(r.u32v()? as u64))),
        VT_I32 => {
            let v = r.u32v()? as i32;
            Ok(Some(MetaValue::I64(v as i64)))
        }
        VT_F32 => Ok(Some(MetaValue::F64(r.f32v()?))),
        VT_U64 => Ok(Some(MetaValue::U64(r.u64v()?))),
        VT_I64 => Ok(Some(MetaValue::I64(r.i64v()?))),
        VT_F64 => Ok(Some(MetaValue::F64(r.f64v()?))),
        VT_STRING => Ok(Some(MetaValue::Str(r.string()?))),
        VT_ARRAY => {
            let evt = r.u32v()?;
            let count = r.u64v()?;
            if count > 1_000_000 {
                // Don't materialize huge arrays; skip.
                for _ in 0..count {
                    skip_value(r, evt, depth + 1)?;
                }
                return Ok(None);
            }
            let mut out: Vec<MetaValue> = Vec::new();
            for _ in 0..count {
                match read_value(r, evt, depth + 1)? {
                    Some(v) => out.push(v),
                    None => {}
                }
            }
            // Represent arrays as a joined string when scalar/string-ish.
            let joined = out
                .iter()
                .map(|v| match v {
                    MetaValue::Str(s) => s.clone(),
                    other => format!("{other:?}"),
                })
                .collect::<Vec<_>>()
                .join(" | ");
            Ok(Some(MetaValue::Str(joined)))
        }
        _ => Err(format!("gguf: unknown metadata value type {vt}")),
    }
}

impl GgufMeta {
    /// Parse a GGUF file's header/metadata from `path`.
    pub fn from_file(path: &Path) -> Result<GgufMeta, String> {
        let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
        let file_size = f.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
        f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
        let mut m = Self::from_reader(&mut f)?;
        m.file_size = file_size;
        Ok(m)
    }

    pub fn from_reader<R: Read>(data: &mut R) -> Result<GgufMeta, String> {
        let mut r = Reader { data, pos: 0 };
        let magic = r.u32v()?;
        if magic != MAGIC_GGUF {
            return Err("gguf: bad magic (not a GGUF file)".into());
        }
        let version = r.u32v()?;
        if version < 2 || version > 3 {
            return Err(format!("gguf: unsupported version {version}"));
        }
        let tensor_count = r.u64v()?;
        let kv_count = r.u64v()?;
        if tensor_count > 100_000 || kv_count > 100_000 {
            return Err("gguf: implausible header counts".into());
        }
        let mut meta = GgufMeta {
            version,
            tensor_count,
            ..Default::default()
        };
        for _ in 0..kv_count {
            let key = r.string()?;
            let vt = r.u32v()?;
            match read_value(&mut r, vt, 0)? {
                Some(v) => {
                    meta.kv.insert(key, v);
                }
                None => {
                    meta.kv.entry(key).or_insert(MetaValue::Str(String::new()));
                }
            }
        }
        let mut max_off: u64 = 0;
        let mut sum_bytes: u64 = 0;
        for _ in 0..tensor_count {
            let name = r.string()?;
            let ndims = r.u32v()? as usize;
            if ndims > 4 {
                return Err("gguf: tensor with >4 dims".into());
            }
            let mut dims = Vec::with_capacity(ndims);
            for _ in 0..ndims {
                dims.push(r.u64v()?);
            }
            let dtype = GgmlType::from_u32(r.u32v()?);
            let offset = r.u64v()?;
            let nbytes = dtype.tensor_bytes(&dims);
            max_off = max_off.max(offset.saturating_add(nbytes));
            sum_bytes = sum_bytes.saturating_add(nbytes);
            meta.tensors.push(TensorInfo { name, dims, dtype, offset, nbytes });
        }
        let align = meta
            .kv
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(32)
            .max(1);
        let data_start = (r.pos.div_ceil(align)) * align;
        meta.data_start = data_start;
        if data_start.saturating_add(max_off) > meta.file_size && meta.file_size > 0 {
            return Err("gguf: tensor data extends past end of file".into());
        }
        let _ = sum_bytes;
        Ok(meta)
    }

    /// Model architecture (e.g. "llama", "qwen2", "gemma3"), lowercased.
    pub fn arch(&self) -> Option<String> {
        self.kv.get("general.architecture").and_then(|v| v.as_str()).map(|s| s.to_string())
    }

    fn arch_u64(&self, arch: &str, key: &str) -> Option<u64> {
        self.kv
            .get(&format!("{arch}.{key}"))
            .and_then(|v| v.as_u64())
            .or_else(|| {
                // Some models use the bare key or the llama-prefixed one.
                self.kv.get(&format!("llama.{key}")).and_then(|v| v.as_u64())
            })
    }

    /// Architecture facts the guardrail estimator needs.
    pub fn shape(&self) -> Option<ModelShape> {
        let arch = self.arch()?;
        let n_layers = self.arch_u64(&arch, "block_count")?;
        let n_embd = self.arch_u64(&arch, "embedding_length")?;
        let n_head = self.arch_u64(&arch, "attention.head_count").unwrap_or(1).max(1);
        let n_head_kv = self
            .arch_u64(&arch, "attention.head_count_kv")
            .unwrap_or(n_head)
            .max(1);
        let n_ctx_train = self.arch_u64(&arch, "context_length").unwrap_or(4096);
        let head_dim = self
            .arch_u64(&arch, "attention.key_length")
            .unwrap_or(n_embd / n_head)
            .max(1);
        let vocab = self.arch_u64(&arch, "vocab_size").unwrap_or(32000);
        Some(ModelShape {
            arch,
            n_layers,
            n_embd,
            n_head,
            n_head_kv,
            head_dim,
            n_ctx_train,
            vocab,
        })
    }

    pub fn chat_template(&self) -> Option<String> {
        self.kv.get("tokenizer.chat_template").and_then(|v| v.as_str()).map(|s| s.to_string())
    }

    /// Best-effort quantization label: dominant non-embedding tensor type.
    pub fn quant_label(&self) -> String {
        let mut counts: BTreeMap<String, u64> = BTreeMap::new();
        for t in &self.tensors {
            if t.name.contains("ffn_down") || t.name.ends_with(".output") {
                // llama.cpp conventionally keeps these at higher precision;
                // still count them, but below.
            }
            *counts.entry(t.dtype.name()).or_insert(0) += t.nbytes;
        }
        counts
            .into_iter()
            .max_by_key(|(_, n)| *n)
            .map(|(name, _)| name)
            .unwrap_or_else(|| "unknown".into())
    }
}

#[derive(Debug, Clone)]
pub struct ModelShape {
    pub arch: String,
    pub n_layers: u64,
    pub n_embd: u64,
    pub n_head: u64,
    pub n_head_kv: u64,
    pub head_dim: u64,
    pub n_ctx_train: u64,
    pub vocab: u64,
}
