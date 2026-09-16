//! Memory guardrails — the backend enforcement point behind the web UI's
//! estimates. All RAM math lives here so the UI preview and the engine's
//! load-time refusal can never disagree.

use crate::gguf::ModelShape;

/// Extra bytes of a KV-cache element over raw 16-bit storage for a KV
/// quantization. f16 = 2.0 B/elem; Q8_0 = 34 bytes per 32 elems ≈ 1.0625;
/// Q4_0 = 18/32 ≈ 0.5625.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize, Default)]
pub enum KvQuant {
    #[default]
    F16,
    Q8_0,
    Q4_0,
}

impl KvQuant {
    pub fn bytes_per_elem(&self) -> f64 {
        match self {
            KvQuant::F16 => 2.0,
            KvQuant::Q8_0 => 34.0 / 32.0,
            KvQuant::Q4_0 => 18.0 / 32.0,
        }
    }
    #[allow(dead_code)] // used by the tls/llama target builds
    pub fn name(&self) -> &'static str {
        match self {
            KvQuant::F16 => "f16",
            KvQuant::Q8_0 => "q8_0",
            KvQuant::Q4_0 => "q4_0",
        }
    }
}

/// OS + page-cache floor we keep out of the model budget. The Pi 5 runs
/// Buildroot (~60 MB resident) plus page cache headroom for the engine
/// binary, logs and I/O.
pub const OS_RESERVE_BYTES: u64 = 340 * 1024 * 1024;
/// Engine process self (arena, graphs, buffers not accounted per-model).
pub const ENGINE_SELF_BYTES: u64 = 24 * 1024 * 1024;
/// Fallback "usable" on hosts where /proc/meminfo is unavailable.
pub const DEFAULT_TOTAL_BYTES: u64 = 15 * 1024 * 1024 * 1024 + 896 * 1024 * 1024;

/// RAM estimate for a specific model shape + serving configuration.
#[derive(Debug, Clone, serde::Serialize)]
pub struct RamEstimate {
    pub weights_bytes: u64,
    pub kv_cache_bytes: u64,
    pub compute_bytes: u64,
    pub margin_bytes: u64,
    pub total_bytes: u64,
}

/// Compute a RAM estimate. `weights_bytes` is the GGUF file size (weights are
/// mmap'd; we count them fully resident — page cache is reclaimable, but
/// refusing oversized files is exactly the failure mode we want).
pub fn estimate_ram(shape: &ModelShape, weights_bytes: u64, n_ctx: u64, n_batch: u64, kv: KvQuant) -> RamEstimate {
    let n_ctx = n_ctx.max(1);
    let n_batch = n_batch.clamp(1, 4096);

    // KV cache: K and V per layer per token.
    let kv_per_token: f64 = 2.0
        * shape.n_layers as f64
        * (shape.n_head_kv * shape.head_dim) as f64
        * kv.bytes_per_elem();
    let kv_cache_bytes = (kv_per_token * n_ctx as f64).ceil() as u64;

    // Compute/graph buffers. With flash attention enabled these are dominated
    // by the output logits (n_vocab × n_batch f32) and f32 activation staging;
    // graph nodes add a small per-layer constant. Documented approximation,
    // covered by the margin below.
    let logits = 4u64 * shape.vocab.max(1) * n_batch;
    let act = 36u64 * n_batch * shape.n_embd.max(1);
    let graph = shape.n_layers * 96 * 1024;
    let compute_bytes = logits + act + graph;

    let base = weights_bytes + kv_cache_bytes + compute_bytes;
    let margin_bytes = base / 100 * 15; // +15% safety margin
    RamEstimate {
        weights_bytes,
        kv_cache_bytes,
        compute_bytes,
        margin_bytes,
        total_bytes: base + margin_bytes,
    }
}

/// A refusal with numbers, surfaced to both the API caller and the UI.
#[derive(Debug, Clone, serde::Serialize)]
pub struct GuardrailRejection {
    pub reason: String,
    pub required_bytes: u64,
    pub available_bytes: u64,
}

pub fn format_bytes(b: u64) -> String {
    let units = ["B", "KiB", "MiB", "GiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < units.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{b} B")
    } else {
        format!("{v:.1} {}", units[u])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn llama7b() -> ModelShape {
        ModelShape {
            arch: "llama".into(),
            n_layers: 32,
            n_embd: 4096,
            n_head: 32,
            n_head_kv: 32,
            head_dim: 128,
            n_ctx_train: 4096,
            vocab: 32000,
        }
    }

    fn tiny() -> ModelShape {
        ModelShape {
            arch: "qwen2".into(),
            n_layers: 28,
            n_embd: 1024,
            n_head: 16,
            n_head_kv: 8,
            head_dim: 128,
            n_ctx_train: 40960,
            vocab: 151936,
        }
    }

    #[test]
    fn kv_cache_scales_linearly_with_ctx() {
        let e1 = estimate_ram(&llama7b(), 4_000_000_000, 2048, 512, KvQuant::F16);
        let e2 = estimate_ram(&llama7b(), 4_000_000_000, 4096, 512, KvQuant::F16);
        assert_eq!(e2.kv_cache_bytes, e1.kv_cache_bytes * 2);
        // 7B f16 KV: 2 * 32 layers * 4096 heads*dim 4096 * 2B * 4096 ctx
        assert_eq!(e2.kv_cache_bytes, 2 * 32 * 4096 * 2 * 4096);
    }

    #[test]
    fn kv_quant_shrinks_cache() {
        let f16 = estimate_ram(&llama7b(), 1, 4096, 512, KvQuant::F16);
        let q8 = estimate_ram(&llama7b(), 1, 4096, 512, KvQuant::Q8_0);
        let q4 = estimate_ram(&llama7b(), 1, 4096, 512, KvQuant::Q4_0);
        assert!(q8.kv_cache_bytes < f16.kv_cache_bytes);
        assert!(q4.kv_cache_bytes < q8.kv_cache_bytes);
    }

    #[test]
    fn total_is_margin_plus_parts() {
        let e = estimate_ram(&tiny(), 700_000_000, 8192, 512, KvQuant::F16);
        assert_eq!(
            e.total_bytes,
            e.weights_bytes + e.kv_cache_bytes + e.compute_bytes + e.margin_bytes
        );
        // Qwen3-0.6B at 8k ctx: must fit comfortably in a 16 GB budget.
        assert!(e.total_bytes < 2_500_000_000);
    }

    #[test]
    fn format_bytes() {
        assert_eq!(super::format_bytes(512), "512 B");
        assert_eq!(super::format_bytes(2048), "2.0 KiB");
        assert_eq!(super::format_bytes(3 * 1024 * 1024 * 1024), "3.0 GiB");
    }
}
