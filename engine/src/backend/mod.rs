//! Backend abstraction: the boundary between the custom serving layer and the
//! tensor/quantization engine (see docs/design.md §7). The serving layer,
//! scheduler, guardrails and API never touch ggml types — they see this trait
//! only.

#![allow(dead_code)] // `sampling` is consumed by the llama backend build

use crate::config::SamplingParams;

#[derive(Debug, Clone, serde::Serialize)]
pub struct BackendInfo {
    pub id: String,
    pub arch: String,
    pub n_ctx_train: u64,
    pub vocab: u64,
    pub weights_bytes: u64,
}

#[derive(Debug, Clone)]
pub struct GenParams {
    pub prompt: String,
    pub max_tokens: u32,
    pub stop: Vec<String>,
    pub sampling: SamplingParams,
    /// Cooperative cancel flag polled between tokens.
    pub cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GenStats {
    pub prompt_tokens: u64,
    pub gen_tokens: u64,
    /// Trimmed at the stop sequence if one fired.
    pub output: String,
    pub stop_reason: String, // "stop" | "length" | "canceled" | "error"
}

#[derive(Debug)]
pub enum BackendError {
    /// Context length exceeded by the prompt.
    ContextOverflow { prompt_tokens: u64, n_ctx: u64 },
    Other(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::ContextOverflow { prompt_tokens, n_ctx } => write!(
                f,
                "prompt is {prompt_tokens} tokens, exceeding context length {n_ctx}"
            ),
            BackendError::Other(s) => write!(f, "{s}"),
        }
    }
}

pub trait Backend: Send {
    fn info(&self) -> BackendInfo;
    /// Run one greedy/sampled generation, invoking `on_token` per piece of
    /// generated text (must be valid UTF-8 chunks; caller concatenates).
    fn generate(
        &mut self,
        params: &GenParams,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<GenStats, BackendError>;
}

pub mod mock;
#[cfg(feature = "llama")]
pub mod llama;
#[cfg(feature = "axcl")]
pub mod axcl;


/// Construct the production backend for a GGUF file. Used by the model
/// manager; on non-llama builds this fails at load time with a clear message
/// (tests use `mock::load` directly).
#[cfg(feature = "axcl")]
pub fn open_gguf(
    path: &std::path::Path,
    n_ctx: u64,
    n_batch: u64,
    threads: usize,
    kv: crate::guard::KvQuant,
    info: BackendInfo,
    meta: crate::gguf::GgufMeta,
) -> Result<Box<dyn Backend>, String> {
    axcl::load(path, n_ctx, n_batch, threads, kv, info, &meta)
}

#[cfg(all(not(feature = "axcl"), feature = "llama"))]
pub fn open_gguf(
    path: &std::path::Path,
    n_ctx: u64,
    n_batch: u64,
    threads: usize,
    kv: crate::guard::KvQuant,
    info: BackendInfo,
    _meta: crate::gguf::GgufMeta,
) -> Result<Box<dyn Backend>, String> {
    llama::load(path, n_ctx, n_batch, threads, kv, info)
}

#[cfg(not(any(feature = "llama", feature = "axcl")))]
pub fn open_gguf(
    path: &std::path::Path,
    n_ctx: u64,
    n_batch: u64,
    threads: usize,
    _kv: crate::guard::KvQuant,
    _info: BackendInfo,
    _meta: crate::gguf::GgufMeta,
) -> Result<Box<dyn Backend>, String> {
    #[cfg(feature = "mock-backend")]
    {
        mock::load(path, n_ctx, n_batch, threads)
    }
    #[cfg(not(feature = "mock-backend"))]
    {
        let _ = (path, n_ctx, n_batch, threads);
        Err("engine built without the `llama` feature: no tensor backend available \
             (the MarkOS image builds with --features llama,tls)"
            .into())
    }
}
