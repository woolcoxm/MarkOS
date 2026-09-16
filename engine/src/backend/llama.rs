//! llama.cpp/ggml tensor backend (feature `llama`, enabled for the target
//! build in the OS image). This is the *only* module that knows about ggml;
//! everything above the `Backend` trait is MarkOS serving layer.
//!
//! Explicit design points:
//! - CPU/NEON only: no GPU layers are ever requested (there is no viable GPU
//!   on the Pi 5; see design doc §1). Codegen targets cortex-a76 via the
//!   Buildroot toolchain (`-mcpu=cortex-a76`), so ggml's DotProd kernels run.
//! - A fresh context is created per generation request: KV state resets
//!   cleanly, and the appliance runs one generation per model at a time, so
//!   context re-creation cost is negligible.
//! - The context KV quantization (f16/q8_0/q4_0) is wired from the model
//!   config so the guardrail estimate and the runtime agree.

use super::{Backend, BackendError, BackendInfo, GenParams, GenStats};
use crate::backend::mock::apply_stops;
use crate::guard::KvQuant;
use std::sync::OnceLock;

static BACKEND: OnceLock<llama_cpp_2::llama_backend::LlamaBackend> = OnceLock::new();

fn backend() -> &'static llama_cpp_2::llama_backend::LlamaBackend {
    BACKEND.get_or_init(|| {
        llama_cpp_2::llama_backend::LlamaBackend::init().expect("llama backend init")
    })
}

/// Emit the valid UTF-8 prefix of `pending`, keeping any partial multi-byte
/// tail buffered so streamed characters stay intact.
fn emit_utf8(pending: &mut Vec<u8>, out: &mut Vec<u8>, on_token: &mut dyn FnMut(&str)) {
    loop {
        match std::str::from_utf8(pending) {
            Ok(s) => {
                if !s.is_empty() {
                    on_token(s);
                    out.extend_from_slice(s.as_bytes());
                    *pending = Vec::new();
                }
                return;
            }
            Err(e) => {
                if e.valid_up_to() == 0 {
                    return;
                }
                let s = String::from_utf8_lossy(&pending[..e.valid_up_to()]).into_owned();
                on_token(&s);
                out.extend_from_slice(s.as_bytes());
                pending.drain(..e.valid_up_to());
            }
        }
    }
}

pub struct LlamaHandle {
    model: llama_cpp_2::model::LlamaModel,
    info: BackendInfo,
    n_ctx: u32,
    n_batch: u32,
    threads: i32,
    kv: KvQuant,
}

pub fn load(
    path: &std::path::Path,
    n_ctx: u64,
    n_batch: u64,
    threads: usize,
    kv: KvQuant,
    info: BackendInfo,
) -> Result<Box<dyn Backend>, String> {
    let _ = backend(); // init once before any model load
    // CPU-only by construction: default params request 0 GPU layers.
    let model_params = llama_cpp_2::model::params::LlamaModelParams::default();
    let model = llama_cpp_2::model::LlamaModel::load_from_file(backend(), path, &model_params)
        .map_err(|e| format!("load model {}: {e}", path.display()))?;
    Ok(Box::new(LlamaHandle {
        info,
        n_ctx: n_ctx.clamp(1, u32::MAX as u64) as u32,
        n_batch: n_batch.clamp(1, 4096) as u32,
        threads: threads.max(1) as i32,
        kv,
        model,
    }))
}

impl LlamaHandle {
    fn kv_cache_type(t: KvQuant) -> llama_cpp_2::context::params::KvCacheType {
        use llama_cpp_2::context::params::KvCacheType;
        match t {
            KvQuant::Q8_0 => KvCacheType::Q8_0,
            KvQuant::Q4_0 => KvCacheType::Q4_0,
            KvQuant::F16 => KvCacheType::F16,
        }
    }

    fn build_sampler(&self, s: &crate::config::SamplingParams) -> llama_cpp_2::sampling::LlamaSampler {
        use llama_cpp_2::sampling::LlamaSampler;
        let seed = if s.seed == 0 { u32::MAX } else { s.seed };
        let pieces: Vec<LlamaSampler> = if s.mirostat == 1 {
            vec![LlamaSampler::mirostat(
                self.model.n_vocab(),
                seed,
                s.mirostat_tau as f32,
                s.mirostat_eta as f32,
                1,
            )]
        } else if s.mirostat == 2 {
            vec![LlamaSampler::mirostat_v2(seed, s.mirostat_tau as f32, s.mirostat_eta as f32)]
        } else {
            let mut v = vec![LlamaSampler::penalties(
                self.model.n_vocab(),
                s.repeat_last_n,
                s.repeat_penalty as f32,
                s.frequency_penalty as f32,
                s.presence_penalty as f32,
            )];
            v.push(LlamaSampler::top_k(s.top_k));
            v.push(LlamaSampler::top_p(s.top_p as f32, 1));
            v.push(LlamaSampler::min_p(s.min_p as f32, 1));
            if s.temperature <= 0.0 {
                v.push(LlamaSampler::greedy());
            } else {
                v.push(LlamaSampler::temp(s.temperature as f32));
                v.push(LlamaSampler::dist(seed));
            }
            v
        };
        LlamaSampler::chain_simple(pieces)
    }
}

impl Backend for LlamaHandle {
    fn info(&self) -> BackendInfo {
        self.info.clone()
    }

    fn generate(
        &mut self,
        params: &GenParams,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<GenStats, BackendError> {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::token::LlamaToken;

        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(std::num::NonZeroU32::new(self.n_ctx).expect("n_ctx >= 1")))
            .with_n_batch(self.n_batch)
            .with_n_ubatch(self.n_batch)
            .with_n_threads(self.threads)
            .with_n_threads_batch(self.threads)
            .with_type_k(Self::kv_cache_type(self.kv))
            .with_type_v(Self::kv_cache_type(self.kv))
            // Flash attention keeps the compute buffer small; the guardrail
            // estimator assumes it (engine/src/guard.rs).
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED);

        let mut ctx = self
            .model
            .new_context(backend(), ctx_params)
            .map_err(|e| BackendError::Other(format!("context: {e}")))?;

        let tokens = self
            .model
            .str_to_token(&params.prompt, AddBos::Always)
            .map_err(|e| BackendError::Other(format!("tokenize: {e}")))?;
        let prompt_tokens = tokens.len() as u64;
        if prompt_tokens > self.n_ctx as u64 {
            return Err(BackendError::ContextOverflow {
                prompt_tokens,
                n_ctx: self.n_ctx as u64,
            });
        }

        let mut sampler = self.build_sampler(&params.sampling);
        let n_batch = self.n_batch as i32;
        let mut batch = LlamaBatch::new(self.n_batch as usize, 1);

        let mut n_cur: i32 = 0;
        let mut pending: Vec<u8> = Vec::new();
        let mut out: Vec<u8> = Vec::new();
        let mut gen_tokens: u64 = 0;
        let max_tokens = params.max_tokens.max(1) as u64;
        let mut stop_reason: Option<&str> = None;

        // Prefill in n_batch chunks; logits enabled on each chunk's last token.
        for (i, tok) in tokens.iter().enumerate() {
            let i = i as i32;
            let last_of_chunk = (i + 1) % n_batch == 0 || i + 1 == tokens.len() as i32;
            batch
                .add(*tok, i, &[0], last_of_chunk)
                .map_err(|e| BackendError::Other(format!("batch: {e}")))?;
            if last_of_chunk {
                ctx.decode(&mut batch)
                    .map_err(|e| BackendError::Other(format!("prefill: {e}")))?;
                n_cur = i + 1;
            }
        }

        // Decode loop: sample → emit → feed back.
        let eos = self.model.token_eos();
        loop {
            let tok = sampler.sample(&mut ctx, -1);
            if tok == eos {
                // Natural end-of-generation: don't emit the special token.
                stop_reason = Some("stop");
                break;
            }
            sampler.accept(tok);
            let piece = token_piece(&self.model, tok);
            pending.extend_from_slice(piece.as_bytes());
            gen_tokens += 1;
            emit_utf8(&mut pending, &mut out, on_token);

            let view = String::from_utf8_lossy(&out).into_owned();
            for s in &params.stop {
                if !s.is_empty() && view.contains(s.as_str()) {
                    stop_reason = Some("stop");
                    break;
                }
            }
            if stop_reason.is_none() && gen_tokens >= max_tokens {
                stop_reason = Some("length");
            }
            if stop_reason.is_none() && params.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                stop_reason = Some("canceled");
            }
            if stop_reason.is_some() {
                break;
            }

            batch.clear();
            batch
                .add(LlamaToken(tok.0), n_cur, &[0], true)
                .map_err(|e| BackendError::Other(format!("batch: {e}")))?;
            ctx.decode(&mut batch)
                .map_err(|e| BackendError::Other(format!("decode: {e}")))?;
            n_cur += 1;
        }

        let full = String::from_utf8_lossy(&out).into_owned();
        let output = apply_stops(&full, &params.stop);
        let stop_reason = stop_reason.unwrap_or("stop").to_string();
        Ok(GenStats {
            prompt_tokens,
            gen_tokens,
            output,
            stop_reason,
        })
    }
}

#[allow(deprecated)] // token_to_str is deprecated in favor of token_to_piece,
// which needs a decoder object; the string variant is exactly equivalent here.
fn token_piece(model: &llama_cpp_2::model::LlamaModel, tok: llama_cpp_2::token::LlamaToken) -> String {
    model
        .token_to_str(tok, llama_cpp_2::model::Special::Tokenize)
        .unwrap_or_default()
}
