//! llama.cpp/ggml tensor backend (feature `llama`, enabled for the target
//! build in the OS image). This is the *only* module that knows about ggml;
//! everything above the `Backend` trait is MarkOS serving layer.
//!
//! Explicit design points:
//! - CPU/NEON only: no GPU layers are ever requested (there is no viable GPU
//!   on the Pi 5; see design doc §1). Codegen targets cortex-a76 via the
//!   Buildroot toolchain (`-mcpu=cortex-a76`), so ggml's DotProd kernels run.
//! - The inference context is created once per model slot and kept across
//!   requests (KV cleared or prefix-trimmed between them). Fresh contexts
//!   per request paid a KV-cache allocation + graph reserve on every call —
//!   the same pathology removed from the axcl backend, and it also erased
//!   prompt-prefix reuse. Multi-turn chat now prefills only the new suffix.
//! - The context KV quantization (f16/q8_0/q4_0) is wired from the model
//!   config so the guardrail estimate and the runtime agree.

use super::{Backend, BackendError, BackendInfo, GenParams, GenStats};
use crate::backend::mock::apply_stops;
use crate::guard::KvQuant;
use llama_cpp_2::context::LlamaContext;
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
                    pending.clear();
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

/// A context whose borrow of the model has been erased to `'static`.
///
/// Soundness: the context is stored in the same `LlamaHandle` as its model
/// and is dropped in `Drop` BEFORE the model; nothing else can free or move
/// the model meanwhile (it is `ManuallyDrop`, touched only in `Drop` and
/// behind `&mut self` everywhere else).
struct SendContext(LlamaContext<'static>);

// llama.cpp contexts carry no thread-local state; the appliance serializes
// generations per model slot through the manager (one guard = exclusive use),
// the same arrangement as the axcl backend's handle.
unsafe impl Send for SendContext {}

pub struct LlamaHandle {
    /// Owned model. `ManuallyDrop` because the cached context below
    /// references its data; `Drop` frees the context first, then the model.
    model: std::mem::ManuallyDrop<llama_cpp_2::model::LlamaModel>,
    /// Inference context, created on the first generate() and reused for the
    /// handle's lifetime.
    ctx: Option<SendContext>,
    /// Token history resident in the cached context's KV (prompt + generated
    /// of the last request). Enables prompt-prefix reuse: a multi-turn chat
    /// only prefills the new suffix instead of the whole conversation.
    last_tokens: Vec<i32>,
    info: BackendInfo,
    n_ctx: u32,
    n_batch: u32,
    /// Decode threads (token generation) — see ModelConfig::decode_threads.
    threads: i32,
    /// Prefill threads (prompt processing) — every core.
    threads_batch: i32,
    kv: KvQuant,
}

pub fn load(
    path: &std::path::Path,
    n_ctx: u64,
    n_batch: u64,
    threads: usize,
    threads_decode: usize,
    kv: KvQuant,
    info: BackendInfo,
) -> Result<Box<dyn Backend>, String> {
    let _ = backend(); // init once before any model load
    // CPU-only by construction: default params request 0 GPU layers.
    let model_params = llama_cpp_2::model::params::LlamaModelParams::default();
    let model = llama_cpp_2::model::LlamaModel::load_from_file(backend(), path, &model_params)
        .map_err(|e| format!("load model {}: {e}", path.display()))?;
    Ok(Box::new(LlamaHandle {
        model: std::mem::ManuallyDrop::new(model),
        ctx: None,
        last_tokens: Vec::new(),
        info,
        n_ctx: n_ctx.clamp(1, u32::MAX as u64) as u32,
        n_batch: n_batch.clamp(1, 4096) as u32,
        threads: threads_decode.max(1) as i32,
        threads_batch: threads.max(1) as i32,
        kv,
    }))
}

impl Drop for LlamaHandle {
    fn drop(&mut self) {
        // The context borrows the model: it MUST go first (see SendContext).
        self.ctx.take();
        unsafe {
            std::mem::ManuallyDrop::drop(&mut self.model);
        }
    }
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

    /// Create the context on first use. The returned lifetime is erased to
    /// 'static; see `SendContext` for why that is sound here.
    fn ensure_ctx(&mut self) -> Result<(), BackendError> {
        if self.ctx.is_some() {
            return Ok(());
        }
        use llama_cpp_2::context::params::LlamaContextParams;
        let ctx_params = LlamaContextParams::default()
            .with_n_ctx(Some(std::num::NonZeroU32::new(self.n_ctx).expect("n_ctx >= 1")))
            .with_n_batch(self.n_batch)
            .with_n_ubatch(self.n_batch)
            .with_n_threads(self.threads)
            .with_n_threads_batch(self.threads_batch)
            .with_type_k(Self::kv_cache_type(self.kv))
            .with_type_v(Self::kv_cache_type(self.kv))
            // Flash attention keeps the compute buffer small; the guardrail
            // estimator assumes it (engine/src/guard.rs).
            .with_flash_attention_policy(llama_cpp_sys_2::LLAMA_FLASH_ATTN_TYPE_ENABLED);
        let ctx = self
            .model
            .new_context(backend(), ctx_params)
            .map_err(|e| BackendError::Other(format!("context: {e}")))?;
        // SAFETY: lifetime erase only — layout is identical. The context is
        // owned by self and dropped before the model (see Drop / SendContext).
        let ctx: LlamaContext<'static> = unsafe { std::mem::transmute(ctx) };
        self.ctx = Some(SendContext(ctx));
        Ok(())
    }

    /// Raw token bytes into the pending buffer. Raw bytes (not a lossy
    /// String) are load-bearing: byte-fallback tokens split a multi-byte
    /// character across tokens, and `emit_utf8` exists to reassemble them.
    /// A free function over `&LlamaModel` (not a method) so the decode loop
    /// can use it while holding the `&mut` context borrow.
    fn push_token_piece(
        model: &llama_cpp_2::model::LlamaModel,
        tok: llama_cpp_2::token::LlamaToken,
        pending: &mut Vec<u8>,
    ) {
        const BUF: usize = 48;
        let bytes = match model.token_to_piece_bytes(tok, BUF, true, None) {
            Err(llama_cpp_2::TokenToStringError::InsufficientBufferSpace(n)) => {
                model.token_to_piece_bytes(tok, n.max(1) as usize, true, None)
            }
            other => other,
        };
        if let Ok(bytes) = bytes {
            pending.extend_from_slice(&bytes);
        }
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
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::token::LlamaToken;

        self.ensure_ctx()?;
        let mut sampler = self.build_sampler(&params.sampling);
        // Mutable borrow of the context field; everything below touches only
        // disjoint fields (`self.model`, `self.last_tokens`).
        let ctx: &mut LlamaContext<'static> = &mut self.ctx.as_mut().expect("ensure_ctx above").0;

        let tokens = self
            .model
            .str_to_token(&params.prompt, AddBos::Always)
            .map_err(|e| BackendError::Other(format!("tokenize: {e}")))?
        ;
        let prompt_tokens = tokens.len() as u64;
        if prompt_tokens > self.n_ctx as u64 {
            return Err(BackendError::ContextOverflow {
                prompt_tokens,
                n_ctx: self.n_ctx as u64,
            });
        }

        let n_batch = self.n_batch as i32;
        let mut batch = LlamaBatch::new(self.n_batch as usize, 1);

        // ---- KV strategy: prompt-prefix reuse (mirrors backend/axcl.rs) ----
        // Keep the cells of the shared prefix with the previous request,
        // remove the divergent tail, prefill only the suffix.
        let mut common = self
            .last_tokens
            .iter()
            .zip(tokens.iter())
            .take_while(|(a, b)| **a == b.0)
            .count();
        // Force at least one prefill step so the sampler reads fresh logits
        // (also covers the identical-prompt-resubmitted case).
        common = common.min(tokens.len().saturating_sub(1));
        let trimmed = if common == 0 {
            false
        } else {
            ctx.clear_kv_cache_seq(Some(0), Some(common as u32), None)
                .unwrap_or(false)
        };
        if trimmed {
            self.last_tokens.truncate(common);
        } else {
            ctx.clear_kv_cache();
            self.last_tokens.clear();
            common = 0;
        }

        let gen = {
            // Reborrow so the closure consumes only a borrow of `ctx` and the
            // error path below can still clear the cache.
            let ctx = &mut *ctx;
            (|| -> Result<GenStats, BackendError> {
            let mut n_cur: i32 = common as i32;
            let mut pending: Vec<u8> = Vec::new();
            let mut out: Vec<u8> = Vec::new();
            let mut gen_tokens: u64 = 0;
            let max_tokens = params.max_tokens.max(1) as u64;
            let mut stop_reason: Option<&str> = None;
            let stop_max = params
                .stop
                .iter()
                .map(|s| s.as_bytes().len())
                .max()
                .unwrap_or(0);

            // Prefill the uncached suffix in n_batch chunks; logits enabled
            // on each chunk's last token.
            for (i, tok) in tokens[common..].iter().enumerate() {
                let pos = common as i32 + i as i32;
                let last_of_chunk =
                    (i + 1) % n_batch as usize == 0 || common + i + 1 == tokens.len();
                batch
                    .add(*tok, pos, &[0], last_of_chunk)
                    .map_err(|e| BackendError::Other(format!("batch: {e}")))?;
                if last_of_chunk {
                    ctx.decode(&mut batch)
                        .map_err(|e| BackendError::Other(format!("prefill: {e}")))?;
                    n_cur = pos + 1;
                }
            }
            // The whole prompt is now resident in KV — record it so the next
            // request can reuse the prefix.
            if self.last_tokens.len() < tokens.len() {
                self.last_tokens.extend(
                    tokens[self.last_tokens.len()..].iter().map(|t| t.0),
                );
            }

            // Decode loop: sample → emit → feed back.
            let eos = self.model.token_eos();
            loop {
                let tok = sampler.sample(ctx, -1);
                if tok == eos {
                    // Natural end-of-generation: don't emit the special token.
                    stop_reason = Some("stop");
                    break;
                }
                sampler.accept(tok);
                Self::push_token_piece(&self.model, tok, &mut pending);
                gen_tokens += 1;
                let before = out.len();
                emit_utf8(&mut pending, &mut out, on_token);

                if stop_max > 0 {
                    // bounded tail scan: a stop string ending in this piece
                    // starts at most stop_max-1 bytes before `before`
                    let mut from = before.saturating_sub(stop_max - 1);
                    let view = loop {
                        match std::str::from_utf8(&out[from..]) {
                            Ok(v) => break v,
                            Err(_) => from += 1,
                        }
                    };
                    for s in &params.stop {
                        if !s.is_empty() && view.contains(s.as_str()) {
                            stop_reason = Some("stop");
                            break;
                        }
                    }
                }
                if stop_reason.is_none() && gen_tokens >= max_tokens {
                    stop_reason = Some("length");
                }
                if stop_reason.is_none()
                    && params.cancel.load(std::sync::atomic::Ordering::Relaxed)
                {
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
                self.last_tokens.push(tok.0);
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
            })()
        };

        if gen.is_err() {
            // KV contents no longer correspond to last_tokens — force a full
            // clear on the next request.
            ctx.clear_kv_cache();
            self.last_tokens.clear();
        }
        gen
    }
}
