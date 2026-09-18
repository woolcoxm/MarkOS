//! Axera NPU tensor backend (feature `axcl`): the ggml-axcl build of the
//! fork, driven through markos-llama-sys's C shim. Same serving contract as
//! `llama.rs` (fresh context per request, chunked prefill, UTF-8-safe
//! streaming) — see that module for the design notes; only the FFI layer
//! differs (no struct-by-value crossings, see markos-llama-sys).
//!
//! Accelerator policy lives in `crate::accel`: whole-layer mode is armed by
//! the fork when an engine set matches the model's graph geometry; every
//! other GGUF still runs, on per-op matmul engines or the CPU reference.

use super::mock::apply_stops;
use super::{Backend, BackendError, BackendInfo, GenParams, GenStats};
use crate::guard::KvQuant;
use markos_llama_sys as sys;
use std::ffi::CString;
use std::sync::OnceLock;

static BACKEND_INIT: OnceLock<()> = OnceLock::new();

fn backend_init() {
    BACKEND_INIT.get_or_init(|| unsafe {
        sys::markos_llama_backend_init();
    });
}

/// UTF-8-safe emit (identical to llama.rs): keep partial multibyte tails
/// buffered between tokens so streamed characters stay intact.
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

pub struct AxclHandle {
    model: *mut std::ffi::c_void,
    info: BackendInfo,
    n_ctx: u32,
    n_batch: u32,
    /// Threads for token generation (decode). The Pi 5 is memory-bandwidth
    /// bound at decode: ~half the cores is measurably faster than all of
    /// them (llama-bench tg64: 2T 22.6 t/s vs 4T 16.6 t/s, 0.5B Q4_K_M).
    threads: i32,
    /// Threads for prompt processing (prefill) — every core.
    threads_batch: i32,
    kv: KvQuant,
    /// Whole-layer NPU serving for this model (engine set matched). The
    /// card's memory implementation has no partial KV removal, so prompt
    /// prefix reuse is CPU-tier only.
    npu_layer: bool,
    /// Cached inference context — created once at model load, reused across
    /// requests (KV cleared between). Fresh-context-per-request cost was the
    /// dominant performance killer: a 4096-token KV cache allocation +
    /// graph setup per generation reduced a 0.5B model to 1 t/s.
    ctx: *mut std::ffi::c_void,
    /// Token history resident in the cached context's KV (prompt + generated
    /// of the last request). Enables prompt-prefix reuse: a multi-turn chat
    /// only prefills the new suffix instead of the whole conversation.
    last_tokens: Vec<i32>,
}

// The model handle is used from the (single) generation thread only; the
// appliance serializes generations per model through the scheduler.
unsafe impl Send for AxclHandle {}

impl Drop for AxclHandle {
    fn drop(&mut self) {
        unsafe {
            if !self.ctx.is_null() {
                sys::markos_llama_context_free(self.ctx);
            }
            sys::markos_llama_model_free(self.model);
        }
    }
}

fn sampler_cfg(s: &crate::config::SamplingParams) -> sys::SamplerCfg {
    let seed = if s.seed == 0 { u32::MAX } else { s.seed };
    sys::SamplerCfg {
        top_k: s.top_k as i32,
        top_p: s.top_p as f32,
        min_p: s.min_p as f32,
        temperature: s.temperature as f32,
        repeat_penalty: s.repeat_penalty as f32,
        frequency_penalty: s.frequency_penalty as f32,
        presence_penalty: s.presence_penalty as f32,
        repeat_last_n: s.repeat_last_n as i32,
        mirostat: s.mirostat as i32,
        mirostat_tau: s.mirostat_tau as f32,
        mirostat_eta: s.mirostat_eta as f32,
        seed,
    }
}

fn kv_code(kv: KvQuant) -> i32 {
    match kv {
        KvQuant::Q8_0 => 1,
        KvQuant::Q4_0 => 2,
        KvQuant::F16 => 0,
    }
}

pub fn load(
    path: &std::path::Path,
    n_ctx: u64,
    n_batch: u64,
    threads: usize,
    threads_decode: usize,
    kv: KvQuant,
    info: BackendInfo,
    meta: &crate::gguf::GgufMeta,
) -> Result<Box<dyn Backend>, String> {
    backend_init();

    // Decide the serving tier BEFORE loading: only route to the NPU backend
    // (n_gpu_layers > 0) when the model's geometry matches an installed
    // engine set. Non-matching models MUST use n_gpu_layers=0 — routing
    // them to the axcl backend causes graph-splitting between buffer types
    // that corrupts logits (hardware-verified: every non-matching model
    // produced pure '?' tokens with n_gpu_layers=99).
    let sets = crate::axsets::scan(&std::path::Path::new(&crate::accel::engines_root()));
    let card_present = crate::accel::detect().present;
    let mode = if card_present {
        match meta.shape() { Some(s) => crate::axsets::match_mode(&s, &sets), None => crate::axsets::AccelMode::Cpu }
    } else {
        crate::axsets::AccelMode::Cpu
    };
    let npu_layer = matches!(mode, crate::axsets::AccelMode::NpuLayer { .. });
    let n_gpu_layers: i32 = if npu_layer {
        // Matching model: arm the whole-layer NPU path. These env vars
        // are read by the fork's backend at first graph compute (lazily,
        // not cached at init) — safe to set here, just before model load.
        std::env::set_var("GGML_AXCL_LAYER", "1");
        std::env::set_var("GGML_AXCL_GGUF", "1");
        99
    } else {
        // Non-matching: route to CPU only (n_gpu_layers=0). The axcl
        // backend is still registered but receives no work from the
        // scheduler, so the graph computes entirely on the CPU backend.
        0
    };
    eprintln!(
        "markos-engine: model {} -> {:?} (n_gpu_layers={})",
        info.id, mode, n_gpu_layers
    );

    let cpath = CString::new(path.as_os_str().to_string_lossy().as_bytes())
        .map_err(|_| "model path contains NUL".to_string())?;
    let model = unsafe { sys::markos_llama_model_load(cpath.as_ptr(), n_gpu_layers) };
    sys::assert_handle(model, "model load")?;
    // Context is created lazily on first generate() and then reused for the
    // lifetime of the slot (KV cleared / trimmed between requests).
    let ctx: *mut std::ffi::c_void = std::ptr::null_mut();
    Ok(Box::new(AxclHandle {
        model,
        info,
        n_ctx: n_ctx.clamp(1, u32::MAX as u64) as u32,
        n_batch: n_batch.clamp(1, 4096) as u32,
        threads: threads_decode.max(1) as i32,
        threads_batch: threads.max(1) as i32,
        kv,
        npu_layer,
        ctx,
        last_tokens: Vec::new(),
    }))
}

impl AxclHandle {
    fn token_piece(&self, tok: i32) -> String {
        // c_char is i8 on x86_64 but u8 on aarch64 — keep the buffer typed
        // as c_char so this compiles for both hosts
        let mut buf = [0 as std::ffi::c_char; 256];
        let n = unsafe { sys::markos_llama_token_to_piece(self.model, tok, buf.as_mut_ptr(), 256) };
        if n <= 0 {
            return String::new();
        }
        let len = (n as usize).min(buf.len() - 1);
        let bytes: Vec<u8> = buf[..len].iter().map(|&b| b as u8).collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

impl Backend for AxclHandle {
    fn info(&self) -> BackendInfo {
        self.info.clone()
    }

    fn generate(
        &mut self,
        params: &GenParams,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<GenStats, BackendError> {
        unsafe {
            let n_vocab = sys::markos_llama_model_n_vocab(self.model);
            // Context is created on first use and then reused for the
            // lifetime of the slot. Creation cost (KV alloc + graph build)
            // is paid once per model load, not once per request.
            if self.ctx.is_null() {
                let c = sys::markos_llama_context_create(
                    self.model,
                    self.n_ctx,
                    self.n_batch,
                    self.threads,
                    self.threads_batch,
                    kv_code(self.kv),
                    1, // flash attention
                );
                sys::assert_handle(c, "context create").map_err(BackendError::Other)?;
                self.ctx = c;
            }
            self.generate_with_ctx(self.ctx, params, n_vocab, on_token)
        }
    }
}

impl AxclHandle {
    unsafe fn generate_with_ctx(
        &mut self,
        ctx: *mut std::ffi::c_void,
        params: &GenParams,
        n_vocab: i32,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<GenStats, BackendError> {
        // tokenize
        let ctext = CString::new(params.prompt.replace('\0', " "))
            .map_err(|_| BackendError::Other("prompt contains NUL".into()))?;
        let mut tokens = vec![0i32; (params.prompt.len() + 8).max(64)];
        let n = sys::markos_llama_tokenize(self.model, ctext.as_ptr(), 1, tokens.as_mut_ptr(),
                                           tokens.len() as i32);
        if n < 0 {
            return Err(BackendError::Other(format!(
                "tokenize: prompt too long for the staging buffer ({})"
                , tokens.len()
            )));
        }
        tokens.truncate(n as usize);
        let prompt_tokens = tokens.len() as u64;
        if prompt_tokens > self.n_ctx as u64 {
            return Err(BackendError::ContextOverflow {
                prompt_tokens,
                n_ctx: self.n_ctx as u64,
            });
        }

        // ---- KV strategy: prompt-prefix reuse ----
        // The cached context still holds last_tokens' KV cells. When the new
        // prompt shares a prefix with that history (multi-turn chat: system
        // + prior turns), keep the shared cells and prefill only the suffix.
        // Whole-layer NPU serving has no partial-removal primitive — it
        // always clears.
        let mut common = if self.npu_layer {
            0
        } else {
            self.last_tokens
                .iter()
                .zip(tokens.iter())
                .take_while(|(a, b)| a == b)
                .count()
        };
        // Force at least one prefill step so the sampler reads fresh logits
        // (also covers the identical-prompt-resubmitted case).
        common = common.min(tokens.len().saturating_sub(1));
        let mut reused = false;
        if self.npu_layer || common == 0 {
            sys::markos_llama_kv_clear(ctx);
            self.last_tokens.clear();
        } else {
            if sys::markos_llama_memory_seq_rm(ctx, 0, common as i32, -1) == 1 {
                // drop the trimmed tail from the tracked history too
                self.last_tokens.truncate(common);
                reused = true;
            } else {
                // memory backend can't trim: start clean
                sys::markos_llama_kv_clear(ctx);
                self.last_tokens.clear();
                common = 0;
            }
        }

        let batch = sys::markos_llama_batch_create(self.n_batch as i32);
        if batch.is_null() {
            return Err(BackendError::Other("batch alloc".into()));
        }
        let rc = (|| -> Result<GenStats, BackendError> {
            let smpl = sys::markos_llama_sampler_build(&sampler_cfg(&params.sampling), n_vocab);
            if smpl.is_null() {
                return Err(BackendError::Other("sampler build".into()));
            }
            let rc = (|| -> Result<GenStats, BackendError> {
                let eos = sys::markos_llama_vocab_eot(self.model);
                let mut n_cur: i32 = common as i32;
                let mut pending: Vec<u8> = Vec::new();
                let mut out: Vec<u8> = Vec::new();
                let mut gen_tokens: u64 = 0;
                let max_tokens = params.max_tokens.max(1) as u64;
                let mut stop_reason: Option<&'static str> = None;
                let n_batch = self.n_batch as i32;
                // Longest stop string; suffix scanning only needs to look
                // back this far per token (a match ending in earlier bytes
                // was already detected then).
                let stop_max = params
                    .stop
                    .iter()
                    .map(|s| s.as_bytes().len())
                    .max()
                    .unwrap_or(0);

                // chunked prefill of the uncached suffix, logits on each
                // chunk's last token
                for (i, tok) in tokens[common..].iter().enumerate() {
                    let pos = common as i32 + i as i32;
                    let last_of_chunk =
                        (i + 1) % n_batch as usize == 0 || common + i + 1 == tokens.len();
                    sys::markos_llama_batch_add(batch, *tok, pos, i32::from(last_of_chunk));
                    if last_of_chunk {
                        let rc = sys::markos_llama_decode(ctx, batch);
                        if rc != 0 {
                            return Err(BackendError::Other(format!("prefill decode rc={rc}")));
                        }
                        sys::markos_llama_batch_clear(batch);
                        n_cur = pos + 1;
                    }
                }
                // the whole prompt is now resident in KV — record it so the
                // next request can reuse the prefix
                let recorded = self.last_tokens.len();
                self.last_tokens.extend_from_slice(&tokens[recorded..]);

                loop {
                    let tok = sys::markos_llama_sampler_sample(smpl, ctx, -1);
                    if tok == eos {
                        stop_reason = Some("stop");
                        break;
                    }
                    sys::markos_llama_sampler_accept(smpl, tok);
                    let piece = self.token_piece(tok);
                    pending.extend_from_slice(piece.as_bytes());
                    gen_tokens += 1;
                    let before = out.len();
                    emit_utf8(&mut pending, &mut out, on_token);

                    if stop_max > 0 {
                        // scan a bounded tail instead of the whole output
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

                    sys::markos_llama_batch_clear(batch);
                    sys::markos_llama_batch_add(batch, tok, n_cur, 1);
                    let rc = sys::markos_llama_decode(ctx, batch);
                    if rc != 0 {
                        return Err(BackendError::Other(format!("decode rc={rc}")));
                    }
                    // the token is now resident in KV (position n_cur)
                    self.last_tokens.push(tok);
                    n_cur += 1;
                }

                let full = String::from_utf8_lossy(&out).into_owned();
                let output = apply_stops(&full, &params.stop);
                Ok(GenStats {
                    prompt_tokens,
                    gen_tokens,
                    output,
                    stop_reason: stop_reason.unwrap_or("stop").to_string(),
                })
            })();
            if rc.is_err() {
                // KV contents no longer correspond to last_tokens — force a
                // full clear on the next request.
                self.last_tokens.clear();
                sys::markos_llama_kv_clear(ctx);
            }
            sys::markos_llama_sampler_free(smpl);
            rc
        })();
        sys::markos_llama_batch_free(batch);
        if reused {
            eprintln!(
                "markos-engine: prompt cache hit: {}/{} tokens reused",
                common,
                tokens.len()
            );
        }
        rc
    }
}
