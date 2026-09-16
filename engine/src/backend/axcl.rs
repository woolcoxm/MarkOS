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
    threads: i32,
    kv: KvQuant,
}

// The model handle is used from the (single) generation thread only; the
// appliance serializes generations per model through the scheduler.
unsafe impl Send for AxclHandle {}

impl Drop for AxclHandle {
    fn drop(&mut self) {
        unsafe { sys::markos_llama_model_free(self.model) };
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
    kv: KvQuant,
    info: BackendInfo,
) -> Result<Box<dyn Backend>, String> {
    backend_init();
    let cpath = CString::new(path.as_os_str().to_string_lossy().as_bytes())
        .map_err(|_| "model path contains NUL".to_string())?;
    // n_gpu_layers > 0 lets the scheduler route through the registered
    // ggml-axcl device (whole-layer mode claims the graph there). Without
    // a card, llama.cpp falls back to CPU transparently.
    let model = unsafe { sys::markos_llama_model_load(cpath.as_ptr(), 99) };
    sys::assert_handle(model, "model load")?;
    Ok(Box::new(AxclHandle {
        model,
        info,
        n_ctx: n_ctx.clamp(1, u32::MAX as u64) as u32,
        n_batch: n_batch.clamp(1, 4096) as u32,
        threads: threads.max(1) as i32,
        kv,
    }))
}

impl AxclHandle {
    fn token_piece(&self, tok: i32) -> String {
        let mut buf = [0i8; 256];
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
            let ctx = sys::markos_llama_context_create(
                self.model,
                self.n_ctx,
                self.n_batch,
                self.threads,
                kv_code(self.kv),
                1, // flash attention: keeps the compute buffer small
            );
            sys::assert_handle(ctx, "context create")
                .map_err(|e| BackendError::Other(e))?;

            let result = self.generate_with_ctx(ctx, params, n_vocab, on_token);
            sys::markos_llama_context_free(ctx);
            result
        }
    }
}

impl AxclHandle {
    unsafe fn generate_with_ctx(
        &self,
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
                let mut n_cur: i32 = 0;
                let mut pending: Vec<u8> = Vec::new();
                let mut out: Vec<u8> = Vec::new();
                let mut gen_tokens: u64 = 0;
                let max_tokens = params.max_tokens.max(1) as u64;
                let mut stop_reason: Option<&'static str> = None;
                let n_batch = self.n_batch as i32;

                // chunked prefill, logits on each chunk's last token
                for (i, tok) in tokens.iter().enumerate() {
                    let i = i as i32;
                    let last_of_chunk = (i + 1) % n_batch == 0 || i + 1 == tokens.len() as i32;
                    sys::markos_llama_batch_add(batch, *tok, i, i32::from(last_of_chunk));
                    if last_of_chunk {
                        let rc = sys::markos_llama_decode(ctx, batch);
                        if rc != 0 {
                            return Err(BackendError::Other(format!("prefill decode rc={rc}")));
                        }
                        sys::markos_llama_batch_clear(batch);
                        n_cur = i + 1;
                    }
                }

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
            sys::markos_llama_sampler_free(smpl);
            rc
        })();
        sys::markos_llama_batch_free(batch);
        rc
    }
}
