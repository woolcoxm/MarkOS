//! Raw bindings to the Axera-GGUF llama.cpp fork through the C shim.
//!
//! The shim (`shim/markos_llama_shim.c`) is the ONLY thing that touches
//! llama.h, and it exposes exclusively scalar/opaque-pointer functions —
//! no llama.cpp struct ever crosses into Rust, so fork-side struct churn
//! can never silently desync the engine.
//!
//! Only built with the `axcl` feature (the MarkOS target image); host
//! builds use the mock backend and never link this.

#![allow(clippy::missing_safety_doc)]

use std::ffi::{c_char, c_void};

/// Mirror of the shim's `struct markos_sampler_cfg` — both layouts are ours.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SamplerCfg {
    pub top_k: i32,
    pub top_p: f32,
    pub min_p: f32,
    pub temperature: f32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub repeat_last_n: i32,
    pub mirostat: i32,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub seed: u32,
}

extern "C" {
    pub fn markos_llama_backend_init() -> i32;
    pub fn markos_llama_model_load(path: *const c_char, n_gpu_layers: i32) -> *mut c_void;
    pub fn markos_llama_model_free(model: *mut c_void);
    pub fn markos_llama_model_n_vocab(model: *mut c_void) -> i32;
    pub fn markos_llama_vocab_eot(model: *mut c_void) -> i32;
    pub fn markos_llama_vocab_bos(model: *mut c_void) -> i32;
    pub fn markos_llama_tokenize(
        model: *mut c_void,
        text: *const c_char,
        add_bos: i32,
        out: *mut i32,
        cap: i32,
    ) -> i32;
    pub fn markos_llama_context_create(
        model: *mut c_void,
        n_ctx: u32,
        n_batch: u32,
        threads: i32,
        kv_type: i32,
        flash_attn: i32,
    ) -> *mut c_void;
    pub fn markos_llama_context_free(ctx: *mut c_void);
    pub fn markos_llama_kv_clear(ctx: *mut c_void);
    pub fn markos_llama_batch_create(n_tokens: i32) -> *mut c_void;
    pub fn markos_llama_batch_free(batch: *mut c_void);
    pub fn markos_llama_batch_clear(batch: *mut c_void);
    pub fn markos_llama_batch_add(batch: *mut c_void, token: i32, pos: i32, logits: i32) -> i32;
    pub fn markos_llama_decode(ctx: *mut c_void, batch: *mut c_void) -> i32;
    pub fn markos_llama_sampler_build(cfg: *const SamplerCfg, n_vocab: i32) -> *mut c_void;
    pub fn markos_llama_sampler_free(smpl: *mut c_void);
    pub fn markos_llama_sampler_sample(smpl: *mut c_void, ctx: *mut c_void, idx: i32) -> i32;
    pub fn markos_llama_sampler_accept(smpl: *mut c_void, token: i32);
    pub fn markos_llama_token_to_piece(
        model: *mut c_void,
        token: i32,
        buf: *mut c_char,
        cap: i32,
    ) -> i32;
}

/// Panic-free non-null assertion for opaque handles crossing back.
#[track_caller]
pub fn assert_handle(p: *mut c_void, what: &str) -> Result<*mut c_void, String> {
    if p.is_null() {
        Err(format!("{what}: null handle from llama.cpp"))
    } else {
        Ok(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampler_cfg_layout_is_pod_and_stable() {
        // 12 scalars, no padding surprises on aarch64 or x86_64: 4*7 + 4 + 4*2 + 4
        assert_eq!(std::mem::size_of::<SamplerCfg>(), 48);
        assert_eq!(std::mem::align_of::<SamplerCfg>(), 4);
    }
}
