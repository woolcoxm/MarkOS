//! Deterministic mock backend: no model required. Used for host tests, CI and
//! API contract validation. Produces a stable pseudo-text stream so the full
//! serving path (templates, queue, SSE, stop sequences, metrics) is exercised
//! end to end without a tensor backend.

use super::{Backend, BackendError, BackendInfo, GenParams, GenStats};

pub struct MockBackend {
    info: BackendInfo,
    n_ctx: u64,
}

/// Build a mock backend "from" a GGUF path (metadata is read so inventory
/// stays honest), or from raw numbers in tests.
pub fn load(path: &std::path::Path, n_ctx: u64, _n_batch: u64, _threads: usize) -> Result<Box<dyn Backend>, String> {
    let meta = crate::gguf::GgufMeta::from_file(path)?;
    let shape = meta.shape().unwrap_or(crate::gguf::ModelShape {
        arch: "mock".into(),
        n_layers: 1,
        n_embd: 64,
        n_head: 1,
        n_head_kv: 1,
        head_dim: 64,
        n_ctx_train: n_ctx,
        vocab: 256,
    });
    Ok(Box::new(MockBackend {
        info: BackendInfo {
            id: "mock".into(),
            arch: shape.arch,
            n_ctx_train: shape.n_ctx_train,
            vocab: shape.vocab,
            weights_bytes: meta.file_size,
        },
        n_ctx,
    }))
}

impl MockBackend {
    pub fn with_info(info: BackendInfo, n_ctx: u64) -> Self {
        MockBackend { info, n_ctx }
    }
}

impl Backend for MockBackend {
    fn info(&self) -> BackendInfo {
        self.info.clone()
    }

    fn generate(
        &mut self,
        params: &GenParams,
        on_token: &mut dyn FnMut(&str),
    ) -> Result<GenStats, BackendError> {
        // Rough prompt token estimate (4 bytes/token) to exercise overflow.
        let prompt_tokens = (params.prompt.len() as u64 / 4).max(1);
        if prompt_tokens > self.n_ctx {
            return Err(BackendError::ContextOverflow { prompt_tokens, n_ctx: self.n_ctx });
        }

        // Vocabulary: deterministic word list driven by the prompt bytes.
        let words = [
            "the", "model", "serves", "tokens", "quickly", "on", "four", "a76",
            "cores", "with", "neon", "and", "no", "gpu", "in", "sight",
        ];
        let mut out = String::new();
        let mut emitted: u64 = 0;
        let max = params.max_tokens.max(1) as u64;

        for i in 0..max {
            if params.cancel.load(std::sync::atomic::Ordering::Relaxed) {
                return Ok(GenStats {
                    prompt_tokens,
                    gen_tokens: emitted,
                    output: apply_stops(&out, &params.stop),
                    stop_reason: "canceled".into(),
                });
            }
            // Small deterministic delay keeps streaming observable.
            std::thread::sleep(std::time::Duration::from_millis(1));
            let seed = params.prompt.as_bytes().get((i % 16) as usize).copied().unwrap_or(i as u8);
            let word = words[((seed as u64 + i * 7) % words.len() as u64) as usize];
            let piece = if i == 0 { word.to_string() } else { format!(" {word}") };
            on_token(&piece);
            out.push_str(&piece);
            emitted += 1;

            // Stop sequence check.
            for s in &params.stop {
                if !s.is_empty() && out.ends_with(s.as_str()) {
                    let trimmed = apply_stops(&out, &params.stop);
                    return Ok(GenStats {
                        prompt_tokens,
                        gen_tokens: emitted,
                        output: trimmed,
                        stop_reason: "stop".into(),
                    });
                }
            }
        }

        Ok(GenStats {
            prompt_tokens,
            gen_tokens: emitted,
            output: apply_stops(&out, &params.stop),
            stop_reason: "length".into(),
        })
    }
}

/// Trim output at the first stop sequence occurrence.
pub fn apply_stops(out: &str, stops: &[String]) -> String {
    let mut best = out;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(pos) = out.find(s.as_str()) {
            if pos < best.len() {
                best = &out[..pos];
            }
        }
    }
    best.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(prompt: &str, max: u32, stop: Vec<String>) -> GenParams {
        GenParams {
            prompt: prompt.into(),
            max_tokens: max,
            stop,
            sampling: Default::default(),
            cancel: Default::default(),
        }
    }

    #[test]
    fn deterministic_output_and_length_stop() {
        let mut b = MockBackend::with_info(
            BackendInfo { id: "m".into(), arch: "mock".into(), n_ctx_train: 2048, vocab: 16, weights_bytes: 1 },
            2048,
        );
        let mut count = 0;
        let st = b.generate(&params("hello world", 8, vec![]), &mut |_| count += 1).unwrap();
        assert_eq!(st.stop_reason, "length");
        assert_eq!(st.gen_tokens, 8);
        assert_eq!(count, 8);
        // Determinism: same prompt → same output.
        let st2 = b.generate(&params("hello world", 8, vec![]), &mut |_| {}).unwrap();
        assert_eq!(st.output, st2.output);
    }

    #[test]
    fn stop_sequence_trims() {
        let mut b = MockBackend::with_info(
            BackendInfo { id: "m".into(), arch: "mock".into(), n_ctx_train: 2048, vocab: 16, weights_bytes: 1 },
            2048,
        );
        let st = b
            .generate(&params("repeat the model serves tokens", 16, vec!["serves".into()]), &mut |_| {})
            .unwrap();
        assert_eq!(st.stop_reason, "stop");
        assert!(!st.output.contains("serves"));
    }

    #[test]
    fn context_overflow_is_refused() {
        let mut b = MockBackend::with_info(
            BackendInfo { id: "m".into(), arch: "mock".into(), n_ctx_train: 8, vocab: 16, weights_bytes: 1 },
            8,
        );
        let err = b.generate(&params(&"x".repeat(400), 4, vec![]), &mut |_| {}).unwrap_err();
        assert!(matches!(err, BackendError::ContextOverflow { .. }));
    }
}
