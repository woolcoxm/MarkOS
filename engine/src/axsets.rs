//! Axera engine sets: manifests describing whole-layer NPU template sets,
//! plus the match logic that decides how a given GGUF will run.
//!
//! An engine set is a directory under the engines root (default
//! `/data/axcl/sets`) containing:
//!   - `set.txt`  — the manifest (key=value lines, see below)
//!   - one layer template `.axmodel` per transformer layer
//!   - a post (final-norm + lm_head) engine
//!   - optionally a `layout_v4.bin` weight sidecar for GGUF patching
//!
//! Manifest keys:
//!   family=qwen3            qwen3 | qwen35-hybrid
//!   pattern=qwen3_p128_l%d_together.axmodel
//!   post=qwen3_post.axmodel
//!   layout=layout_v4.bin
//!   hidden=1024 vocab=151936 layers=28 ctx=2048
//!
//! The serving ladder for ANY gguf (this is the contract the backend
//! implements, mirrored here for reporting):
//!   1. an engine set whose geometry matches the GGUF  -> whole-layer NPU
//!      with GGUF weights patched into the templates (24-30 t/s class)
//!   2. no matching set, but shape-keyed matmul engines present under
//!      `matmul/`  -> per-op NPU matmuls, host attention
//!   3. neither  -> CPU/NEON reference (still correct, just Pi-speed)

use crate::gguf::ModelShape;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct SetManifest {
    pub dir: PathBuf,
    /// 0 = dense attention family, 1 = hybrid (delta-net + attention)
    pub family: u8,
    pub pattern: String,
    pub post: String,
    pub layout: Option<String>,
    pub hidden: Option<u64>,
    pub vocab: Option<u64>,
    pub layers: Option<u64>,
    pub ctx: Option<u64>,
}

/// Parse one `set.txt`. `dir` is the containing directory (used verbatim in
/// errors/reports). Returns None when the manifest is unusable (missing the
/// mandatory pattern/post keys).
pub fn parse_set_txt(dir: PathBuf, text: &str) -> Option<SetManifest> {
    let mut m = SetManifest {
        dir,
        family: 0,
        pattern: String::new(),
        post: String::new(),
        layout: None,
        hidden: None,
        vocab: None,
        layers: None,
        ctx: None,
    };
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((k, v)) = line.split_once('=') else { continue };
        let (k, v) = (k.trim(), v.trim());
        match k {
            "family" => {
                m.family = match v {
                    "qwen35-hybrid" | "hybrid" => 1,
                    _ => 0,
                }
            }
            "pattern" => m.pattern = v.to_string(),
            "post" => m.post = v.to_string(),
            "layout" => m.layout = Some(v.to_string()),
            "hidden" => m.hidden = v.parse().ok(),
            "vocab" => m.vocab = v.parse().ok(),
            "layers" => m.layers = v.parse().ok(),
            "ctx" => m.ctx = v.parse().ok(),
            _ => {}
        }
    }
    if m.pattern.is_empty() || m.post.is_empty() {
        return None;
    }
    Some(m)
}

/// Does `pattern` (with its single %d) name `index`? Used by tooling/tests
/// and by future engine-set install helpers.
#[allow(dead_code)]
pub fn pattern_names(pattern: &str, index: u64) -> String {
    match pattern.find('%') {
        Some(p) => {
            let mut rest = &pattern[p + 1..];
            while let Some(c) = rest.chars().next() {
                if c == 'd' {
                    break;
                }
                rest = &rest[1..];
            }
            let end = pattern.len() - rest.len() + 1; // include the 'd'
            format!("{}{}{}", &pattern[..p], index, &pattern[end..])
        }
        None => pattern.to_string(),
    }
}

/// How a given model will run on this box.
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum AccelMode {
    /// whole-layer engines with GGUF weights patched in
    NpuLayer { set: String },
    /// no matching template set; shape-keyed matmul engines may still
    /// accelerate the projections if installed under the engines root
    PerOp,
    /// CPU/NEON only
    Cpu,
}

/// Decide the serving mode for a model given the available sets.
///
/// Geometry gate (mirrors the fork's `axcl_set_select`):
///   family must match the model arch class, and any manifest value that is
///   present must equal the GGUF's. Vocab is advisory (trimmed heads exist)
///   EXCEPT when the manifest pins it.
pub fn match_mode(shape: &ModelShape, sets: &[SetManifest]) -> AccelMode {
    let hybrid = shape_is_hybrid(shape);
    for s in sets {
        if s.family != u8::from(hybrid) {
            continue;
        }
        if let Some(h) = s.hidden {
            if h != shape.n_embd {
                continue;
            }
        }
        if let Some(v) = s.vocab {
            if v != shape.vocab {
                continue;
            }
        }
        if let Some(l) = s.layers {
            if l != shape.n_layers {
                continue;
            }
        }
        // layer-file existence is re-checked by the fork at engine load;
        // the Rust gate is advisory (UI reporting)
        return AccelMode::NpuLayer {
            set: s.dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
        };
    }
    AccelMode::PerOp
}

fn shape_is_hybrid(shape: &ModelShape) -> bool {
    // hybrid archs carry SSM/conv state tensors; the fork detects them in
    // the live graph — the Rust gate approximates by the known arch names
    matches!(shape.arch.as_str(), "qwen35" | "qwen3.5" | "gdn")
}

/// Scan a sets root: one level of directories, each with set.txt.
pub fn scan(root: &Path) -> Vec<SetManifest> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(root) else {
        return out;
    };
    for ent in rd.flatten() {
        let dir = ent.path();
        let Some(txt) = std::fs::read_to_string(dir.join("set.txt")).ok() else {
            continue;
        };
        if let Some(m) = parse_set_txt(dir, &txt) {
            out.push(m);
        }
    }
    out.sort_by(|a, b| a.dir.cmp(&b.dir));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(arch: &str, n_embd: u64, n_layers: u64, vocab: u64) -> ModelShape {
        ModelShape {
            arch: arch.into(),
            n_layers,
            n_embd,
            n_head: 16,
            n_head_kv: 8,
            head_dim: 64,
            n_ctx_train: 32768,
            vocab,
        }
    }

    const QWEN3_SET: &str = "family=qwen3\npattern=qwen3_p128_l%d_together.axmodel\npost=qwen3_post.axmodel\nlayout=layout_v4.bin\nhidden=1024\nvocab=151936\nlayers=28\nctx=2048\n";

    #[test]
    fn parse_full_manifest() {
        let m = parse_set_txt(PathBuf::from("/sets/q3"), QWEN3_SET).unwrap();
        assert_eq!(m.family, 0);
        assert_eq!(m.pattern, "qwen3_p128_l%d_together.axmodel");
        assert_eq!(m.hidden, Some(1024));
        assert_eq!(m.vocab, Some(151936));
        assert_eq!(m.layers, Some(28));
        assert_eq!(m.ctx, Some(2048));
        assert_eq!(m.layout.as_deref(), Some("layout_v4.bin"));
    }

    #[test]
    fn parse_rejects_incomplete() {
        assert!(parse_set_txt(PathBuf::from("/x"), "family=qwen3\n").is_none());
        assert!(parse_set_txt(PathBuf::from("/x"), "").is_none());
    }

    #[test]
    fn pattern_names_splices_index() {
        assert_eq!(
            pattern_names("qwen3_p128_l%d_together.axmodel", 7),
            "qwen3_p128_l7_together.axmodel"
        );
        assert_eq!(pattern_names("layer_%d.axmodel", 0), "layer_0.axmodel");
        // no %d: returned verbatim
        assert_eq!(pattern_names("fixed.axmodel", 3), "fixed.axmodel");
    }

    #[test]
    fn exact_geometry_match_selects_layer_mode() {
        let m = parse_set_txt(PathBuf::from("/sets/q3"), QWEN3_SET).unwrap();
        let mode = match_mode(&shape("qwen3", 1024, 28, 151936), &[m.clone()]);
        assert!(matches!(mode, AccelMode::NpuLayer { ref set } if set == "q3"));

        // different hidden width -> no whole-layer
        let mode = match_mode(&shape("qwen3", 2048, 28, 151936), &[m.clone()]);
        assert_eq!(mode, AccelMode::PerOp);

        // different layer count -> no whole-layer
        let mode = match_mode(&shape("qwen3", 1024, 36, 151936), &[m.clone()]);
        assert_eq!(mode, AccelMode::PerOp);

        // vocab-trimmed GGUF head vs full manifest vocab: manifest pins it
        let mode = match_mode(&shape("qwen3", 1024, 28, 248320), &[m]);
        assert_eq!(mode, AccelMode::PerOp);
    }

    #[test]
    fn partial_manifests_match_loosely() {
        // hidden-only manifest: any vocab/layer count with that hidden width
        let m = parse_set_txt(
            PathBuf::from("/sets/h1024"),
            "family=qwen3\npattern=qwen3_p128_l%d_together.axmodel\npost=qwen3_post.axmodel\nhidden=1024\n",
        )
        .unwrap();
        let mode = match_mode(&shape("qwen3", 1024, 28, 151936), &[m]);
        assert!(matches!(mode, AccelMode::NpuLayer { .. }));
    }

    #[test]
    fn other_arches_fall_off_the_ladder_cleanly() {
        let m = parse_set_txt(PathBuf::from("/sets/q3"), QWEN3_SET).unwrap();
        // llama-1B: family mismatch (hybrid flag aside, hidden/layer gates)
        let mode = match_mode(&shape("llama", 2048, 16, 128256), &[m]);
        assert_eq!(mode, AccelMode::PerOp);
        // no sets at all
        let mode = match_mode(&shape("llama", 2048, 16, 128256), &[]);
        assert_eq!(mode, AccelMode::PerOp);
    }

    #[test]
    fn hybrid_family_gate() {
        let m = parse_set_txt(
            PathBuf::from("/sets/q35"),
            "family=qwen35-hybrid\npattern=qwen3_5_text_p128_l%d_together.axmodel\npost=qwen3_5_text_post.axmodel\nhidden=1024\n",
        )
        .unwrap();
        let mode = match_mode(&shape("qwen3.5", 1024, 24, 248320), &[m.clone()]);
        assert!(matches!(mode, AccelMode::NpuLayer { .. }));
        let mode = match_mode(&shape("qwen3", 1024, 28, 151936), &[m]);
        assert_eq!(mode, AccelMode::PerOp);
    }
}
