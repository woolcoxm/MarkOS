//! Persisted configuration model: engine config, per-model runtime config,
//! sampling profiles, users. Stored as JSON under the data dir
//! (`/var/lib/markos` on MarkOS, `./data` in host dev), written atomically.

use crate::guard::KvQuant;
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// Inference API bind, e.g. "0.0.0.0:8080".
    pub api_bind: String,
    /// Web UI / control-plane binds. The OS adds the IPv4LL rescue bind
    /// "0.0.0.0:4444" so a direct-cable connection always reaches the UI.
    pub ui_binds: Vec<String>,
    pub tls_enabled: bool,
    pub tls_cert_path: Option<String>,
    pub tls_key_path: Option<String>,
    /// Require `Authorization: Bearer <key>` on /v1/* (key itself is stored
    /// hashed; set via admin API).
    pub api_key_required: bool,
    /// sha256 hex of the API key (never the key itself).
    pub api_key_hash: Option<String>,
    /// Optional allowlist of CIDRs/IPv4s; empty = allow all.
    pub ip_allowlist: Vec<String>,
    /// Hostname advertised over mDNS (informational; OS owns avahi).
    pub mdns_name: String,
}

impl Default for ServerConfig {
    fn default() -> Self {
        ServerConfig {
            api_bind: "0.0.0.0:8080".into(),
            ui_binds: vec!["0.0.0.0:80".into(), "0.0.0.0:4444".into()],
            tls_enabled: true, // self-signed cert generated on first boot (engine/src/tls.rs)
            tls_cert_path: None,
            tls_key_path: None,
            api_key_required: false,
            api_key_hash: None,
            ip_allowlist: Vec::new(),
            mdns_name: "pi-inference".into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineConfig {
    pub server: ServerConfig,
    pub models_dir: String,
    pub log_dir: String,
    /// Max models resident in RAM simultaneously (design §8.4: default 2).
    pub max_resident: usize,
    /// Request queue depth per model before 429.
    pub queue_depth: usize,
    /// How long a request may wait in queue before 504.
    pub queue_timeout_s: u64,
    /// Default generation threads (Pi 5: 4).
    pub threads: usize,
    /// Default prompt batch size.
    pub n_batch: usize,
    /// KV cache quantization default.
    pub kv_quant_default: KvQuant,
    /// Models to load at startup, in order, subject to guardrails.
    pub auto_load: Vec<String>,
    /// Update manifest URL (manual check only — nothing phones home).
    pub update_manifest_url: Option<String>,
    pub update_channel: String, // "stable" | "manual"
}

impl Default for EngineConfig {
    fn default() -> Self {
        EngineConfig {
            server: ServerConfig::default(),
            models_dir: "models".into(),
            log_dir: "logs".into(),
            max_resident: 2,
            queue_depth: 4,
            queue_timeout_s: 300,
            threads: 4,
            n_batch: 512,
            kv_quant_default: KvQuant::F16,
            auto_load: Vec::new(),
            update_manifest_url: None,
            update_channel: "stable".into(),
        }
    }
}

/// Named sampling profile — the "creative"/"precise"/"coding" switch the UI
/// exposes per model. These are the defaults the engine itself falls back to.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default)]
pub struct SamplingParams {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: i32,
    pub min_p: f64,
    pub repeat_penalty: f64,
    pub repeat_last_n: i32,
    pub presence_penalty: f64,
    pub frequency_penalty: f64,
    pub mirostat: i32, // 0 off, 1 v1, 2 v2
    pub mirostat_tau: f64,
    pub mirostat_eta: f64,
    pub seed: u32,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 0.8,
            top_p: 0.95,
            top_k: 40,
            min_p: 0.05,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            mirostat: 0,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            seed: 0, // 0 = random
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Profile {
    pub name: String,
    pub description: String,
    pub sampling: SamplingParams,
    pub max_tokens: u32,
    pub stop: Vec<String>,
    /// Optional per-profile system prompt override.
    pub system_prompt: Option<String>,
}

impl Profile {
    pub fn creative() -> Profile {
        Profile {
            name: "creative".into(),
            description: "Higher temperature, broader sampling".into(),
            sampling: SamplingParams { temperature: 1.0, top_p: 0.98, min_p: 0.02, repeat_penalty: 1.05, ..Default::default() },
            max_tokens: 1024,
            stop: vec![],
            system_prompt: None,
        }
    }
    pub fn precise() -> Profile {
        Profile {
            name: "precise".into(),
            description: "Low temperature, tight sampling for factual work".into(),
            sampling: SamplingParams { temperature: 0.2, top_p: 0.9, top_k: 20, min_p: 0.1, ..Default::default() },
            max_tokens: 1024,
            stop: vec![],
            system_prompt: None,
        }
    }
    pub fn coding() -> Profile {
        Profile {
            name: "coding".into(),
            description: "Deterministic, code-friendly stops".into(),
            sampling: SamplingParams { temperature: 0.1, top_p: 0.85, min_p: 0.1, repeat_penalty: 1.05, ..Default::default() },
            max_tokens: 2048,
            stop: vec!["```".into()],
            system_prompt: None,
        }
    }
}

impl Default for Profile {
    fn default() -> Self {
        Profile {
            name: "default".into(),
            description: "Balanced defaults".into(),
            sampling: SamplingParams::default(),
            max_tokens: 512,
            stop: vec![],
            system_prompt: None,
        }
    }
}

/// Per-model runtime configuration. `n_ctx` is the context actually allocated
/// (clamped to the model's training context by the manager).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub id: String,
    pub name: String,
    /// File name (within models_dir) of the GGUF.
    pub file: String,
    pub n_ctx: u64,
    pub n_batch: u64,
    pub threads: usize,
    pub kv_quant: KvQuant,
    /// Custom Jinja2-style chat template (minijinja). None = use the GGUF's
    /// embedded template, falling back to a plain-text renderer.
    pub chat_template: Option<String>,
    pub system_prompt: Option<String>,
    pub profiles: Vec<Profile>,
    pub default_profile: String,
    pub auto_load: bool,
    /// HF repo this came from, if any (display + re-download).
    pub source: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            id: String::new(),
            name: String::new(),
            file: String::new(),
            n_ctx: 4096,
            n_batch: 512,
            threads: 4,
            kv_quant: KvQuant::F16,
            chat_template: None,
            system_prompt: None,
            profiles: vec![Profile::default(), Profile::creative(), Profile::precise(), Profile::coding()],
            default_profile: "default".into(),
            auto_load: false,
            source: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum Role {
    Admin,
    Viewer,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub username: String,
    /// argon2id PHC string.
    pub pwhash: String,
    pub role: Role,
}

#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct UserDb {
    pub users: Vec<User>,
}

/// Root persisted document (single file, atomic replace).
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Store {
    pub engine: EngineConfig,
    pub models: Vec<ModelConfig>,
    pub users: UserDb,
}

impl Store {
    pub fn load(path: &Path) -> Result<Store, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let mut store: Store = serde_json::from_str(&text).map_err(|e| format!("parse {}: {e}", path.display()))?;
        if store.engine.max_resident == 0 {
            store.engine.max_resident = 1;
        }
        if store.engine.threads == 0 {
            store.engine.threads = 4;
        }
        Ok(store)
    }

    pub fn save(&self, path: &Path) -> Result<(), String> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let tmp = path.with_extension("json.tmp");
        {
            let f = std::fs::File::create(&tmp).map_err(|e| e.to_string())?;
            let mut w = std::io::BufWriter::new(f);
            serde_json::to_writer_pretty(&mut w, self).map_err(|e| e.to_string())?;
            w.flush().map_err(|e| e.to_string())?;
        }
        // Atomic on POSIX; best-effort on Windows dev hosts.
        std::fs::rename(&tmp, path).map_err(|e| e.to_string())?;
        Ok(())
    }
}

pub fn data_paths(data_dir: &Path) -> (PathBuf, PathBuf, PathBuf) {
    (
        data_dir.join("markos.json"),
        data_dir.join("models"),
        data_dir.join("logs"),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_roundtrip_with_defaults() {
        let mut store = Store::default();
        store.models.push(ModelConfig {
            id: "m1".into(),
            file: "m1.gguf".into(),
            n_ctx: 8192,
            ..Default::default()
        });
        let dir = std::env::temp_dir().join(format!("markos-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("markos.json");
        store.save(&p).unwrap();
        let back = Store::load(&p).unwrap();
        assert_eq!(back.models.len(), 1);
        assert_eq!(back.models[0].n_ctx, 8192);
        assert_eq!(back.engine.server.api_bind, "0.0.0.0:8080");
        // New fields added later must deserialize as defaults.
        assert_eq!(back.models[0].profiles.len(), 4);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn sampling_defaults_stable() {
        let s = SamplingParams::default();
        assert_eq!(s.mirostat, 0);
        assert_eq!(s.seed, 0);
    }
}
