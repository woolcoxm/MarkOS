//! Shared engine context: persisted store, metrics, sessions, model manager,
//! guardrail-checked model loading, and download tracking.

use crate::config::{EngineConfig, ModelConfig, Store};
use crate::gguf::GgufMeta;
use crate::guard;
use crate::metrics::Metrics;
use crate::models::{Manager, SlotSummary};
use crate::sysinfo;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, RwLock};

pub struct EngineCtx {
    pub data_dir: PathBuf,
    pub store: RwLock<Store>,
    pub metrics: Metrics,
    pub sessions: crate::auth::Sessions,
    pub manager: Manager,
    /// model id -> active download status
    pub downloads: RwLock<std::collections::BTreeMap<String, DownloadStatus>>,
    pub started_at: std::time::Instant,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct DownloadStatus {
    pub url: String,
    pub state: String, // "running" | "done" | "error"
    pub error: Option<String>,
    pub file: String,
}

impl EngineCtx {
    pub fn open(data_dir: &Path) -> Result<Arc<EngineCtx>, String> {
        let (store_path, models_dir, logs_dir) = crate::config::data_paths(data_dir);
        let mut store = if store_path.exists() {
            Store::load(&store_path)?
        } else {
            Store::default()
        };
        // Paths default relative to the data dir when unset.
        if store.engine.models_dir.is_empty() || store.engine.models_dir == "models" {
            store.engine.models_dir = models_dir.to_string_lossy().into_owned();
        }
        if store.engine.log_dir.is_empty() || store.engine.log_dir == "logs" {
            store.engine.log_dir = logs_dir.to_string_lossy().into_owned();
        }
        store.save(&store_path)?;
        std::fs::create_dir_all(&store.engine.models_dir).map_err(|e| e.to_string())?;
        std::fs::create_dir_all(&store.engine.log_dir).map_err(|e| e.to_string())?;
        let ctx = Arc::new(EngineCtx {
            data_dir: data_dir.to_path_buf(),
            store: RwLock::new(store),
            metrics: Metrics::new(&logs_dir),
            sessions: crate::auth::Sessions::new(),
            manager: Manager::new(),
            downloads: RwLock::new(std::collections::BTreeMap::new()),
            started_at: std::time::Instant::now(),
        });
        ctx.auto_load();
        Ok(ctx)
    }

    pub fn persist(&self) -> Result<(), String> {
        let (store_path, _, _) = crate::config::data_paths(&self.data_dir);
        self.store.read().unwrap().save(&store_path)
    }

    pub fn engine_config(&self) -> EngineConfig {
        self.store.read().unwrap().engine.clone()
    }

    pub fn model_config(&self, id: &str) -> Option<ModelConfig> {
        self.store
            .read()
            .unwrap()
            .models
            .iter()
            .find(|m| m.id == id)
            .cloned()
    }

    pub fn model_path(&self, cfg: &ModelConfig) -> PathBuf {
        Path::new(&self.engine_config().models_dir).join(&cfg.file)
    }

    /// Guardrail check + backend construction for one model config.
    /// Returns (backend, ram estimate total).
    pub fn load_backend(
        self: &Arc<Self>,
        cfg: &ModelConfig,
    ) -> Result<(Box<dyn crate::backend::Backend>, guard::RamEstimate), crate::models::LoadError> {
        let path = self.model_path(cfg);
        if !path.exists() {
            return Err(crate::models::LoadError::Other(format!(
                "model file {} not found (download or remove this model)",
                cfg.file
            )));
        }
        let meta = GgufMeta::from_file(&path)
            .map_err(|e| crate::models::LoadError::Other(format!("{}: {e}", cfg.file)))?;
        let shape = meta
            .shape()
            .ok_or_else(|| crate::models::LoadError::Other(format!("{}: no architecture metadata in GGUF", cfg.file)))?;
        let _ec = self.engine_config();
        let n_ctx = cfg.n_ctx.clamp(1, shape.n_ctx_train);
        let est = guard::estimate_ram(&shape, meta.file_size, n_ctx, cfg.n_batch, cfg.kv_quant);
        let others = self.manager.resident_bytes_excluding(&cfg.id);
        let available = sysinfo::usable_ram().saturating_sub(others);
        if est.total_bytes > available {
            return Err(crate::models::LoadError::Guardrail(guard::GuardrailRejection {
                reason: format!(
                    "{} + {} ctx would use {} (weights {}, kv {}, compute {} + margin), but only {} is free after reserving the OS and resident models",
                    cfg.id,
                    n_ctx,
                    guard::format_bytes(est.total_bytes),
                    guard::format_bytes(est.weights_bytes),
                    guard::format_bytes(est.kv_cache_bytes),
                    guard::format_bytes(est.compute_bytes),
                    guard::format_bytes(available)
                ),
                required_bytes: est.total_bytes,
                available_bytes: available,
            }));
        }
        let info = crate::backend::BackendInfo {
            id: cfg.id.clone(),
            arch: shape.arch.clone(),
            n_ctx_train: shape.n_ctx_train,
            vocab: shape.vocab,
            weights_bytes: meta.file_size,
        };
        sysinfo::refresh();
        let backend = crate::backend::open_gguf(&path, n_ctx, cfg.n_batch, cfg.threads.max(1), cfg.kv_quant, info.clone(), meta)
            .map_err(crate::models::LoadError::Other)?;
        Ok((backend, est))
    }

    fn auto_load(self: &Arc<Self>) {
        // Non-fatal at boot: a guardrail refusal here is logged, the box still
        // serves its web UI (that's the whole point of not crashing).
        let ids: Vec<String> = {
            let store = self.store.read().unwrap();
            store
                .models
                .iter()
                .filter(|m| m.auto_load || store.engine.auto_load.iter().any(|a| a == &m.id))
                .map(|m| m.id.clone())
                .collect()
        };
        for id in ids {
            let Some(cfg) = self.model_config(&id) else { continue };
            let this = self.clone();
            std::thread::Builder::new()
                .name(format!("autoload-{id}"))
                .spawn(move || {
                    let ec = this.engine_config();
                    let mut loader = |c: &ModelConfig| this.load_backend(c).map(|(b, _)| b);
                    match this.manager.acquire(&id, &ec, &mut loader, &cfg) {
                        Ok(_g) => {
                            // keep the slot resident (guard drops but slot stays)
                            this.metrics.record(crate::metrics::LogEntry {
                                ts: Metrics::now_ms(),
                                kind: "model".into(),
                                model: Some(id.clone()),
                                peer: None,
                                path: None,
                                status: 200,
                                prompt_tokens: None,
                                gen_tokens: None,
                                ms: None,
                                tokps: None,
                                message: Some("auto-load complete".into()),
                            });
                        }
                        Err(e) => {
                            this.metrics.record(crate::metrics::LogEntry {
                                ts: Metrics::now_ms(),
                                kind: "error".into(),
                                model: Some(id.clone()),
                                peer: None,
                                path: None,
                                status: 507,
                                prompt_tokens: None,
                                gen_tokens: None,
                                ms: None,
                                tokps: None,
                                message: Some(format!("auto-load failed: {e:?}")),
                            });
                        }
                    }
                })
                .ok();
        }
    }

    /// resident slots + guardrail estimate for the UI in one shot.
    pub fn inventory(&self, id: &str, n_ctx: Option<u64>) -> Result<serde_json::Value, String> {
        let cfg = self.model_config(id).ok_or_else(|| format!("unknown model {id}"))?;
        let path = self.model_path(&cfg);
        let meta = GgufMeta::from_file(&path)?;
        let shape = meta.shape().ok_or("no architecture metadata")?;
        let n_ctx = n_ctx.unwrap_or(cfg.n_ctx).clamp(1, shape.n_ctx_train);
        let est = guard::estimate_ram(&shape, meta.file_size, n_ctx, cfg.n_batch, cfg.kv_quant);
        let others = self.manager.resident_bytes_excluding(id);
        let available = sysinfo::usable_ram().saturating_sub(others);
        Ok(serde_json::json!({
            "model": id,
            "n_ctx": n_ctx,
            "n_ctx_train": shape.n_ctx_train,
            "quant": meta.quant_label(),
            "chat_template_embedded": meta.chat_template().is_some(),
            "estimate": est,
            "available_bytes": available,
            "fits": est.total_bytes <= available,
            "resident": self.manager.resident_ids().contains(&id.to_string()),
        }))
    }

    pub fn slots(&self) -> Vec<SlotSummary> {
        self.manager.slot_summaries()
    }

    pub fn update_resident_metrics(&self) {
        let ids = self.manager.resident_ids();
        self.metrics.models_loaded.store(ids.len() as u64, Ordering::Relaxed);
        let mut bytes = 0u64;
        for id in &ids {
            if let Some(cfg) = self.model_config(id) {
                if let Ok(meta) = GgufMeta::from_file(&self.model_path(&cfg)) {
                    bytes += meta.file_size;
                }
            }
        }
        self.metrics.resident_bytes.store(bytes, Ordering::Relaxed);
    }
}
