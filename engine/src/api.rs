//! OpenAI-compatible inference API: POST /v1/chat/completions,
//! POST /v1/completions, GET /v1/models (+ /v1/models/{id}). Streaming via
//! SSE. This is the integration surface for everything else on the network.

use crate::backend::GenParams;
use crate::config::{ModelConfig, Profile, SamplingParams};
use crate::http::{sse_event, Request, ResponseOut};
use crate::metrics::{LogEntry, Metrics};
use crate::models::AcquireError;
use crate::state::EngineCtx;
use crate::templates::{render_chat, ChatMessage, TemplateInput};
use serde::Deserialize;
use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Instant;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    // Sampling overrides (all optional; profile/engine defaults otherwise)
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i32>,
    pub min_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub stop: Option<serde_json::Value>,
    pub presence_penalty: Option<f64>,
    pub frequency_penalty: Option<f64>,
    pub repeat_penalty: Option<f64>,
    pub mirostat: Option<i32>,
    pub mirostat_tau: Option<f64>,
    pub mirostat_eta: Option<f64>,
    pub seed: Option<u32>,
    /// MarkOS extension: named profile to source defaults from.
    pub profile: Option<String>,
}

#[derive(Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: serde_json::Value, // string or [string]
    #[serde(default)]
    pub stream: bool,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i32>,
    pub stop: Option<serde_json::Value>,
    pub seed: Option<u32>,
}

fn stops_from(v: &Option<serde_json::Value>) -> Vec<String> {
    match v {
        Some(serde_json::Value::String(s)) => vec![s.clone()],
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|x| x.as_str().map(|s| s.to_string()))
            .collect(),
        _ => Vec::new(),
    }
}

fn resolve_sampling(profile: Option<&Profile>, req: &ChatRequest) -> (SamplingParams, u32, Vec<String>) {
    let base: SamplingParams = profile.map(|p| p.sampling.clone()).unwrap_or_default();
    let max_tokens = req
        .max_tokens
        .or(req.max_completion_tokens)
        .or(profile.map(|p| p.max_tokens))
        .unwrap_or(512);
    let mut stop = profile.map(|p| p.stop.clone()).unwrap_or_default();
    let req_stops = stops_from(&req.stop);
    if !req_stops.is_empty() {
        stop = req_stops;
    }
    let mut s = base;
    if let Some(v) = req.temperature { s.temperature = v; }
    if let Some(v) = req.top_p { s.top_p = v; }
    if let Some(v) = req.top_k { s.top_k = v; }
    if let Some(v) = req.min_p { s.min_p = v; }
    if let Some(v) = req.repeat_penalty { s.repeat_penalty = v; }
    if let Some(v) = req.presence_penalty { s.presence_penalty = v; }
    if let Some(v) = req.frequency_penalty { s.frequency_penalty = v; }
    if let Some(v) = req.mirostat { s.mirostat = v; }
    if let Some(v) = req.mirostat_tau { s.mirostat_tau = v; }
    if let Some(v) = req.mirostat_eta { s.mirostat_eta = v; }
    if let Some(v) = req.seed { s.seed = v; }
    (s, max_tokens, stop)
}

pub struct Resolved {
    pub cfg: ModelConfig,
    pub profile_name: String,
    pub profile: Option<Profile>,
    pub gguf_template: Option<String>,
}

pub fn resolve_model(ctx: &EngineCtx, want: Option<&str>, profile: Option<&str>) -> Result<Resolved, (u16, String)> {
    let store = ctx.store.read().unwrap();
    let cfg = match want {
        Some(w) => store
            .models
            .iter()
            .find(|m| m.id == w || m.name == w || m.file == w)
            .cloned()
            .ok_or_else(|| {
                (
                    404,
                    format!("model '{w}' is not configured; see GET /v1/models"),
                )
            })?,
        None => store
            .models
            .first()
            .cloned()
            .ok_or((503, "no models are configured on this appliance".to_string()))?,
    };
    let pname = profile.unwrap_or(&cfg.default_profile).to_string();
    let prof = cfg.profiles.iter().find(|p| p.name == pname).cloned();
    drop(store);
    let gguf_template = ctx
        .model_config(&cfg.id)
        .map(|c| ctx.model_path(&c))
        .and_then(|p| crate::gguf::GgufMeta::from_file(&p).ok())
        .and_then(|m| m.chat_template());
    Ok(Resolved { cfg, profile_name: pname, profile: prof, gguf_template })
}

/// GET /v1/models
pub fn list_models(ctx: &EngineCtx, out: &mut ResponseOut) {
    let store = ctx.store.read().unwrap();
    let resident = ctx.manager.resident_ids();
    let mut list: Vec<serde_json::Value> = Vec::new();
    for m in &store.models {
        let path = Path::new(&store.engine.models_dir).join(&m.file);
        let size = std::fs::metadata(&path).map(|x| x.len()).unwrap_or(0);
        list.push(serde_json::json!({
            "id": m.id,
            "object": "model",
            "owned_by": "markos",
            "created": 0,
            "markos": {
                "resident": resident.contains(&m.id),
                "quant": crate::gguf::GgufMeta::from_file(&path).map(|g| g.quant_label()).unwrap_or_else(|_| "unknown".into()),
                "size_bytes": size,
            }
        }));
    }
    drop(store);
    out.json(200, &serde_json::json!({ "object": "list", "data": list }));
}

/// Generate + stream/collect, shared by chat and completions.
fn run_generation(
    ctx: &Arc<EngineCtx>,
    resolved: &Resolved,
    sampling: SamplingParams,
    max_tokens: u32,
    stop: Vec<String>,
    prompt: String,
    stream: bool,
    out: &mut ResponseOut,
    path: &str,
) {
    let ec = ctx.engine_config();
    ctx.metrics.queued.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let queued_at = Instant::now();
    let model_id = resolved.cfg.id.clone();

    let cancel = Arc::new(AtomicBool::new(false));
    let mut loader = |c: &ModelConfig| ctx.load_backend(c).map(|(b, _)| b);
    let guard = match ctx.manager.acquire(&model_id, &ec, &mut loader, &resolved.cfg) {
        Ok(g) => g,
        Err(e) => {
            ctx.metrics.queued.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            match e {
                AcquireError::QueueFull => {
                    out.add_header("Retry-After", "2");
                    out.error(429, "request queue is full for this model; retry shortly");
                }
                AcquireError::Timeout => {
                    out.error(504, "timed out waiting for a free generation slot");
                }
                AcquireError::LoadFailed(le) => {
                    let msg = le.to_string();
                    let code = if msg.contains("memory guardrail") { 409 } else { 502 };
                    out.error(code, &msg);
                }
            }
            return;
        }
    };
    ctx.metrics.queued.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    ctx.metrics.active.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let wait_ms = queued_at.elapsed().as_millis() as u64;

    // Stream or buffer.
    let started = Instant::now();
    let id = format!("chatcmpl-{:x}", crate::metrics::Metrics::now_ms());
    let created = crate::metrics::Metrics::now_ms() / 1000;
    let mut first = true;

    let result = guard.with_backend(|be| {
        let gen = GenParams { prompt, max_tokens, stop: stop.clone(), sampling, cancel: cancel.clone() };
        let mut collected = String::new();
        let on_token = &mut |piece: &str| {
            collected.push_str(piece);
            ctx.metrics.note_token();
            if stream {
                if first {
                    first = false;
                    let chunk = serde_json::json!({
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]
                    });
                    let _ = sse_event(out, "", &chunk.to_string());
                }
                let chunk = serde_json::json!({
                    "id": id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                    "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": null}]
                });
                let _ = sse_event(out, "", &chunk.to_string());
            }
        };
        be.generate(&gen, on_token)
    });

    ctx.metrics.active.fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
    drop(guard);
    ctx.update_resident_metrics();

    let ms = started.elapsed().as_millis() as u64;
    let finish = |out: &mut ResponseOut, result: Result<crate::backend::GenStats, crate::backend::BackendError>| {
        match result {
            Ok(stats) => {
                let tokps = if ms > 0 { stats.gen_tokens as f64 / (ms as f64 / 1000.0) } else { 0.0 };
                ctx.metrics.record(LogEntry {
                    ts: Metrics::now_ms(),
                    kind: "request".into(),
                    model: Some(model_id.clone()),
                    peer: None,
                    path: Some(path.to_string()),
                    status: 200,
                    prompt_tokens: Some(stats.prompt_tokens),
                    gen_tokens: Some(stats.gen_tokens),
                    ms: Some(ms),
                    tokps: Some(tokps),
                    message: Some(format!(
                        "profile={} queue_wait_ms={wait_ms} stop={}",
                        resolved.profile_name,
                        stats.stop_reason
                    )),
                });
                if stream {
                    let chunk = serde_json::json!({
                        "id": id, "object": "chat.completion.chunk", "created": created, "model": model_id,
                        "choices": [{"index": 0, "delta": {}, "finish_reason": stats.stop_reason}]
                    });
                    let _ = sse_event(out, "", &chunk.to_string());
                    let _ = sse_event(out, "", "[DONE]");
                } else {
                    out.json(200, &serde_json::json!({
                        "id": id,
                        "object": "chat.completion",
                        "created": created,
                        "model": model_id,
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant", "content": stats.output},
                            "finish_reason": stats.stop_reason
                        }],
                        "usage": {
                            "prompt_tokens": stats.prompt_tokens,
                            "completion_tokens": stats.gen_tokens,
                            "total_tokens": stats.prompt_tokens + stats.gen_tokens
                        }
                    }));
                }
            }
            Err(e) => {
                let (code, msg) = match &e {
                    crate::backend::BackendError::ContextOverflow { .. } => (400, e.to_string()),
                    crate::backend::BackendError::Other(_) => (502, e.to_string()),
                };
                ctx.metrics.record(LogEntry {
                    ts: Metrics::now_ms(),
                    kind: "error".into(),
                    model: Some(model_id.clone()),
                    peer: None,
                    path: Some(path.to_string()),
                    status: code,
                    prompt_tokens: None,
                    gen_tokens: None,
                    ms: Some(ms),
                    tokps: None,
                    message: Some(msg.clone()),
                });
                if out.is_streaming() {
                    // Stream already started; emit the error as an event.
                    let _ = sse_event(out, "error", &serde_json::json!({"message": msg}).to_string());
                    let _ = sse_event(out, "", "[DONE]");
                } else {
                    out.error(code, &msg);
                }
            }
        }
    };
    finish(out, result);
}

pub fn chat_completions(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut) {
    let body: ChatRequest = match req.json() {
        Ok(b) => b,
        Err(e) => {
            out.error(400, &e);
            return;
        }
    };
    let resolved = match resolve_model(ctx, body.model.as_deref(), body.profile.as_deref()) {
        Ok(r) => r,
        Err((code, msg)) => {
            out.error(code, &msg);
            return;
        }
    };
    let (sampling, max_tokens, stop) = resolve_sampling(resolved.profile.as_ref(), &body);
    let system = resolved
        .profile
        .as_ref()
        .and_then(|p| p.system_prompt.clone())
        .or_else(|| resolved.cfg.system_prompt.clone());
    let mut messages = body.messages.clone();
    if let Some(sys) = system {
        if !messages.iter().any(|m| m.role == "system") {
            messages.insert(0, ChatMessage { role: "system".into(), content: sys });
        }
    }
    let prompt = match render_chat(
        resolved.cfg.chat_template.as_deref(),
        resolved.gguf_template.as_deref(),
        &TemplateInput { messages: &messages, add_generation_prompt: true },
    ) {
        Ok(p) => p,
        Err(e) => {
            out.error(400, &e);
            return;
        }
    };
    run_generation(ctx, &resolved, sampling, max_tokens, stop, prompt, body.stream, out, "/v1/chat/completions");
}

pub fn completions(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut) {
    let body: CompletionRequest = match req.json() {
        Ok(b) => b,
        Err(e) => {
            out.error(400, &e);
            return;
        }
    };
    let prompt = match &body.prompt {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(a) => a
            .iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .join("\n"),
        _ => {
            out.error(400, "prompt must be a string or array of strings");
            return;
        }
    };
    let resolved = match resolve_model(ctx, body.model.as_deref(), None) {
        Ok(r) => r,
        Err((code, msg)) => {
            out.error(code, &msg);
            return;
        }
    };
    let profile = resolved.profile.as_ref();
    let base = profile.map(|p| p.sampling.clone()).unwrap_or_default();
    let mut s = base;
    if let Some(v) = body.temperature { s.temperature = v; }
    if let Some(v) = body.top_p { s.top_p = v; }
    if let Some(v) = body.top_k { s.top_k = v; }
    if let Some(v) = body.seed { s.seed = v; }
    let max_tokens = body
        .max_tokens
        .or(profile.map(|p| p.max_tokens))
        .unwrap_or(512);
    let stop = {
        let st = stops_from(&body.stop);
        if st.is_empty() { profile.map(|p| p.stop.clone()).unwrap_or_default() } else { st }
    };
    run_generation(ctx, &resolved, s, max_tokens, stop, prompt, body.stream, out, "/v1/completions");
}
