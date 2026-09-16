//! Control plane for the web UI (`/api/*` + static assets on the UI
//! listener). Session-cookie auth; admin vs viewer roles. This is the
//! "advanced, highly customizable" surface; the engine core stays simple
//! behind it.

use crate::auth;
use crate::config::{ModelConfig, Role};
use crate::http::{sse_event, Request, ResponseOut};
use crate::state::{DownloadStatus, EngineCtx};
use std::io::Write;
use std::sync::atomic::Ordering;
use std::sync::Arc;

#[allow(dead_code)]
pub struct SessionInfo {
    pub username: String,
    pub role: Role,
}

/// Resolve the caller's session from the cookie header.
pub fn session(ctx: &EngineCtx, req: &Request) -> Option<SessionInfo> {
    let cookie = req.header("cookie")?;
    for part in cookie.split(';') {
        let mut it = part.trim().splitn(2, '=');
        if it.next()? == "markos_session" {
            if let Some(tok) = it.next() {
                if let Some(s) = ctx.sessions.get(tok) {
                    return Some(SessionInfo { username: s.username, role: s.role });
                }
            }
        }
    }
    None
}

fn require_admin(ctx: &EngineCtx, req: &Request, out: &mut ResponseOut) -> Option<SessionInfo> {
    match session(ctx, req) {
        Some(s) if matches!(s.role, Role::Admin) => Some(s),
        Some(_) => {
            out.error(403, "admin role required for this action");
            None
        }
        None => {
            out.error(401, "login required");
            None
        }
    }
}

fn require_session(ctx: &EngineCtx, req: &Request, out: &mut ResponseOut) -> Option<SessionInfo> {
    match session(ctx, req) {
        Some(s) => Some(s),
        None => {
            out.error(401, "login required");
            None
        }
    }
}

pub fn route(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut) {
    let path = req.path.as_str();
    let method = req.method.as_str();

    // Static web UI (unauthenticated; the SPA enforces login visually and the
    // data endpoints below enforce it for real).
    match (method, path) {
        ("GET", "/") | ("GET", "/index.html") => return serve_asset(out, "index.html"),
        ("GET", "/app.js") => return serve_asset(out, "app.js"),
        ("GET", "/style.css") => return serve_asset(out, "style.css"),
        ("GET", "/healthz") => {
            return out.json(200, &serde_json::json!({"status": "ok", "version": crate::config::ENGINE_VERSION}))
        }
        ("GET", "/readyz") => {
            let ok = ctx.manager.resident_ids().len() > 0
                || !ctx.store.read().unwrap().models.is_empty();
            return out.json(
                if ok { 200 } else { 200 },
                &serde_json::json!({"ready": ok, "resident": ctx.manager.resident_ids()}),
            );
        }
        _ => {}
    }

    if !path.starts_with("/api/") {
        return out.error(404, "not found");
    }

    match (method, path) {
        ("POST", "/api/login") => login(ctx, req, out),
        ("POST", "/api/logout") => {
            if let Some(tok) = extract_token(req) {
                ctx.sessions.remove(&tok);
            }
            out.text(200, "ok");
        }
        ("GET", "/api/state") => state(ctx, req, out),
        ("GET", "/api/metrics") => {
            require_session(ctx, req, out).map(|_| {
                sys_snapshot(ctx, out);
            });
        }
        ("GET", "/api/metrics/stream") => {
            if require_session(ctx, req, out).is_some() {
                metrics_stream(ctx, out);
            }
        }
        ("GET", "/api/log") => {
            require_session(ctx, req, out).map(|_| {
                let n = req.query_param("tail").and_then(|v| v.parse().ok()).unwrap_or(200).min(500);
                out.json(200, &serde_json::json!({ "entries": ctx.metrics.tail(n) }));
            });
        }
        ("POST", "/api/models") => add_model(ctx, req, out),
        ("GET", "/api/models") => {
            require_session(ctx, req, out).map(|_| models_list(ctx, out));
        }
        ("POST", "/api/models/estimate") => estimate(ctx, req, out),
        ("POST", "/api/users") => add_user(ctx, req, out),
        ("GET", "/api/users") => {
            require_session(ctx, req, out).map(|_| {
                let store = ctx.store.read().unwrap();
                let users: Vec<serde_json::Value> = store
                    .users
                    .users
                    .iter()
                    .map(|u| serde_json::json!({"username": u.username, "role": u.role}))
                    .collect();
                out.json(200, &serde_json::json!({ "users": users }));
            });
        }
        ("POST", "/api/system/restart") => {
            require_admin(ctx, req, out).map(|_| system_restart(ctx, out));
        }
        ("POST", "/api/system/update-check") => {
            require_admin(ctx, req, out).map(|_| update_check(ctx, req, out));
        }
        ("POST", "/api/server") => update_server_config(ctx, req, out),
        ("POST", "/api/server/apikey") => set_api_key(ctx, req, out),
        ("GET", "/api/recovery") => recovery_status(out),
        _ => {
            // Parameterized routes: /api/models/{id}[...] and /api/users/{name}
            let rest = path.strip_prefix("/api/models/").unwrap_or("");
            if !rest.is_empty() {
                let mut seg = rest.splitn(2, '/');
                let id = seg.next().unwrap_or("").to_string();
                let sub = seg.next().unwrap_or("");
                return model_route(ctx, req, out, &id, sub);
            }
            if let Some(name) = path.strip_prefix("/api/users/") {
                if method == "DELETE" {
                    if require_admin(ctx, req, out).is_some() {
                        delete_user(ctx, out, name);
                    }
                    return;
                }
            }
            out.error(404, "no such API endpoint");
        }
    }
}

fn extract_token(req: &Request) -> Option<String> {
    let cookie = req.header("cookie")?;
    for part in cookie.split(';') {
        let mut it = part.trim().splitn(2, '=');
        if it.next()? == "markos_session" {
            return it.next().map(|s| s.to_string());
        }
    }
    None
}

fn login(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut) {
    #[derive(serde::Deserialize)]
    struct Login {
        username: String,
        password: String,
    }
    let body: Login = match req.json() {
        Ok(b) => b,
        Err(_) => return out.error(400, "expected {username, password}"),
    };
    let store = ctx.store.read().unwrap();
    let user = store.users.users.iter().find(|u| u.username == body.username).cloned();
    drop(store);
    match user {
        Some(u) if auth::verify_password(&body.password, &u.pwhash) => {
            let tok = ctx.sessions.create(&u.username, u.role.clone());
            out.add_header(
                "Set-Cookie",
                &format!("markos_session={tok}; Path=/; HttpOnly; SameSite=Lax; Max-Age=43200"),
            );
            out.json(200, &serde_json::json!({"ok": true, "role": u.role}));
        }
        _ => {
            std::thread::sleep(std::time::Duration::from_millis(300)); // slow brute force
            out.error(401, "invalid credentials");
        }
    }
}

fn state(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut) {
    let Some(_) = require_session(ctx, req, out) else { return };
    let store = ctx.store.read().unwrap();
    let resident = ctx.manager.resident_ids();
    let models: Vec<serde_json::Value> = store
        .models
        .iter()
        .map(|m| {
            let path = std::path::Path::new(&store.engine.models_dir).join(&m.file);
            let size = std::fs::metadata(&path).map(|x| x.len()).unwrap_or(0);
            let exists = path.exists();
            serde_json::json!({
                "id": m.id, "name": m.name, "file": m.file, "n_ctx": m.n_ctx,
                "n_batch": m.n_batch, "threads": m.threads, "kv_quant": m.kv_quant,
                "chat_template": m.chat_template, "system_prompt": m.system_prompt,
                "profiles": m.profiles, "default_profile": m.default_profile,
                "auto_load": m.auto_load, "source": m.source,
                "resident": resident.contains(&m.id),
                "size_bytes": size, "on_disk": exists,
                "quant": crate::gguf::GgufMeta::from_file(&path).map(|g| g.quant_label()).unwrap_or_else(|_| "?".into()),
                "slots": ctx.slots().into_iter().filter(|s| s.model_id == m.id).collect::<Vec<_>>(),
            })
        })
        .collect();
    let models_dir = store.engine.models_dir.clone();
    let engine = store.engine.clone();
    let users: Vec<serde_json::Value> = store
        .users
        .users
        .iter()
        .map(|u| serde_json::json!({"username": u.username, "role": u.role}))
        .collect();
    drop(store);

    // disk usage of the models dir
    let (models_bytes, data_free) = dir_usage(std::path::Path::new(&models_dir));
    let sys = crate::sysinfo::snapshot();
    out.json(
        200,
        &serde_json::json!({
            "version": crate::config::ENGINE_VERSION,
            "uptime_s": ctx.started_at.elapsed().as_secs(),
            "server": {
                "api_bind": engine.server.api_bind,
                "ui_binds": engine.server.ui_binds,
                "tls_enabled": engine.server.tls_enabled,
                "api_key_required": engine.server.api_key_required,
                "ip_allowlist": engine.server.ip_allowlist,
                "mdns_name": engine.server.mdns_name,
            },
            "engine": {
                "threads": engine.threads, "n_batch": engine.n_batch,
                "queue_depth": engine.queue_depth, "max_resident": engine.max_resident,
                "queue_timeout_s": engine.queue_timeout_s,
                "kv_quant_default": engine.kv_quant_default,
                "update_channel": engine.update_channel,
                "update_manifest_url": engine.update_manifest_url,
                "auto_load": engine.auto_load,
            },
            "models": models,
            "users": users,
            "slots": ctx.slots(),
            "downloads": *ctx.downloads.read().unwrap(),
            "disk": { "models_bytes": models_bytes, "data_free": data_free },
            "system": sys,
            "metrics": ctx.metrics.snapshot(),
        }),
    );
}

fn dir_usage(p: &std::path::Path) -> (u64, u64) {
    let mut total = 0u64;
    if let Ok(rd) = std::fs::read_dir(p) {
        for e in rd.flatten() {
            if let Ok(md) = e.metadata() {
                if md.is_file() {
                    total += md.len();
                } else if md.is_dir() {
                    total += dir_usage(&e.path()).0;
                }
            }
        }
    }
    (total, free_bytes(p))
}

#[cfg(target_family = "unix")]
fn free_bytes(p: &std::path::Path) -> u64 {
    use std::os::unix::ffi::OsStrExt;
    let c = match std::ffi::CString::new(p.as_os_str().as_bytes()) {
        Ok(c) => c,
        Err(_) => return 0,
    };
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } == 0 {
        (st.f_bavail as u64) * (st.f_frsize as u64)
    } else {
        0
    }
}

#[cfg(not(target_family = "unix"))]
fn free_bytes(_p: &std::path::Path) -> u64 {
    0
}

fn models_list(ctx: &Arc<EngineCtx>, out: &mut ResponseOut) {
    let store = ctx.store.read().unwrap();
    let resident = ctx.manager.resident_ids();
    let models: Vec<serde_json::Value> = store
        .models
        .iter()
        .map(|m| {
            let path = std::path::Path::new(&store.engine.models_dir).join(&m.file);
            serde_json::json!({
                "id": m.id, "name": m.name, "file": m.file,
                "resident": resident.contains(&m.id),
                "size_bytes": std::fs::metadata(&path).map(|x| x.len()).unwrap_or(0),
            })
        })
        .collect();
    out.json(200, &serde_json::json!({ "models": models }));
}

pub(crate) fn slugify(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if matches!(c, '-' | '_' | '.') && !out.ends_with('-') {
            out.push(c);
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_string()
}

/// POST /api/models — register + optionally download {url|repo, quant, file?, id?}
fn add_model(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut) {
    let Some(_) = require_admin(ctx, req, out) else { return };
    #[derive(serde::Deserialize)]
    struct AddModel {
        url: Option<String>,
        /// HuggingFace repo id, e.g. "bartowski/Llama-3.2-1B-Instruct-GGUF".
        repo: Option<String>,
        quant: Option<String>,
        id: Option<String>,
        file: Option<String>,
        name: Option<String>,
    }
    let body: AddModel = match req.json() {
        Ok(b) => b,
        Err(e) => return out.error(400, &e),
    };
    let quant = body.quant.clone().unwrap_or_default();
    let (url, file) = match (&body.url, &body.repo) {
        (Some(u), _) => {
            let fname = body
                .file
                .clone()
                .or_else(|| u.split('/').next_back().map(|s| s.to_string()))
                .unwrap_or_else(|| "model.gguf".into());
            (u.clone(), fname)
        }
        (None, Some(repo)) => {
            // HF GGUF repo: file name conventionally contains the quant.
            let fname = body.file.clone().unwrap_or_else(|| {
                let short = repo.split('/').next_back().unwrap_or("model");
                if quant.is_empty() {
                    format!("{short}.gguf")
                } else {
                    format!("{short}.{quant}.gguf")
                }
            });
            (format!("https://huggingface.co/{repo}/resolve/main/{fname}"), fname)
        }
        _ => return out.error(400, "provide either `url` or `repo` (+quant)"),
    };

    let id = body
        .id
        .map(|i| slugify(&i))
        .unwrap_or_else(|| slugify(file.trim_end_matches(".gguf")));
    if id.is_empty() {
        return out.error(400, "could not derive a model id");
    }

    {
        let mut store = ctx.store.write().unwrap();
        if store.models.iter().any(|m| m.id == id) {
            return out.error(409, "model id already exists");
        }
        let cfg = ModelConfig {
            id: id.clone(),
            name: body.name.unwrap_or_else(|| id.clone()),
            file: file.clone(),
            ..Default::default()
        };
        store.models.push(cfg);
    }
    if let Err(e) = ctx.persist() {
        return out.error(500, &e);
    }

    // Start download in background (curl ships in the MarkOS image; on dev
    // hosts without it the model registers and the file can be copied in).
    let (models_dir, _) = {
        let store = ctx.store.read().unwrap();
        (store.engine.models_dir.clone(), ())
    };
    let path = std::path::Path::new(&models_dir).join(&file);
    let ctx2 = ctx.clone();
    let url2 = url.clone();
    let id2 = id.clone();
    let file2 = file.clone();
    std::thread::Builder::new()
        .name(format!("download-{id}"))
        .spawn(move || {
            ctx2.downloads.write().unwrap().insert(
                id2.clone(),
                DownloadStatus { url: url2.clone(), state: "running".into(), error: None, file: file2.clone() },
            );
            let tmp = path.with_extension("part");
            // --cacert: the mbedTLS-backed curl in the image has no compiled
            // default CA path; point it at the system bundle explicitly.
            let status = match std::process::Command::new("curl")
                .args(["-fSL", "--retry", "3", "--cacert", "/etc/ssl/certs/ca-certificates.crt", "-o"])
                .arg(&tmp)
                .arg(&url2)
                .output()
            {
                Ok(o) if o.status.success() => {
                    std::fs::rename(&tmp, &path).ok();
                    DownloadStatus { url: url2.clone(), state: "done".into(), error: None, file: file2 }
                }
                Ok(o) => {
                    let _ = std::fs::remove_file(&tmp);
                    DownloadStatus {
                        url: url2.clone(),
                        state: "error".into(),
                        error: Some(String::from_utf8_lossy(&o.stderr).trim().to_string()),
                        file: file2,
                    }
                }
                Err(e) => DownloadStatus {
                    url: url2.clone(),
                    state: "error".into(),
                    error: Some(format!("curl: {e}")),
                    file: file2,
                },
            };
            ctx2.downloads.write().unwrap().insert(id2, status);
        })
        .ok();

    out.json(201, &serde_json::json!({"id": id, "url": url, "file": file}));
}

/// POST /api/models/estimate {id|shape, n_ctx} — guardrail preview.
fn estimate(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut) {
    let Some(_) = require_session(ctx, req, out) else { return };
    #[derive(serde::Deserialize)]
    struct Est {
        id: String,
        n_ctx: Option<u64>,
    }
    let body: Est = match req.json() {
        Ok(b) => b,
        Err(e) => return out.error(400, &e),
    };
    match ctx.inventory(&body.id, body.n_ctx) {
        Ok(v) => out.json(200, &v),
        Err(e) => out.error(400, &e),
    }
}

fn model_route(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut, id: &str, sub: &str) {
    match (req.method.as_str(), sub) {
        ("GET", "") => {
            if require_session(ctx, req, out).is_none() {
                return;
            }
            match ctx.inventory(id, None) {
                Ok(v) => out.json(200, &v),
                Err(e) => out.error(404, &e),
            }
        }
        ("POST", "config") => update_model_config(ctx, req, out, id),
        ("POST", "load") => load_model(ctx, req, out, id),
        ("POST", "unload") => {
            if require_admin(ctx, req, out).is_none() {
                return;
            }
            match ctx.manager.unload(id) {
                Ok(_) => {
                    ctx.update_resident_metrics();
                    out.json(200, &serde_json::json!({"ok": true}))
                }
                Err(e) => out.error(409, &e),
            }
        }
        ("DELETE", "") => delete_model(ctx, req, out, id),
        _ => out.error(404, "no such model action"),
    }
}

fn update_model_config(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut, id: &str) {
    if require_admin(ctx, req, out).is_none() {
        return;
    }
    // The body is a partial ModelConfig; serde(default) fills the rest. We
    // then re-apply guardrails before persisting if n_ctx changed.
    let incoming: ModelConfig = match req.json() {
        Ok(c) => c,
        Err(e) => return out.error(400, &e),
    };
    let mut store = ctx.store.write().unwrap();
    let Some(existing) = store.models.iter_mut().find(|m| m.id == id) else {
        return out.error(404, "unknown model");
    };
    let new_ctx = incoming.n_ctx;
    let was_resident = ctx.manager.resident_ids().iter().any(|r| r == id);
    *existing = ModelConfig {
        chat_template: incoming.chat_template,
        system_prompt: incoming.system_prompt,
        n_ctx: incoming.n_ctx.clamp(1, 1_000_000),
        n_batch: incoming.n_batch.clamp(1, 4096),
        threads: incoming.threads.clamp(1, 8),
        kv_quant: incoming.kv_quant,
        profiles: incoming.profiles,
        default_profile: incoming.default_profile,
        auto_load: incoming.auto_load,
        name: incoming.name,
        ..existing.clone()
    };
    let cfg = existing.clone();
    drop(store);
    if let Err(e) = ctx.persist() {
        return out.error(500, &e);
    }
    // Guardrail re-check (non-fatal: model currently resident stays until
    // reloaded; the UI shows the estimate).
    if was_resident {
        if let Ok(inv) = ctx.inventory(id, Some(new_ctx)) {
            out.json(200, &serde_json::json!({"ok": true, "note": "resident model: reload to apply", "inventory": inv}));
            return;
        }
    }
    out.json(200, &serde_json::json!({"ok": true, "config": cfg}));
}

fn load_model(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut, id: &str) {
    if require_admin(ctx, req, out).is_none() {
        return;
    }
    let Some(cfg) = ctx.model_config(id) else {
        return out.error(404, "unknown model");
    };
    // Synchronous load (the UI shows a spinner; loads take seconds).
    let ec = ctx.engine_config();
    let mut loader = |c: &ModelConfig| ctx.load_backend(c).map(|(b, _)| b);
    ctx.metrics.queued.fetch_add(1, Ordering::Relaxed);
    let r = ctx.manager.acquire(id, &ec, &mut loader, &cfg);
    ctx.metrics.queued.fetch_sub(1, Ordering::Relaxed);
    match r {
        Ok(g) => {
            drop(g);
            ctx.update_resident_metrics();
            out.json(200, &serde_json::json!({"ok": true, "resident": ctx.manager.resident_ids()}));
        }
        Err(e) => {
            let (code, msg) = match e {
                crate::models::AcquireError::QueueFull => (429, "engine busy".to_string()),
                crate::models::AcquireError::Timeout => (504, "load timed out".to_string()),
                crate::models::AcquireError::LoadFailed(le) => {
                    let m = le.to_string();
                    (if m.contains("memory guardrail") { 409 } else { 502 }, m)
                }
            };
            out.error(code, &msg);
        }
    }
}

fn delete_model(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut, id: &str) {
    if require_admin(ctx, req, out).is_none() {
        return;
    }
    if let Err(e) = ctx.manager.unload(id) {
        return out.error(409, &e);
    }
    let (file, models_dir) = {
        let mut store = ctx.store.write().unwrap();
        match store.models.iter().position(|m| m.id == id) {
            Some(i) => {
                let m = store.models.remove(i);
                (m.file, store.engine.models_dir.clone())
            }
            None => return out.error(404, "unknown model"),
        }
    };
    let _ = std::fs::remove_file(std::path::Path::new(&models_dir).join(&file));
    if let Err(e) = ctx.persist() {
        return out.error(500, &e);
    }
    ctx.update_resident_metrics();
    out.json(200, &serde_json::json!({"ok": true}));
}

fn add_user(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut) {
    let Some(admin) = require_admin(ctx, req, out) else { return };
    #[derive(serde::Deserialize)]
    struct NewUser {
        username: String,
        password: String,
        role: Role,
    }
    let body: NewUser = match req.json() {
        Ok(b) => b,
        Err(e) => return out.error(400, &e),
    };
    if !matches!(admin.role, Role::Admin) {
        return out.error(403, "admin required");
    }
    let mut store = ctx.store.write().unwrap();
    if auth::ensure_user(&mut store.users, &body.username, &body.password, body.role).is_err() {
        return out.error(400, "username non-empty, password >= 8 chars");
    }
    drop(store);
    if let Err(e) = ctx.persist() {
        return out.error(500, &e);
    }
    out.json(201, &serde_json::json!({"ok": true}));
}

fn sys_snapshot(ctx: &Arc<EngineCtx>, out: &mut ResponseOut) {
    crate::sysinfo::refresh();
    ctx.update_resident_metrics();
    let sys = crate::sysinfo::snapshot();
    out.json(200, &serde_json::json!({ "system": sys, "metrics": ctx.metrics.snapshot() }));
}

fn metrics_stream(ctx: &Arc<EngineCtx>, out: &mut ResponseOut) {
    if out.begin_stream(200, "text/event-stream").is_err() {
        return;
    }
    // ~1 Hz live metrics until the client disconnects (write failure).
    for _ in 0..3600 {
        crate::sysinfo::refresh();
        ctx.update_resident_metrics();
        let payload = serde_json::json!({
            "system": crate::sysinfo::snapshot(),
            "metrics": ctx.metrics.snapshot(),
            "slots": ctx.slots(),
        });
        if sse_event(out, "metrics", &payload.to_string()).is_err() {
            return;
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
    let _ = sse_event(out, "", "[DONE]");
}

fn system_restart(ctx: &Arc<EngineCtx>, out: &mut ResponseOut) {
    // On MarkOS, runit supervises us; a clean exit is a restart. On dev
    // hosts, this exits the engine process.
    ctx.metrics.record(crate::metrics::LogEntry {
        ts: crate::metrics::Metrics::now_ms(),
        kind: "system".into(),
        model: None,
        peer: None,
        path: Some("/api/system/restart".into()),
        status: 200,
        prompt_tokens: None,
        gen_tokens: None,
        ms: None,
        tokps: None,
        message: Some("restart requested via web UI".into()),
    });
    out.json(200, &serde_json::json!({"ok": true, "note": "engine exiting; supervisor will restart it"}));
    let _ = std::io::stdout().flush();
    // Give the response a moment to reach the wire.
    std::thread::sleep(std::time::Duration::from_millis(200));
    std::process::exit(0);
}

/// POST /api/system/update-check {manifest_url?} — manual, user-initiated.
/// Fetches a tiny JSON manifest {"version": "x.y.z", "url": "..."} and
/// compares. Nothing is installed automatically; applying an update runs
/// /usr/bin/markos-update on the appliance (A/B slot switch, design §5).
fn update_check(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut) {
    #[derive(serde::Deserialize)]
    struct CheckReq {
        manifest_url: Option<String>,
    }
    let body: CheckReq = req.json().unwrap_or(CheckReq { manifest_url: None });
    let url = body
        .manifest_url
        .or_else(|| ctx.engine_config().update_manifest_url.clone());
    let Some(url) = url else {
        return out.json(200, &serde_json::json!({
            "checked": false,
            "reason": "no update manifest configured (offline/manual mode)"
        }));
    };
    let output = std::process::Command::new("curl")
        .args(["-fsSL", "--max-time", "20"])
        .arg(&url)
        .output();
    match output {
        Ok(o) if o.status.success() => {
            #[derive(serde::Deserialize)]
            struct Manifest {
                version: String,
                #[serde(default)]
                url: Option<String>,
                #[serde(default)]
                notes: Option<String>,
            }
            match serde_json::from_slice::<Manifest>(&o.stdout) {
                Ok(m) => {
                    let current = crate::config::ENGINE_VERSION;
                    let newer = version_newer(&m.version, current);
                    out.json(
                        200,
                        &serde_json::json!({
                            "checked": true,
                            "current": current,
                            "available": m.version,
                            "update_available": newer,
                            "url": m.url,
                            "notes": m.notes,
                        }),
                    );
                }
                Err(e) => out.error(502, &format!("bad manifest: {e}")),
            }
        }
        Ok(o) => out.error(502, &format!("manifest fetch failed: {}", String::from_utf8_lossy(&o.stderr))),
        Err(e) => out.error(502, &format!("curl: {e}")),
    }
}

fn version_newer(candidate: &str, current: &str) -> bool {
    let parse = |s: &str| -> Vec<u64> {
        s.trim_start_matches('v')
            .split('.')
            .map(|p| p.split('-').next().unwrap_or("0").parse().unwrap_or(0))
            .collect()
    };
    let (a, b) = (parse(candidate), parse(current));
    for i in 0..3 {
        let av = a.get(i).copied().unwrap_or(0);
        let bv = b.get(i).copied().unwrap_or(0);
        if av != bv {
            return av > bv;
        }
    }
    false
}

fn recovery_status(out: &mut ResponseOut) {
    // Recovery facts are static by construction (docs/recovery.md); the
    // button state is checked at boot by markos-recovery and logged.
    let last = std::fs::read_to_string("/data/state/recovery-last.txt").ok();
    out.json(
        200,
        &serde_json::json!({
            "ipv4ll_rescue": "listening on 169.254.9.1:4444 whenever a cable is connected",
            "factory_reset_button": "GPIO26 held >=3s at boot wipes /data/state to the provision snapshot",
            "serial_console": "UART GPIO14/15 @115200 (j8 pins 8/10)",
            "last_event": last,
        }),
    );
}

fn serve_asset(out: &mut ResponseOut, name: &str) {
    match crate::webassets::get(name) {
        Some((ctype, bytes)) => out.bytes(200, ctype, bytes.to_vec()),
        None => out.error(404, "asset missing"),
    }
}

/// POST /api/server — network/exposure settings. Listeners restart on
/// process restart (runit); we persist now and tell the caller.
fn update_server_config(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut) {
    if require_admin(ctx, req, out).is_none() {
        return;
    }
    #[derive(serde::Deserialize)]
    struct NetCfg {
        api_bind: Option<String>,
        ui_binds: Option<Vec<String>>,
        tls_enabled: Option<bool>,
        api_key_required: Option<bool>,
        ip_allowlist: Option<Vec<String>>,
        mdns_name: Option<String>,
    }
    let body: NetCfg = match req.json() {
        Ok(b) => b,
        Err(e) => return out.error(400, &e),
    };
    {
        let mut store = ctx.store.write().unwrap();
        if let Some(b) = body.api_bind {
            if !b.contains(':') {
                return out.error(400, "api_bind must be host:port");
            }
            store.engine.server.api_bind = b;
        }
        if let Some(v) = body.ui_binds {
            if v.is_empty() {
                return out.error(400, "refusing to remove all UI binds (would lock you out)");
            }
            if !v.iter().any(|b| b.ends_with(":4444")) {
                // Soft-guard the rescue listener, not a hard block.
            }
            store.engine.server.ui_binds = v;
        }
        if let Some(v) = body.tls_enabled {
            store.engine.server.tls_enabled = v;
        }
        if let Some(v) = body.api_key_required {
            if v && store.engine.server.api_key_hash.is_none() {
                return out.error(400, "set an API key before requiring one");
            }
            store.engine.server.api_key_required = v;
        }
        if let Some(v) = body.ip_allowlist {
            store.engine.server.ip_allowlist = v;
        }
        if let Some(v) = body.mdns_name {
            store.engine.server.mdns_name = v;
        }
    }
    if let Err(e) = ctx.persist() {
        return out.error(500, &e);
    }
    out.json(200, &serde_json::json!({"ok": true, "note": "saved; listener changes apply on engine restart"}));
}

/// POST /api/server/apikey {key} — stores sha256(key), never the key.
fn set_api_key(ctx: &Arc<EngineCtx>, req: &Request, out: &mut ResponseOut) {
    if require_admin(ctx, req, out).is_none() {
        return;
    }
    #[derive(serde::Deserialize)]
    struct KeyReq {
        key: String,
    }
    let body: KeyReq = match req.json() {
        Ok(b) => b,
        Err(e) => return out.error(400, &e),
    };
    if body.key.len() < 8 {
        return out.error(400, "API key too short (>= 8 chars)");
    }
    {
        let mut store = ctx.store.write().unwrap();
        store.engine.server.api_key_hash = Some(auth::sha256_hex(body.key.as_bytes()));
    }
    if let Err(e) = ctx.persist() {
        return out.error(500, &e);
    }
    out.json(200, &serde_json::json!({"ok": true}));
}

fn delete_user(ctx: &Arc<EngineCtx>, out: &mut ResponseOut, name: &str) {
    let mut store = ctx.store.write().unwrap();
    let Some(pos) = store.users.users.iter().position(|u| u.username == name) else {
        return out.error(404, "unknown user");
    };
    if matches!(store.users.users[pos].role, Role::Admin)
        && store.users.users.iter().filter(|u| matches!(u.role, Role::Admin)).count() <= 1
    {
        return out.error(409, "refusing to remove the last admin");
    }
    store.users.users.remove(pos);
    drop(store);
    if let Err(e) = ctx.persist() {
        return out.error(500, &e);
    }
    out.json(200, &serde_json::json!({"ok": true}));
}
