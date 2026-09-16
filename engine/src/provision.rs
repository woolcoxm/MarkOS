//! First-boot provisioning: consumes the installer-written provision file
//! (`/boot/markos/provision.env` on the appliance) to create the initial
//! admin account and optionally pre-seed a model. Runs exactly once — the
//! file is renamed to `provision.done` afterward. There are no default
//! credentials anywhere; this is the only way an admin comes into existence.

use crate::auth;
use crate::config::{ModelConfig, Role};

pub fn provision_path() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("MARKOS_PROVISION") {
        // Empty = explicitly disabled (host-dev mode).
        return if p.is_empty() { None } else { Some(std::path::PathBuf::from(p)) };
    }
    #[cfg(target_family = "unix")]
    {
        let p = std::path::PathBuf::from("/boot/markos/provision.env");
        if p.exists() {
            return Some(p);
        }
    }
    None
}

fn parse_env(text: &str) -> std::collections::BTreeMap<String, String> {
    let mut map = std::collections::BTreeMap::new();
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once('=') {
            let v = v.trim().trim_matches('"').to_string();
            map.insert(k.trim().to_string(), v);
        }
    }
    map
}

/// Called once at engine startup. Returns a human-readable summary of what
/// was applied (for the boot log).
pub fn run_provisioning(ctx: &std::sync::Arc<crate::state::EngineCtx>) -> String {
    let Some(path) = provision_path() else {
        return "no provision file (already provisioned or never provided)".into();
    };
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => return format!("provision file unreadable: {e}"),
    };
    let env = parse_env(&text);
    let mut notes = Vec::new();

    let has_users = !ctx.store.read().unwrap().users.users.is_empty();
    if has_users {
        return "provision file present but users already exist; ignoring".into();
    }

    // System-level files the OS boot stages read (tz, locale, hostname).
    if let Some(tz) = env.get("MARKOS_TZ") {
        let _ = std::fs::write(ctx.data_dir.join("tz"), tz);
    }
    if let Some(lang) = env.get("MARKOS_LANG") {
        let _ = std::fs::write(ctx.data_dir.join("lang"), lang);
    }
    if let Some(hn) = env.get("MARKOS_HOSTNAME") {
        let _ = std::fs::write(ctx.data_dir.join("hostname"), format!("{hn}
"));
    }

    let username = env.get("MARKOS_ADMIN_USER").cloned().unwrap_or_else(|| "admin".into());
    let password = env.get("MARKOS_ADMIN_PASSWORD").cloned();
    let pwhash = env.get("MARKOS_ADMIN_PASSWORD_HASH").cloned();
    let created = match (password, pwhash) {
        (Some(pw), _) => {
            let mut store = ctx.store.write().unwrap();
            auth::ensure_user(&mut store.users, &username, &pw, Role::Admin)
                .map(|_| "admin account created".to_string())
        }
        (None, Some(h)) => {
            let mut store = ctx.store.write().unwrap();
            store.users.users.push(crate::config::User {
                username: username.clone(),
                pwhash: h,
                role: Role::Admin,
            });
            Ok("admin account created (pre-hashed)".to_string())
        }
        _ => Err("no MARKOS_ADMIN_PASSWORD/HASH in provision file".into()),
    };
    match created {
        Ok(msg) => {
            if let Err(e) = ctx.persist() {
                notes.push(format!("persist failed: {e}"));
            }
            notes.push(msg);
        }
        Err(e) => notes.push(format!("provisioning refused: {e}")),
    }

    // Optional model pre-seed: registers + starts a background download.
    let repo = env.get("MARKOS_MODEL_PRESEED_REPO");
    let url = env.get("MARKOS_MODEL_PRESEED_URL");
    if let Some(src) = repo.or(url) {
        let quant = env.get("MARKOS_MODEL_PRESEED_QUANT").cloned().unwrap_or_default();
        // HF GGUF convention: <RepoShort>.<QUANT>.gguf — mirrors admin::add_model.
        let short = src
            .trim_end_matches(".gguf")
            .split('/')
            .next_back()
            .unwrap_or("model")
            .to_string();
        let file = if repo.is_some() && !quant.is_empty() {
            format!("{short}.{quant}.gguf")
        } else {
            format!("{short}.gguf")
        };
        let id = crate::admin::slugify(&format!("{short}-{quant}"));
        {
            let mut store = ctx.store.write().unwrap();
            if !store.models.iter().any(|m| m.id == id) {
                store.models.push(ModelConfig {
                    id: id.clone(),
                    name: id.clone(),
                    file: file.clone(),
                    source: Some(src.to_string()),
                    ..Default::default()
                });
            }
        }
        let _ = ctx.persist();
        let models_dir = ctx.engine_config().models_dir;
        let ctx2 = ctx.clone();
        let id2 = id.clone();
        let src2 = if let Some(r) = repo {
            format!("https://huggingface.co/{r}/resolve/main/{file}")
        } else {
            src.to_string()
        };
        std::thread::Builder::new()
            .name("preseed-download".into())
            .spawn(move || {
                let path = std::path::Path::new(&models_dir).join(&file);
                ctx2.downloads.write().unwrap().insert(
                    id2.clone(),
                    crate::state::DownloadStatus {
                        url: src2.clone(),
                        state: "running".into(),
                        error: None,
                        file: file.clone(),
                    },
                );
                let ok = std::process::Command::new("curl")
                    .args(["-fSL", "--retry", "3", "-o"])
                    .arg(&path)
                    .arg(&src2)
                    .status()
                    .map(|s| s.success())
                    .unwrap_or(false);
                let st = if ok {
                    crate::state::DownloadStatus { url: src2, state: "done".into(), error: None, file }
                } else {
                    crate::state::DownloadStatus {
                        url: src2,
                        state: "error".into(),
                        error: Some("download failed".into()),
                        file,
                    }
                };
                ctx2.downloads.write().unwrap().insert(id2, st);
            })
            .ok();
        notes.push(format!("model pre-seed '{id}' started"));
    }

    // Mark done regardless of content: never loop on a bad provision file.
    let done = path.with_file_name("provision.done");
    std::fs::rename(&path, &done).ok();
    format!("provisioned: {}", notes.join("; "))
}
