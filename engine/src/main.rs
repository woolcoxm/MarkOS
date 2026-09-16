//! markos-engine — the MarkOS inference engine + web UI (one binary, two
//! listeners: OpenAI-compatible inference API and the admin control plane).

mod admin;
mod api;
mod auth;
mod backend;
mod config;
mod gguf;
mod guard;
mod http;
mod metrics;
mod models;
mod provision;
mod state;
mod sysinfo;
mod templates;
#[cfg(feature = "tls")]
mod tls;
mod webassets;

use http::{Handler, Request, ResponseOut, Server};
use metrics::{LogEntry, Metrics};
use state::EngineCtx;
use std::net::TcpListener;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Role {
    Api,
    Ui,
}

struct Dispatcher {
    ctx: Arc<EngineCtx>,
    kind: Role,
}

impl Dispatcher {
    fn ip_allowed(&self, req: &Request) -> bool {
        let list = &self.ctx.engine_config().server.ip_allowlist;
        if list.is_empty() {
            return true;
        }
        let peer_ip = req.peer.rsplit_once(':').map(|(ip, _)| ip.trim_matches(['[', ']'])).unwrap_or("");
        for entry in list {
            if let Some((net, bits)) = entry.split_once('/') {
                if cidr_contains(net, peer_ip, bits.parse().unwrap_or(32)) {
                    return true;
                }
            } else if entry == peer_ip {
                return true;
            }
        }
        false
    }
}

impl Handler for Dispatcher {
    fn handle(&self, req: &Request, out: &mut ResponseOut) {
        let t0 = Instant::now();
        self.metrics_requests_add();

        // Health/readiness bypass auth and allowlist (supervisor + watchdog
        // poll these locally).
        match req.path.as_str() {
            "/healthz" => {
                return out.json(
                    200,
                    &serde_json::json!({
                        "status": "ok",
                        "version": config::ENGINE_VERSION,
                        "uptime_s": self.ctx.started_at.elapsed().as_secs(),
                    }),
                );
            }
            "/readyz" => {
                let resident = self.ctx.manager.resident_ids();
                let configured = self.ctx.store.read().unwrap().models.len();
                return out.json(
                    200,
                    &serde_json::json!({
                        "ready": true,
                        "resident": resident,
                        "configured": configured,
                    }),
                );
            }
            _ => {}
        }

        if !self.ip_allowed(req) {
            out.error(403, "source address not in allowlist");
            return;
        }

        match self.kind {
            Role::Api => {
                // Bearer API key on the inference surface (optional).
                let bearer = req
                    .header("authorization")
                    .and_then(|a| a.strip_prefix("Bearer "));
                if !auth::api_key_ok(&self.ctx.engine_config().server, bearer) {
                    out.error(401, "missing or invalid API key");
                } else {
                    match req.path.as_str() {
                        "/v1/models" if req.method == "GET" => api::list_models(&self.ctx, out),
                        "/v1/chat/completions" if req.method == "POST" => api::chat_completions(&self.ctx, req, out),
                        "/v1/completions" if req.method == "POST" => api::completions(&self.ctx, req, out),
                        _ => out.error(404, "unknown API endpoint (POST /v1/chat/completions, /v1/completions, GET /v1/models)"),
                    }
                }
            }
            Role::Ui => admin::route(&self.ctx, req, out),
        }

        self.log_access(req, out, t0);
    }
}

impl Dispatcher {
    fn metrics_requests_add(&self) {
        self.ctx.metrics.requests_total.fetch_add(1, Ordering::Relaxed);
    }
    fn log_access(&self, req: &Request, out: &mut ResponseOut, t0: Instant) {
        let st = out.status();
        if st >= 400 {
            if st >= 500 {
                self.ctx.metrics.requests_5xx.fetch_add(1, Ordering::Relaxed);
            } else {
                self.ctx.metrics.requests_4xx.fetch_add(1, Ordering::Relaxed);
            }
            // Don't log polling 401s from the SPA's initial state fetch.
            if !(st == 401 && req.path == "/api/state") {
                self.ctx.metrics.record(LogEntry {
                    ts: Metrics::now_ms(),
                    kind: "request".into(),
                    model: None,
                    peer: Some(req.peer.clone()),
                    path: Some(req.path.clone()),
                    status: st,
                    prompt_tokens: None,
                    gen_tokens: None,
                    ms: Some(t0.elapsed().as_millis() as u64),
                    tokps: None,
                    message: None,
                });
            }
        }
    }
}

fn cidr_contains(net: &str, ip: &str, bits: u8) -> bool {
    let parse4 = |s: &str| -> Option<u32> {
        let parts: Vec<u8> = s.split('.').filter_map(|p| p.parse().ok()).collect();
        if parts.len() != 4 {
            return None;
        }
        Some(u32::from_be_bytes([parts[0], parts[1], parts[2], parts[3]]))
    };
    match (parse4(net), parse4(ip)) {
        (Some(n), Some(p)) => {
            let b = bits.min(32);
            let mask = if b == 0 { 0 } else { !0u32 << (32 - b) };
            n & mask == p & mask
        }
        // IPv6 allowlist entries only support exact match (handled above).
        _ => net == ip,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let mut data_dir = std::path::PathBuf::from("/var/lib/markos");
    let mut host_dev = false;
    let mut show_version = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--data-dir" => {
                data_dir = args.next().map(std::path::PathBuf::from).unwrap_or(data_dir);
            }
            "--host-dev" => host_dev = true,
            "--version" => show_version = true,
            // Maintenance flag used by markos-update (A/B slot switch): one
            // boot into the tryboot cmdline. Not part of the serving path.
            "--reboot-tryboot" => reboot_tryboot(),
            "--help" | "-h" => {
                println!(
                    "markos-engine {}\n  --data-dir DIR   state/models/logs root (default /var/lib/markos)\n  --host-dev       bind loopback ports 8080/8081 for development\n  --version        print version",
                    config::ENGINE_VERSION
                );
                return;
            }
            _ => {}
        }
    }
    if show_version {
        println!("markos-engine {}", config::ENGINE_VERSION);
        return;
    }
    sysinfo::refresh();
    let ctx = match EngineCtx::open(&data_dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("markos-engine: cannot open data dir {}: {e}", data_dir.display());
            std::process::exit(1);
        }
    };

    // First-boot provisioning (installer-written admin account, model seed).
    let note = provision::run_provisioning(&ctx);
    ctx.metrics.record(LogEntry {
        ts: Metrics::now_ms(),
        kind: "system".into(),
        model: None,
        peer: None,
        path: None,
        status: 200,
        prompt_tokens: None,
        gen_tokens: None,
        ms: None,
        tokps: None,
        message: Some(format!("engine {} started: {note}", config::ENGINE_VERSION)),
    });
    println!("markos-engine {} up (data: {})", config::ENGINE_VERSION, data_dir.display());

    let ec = ctx.engine_config();
    let (api_bind, ui_binds) = if host_dev {
        ("127.0.0.1:8080".to_string(), vec!["127.0.0.1:8081".to_string()])
    } else {
        (ec.server.api_bind.clone(), ec.server.ui_binds.clone())
    };

    #[cfg(feature = "tls")]
    let tls_wrap = match tls::wrap_if_enabled(&ctx) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("markos-engine: TLS init failed, UI falls back to plain HTTP: {e}");
            None
        }
    };
    #[cfg(not(feature = "tls"))]
    let tls_wrap: Option<Arc<dyn http::StreamWrap>> = None;

    let mut listeners: Vec<(TcpListener, Role, Option<Arc<dyn http::StreamWrap>>)> = Vec::new();
    for (bind, role) in std::iter::once((api_bind, Role::Api)).chain(ui_binds.into_iter().map(|b| (b, Role::Ui))) {
        match TcpListener::bind(&bind) {
            Ok(l) => listeners.push((l, role, None)),
            Err(e) => eprintln!("markos-engine: cannot bind {bind} ({role:?}): {e}"),
        }
    }
    if listeners.is_empty() {
        eprintln!("markos-engine: no listeners bound, exiting");
        std::process::exit(1);
    }
    // TLS applies to UI listeners (the API stays plain for LAN clients unless
    // operators front it themselves).
    for (_l, role, wrap_slot) in listeners.iter_mut() {
        if *role == Role::Ui {
            *wrap_slot = tls_wrap
                .clone()
                .map(|w| w as Arc<dyn http::StreamWrap>);
        }
    }

    let handles: Vec<_> = listeners
        .into_iter()
        .map(|(l, role, wrap)| {
            let ctx = ctx.clone();
            std::thread::Builder::new()
                .name(format!("listener-{role:?}"))
                .spawn(move || {
                    let server = match wrap {
                        Some(w) => Arc::new(Server::with_wrap(128, w)),
                        None => Arc::new(Server::new(128)),
                    };
                    let handler: Arc<dyn Handler> = Arc::new(Dispatcher { ctx, kind: role });
                    server.serve(l, handler)
                })
                .expect("spawn listener thread")
        })
        .collect();

    // Metric refresher: keeps meminfo/thermal current for the UI.
    {
        let ctx = ctx.clone();
        std::thread::Builder::new()
            .name("sysinfo".into())
            .spawn(move || loop {
                sysinfo::refresh();
                ctx.update_resident_metrics();
                std::thread::sleep(std::time::Duration::from_secs(5));
            })
            .ok();
    }

    // Ctrl-C / SIGTERM → clean exit (runit restarts us unless stopping).
    ctrlc_wait();
    for h in handles {
        // Listeners run forever; this is unreachable in practice.
        drop(h);
    }
}

#[cfg(target_family = "unix")]
fn ctrlc_wait() {
    unsafe {
        // Minimal signal handling: exit on SIGINT/SIGTERM so runit can
        // supervise us cleanly. No async-signal work, just _exit.
        let mut sa: libc::sigaction = std::mem::zeroed();
        sa.sa_sigaction = handle_term as *const () as usize;
        sa.sa_flags = libc::SA_RESTART;
        libc::sigaction(libc::SIGINT, &sa, std::ptr::null_mut());
        libc::sigaction(libc::SIGTERM, &sa, std::ptr::null_mut());
    }
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}

/// One-shot reboot into the firmware's `tryboot` cmdline (A/B updates).
/// The Raspberry Pi kernel's watchdog restart handler interprets this
/// reboot command string.
#[cfg(target_family = "unix")]
fn reboot_tryboot() {
    use std::ffi::CString;
    let cmd = CString::new("firmware=tryboot").unwrap();
    unsafe {
        libc::syscall(
            libc::SYS_reboot,
            libc::LINUX_REBOOT_MAGIC1,
            libc::LINUX_REBOOT_MAGIC2,
            libc::LINUX_REBOOT_CMD_RESTART2,
            cmd.as_ptr(),
        );
    }
    eprintln!("markos-engine: tryboot reboot failed (not a Pi kernel?)");
    std::process::exit(1);
}

#[cfg(not(target_family = "unix"))]
fn reboot_tryboot() {
    eprintln!("markos-engine: --reboot-tryboot requires the appliance kernel");
    std::process::exit(1);
}

#[cfg(target_family = "unix")]
extern "C" fn handle_term(_sig: libc::c_int) {
    // Exit directly: no locks are held on this thread.
    unsafe { libc::_exit(0) };
}

#[cfg(not(target_family = "unix"))]
fn ctrlc_wait() {
    loop {
        std::thread::sleep(std::time::Duration::from_secs(3600));
    }
}
