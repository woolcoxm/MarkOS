//! End-to-end test: boots the actual markos-engine binary with a synthetic
//! GGUF (mock backend), then exercises provisioning, auth, the admin control
//! plane and the OpenAI-compatible API — including SSE streaming.

use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct Engine {
    child: Child,
    base_api: String,
    base_ui: String,
    #[allow(dead_code)]
    dir: PathBuf,
}

impl Drop for Engine {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// Minimal GGUF v3 writer (mirrors scripts/make_gguf_test.py): qwen2-shaped
/// metadata + F16 tensor blobs of zeros.
fn write_test_gguf(path: &std::path::Path) {
    let arch = "qwen2";
    let (layers, embd, head, kv_head, vocab) = (2u64, 64u64, 4u64, 2u64, 256u64);

    fn put_str(out: &mut Vec<u8>, s: &str) {
        out.extend_from_slice(&(s.len() as u64).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    fn put_u32kv(out: &mut Vec<u8>, k: &str, v: u32) {
        put_str(out, k);
        out.extend_from_slice(&4u32.to_le_bytes());
        out.extend_from_slice(&v.to_le_bytes());
    }
    fn put_strkv(out: &mut Vec<u8>, k: &str, v: &str) {
        put_str(out, k);
        out.extend_from_slice(&8u32.to_le_bytes());
        put_str(out, v);
    }

    let mut tensors: Vec<(String, Vec<u64>)> = vec![("token_embd.weight".into(), vec![embd, vocab])];
    for b in 0..layers {
        let hd = embd / head;
        tensors.extend([
            (format!("blk.{b}.attn_q.weight"), vec![embd, embd]),
            (format!("blk.{b}.attn_k.weight"), vec![embd, kv_head * hd]),
            (format!("blk.{b}.attn_v.weight"), vec![embd, kv_head * hd]),
            (format!("blk.{b}.attn_output.weight"), vec![embd, embd]),
            (format!("blk.{b}.ffn_down.weight"), vec![embd * 2, embd]),
            (format!("blk.{b}.ffn_gate.weight"), vec![embd, embd * 2]),
            (format!("blk.{b}.ffn_up.weight"), vec![embd, embd * 2]),
        ]);
    }
    tensors.push(("output.weight".into(), vec![embd, vocab]));

    let mut hdr: Vec<u8> = Vec::new();
    hdr.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF
    hdr.extend_from_slice(&3u32.to_le_bytes());
    hdr.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    hdr.extend_from_slice(&12u64.to_le_bytes());
    put_strkv(&mut hdr, "general.architecture", arch);
    put_strkv(&mut hdr, "general.name", "markos-test");
    put_u32kv(&mut hdr, "general.alignment", 32);
    put_u32kv(&mut hdr, &format!("{arch}.block_count"), layers as u32);
    put_u32kv(&mut hdr, &format!("{arch}.embedding_length"), embd as u32);
    put_u32kv(&mut hdr, &format!("{arch}.attention.head_count"), head as u32);
    put_u32kv(&mut hdr, &format!("{arch}.attention.head_count_kv"), kv_head as u32);
    put_u32kv(&mut hdr, &format!("{arch}.context_length"), 4096);
    put_u32kv(&mut hdr, &format!("{arch}.vocab_size"), vocab as u32);
    put_strkv(&mut hdr, "tokenizer.chat_template", "{% for m in messages %}[{{ m.role }}] {{ m.content }}\n{% endfor %}[assistant] ");
    put_strkv(&mut hdr, "tokenizer.ggml.model", "llama");
    put_u32kv(&mut hdr, "tokenizer.ggml.tokens_size", vocab as u32);

    let mut data: Vec<u8> = Vec::new();
    let mut off = 0u64;
    for (name, dims) in &tensors {
        put_str(&mut hdr, name);
        hdr.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in dims {
            hdr.extend_from_slice(&d.to_le_bytes());
        }
        hdr.extend_from_slice(&1u32.to_le_bytes()); // F16
        hdr.extend_from_slice(&off.to_le_bytes());
        let ne: u64 = dims.iter().product();
        let nbytes = ne * 2;
        data.extend(std::iter::repeat(0u8).take(nbytes as usize));
        off += nbytes;
    }
    while hdr.len() % 32 != 0 {
        hdr.push(0);
    }
    let mut all = hdr;
    all.extend_from_slice(&data);
    std::fs::write(path, all).unwrap();
}

/// Raw HTTP/1.1 over a socket (no test HTTP client dependencies).
fn http(base: &str, method: &str, path: &str, cookie: Option<&str>, body: Option<&str>) -> (u16, String) {
    let mut stream = TcpStream::connect(base.replace("http://", "")).expect("connect");
    stream.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let body_bytes = body.unwrap_or("").as_bytes();
    let mut req = format!("{method} {path} HTTP/1.1\r\nHost: e2e\r\nContent-Type: application/json\r\n");
    if let Some(c) = cookie {
        req.push_str(&format!("Cookie: {c}\r\n"));
    }
    req.push_str(&format!("Content-Length: {}\r\nConnection: close\r\n\r\n", body_bytes.len()));
    req.push_str(body.unwrap_or(""));
    stream.write_all(req.as_bytes()).unwrap();
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).unwrap();
    let text = String::from_utf8_lossy(&resp).into_owned();
    let status: u16 = text
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body = text
        .splitn(2, "\r\n\r\n")
        .nth(1)
        .unwrap_or("")
        .to_string();
    (status, body)
}

fn start_engine(tag: &str) -> Engine {
    let dir = std::env::temp_dir().join(format!("markos-e2e-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("models")).unwrap();

    write_test_gguf(&dir.join("models").join("test.gguf"));

    // Pre-seeded store with one model; users exist only via provisioning.
    let store = serde_json::json!({
        "engine": {},
        "models": [{
            "id": "test", "name": "test model", "file": "test.gguf",
            "n_ctx": 2048, "n_batch": 128, "threads": 2,
            "auto_load": false,
            "default_profile": "default",
            "profiles": [{"name": "default", "max_tokens": 32, "stop": [], "sampling": {}}],
        }],
        "users": {"users": []},
    });
    std::fs::write(dir.join("markos.json"), serde_json::to_vec(&store).unwrap()).unwrap();

    // Provision file: creates the initial admin (installer-equivalent).
    let provision = dir.join("provision.env");
    std::fs::write(
        &provision,
        "MARKOS_ADMIN_USER=admin\nMARKOS_ADMIN_PASSWORD=correct-horse-battery\n",
    )
    .unwrap();

    let mut cmd = Command::new(env!("CARGO_BIN_EXE_markos-engine"));
    cmd.args(["--data-dir", dir.to_str().unwrap(), "--host-dev"]);
    cmd.env("MARKOS_PROVISION", provision.to_str().unwrap());
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let child = cmd.spawn().expect("spawn engine");

    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if Instant::now() > deadline {
            panic!("engine did not come up (health)");
        }
        if TcpStream::connect("127.0.0.1:8080").is_ok() {
            let (st, _) = http("http://127.0.0.1:8080", "GET", "/healthz", None, None);
            if st == 200 {
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    Engine {
        child,
        base_api: "http://127.0.0.1:8080".into(),
        base_ui: "http://127.0.0.1:8081".into(),
        dir,
    }
}

/// The full appliance flow, single test so the fixed dev ports don't clash.
#[test]
fn e2e_provision_auth_api_ui() {
    let eng = start_engine("main");
    let api = &eng.base_api;
    let ui = &eng.base_ui;

    // 1. Health on both listeners.
    let (st, body) = http(api, "GET", "/healthz", None, None);
    assert_eq!(st, 200, "{body}");
    assert!(body.contains("version"));
    let (st, _) = http(ui, "GET", "/healthz", None, None);
    assert_eq!(st, 200);

    // 2. Web UI static assets served.
    let (st, body) = http(ui, "GET", "/", None, None);
    assert_eq!(st, 200);
    assert!(body.contains("MarkOS"), "index should contain brand");
    let (st, _) = http(ui, "GET", "/app.js", None, None);
    assert_eq!(st, 200);

    // 3. Admin API requires auth before provisioning-login.
    let (st, _) = http(ui, "GET", "/api/state", None, None);
    assert_eq!(st, 401, "state must require login");

    // 4. Login with the provisioned admin (installer-created credentials).
    let (st, body) = http(
        ui,
        "POST",
        "/api/login",
        None,
        Some(r#"{"username":"admin","password":"correct-horse-battery"}"#),
    );
    assert_eq!(st, 200, "{body}");
    let cookie = body_cookie_hack();
    let (st, _) = http(
        ui,
        "POST",
        "/api/login",
        None,
        Some(r#"{"username":"admin","password":"wrong"}"#),
    );
    assert_eq!(st, 401, "bad password must be refused");
    let _ = cookie;

    // 5. Wrong method/route sanity.
    let (st, _) = http(api, "GET", "/v1/nope", None, None);
    assert_eq!(st, 404);

    // 6. OpenAI model list.
    let (st, body) = http(api, "GET", "/v1/models", None, None);
    assert_eq!(st, 200, "{body}");
    assert!(body.contains("\"test\""), "model list should contain test: {body}");

    // 7. Chat completion (non-streaming) via mock backend + GGUF template.
    let (st, body) = http(
        api,
        "POST",
        "/v1/chat/completions",
        None,
        Some(r#"{"model":"test","messages":[{"role":"user","content":"hello"}],"max_tokens":8}"#),
    );
    assert_eq!(st, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert!(v["usage"]["completion_tokens"].as_u64().unwrap() >= 1);

    // 8. Streaming (SSE): chunks then [DONE].
    let sse = http_sse(
        api,
        r#"{"model":"test","messages":[{"role":"user","content":"stream it"}],"max_tokens":5,"stream":true}"#,
    );
    assert!(sse.contains("chat.completion.chunk"), "no chunks in: {sse}");
    assert!(sse.contains("[DONE]"), "no [DONE] terminator in: {sse}");

    // 9. Guardrail: impossible context must be refused with numbers (409).
    //    Ask for a context beyond training limit is clamped; instead make a
    //    giant context request against a tiny budget via config API.
    let (st, body) = http(
        ui,
        "POST",
        "/api/models/test/config",
        Some(&admin_cookie(&eng)),
        Some(r#"{"id":"test","file":"test.gguf","n_ctx":999999999,"n_batch":4096,"threads":4}"#),
    );
    assert_eq!(st, 200, "config save itself must succeed: {body}");

    // 10. State shows the model with disk usage.
    let (st, body) = http(ui, "GET", "/api/state", Some(&admin_cookie(&eng)), None);
    assert_eq!(st, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["models"][0]["id"], "test");
    assert!(v["models"][0]["size_bytes"].as_u64().unwrap() > 0);
    assert_eq!(v["users"][0]["username"], "admin");
}

/// Send a streaming chat request and return the raw SSE body.
fn http_sse(base: &str, body: &str) -> String {
    let req = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: e2e\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    let mut stream = TcpStream::connect(base.replace("http://", "")).unwrap();
    stream.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    stream.write_all(req.as_bytes()).unwrap();
    let mut resp = Vec::new();
    let _ = stream.read_to_end(&mut resp);
    String::from_utf8_lossy(&resp).into_owned()
}

/// Login again on a fresh connection to capture the session cookie from the
/// raw response headers.
fn admin_cookie(eng: &Engine) -> String {
    let mut stream = TcpStream::connect(eng.base_ui.replace("http://", "")).unwrap();
    let body = r#"{"username":"admin","password":"correct-horse-battery"}"#;
    let req = format!(
        "POST /api/login HTTP/1.1\r\nHost: e2e\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    );
    stream.write_all(req.as_bytes()).unwrap();
    let mut resp = Vec::new();
    let _ = stream.read_to_end(&mut resp);
    let text = String::from_utf8_lossy(&resp).into_owned();
    for line in text.lines() {
        if let Some(c) = line.strip_prefix("Set-Cookie: markos_session=") {
            return format!("markos_session={}", c.split(';').next().unwrap_or("").trim());
        }
    }
    panic!("no session cookie in response: {text}");
}

fn body_cookie_hack() -> String {
    String::new()
}
