//! Custom HTTP/1.1 server — std::net + bounded thread pool. This is the
//! serving layer of MarkOS: keep-alive, Content-Length and chunked request
//! bodies, Expect: 100-continue, and raw streaming (SSE) responses. No async
//! runtime: one thread per connection capped by a semaphore; generation
//! streams write directly to the connection.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

pub const MAX_HEADER_BYTES: usize = 32 * 1024;
pub const MAX_BODY_BYTES: usize = 64 * 1024 * 1024;

pub trait ReadWrite: Read + Write + Send {}
impl<T: Read + Write + Send> ReadWrite for T {}

pub type BoxedStream = Box<dyn ReadWrite>;

/// Optional TLS wrapper around accepted sockets (feature `tls`).
pub trait StreamWrap: Send + Sync {
    fn wrap(&self, sock: TcpStream) -> std::io::Result<BoxedStream>;
}

#[derive(Debug)]
pub struct Request {
    pub method: String,
    /// Path without query string, percent-decoding left to the caller (we
    /// only route on ASCII paths).
    pub path: String,
    pub query: String,
    pub headers: HashMap<String, String>, // keys lowercased
    pub body: Vec<u8>,
    pub peer: String,
}

impl Request {
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(&name.to_ascii_lowercase()).map(|s| s.as_str())
    }
    pub fn query_param(&self, name: &str) -> Option<String> {
        for pair in self.query.split('&') {
            let mut it = pair.splitn(2, '=');
            if it.next()? == name {
                return it.next().map(|v| v.replace("+", " ").percent_decode());
            }
        }
        None
    }
    pub fn json<T: serde::de::DeserializeOwned>(&self) -> Result<T, String> {
        serde_json::from_slice(&self.body).map_err(|e| format!("invalid JSON body: {e}"))
    }
}

trait PercentDecode {
    fn percent_decode(&self) -> String;
}
impl PercentDecode for String {
    fn percent_decode(&self) -> String {
        let b = self.as_bytes();
        let mut out = Vec::with_capacity(b.len());
        let mut i = 0;
        while i < b.len() {
            if b[i] == b'%' && i + 2 < b.len() + 1 && i + 2 < b.len() + 1 {
                if let Ok(v) = u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).unwrap_or(""), 16) {
                    out.push(v);
                    i += 3;
                    continue;
                }
            }
            out.push(b[i]);
            i += 1;
        }
        String::from_utf8_lossy(&out).into_owned()
    }
}

/// Response control passed to handlers.
pub struct ResponseOut {
    stream: BoxedStream,
    head_written: bool,
    status: u16,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
    keep_alive: bool,
    is_head: bool,
    stream_failed: bool,
}

impl ResponseOut {
    pub fn status(&self) -> u16 {
        self.status
    }
    /// True once a streamed head is on the wire.
    pub fn is_streaming(&self) -> bool {
        self.head_written
    }
    /// True once a handler has chosen a status (0 = untouched).
    pub fn answered(&self) -> bool {
        self.status != 0
    }
    fn set_status(&mut self, s: u16) {
        self.status = s;
    }
    pub fn add_header(&mut self, name: &str, value: &str) {
        self.headers.push((name.into(), value.into()));
    }

    pub fn bytes(&mut self, status: u16, content_type: &str, body: Vec<u8>) {
        self.set_status(status);
        self.add_header("Content-Type", content_type);
        self.body = body;
    }

    pub fn text(&mut self, status: u16, s: &str) {
        self.bytes(status, "text/plain; charset=utf-8", s.as_bytes().to_vec());
    }

    pub fn json<V: serde::Serialize>(&mut self, status: u16, v: &V) {
        match serde_json::to_vec(v) {
            Ok(b) => self.bytes(status, "application/json", b),
            Err(e) => self.text(500, &format!("serialize error: {e}")),
        }
    }

    /// OpenAI-style error object.
    pub fn error(&mut self, status: u16, message: &str) {
        let body = serde_json::json!({
            "error": { "message": message, "type": error_type(status), "code": status }
        });
        self.json(status, &body);
    }

    /// Begin a streamed response (no Content-Length; connection closes at the
    /// end). Returns once the head is on the wire.
    pub fn begin_stream(&mut self, status: u16, content_type: &str) -> std::io::Result<()> {
        self.set_status(status);
        self.add_header("Content-Type", content_type);
        self.add_header("Cache-Control", "no-cache");
        self.add_header("X-Accel-Buffering", "no");
        self.keep_alive = false;
        self.write_head(true)?;
        Ok(())
    }

    /// Write a raw chunk of a streamed response.
    pub fn write_stream(&mut self, data: &[u8]) -> std::io::Result<()> {
        if !self.head_written {
            self.begin_stream(200, "text/event-stream")?;
        }
        match self.stream.write_all(data).and_then(|_| self.stream.flush()) {
            Ok(()) => Ok(()),
            Err(e) => {
                self.stream_failed = true;
                Err(e)
            }
        }
    }

    fn write_head(&mut self, no_length: bool) -> std::io::Result<()> {
        if self.head_written {
            return Ok(());
        }
        let reason = reason_phrase(self.status);
        let mut head = format!("HTTP/1.1 {} {}\r\nServer: markos-engine/{}\r\n", self.status, reason, crate::config::ENGINE_VERSION);
        let has_ct = self.headers.iter().any(|(n, _)| n.eq_ignore_ascii_case("content-type"));
        if !has_ct && !no_length {
            head.push_str("Content-Type: application/json\r\n");
        }
        for (n, v) in &self.headers {
            head.push_str(n);
            head.push_str(": ");
            head.push_str(v);
            head.push_str("\r\n");
        }
        if no_length {
            head.push_str("Connection: close\r\n\r\n");
        } else {
            head.push_str(&format!("Content-Length: {}\r\n", self.body.len()));
            head.push_str(if self.keep_alive { "Connection: keep-alive\r\n" } else { "Connection: close\r\n" });
            head.push_str("\r\n");
        }
        self.stream.write_all(head.as_bytes())?;
        self.head_written = true;
        if self.is_head && !no_length {
            self.stream.flush()?;
            return Ok(());
        }
        Ok(())
    }

    fn finish(&mut self) -> std::io::Result<()> {
        if self.stream_failed {
            return Ok(());
        }
        if !self.head_written {
            self.write_head(false)?;
            if !self.is_head {
                self.stream.write_all(&self.body)?;
            }
            self.stream.flush()?;
        } else {
            self.stream.flush()?;
        }
        Ok(())
    }
}

fn reason_phrase(code: u16) -> &'static str {
    match code {
        100 => "Continue",
        200 => "OK",
        201 => "Created",
        202 => "Accepted",
        204 => "No Content",
        302 => "Found",
        304 => "Not Modified",
        400 => "Bad Request",
        401 => "Unauthorized",
        403 => "Forbidden",
        404 => "Not Found",
        405 => "Method Not Allowed",
        408 => "Request Timeout",
        409 => "Conflict",
        413 => "Payload Too Large",
        422 => "Unprocessable Entity",
        429 => "Too Many Requests",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        503 => "Service Unavailable",
        504 => "Gateway Timeout",
        _ => "Unknown",
    }
}

fn error_type(status: u16) -> &'static str {
    match status {
        400..=422 => "invalid_request_error",
        429 => "rate_limit_error",
        _ => "server_error",
    }
}

/// Handler contract. Streaming handlers use `out.begin_stream` /
/// `out.write_stream`; everything else fills `out` with a buffered response.
pub trait Handler: Send + Sync {
    fn handle(&self, req: &Request, out: &mut ResponseOut);
}

/// Encode one SSE frame.
pub fn sse_event(out: &mut ResponseOut, event: &str, data: &str) -> std::io::Result<()> {
    let mut frame = String::with_capacity(data.len() + event.len() + 16);
    if !event.is_empty() {
        frame.push_str("event: ");
        frame.push_str(event);
        frame.push('\n');
    }
    frame.push_str("data: ");
    frame.push_str(data);
    frame.push_str("\n\n");
    out.write_stream(frame.as_bytes())
}

struct ConnGate {
    active: AtomicUsize,
    max: usize,
    cv: Condvar,
    m: Mutex<()>,
}

impl ConnGate {
    fn new(max: usize) -> Self {
        ConnGate { active: AtomicUsize::new(0), max, cv: Condvar::new(), m: Mutex::new(()) }
    }
    fn enter(&self) -> bool {
        let mut g = self.m.lock().unwrap();
        loop {
            if self.active.load(Ordering::SeqCst) < self.max {
                self.active.fetch_add(1, Ordering::SeqCst);
                return true;
            }
            let (gg, _to) = self
                .cv
                .wait_timeout(g, Duration::from_secs(5))
                .unwrap();
            g = gg;
            if _to.timed_out() && self.active.load(Ordering::SeqCst) >= self.max {
                return false;
            }
        }
    }
    fn leave(&self) {
        self.active.fetch_sub(1, Ordering::SeqCst);
        self.cv.notify_one();
    }
}

pub struct Server {
    gate: Arc<ConnGate>,
    pub wrap: Option<Arc<dyn StreamWrap>>,
}

impl Server {
    pub fn new(max_conns: usize) -> Server {
        Server { gate: Arc::new(ConnGate::new(max_conns)), wrap: None }
    }

    pub fn with_wrap(max_conns: usize, wrap: Arc<dyn StreamWrap>) -> Server {
        Server { gate: Arc::new(ConnGate::new(max_conns)), wrap: Some(wrap) }
    }

    /// Accept loop; spawns a thread per connection. One call per listener.
    pub fn serve(self: &Arc<Self>, listener: TcpListener, handler: Arc<dyn Handler>) -> ! {
        loop {
            match listener.accept() {
                Ok((sock, peer)) => {
                    sock.set_read_timeout(Some(Duration::from_secs(300))).ok();
                    sock.set_write_timeout(Some(Duration::from_secs(120))).ok();
                    sock.set_nodelay(true).ok();
                    let h = handler.clone();
                    let gate = self.gate.clone();
                    let stream: Result<BoxedStream, std::io::Error> = match &self.wrap {
                        Some(w) => w.wrap(sock),
                        None => Ok(Box::new(sock)),
                    };
                    match stream {
                        Ok(s) => {
                            if !gate.enter() {
                                drop(s);
                                continue;
                            }
                            std::thread::Builder::new()
                                .name("http-conn".into())
                                .spawn(move || {
                                    let _ = serve_stream(s, peer.to_string(), h);
                                    gate.leave();
                                })
                                .ok();
                        }
                        Err(_) => continue, // failed handshake (TLS junk probe etc.)
                    }
                }
                Err(_) => {
                    std::thread::sleep(Duration::from_millis(50));
                }
            }
        }
    }
}

/// Read requests off a stream until close (keep-alive loop).
pub fn serve_stream(mut stream: BoxedStream, peer: String, handler: Arc<dyn Handler>) -> std::io::Result<()> {
    let mut buf: Vec<u8> = Vec::with_capacity(8 * 1024);
    loop {
        let req = match read_request(&mut stream, &mut buf, &peer) {
            Ok(Some(r)) => r,
            Ok(None) => return Ok(()), // clean EOF between requests
            Err(e) => {
                // Malformed request: best-effort 400.
                let msg = format!("HTTP/1.1 400 Bad Request\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}", e.len(), e);
                let _ = stream.write_all(msg.as_bytes());
                return Ok(());
            }
        };
        let client_close = req
            .header("connection")
            .map(|v| v.to_ascii_lowercase().contains("close"))
            .unwrap_or(false);
        let is_head = req.method == "HEAD";

        let mut out = ResponseOut {
            stream,
            head_written: false,
            status: 0,
            headers: Vec::new(),
            body: Vec::new(),
            keep_alive: !client_close,
            is_head,
            stream_failed: false,
        };
        handler.handle(&req, &mut out);
        if !out.answered() {
            out.error(500, "handler produced no response");
        }
        let keep = out.keep_alive && !out.stream_failed;
        out.finish()?;
        stream = std::mem::replace(&mut out.stream, Box::new(NullStream));
        // Access log is done by the dispatcher; here we just honor keep-alive.
        if !keep {
            return Ok(());
        }
    }
}

/// Placeholder stream used when taking the real stream back out of a
/// finished `ResponseOut`.
struct NullStream;

impl Read for NullStream {
    fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
        Ok(0)
    }
}
impl Write for NullStream {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Parse one request. `Ok(None)` = orderly EOF before any bytes.
fn read_request(stream: &mut BoxedStream, buf: &mut Vec<u8>, peer: &str) -> Result<Option<Request>, String> {
    // 1. header block
    let header_end = loop {
        if let Some(pos) = find_subslice(buf, b"\r\n\r\n") {
            break pos;
        }
        if buf.len() > MAX_HEADER_BYTES {
            return Err("headers too large".into());
        }
        let mut chunk = [0u8; 4096];
        let n = stream.read(&mut chunk).map_err(|e| e.to_string())?;
        if n == 0 {
            if buf.is_empty() {
                return Ok(None);
            }
            return Err("connection closed mid-request".into());
        }
        buf.extend_from_slice(&chunk[..n]);
    };

    let header_block = String::from_utf8_lossy(&buf[..header_end]).into_owned();
    let mut lines = header_block.split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().ok_or("missing method")?.to_string();
    let target = parts.next().ok_or("missing target")?.to_string();
    let version = parts.next().unwrap_or("HTTP/1.1").to_string();
    if !version.starts_with("HTTP/1") {
        return Err("unsupported HTTP version".into());
    }
    let (path, query) = match target.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (target.clone(), String::new()),
    };
    let mut headers = HashMap::new();
    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some((name, value)) = line.split_once(':') {
            headers.insert(name.trim().to_ascii_lowercase(), value.trim().to_string());
        }
    }

    // 100-continue handshake before body read.
    if headers
        .get("expect")
        .map(|v| v.to_ascii_lowercase().contains("100-continue"))
        .unwrap_or(false)
    {
        stream.write_all(b"HTTP/1.1 100 Continue\r\n\r\n").map_err(|e| e.to_string())?;
    }

    // 2. body
    let body: Vec<u8>;
    let chunked = headers
        .get("transfer-encoding")
        .map(|v| v.to_ascii_lowercase().contains("chunked"))
        .unwrap_or(false);
    if chunked {
        let mut rest = buf[header_end + 4..].to_vec();
        body = read_chunked(stream, &mut rest)?;
        *buf = rest;
    } else {
        let content_length: usize = headers
            .get("content-length")
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);
        if content_length > MAX_BODY_BYTES {
            return Err("body too large".into());
        }
        let mut rest: Vec<u8> = buf[header_end + 4..].to_vec();
        if rest.len() < content_length {
            let mut extra = vec![0u8; content_length - rest.len()];
            stream.read_exact(&mut extra).map_err(|e| e.to_string())?;
            rest.extend_from_slice(&extra);
        }
        body = rest[..content_length].to_vec();
        *buf = rest[content_length..].to_vec();
    }
    if buf.len() > 1024 * 1024 {
        buf.drain(..buf.len() - 8192); // keep it bounded between requests
    }

    Ok(Some(Request { method, path, query, headers, body, peer: peer.to_string() }))
}

fn read_chunked(stream: &mut BoxedStream, rest: &mut Vec<u8>) -> Result<Vec<u8>, String> {
    let mut body = Vec::new();
    loop {
        // read a size line
        let line = loop {
            if let Some(pos) = find_subslice(rest, b"\r\n") {
                let l: Vec<u8> = rest.drain(..pos + 2).collect();
                break String::from_utf8_lossy(&l[..pos]).into_owned();
            }
            let mut chunk = [0u8; 1024];
            let n = stream.read(&mut chunk).map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("eof in chunked body".into());
            }
            rest.extend_from_slice(&chunk[..n]);
        };
        let size_str = line.split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_str, 16).map_err(|_| "bad chunk size")?;
        if body.len() + size > MAX_BODY_BYTES {
            return Err("chunked body too large".into());
        }
        if size == 0 {
            // trailers until blank line
            loop {
                if let Some(pos) = find_subslice(rest, b"\r\n") {
                    let _l: Vec<u8> = rest.drain(..pos + 2).collect();
                    if pos == 0 {
                        return Ok(body);
                    }
                    continue;
                }
                let mut chunk = [0u8; 1024];
                let n = stream.read(&mut chunk).map_err(|e| e.to_string())?;
                if n == 0 {
                    return Ok(body); // tolerate missing final CRLF
                }
                rest.extend_from_slice(&chunk[..n]);
            }
        }
        while rest.len() < size + 2 {
            let mut chunk = [0u8; 8192];
            let n = stream.read(&mut chunk).map_err(|e| e.to_string())?;
            if n == 0 {
                return Err("eof mid-chunk".into());
            }
            rest.extend_from_slice(&chunk[..n]);
        }
        body.extend_from_slice(&rest[..size]);
        rest.drain(..size);
        // consume trailing CRLF
        if rest.starts_with(b"\r\n") {
            rest.drain(..2);
        }
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Echo;
    impl Handler for Echo {
        fn handle(&self, req: &Request, out: &mut ResponseOut) {
            if req.path == "/stream" {
                out.begin_stream(200, "text/event-stream").unwrap();
                for i in 0..3 {
                    sse_event(out, "delta", &format!("\"tok{i}\"")).unwrap();
                }
                sse_event(out, "", "[DONE]").unwrap();
                return;
            }
            out.json(200, &serde_json::json!({
                "method": req.method,
                "path": req.path,
                "body_len": req.body.len(),
                "echo": String::from_utf8_lossy(&req.body),
            }));
        }
    }

    #[test]
    fn http_keepalive_and_sse() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let srv = Arc::new(Server::new(16));
        std::thread::spawn(move || {
            srv.serve(listener, Arc::new(Echo));
        });

        let mut child = std::process::Command::new("curl")
            .args([
                "-s", "-i",
                &format!("http://{addr}/hello"),
                "-d", "payload=1",
                &format!("http://{addr}/stream"),
            ])
            .stdout(std::process::Stdio::piped())
            .spawn()
            .expect("curl must exist for http tests");
        // Give curl time to finish both requests and close.
        let mut out = String::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        {
            use std::io::Read;
            let mut so = child.stdout.take().unwrap();
            let mut tmp = [0u8; 4096];
            while std::time::Instant::now() < deadline {
                match so.read(&mut tmp) {
                    Ok(0) => break,
                    Ok(n) => out.push_str(&String::from_utf8_lossy(&tmp[..n])),
                    Err(_) => break,
                }
            }
        }
        let _ = child.wait();
        assert!(out.contains("\"path\":\"/hello\""), "missing json response: {out}");
        assert!(out.contains("\"echo\":\"payload=1\""));
        assert!(out.contains("text/event-stream"), "no SSE head: {out}");
        assert!(out.contains("data: \"tok2\""), "no SSE data: {out}");
        assert!(out.contains("data: [DONE]"));
    }
}
