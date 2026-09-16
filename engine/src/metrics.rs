//! Structured metrics + request log: in-memory ring (web UI) and append-only
//! JSONL (local log store, read back by the UI). No external logging service.

use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub ts: u64, // unix epoch millis
    pub kind: String, // "request" | "error" | "system" | "model"
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peer: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
    pub status: u16,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gen_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

struct Inner {
    ring: std::collections::VecDeque<LogEntry>,
    jsonl: Option<std::fs::File>,
    /// (timestamp, cumulative tokens_out) samples for tok/s windows.
    token_samples: std::collections::VecDeque<(Instant, u64)>,
}

pub struct Metrics {
    pub started: Instant,
    pub requests_total: AtomicU64,
    pub requests_4xx: AtomicU64,
    pub requests_5xx: AtomicU64,
    pub active: AtomicI64,
    pub queued: AtomicI64,
    pub tokens_out: AtomicU64,
    pub models_loaded: AtomicU64,
    /// Bytes currently resident across loaded models.
    pub resident_bytes: AtomicU64,
    inner: Mutex<Inner>,
    #[allow(dead_code)] // surfaced to operators via logs API docs
    pub log_path: PathBuf,
}

const RING_CAP: usize = 512;

impl Metrics {
    pub fn new(log_dir: &std::path::Path) -> Metrics {
        std::fs::create_dir_all(log_dir).ok();
        let log_path = log_dir.join("markos.jsonl");
        let jsonl = OpenOptions::new().create(true).append(true).open(&log_path).ok();
        Metrics {
            started: Instant::now(),
            requests_total: AtomicU64::new(0),
            requests_4xx: AtomicU64::new(0),
            requests_5xx: AtomicU64::new(0),
            active: AtomicI64::new(0),
            queued: AtomicI64::new(0),
            tokens_out: AtomicU64::new(0),
            models_loaded: AtomicU64::new(0),
            resident_bytes: AtomicU64::new(0),
            inner: Mutex::new(Inner {
                ring: std::collections::VecDeque::with_capacity(RING_CAP),
                jsonl,
                token_samples: std::collections::VecDeque::new(),
            }),
            log_path,
        }
    }

    pub fn now_ms() -> u64 {
        SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
    }

    pub fn record(&self, entry: LogEntry) {
        let mut g = self.inner.lock().unwrap();
        if let Some(f) = g.jsonl.as_mut() {
            let mut line = serde_json::to_string(&entry).unwrap_or_default();
            line.push('\n');
            let _ = f.write_all(line.as_bytes());
        }
        if g.ring.len() >= RING_CAP {
            g.ring.pop_front();
        }
        g.ring.push_back(entry);
    }

    pub fn tail(&self, n: usize) -> Vec<LogEntry> {
        let g = self.inner.lock().unwrap();
        g.ring.iter().rev().take(n).cloned().collect()
    }

    pub fn note_token(&self) {
        let total = self.tokens_out.fetch_add(1, Ordering::Relaxed) + 1;
        let mut g = self.inner.lock().unwrap();
        g.token_samples.push_back((Instant::now(), total));
        // Keep ~2 minutes of samples.
        while let Some((t, _)) = g.token_samples.front() {
            if t.elapsed() > Duration::from_secs(120) {
                g.token_samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Tokens/sec over the trailing `secs` window (0 if idle whole window).
    pub fn tokps_window(&self, secs: u64) -> f64 {
        let g = self.inner.lock().unwrap();
        let cutoff = Instant::now() - Duration::from_secs(secs);
        for (t, c) in g.token_samples.iter().rev() {
            if *t <= cutoff {
                let span = t.elapsed().as_secs_f64().max(0.001);
                return (g.token_samples.back().unwrap().1 - c) as f64 / span;
            }
        }
        // All samples are newer than the window: rate within available span.
        if let (Some((old, oc)), Some((_, nc))) = (g.token_samples.front(), g.token_samples.back()) {
            let span = old.elapsed().as_secs_f64().max(0.001);
            if span <= secs as f64 {
                return (nc - oc) as f64 / span;
            }
        }
        0.0
    }

    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "uptime_s": self.started.elapsed().as_secs(),
            "requests_total": self.requests_total.load(Ordering::Relaxed),
            "requests_4xx": self.requests_4xx.load(Ordering::Relaxed),
            "requests_5xx": self.requests_5xx.load(Ordering::Relaxed),
            "active": self.active.load(Ordering::Relaxed).max(0),
            "queued": self.queued.load(Ordering::Relaxed).max(0),
            "tokens_out": self.tokens_out.load(Ordering::Relaxed),
            "tokps_1s": self.tokps_window(1),
            "tokps_10s": self.tokps_window(10),
            "tokps_60s": self.tokps_window(60),
            "models_loaded": self.models_loaded.load(Ordering::Relaxed),
            "resident_bytes": self.resident_bytes.load(Ordering::Relaxed),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ring_and_windows() {
        let dir = std::env::temp_dir().join(format!("markos-metrics-{}", std::process::id()));
        let m = Metrics::new(&dir);
        m.note_token();
        m.note_token();
        m.record(LogEntry {
            ts: Metrics::now_ms(),
            kind: "request".into(),
            model: Some("m".into()),
            peer: Some("127.0.0.1".into()),
            path: Some("/v1/chat/completions".into()),
            status: 200,
            prompt_tokens: Some(5),
            gen_tokens: Some(2),
            ms: Some(10),
            tokps: Some(200.0),
            message: None,
        });
        assert_eq!(m.tail(10).len(), 1);
        assert_eq!(m.tokens_out.load(Ordering::Relaxed), 2);
        let snap = m.snapshot();
        assert_eq!(snap["tokens_out"], 2);
        // JSONL got a line.
        let text = std::fs::read_to_string(dir.join("markos.jsonl")).unwrap();
        assert!(text.trim().starts_with("{\"ts\":"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
