//! Model manager: registry (disk) vs. resident (RAM) with the explicit
//! multi-model policy from design doc §8.4 — load/unload on switch, up to
//! `max_resident` concurrent, LRU eviction of idle models when a new load
//! needs room, bounded FIFO queue with 429 on overflow.

use crate::config::{EngineConfig, ModelConfig};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

#[allow(dead_code)] // Failed is reported via LoadError before insertion
pub enum SlotState {
    Loading,
    Ready(Box<dyn crate::backend::Backend>),
    Failed(String),
}

impl std::fmt::Debug for SlotState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlotState::Loading => write!(f, "Loading"),
            SlotState::Failed(e) => write!(f, "Failed({e})"),
            SlotState::Ready(_) => write!(f, "Ready"),
        }
    }
}

pub struct Slot {
    pub model_id: String,
    pub state: SlotState,
    pub busy: bool,
    pub last_used: Instant,
    pub n_ctx: u64,
}

struct Inner {
    slots: Vec<Slot>,
    waiters: usize,
}

struct Shared {
    inner: Mutex<Inner>,
    cv: Condvar,
}

#[derive(Clone)]
pub struct Manager {
    shared: Arc<Shared>,
}

impl Manager {
    pub fn new() -> Manager {
        Manager {
            shared: Arc::new(Shared { inner: Mutex::new(Inner { slots: Vec::new(), waiters: 0 }), cv: Condvar::new() }),
        }
    }

    pub fn slot_summaries(&self) -> Vec<SlotSummary> {
        let inner = self.shared.inner.lock().unwrap();
        inner
            .slots
            .iter()
            .map(|s| SlotSummary {
                model_id: s.model_id.clone(),
                state: match &s.state {
                    SlotState::Loading => "loading".into(),
                    SlotState::Ready(_) => "ready".into(),
                    SlotState::Failed(e) => format!("failed: {e}"),
                },
                busy: s.busy,
                idle_for_s: s.last_used.elapsed().as_secs(),
                n_ctx: s.n_ctx,
            })
            .collect()
    }

    /// Bytes resident in a Ready slot, excluding `exclude_id` — used by the
    /// guardrail when checking whether another model fits.
    pub fn resident_bytes_excluding(&self, exclude_id: &str) -> u64 {
        let inner = self.shared.inner.lock().unwrap();
        inner
            .slots
            .iter()
            .filter(|s| s.model_id != exclude_id)
            .filter_map(|s| match &s.state {
                SlotState::Ready(b) => Some(b.info().weights_bytes),
                _ => None,
            })
            .sum()
    }

    pub fn resident_ids(&self) -> Vec<String> {
        let inner = self.shared.inner.lock().unwrap();
        inner
            .slots
            .iter()
            .filter(|s| matches!(s.state, SlotState::Ready(_)))
            .map(|s| s.model_id.clone())
            .collect()
    }

    /// Unload a model if present and idle. Ok(true)=was resident and now
    /// unloaded; Ok(false)=not resident; Err=busy.
    pub fn unload(&self, model_id: &str) -> Result<bool, String> {
        let mut inner = self.shared.inner.lock().unwrap();
        let Some(pos) = inner.slots.iter().position(|s| s.model_id == model_id) else {
            return Ok(false);
        };
        if inner.slots[pos].busy {
            return Err("model is busy generating; try again shortly".into());
        }
        inner.slots.remove(pos);
        drop(inner);
        self.shared.cv.notify_all();
        Ok(true)
    }

    /// Acquire exclusive generation access to a model, loading/evicting as
    /// needed. `loader` performs the guardrail check + backend construction
    /// OUTSIDE the manager lock (loading takes seconds).
    pub fn acquire(
        &self,
        model_id: &str,
        cfg: &EngineConfig,
        loader: &mut dyn FnMut(&ModelConfig) -> Result<Box<dyn crate::backend::Backend>, LoadError>,
        model_cfg: &ModelConfig,
    ) -> Result<AcquiredGuard, AcquireError> {
        let deadline = Instant::now() + Duration::from_secs(cfg.queue_timeout_s.max(1));
        let mut inner = self.shared.inner.lock().unwrap();
        loop {
            if Instant::now() > deadline {
                return Err(AcquireError::Timeout);
            }
            if let Some(idx) = inner.slots.iter().position(|s| s.model_id == model_id) {
                match &inner.slots[idx].state {
                    SlotState::Ready(_) => {
                        if !inner.slots[idx].busy {
                            inner.slots[idx].busy = true;
                            inner.slots[idx].last_used = Instant::now();
                            let guard = AcquiredGuard {
                                shared: self.shared.clone(),
                                model_id: model_id.to_string(),
                            };
                            return Ok(guard);
                        }
                        // busy: queue for the slot
                        inner = self.queue_wait(inner, cfg, &deadline)?;
                    }
                    SlotState::Loading => {
                        inner = self.queue_wait(inner, cfg, &deadline)?;
                    }
                    SlotState::Failed(e) => {
                        let err = e.clone();
                        inner.slots.remove(idx);
                        return Err(AcquireError::LoadFailed(LoadError::Other(err)));
                    }
                }
                continue;
            }

            // Model not resident: need to load.
            if inner.slots.len() < cfg.max_resident.max(1) {
                inner.slots.push(Slot {
                    model_id: model_id.to_string(),
                    state: SlotState::Loading,
                    busy: true,
                    last_used: Instant::now(),
                    n_ctx: model_cfg.n_ctx,
                });
                drop(inner);
                let result = loader(model_cfg);
                let mut inner = self.shared.inner.lock().unwrap();
                let idx = inner
                    .slots
                    .iter()
                    .position(|s| s.model_id == model_id)
                    .expect("our Loading slot vanished");
                match result {
                    Ok(backend) => {
                        inner.slots[idx].state = SlotState::Ready(backend);
                        inner.slots[idx].last_used = Instant::now();
                        let guard = AcquiredGuard {
                            shared: self.shared.clone(),
                            model_id: model_id.to_string(),
                        };
                        drop(inner);
                        self.shared.cv.notify_all();
                        return Ok(guard);
                    }
                    Err(e) => {
                        inner.slots.remove(idx);
                        drop(inner);
                        self.shared.cv.notify_all();
                        return Err(AcquireError::LoadFailed(e));
                    }
                }
            }

            // At capacity: evict the LRU idle Ready slot, else queue.
            let evict = inner
                .slots
                .iter()
                .enumerate()
                .filter(|(_, s)| !s.busy && matches!(s.state, SlotState::Ready(_)))
                .min_by_key(|(_, s)| s.last_used)
                .map(|(i, _)| i);
            if let Some(idx) = evict {
                inner.slots.remove(idx);
                drop(inner);
                self.shared.cv.notify_all();
                // loop will now find capacity
                inner = self.shared.inner.lock().unwrap();
                continue;
            }
            inner = self.queue_wait(inner, cfg, &deadline)?;
        }
    }

    fn queue_wait<'a>(
        &'a self,
        mut inner: std::sync::MutexGuard<'a, Inner>,
        cfg: &EngineConfig,
        deadline: &Instant,
    ) -> Result<std::sync::MutexGuard<'a, Inner>, AcquireError> {
        if inner.waiters >= cfg.queue_depth {
            return Err(AcquireError::QueueFull);
        }
        inner.waiters += 1;
        let remaining = deadline.saturating_duration_since(Instant::now());
        let (guard, to) = self
            .shared
            .cv
            .wait_timeout(inner, remaining.min(Duration::from_secs(5)))
            .unwrap();
        let timed_out = to.timed_out() && Instant::now() >= *deadline;
        let mut inner = guard;
        inner.waiters = inner.waiters.saturating_sub(1);
        if timed_out {
            return Err(AcquireError::Timeout);
        }
        Ok(inner)
    }
}

impl Default for Manager {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct SlotSummary {
    pub model_id: String,
    pub state: String,
    pub busy: bool,
    pub idle_for_s: u64,
    pub n_ctx: u64,
}

pub enum LoadError {
    Guardrail(crate::guard::GuardrailRejection),
    Other(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Guardrail(g) => write!(
                f,
                "memory guardrail: {} (need {}, have {})",
                g.reason,
                crate::guard::format_bytes(g.required_bytes),
                crate::guard::format_bytes(g.available_bytes)
            ),
            LoadError::Other(s) => write!(f, "{s}"),
        }
    }
}

impl std::fmt::Debug for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}

impl From<String> for LoadError {
    fn from(s: String) -> Self {
        LoadError::Other(s)
    }
}

pub enum AcquireError {
    QueueFull,
    Timeout,
    LoadFailed(LoadError),
}

impl std::fmt::Debug for AcquireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcquireError::QueueFull => write!(f, "QueueFull"),
            AcquireError::Timeout => write!(f, "Timeout"),
            AcquireError::LoadFailed(e) => write!(f, "LoadFailed({e:?})"),
        }
    }
}

/// Exclusive access to a model slot; releases on drop.
pub struct AcquiredGuard {
    shared: Arc<Shared>,
    pub model_id: String,
}

impl AcquiredGuard {
    /// Run `f` with the resident backend. The slot is busy for the duration,
    /// so nothing can evict or unload it.
    pub fn with_backend<R>(&self, f: impl FnOnce(&mut dyn crate::backend::Backend) -> R) -> R {
        let mut inner = self.shared.inner.lock().unwrap();
        let slot = inner
            .slots
            .iter_mut()
            .find(|s| s.model_id == self.model_id)
            .expect("guard holds a live slot");
        match &mut slot.state {
            SlotState::Ready(b) => f(b.as_mut()),
            _ => panic!("slot not ready while guarded"),
        }
    }
}

impl Drop for AcquiredGuard {
    fn drop(&mut self) {
        let mut inner = self.shared.inner.lock().unwrap();
        if let Some(slot) = inner.slots.iter_mut().find(|s| s.model_id == self.model_id) {
            slot.busy = false;
            slot.last_used = Instant::now();
        }
        drop(inner);
        self.shared.cv.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::mock::MockBackend;
    use crate::backend::BackendInfo;
    use std::sync::atomic::AtomicUsize;

    fn cfg(max_resident: usize, queue_depth: usize) -> EngineConfig {
        EngineConfig {
            max_resident,
            queue_depth,
            queue_timeout_s: 2,
            ..Default::default()
        }
    }

    fn model_cfg(id: &str) -> ModelConfig {
        ModelConfig { id: id.into(), file: format!("{id}.gguf"), n_ctx: 2048, ..Default::default() }
    }

    fn loader(seen: &AtomicUsize) -> impl FnMut(&ModelConfig) -> Result<Box<dyn crate::backend::Backend>, LoadError> + '_ {
        move |c: &ModelConfig| {
            seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Ok(Box::new(MockBackend::with_info(
                BackendInfo {
                    id: c.id.clone(),
                    arch: "mock".into(),
                    n_ctx_train: 4096,
                    vocab: 32,
                    weights_bytes: 10_000_000,
                },
                c.n_ctx,
            )) as Box<dyn crate::backend::Backend>)
        }
    }

    #[test]
    fn load_once_acquire_twice() {
        let m = Manager::new();
        let seen = AtomicUsize::new(0);
        let mc = model_cfg("a");
        {
            let g = m.acquire("a", &cfg(2, 2), &mut loader(&seen), &mc).unwrap();
            g.with_backend(|be| {
                let s = be
                    .generate(&crate::backend::GenParams {
                        prompt: "p".into(),
                        max_tokens: 2,
                        stop: vec![],
                        sampling: Default::default(),
                        cancel: Default::default(),
                    }, &mut |_| {})
                    .unwrap();
                assert_eq!(s.stop_reason, "length");
            });
        }
        // Released; acquire again must NOT reload.
        {
            let mut loader_fn = loader(&seen);
            let g = m.acquire("a", &cfg(2, 2), &mut loader_fn, &mc).unwrap();
            g.with_backend(|_| {});
        }
        assert_eq!(seen.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[test]
    fn lru_eviction_at_capacity() {
        let m = Manager::new();
        let seen = AtomicUsize::new(0);
        let ec = cfg(1, 2);
        let mut loader_fn = loader(&seen);
        let _ = m.acquire("a", &ec, &mut loader_fn, &model_cfg("a")).unwrap();
        let _ = m.acquire("b", &ec, &mut loader_fn, &model_cfg("b")).unwrap();
        // capacity 1: b must have evicted a
        assert_eq!(m.resident_ids(), vec!["b".to_string()]);
        assert_eq!(seen.load(std::sync::atomic::Ordering::SeqCst), 2);
    }

    #[test]
    fn queue_full_rejects() {
        let m = Manager::new();
        let seen = AtomicUsize::new(0);
        let ec = cfg(1, 1);
        let mc = model_cfg("a");
        let mut loader_fn = loader(&seen);
        let g = m.acquire("a", &ec, &mut loader_fn, &mc).unwrap();
        // slot busy, queue depth 1: first extra waits (would succeed), fill queue with 1 waiter thread? 
        // simpler: queue depth 0 → immediate QueueFull
        let ec0 = cfg(1, 0);
        let r = m.acquire("a", &ec0, &mut loader_fn, &mc);
        assert!(matches!(r, Err(AcquireError::QueueFull)));
        drop(g);
    }

    #[test]
    fn concurrent_load_is_deduplicated() {
        // Two threads acquiring the same absent model must load it once.
        let m = Arc::new(Manager::new());
        let seen = Arc::new(AtomicUsize::new(0));
        let ec = cfg(2, 2);
        let mc = model_cfg("x");
        let handles: Vec<_> = (0..2)
            .map(|_| {
                let m = m.clone();
                let seen = seen.clone();
                let ec = ec.clone();
                let mc = mc.clone();
                std::thread::spawn(move || {
                    let mut loader_fn = move |c: &ModelConfig| {
                        // slow load so both threads race
                        std::thread::sleep(Duration::from_millis(50));
                        seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                        Ok(Box::new(MockBackend::with_info(
                            BackendInfo {
                                id: c.id.clone(),
                                arch: "mock".into(),
                                n_ctx_train: 4096,
                                vocab: 32,
                                weights_bytes: 1,
                            },
                            c.n_ctx,
                        )) as Box<dyn crate::backend::Backend>)
                    };
                    // Release the guard inside the thread: join order must
                    // not depend on which thread waits for the slot.
                    let g = m
                        .acquire("x", &ec, &mut loader_fn, &mc)
                        .map_err(|_| std::io::Error::new(std::io::ErrorKind::Other, "acquire failed"))?;
                    drop(g);
                    Ok::<(), std::io::Error>(())
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap().unwrap();
        }
        assert_eq!(seen.load(std::sync::atomic::Ordering::SeqCst), 1);
    }
}
