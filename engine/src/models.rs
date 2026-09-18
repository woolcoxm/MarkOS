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
    /// The backend is checked out into an `AcquiredGuard` while it
    /// generates. The manager mutex is NOT held during generation — other
    /// models can be acquired, the queue/metrics/readyz stay responsive,
    /// and a queued request's prefill overlaps the active decode (design §8.3).
    Generating,
    Failed(String),
}

impl std::fmt::Debug for SlotState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SlotState::Loading => write!(f, "Loading"),
            SlotState::Generating => write!(f, "Generating"),
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
    /// Guardrail budget of the resident backend (weights + kv + compute +
    /// margin), set when the slot becomes Ready. Counted against other
    /// models' loads so a second resident cannot overcommit RAM; kept
    /// distinct from `weights_bytes` because kv/compute/margin are anon
    /// memory that cannot be evicted the way mmap'd weights can.
    pub budget_bytes: u64,
}

impl SlotState {
    /// Weights of the resident backend, when one is here.
    fn backend_weights(&self) -> Option<u64> {
        match self {
            SlotState::Ready(b) => Some(b.info().weights_bytes),
            _ => None,
        }
    }
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
                    SlotState::Generating => "generating".into(),
                    SlotState::Failed(e) => format!("failed: {e}"),
                },
                busy: s.busy,
                idle_for_s: s.last_used.elapsed().as_secs(),
                n_ctx: s.n_ctx,
            })
            .collect()
    }

    /// Guardrail bytes held by Ready/Generating slots, excluding
    /// `exclude_id` — used by the guardrail when checking whether another
    /// model fits. Checked-out (Generating) slots count too: a generation
    /// in flight holds its KV + compute just the same. Loading slots count
    /// once their provisional estimate is known (see `set_loading_budget`),
    /// so two concurrent loads cannot both pass the check before either is
    /// resident.
    pub fn resident_bytes_excluding(&self, exclude_id: &str) -> u64 {
        let inner = self.shared.inner.lock().unwrap();
        inner
            .slots
            .iter()
            .filter(|s| s.model_id != exclude_id)
            .filter(|s| match s.state {
                SlotState::Ready(_) | SlotState::Generating => true,
                SlotState::Loading => s.budget_bytes > 0,
                SlotState::Failed(_) => false,
            })
            .map(|s| {
                if s.budget_bytes > 0 {
                    s.budget_bytes
                } else {
                    s.state
                        .backend_weights()
                        .unwrap_or(0)
                }
            })
            .sum()
    }

    /// Provisional guardrail budget for a Loading slot, recorded by the
    /// loader as soon as the RAM estimate exists (before the model itself
    /// is constructed). Cleared implicitly when the slot resolves.
    pub fn set_loading_budget(&self, model_id: &str, budget_bytes: u64) {
        let mut inner = self.shared.inner.lock().unwrap();
        if let Some(s) = inner.slots.iter_mut().find(|s| s.model_id == model_id) {
            s.budget_bytes = budget_bytes;
        }
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
    /// OUTSIDE the manager lock (loading takes seconds) and returns the
    /// backend plus its guardrail budget in bytes (the RAM estimate total).
    pub fn acquire(
        &self,
        model_id: &str,
        cfg: &EngineConfig,
        loader: &mut dyn FnMut(&ModelConfig) -> Result<(Box<dyn crate::backend::Backend>, u64), LoadError>,
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
                                checked_out: None,
                            };
                            return Ok(guard);
                        }
                        // busy: queue for the slot
                        inner = self.queue_wait(inner, cfg, &deadline)?;
                    }
                    SlotState::Loading | SlotState::Generating => {
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
                    budget_bytes: 0,
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
                    Ok((backend, budget_bytes)) => {
                        inner.slots[idx].state = SlotState::Ready(backend);
                        inner.slots[idx].budget_bytes = budget_bytes;
                        inner.slots[idx].last_used = Instant::now();
                        let guard = AcquiredGuard {
                            shared: self.shared.clone(),
                            model_id: model_id.to_string(),
                            checked_out: None,
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
    /// The backend while a generation is in flight (checked out of the slot
    /// so the manager mutex is not held). Restored by `with_backend` on the
    /// success path and by `Drop` on the unwind path.
    checked_out: Option<Box<dyn crate::backend::Backend>>,
}

impl AcquiredGuard {
    /// Run `f` with the resident backend. The slot is busy for the duration,
    /// so nothing can evict or unload it.
    ///
    /// The backend is checked OUT of the slot for the duration of `f`: the
    /// manager mutex is held only for the two pointer-sized swaps, never
    /// across the generation itself. Holding it (the previous behavior)
    /// froze every other acquire, the queue, metrics and /readyz for the
    /// whole generation and made the second resident model unservable while
    /// the first generated.
    pub fn with_backend<R>(
        &mut self,
        f: impl FnOnce(&mut dyn crate::backend::Backend) -> R,
    ) -> R {
        // Check out: Ready(backend) → Generating, taking the backend with us.
        let backend = {
            let mut inner = self.shared.inner.lock().unwrap();
            let slot = inner
                .slots
                .iter_mut()
                .find(|s| s.model_id == self.model_id)
                .expect("guard holds a live slot");
            match &mut slot.state {
                SlotState::Ready(_) => {
                    let old = std::mem::replace(&mut slot.state, SlotState::Generating);
                    match old {
                        SlotState::Ready(b) => b,
                        _ => unreachable!("state was Ready above"),
                    }
                }
                _ => panic!("slot not ready while guarded"),
            }
        };
        // Stash first so Drop restores the slot even if `f` unwinds.
        self.checked_out = Some(backend);
        let r = f(self.checked_out.as_mut().expect("just stored").as_mut());
        let backend = self.checked_out.take().expect("checked out above");
        // Check back in; the slot stays busy until the guard drops.
        let mut inner = self.shared.inner.lock().unwrap();
        let slot = inner
            .slots
            .iter_mut()
            .find(|s| s.model_id == self.model_id)
            .expect("guard holds a live slot");
        slot.state = SlotState::Ready(backend);
        drop(inner);
        r
    }
}

impl Drop for AcquiredGuard {
    fn drop(&mut self) {
        let mut inner = self.shared.inner.lock().unwrap();
        if let Some(slot) = inner.slots.iter_mut().find(|s| s.model_id == self.model_id) {
            // Unwind path: put a still-checked-out backend back.
            if let Some(b) = self.checked_out.take() {
                slot.state = SlotState::Ready(b);
            }
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

    fn loader(seen: &AtomicUsize) -> impl FnMut(&ModelConfig) -> Result<(Box<dyn crate::backend::Backend>, u64), LoadError> + '_ {
        move |c: &ModelConfig| {
            seen.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let be = Box::new(MockBackend::with_info(
                BackendInfo {
                    id: c.id.clone(),
                    arch: "mock".into(),
                    n_ctx_train: 4096,
                    vocab: 32,
                    weights_bytes: 10_000_000,
                },
                c.n_ctx,
            )) as Box<dyn crate::backend::Backend>;
            // Budget deliberately differs from weights: the guardrail must
            // count the estimate total, not just the file size.
            Ok((be, 50_000_000))
        }
    }

    #[test]
    fn load_once_acquire_twice() {
        let m = Manager::new();
        let seen = AtomicUsize::new(0);
        let mc = model_cfg("a");
        {
            let mut g = m.acquire("a", &cfg(2, 2), &mut loader(&seen), &mc).unwrap();
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
            let mut g = m.acquire("a", &cfg(2, 2), &mut loader_fn, &mc).unwrap();
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
                        let be = Box::new(MockBackend::with_info(
                            BackendInfo {
                                id: c.id.clone(),
                                arch: "mock".into(),
                                n_ctx_train: 4096,
                                vocab: 32,
                                weights_bytes: 1,
                            },
                            c.n_ctx,
                        )) as Box<dyn crate::backend::Backend>;
                        Ok((be, 1))
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

    #[test]
    fn generation_does_not_hold_the_manager_lock() {
        let m = Manager::new();
        let seen = AtomicUsize::new(0);
        let mut g = m
            .acquire("a", &cfg(2, 2), &mut loader(&seen), &model_cfg("a"))
            .unwrap();
        g.with_backend(|_be| {
            // Mid-generation the manager must answer immediately (previously
            // this deadlocked: with_backend held the mutex for the whole
            // generation) and the slot must read as generating.
            let sums = m.slot_summaries();
            assert_eq!(sums.len(), 1);
            assert_eq!(sums[0].state, "generating");
            assert!(sums[0].busy);
            // The checked-out model still counts against the guardrail.
            assert_eq!(m.resident_bytes_excluding("b"), 50_000_000);
        });
        // Backend is back in the slot after the call.
        let sums = m.slot_summaries();
        assert_eq!(sums[0].state, "ready");
    }

    #[test]
    fn guardrail_counts_full_budget_not_just_weights() {
        let m = Manager::new();
        let seen = AtomicUsize::new(0);
        let mut loader_fn = loader(&seen);
        {
            let mut g = m
            .acquire("a", &cfg(2, 2), &mut loader_fn, &model_cfg("a"))
            .unwrap();
        // Touch the backend so the slot is exercised, then release.
        g.with_backend(|_| {});
        }
        // Resident budget (kv+compute+margin included) ≠ weights_bytes.
        assert_eq!(m.resident_bytes_excluding("a"), 0);
        assert_eq!(m.resident_bytes_excluding("b"), 50_000_000);
        assert_ne!(m.resident_bytes_excluding("b"), 10_000_000);
    }
}
