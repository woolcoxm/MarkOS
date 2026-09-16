//! System facts the UI/watchdog surface: memory, load, SoC temperature.
//! Linux (the appliance) reads /proc and /sys directly; non-Linux hosts get
//! safe fallbacks so dev/testing still works.

use std::sync::atomic::{AtomicU64, Ordering};

pub static TOTAL_MEM: AtomicU64 = AtomicU64::new(0);
pub static AVAIL_MEM: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SystemInfo {
    pub total_mem: u64,
    pub avail_mem: u64,
    pub load1: f64,
    pub load5: f64,
    pub soc_temp_c: Option<f64>,
    /// 1 = thermal throttling right now (Pi: below 80 °C threshold logic
    /// approximated in the UI from temp + freq).
    pub throttling: bool,
    pub cores: usize,
    pub os_name: String,
}

pub fn refresh() {
    if let Some((total, avail)) = meminfo() {
        TOTAL_MEM.store(total, Ordering::Relaxed);
        AVAIL_MEM.store(avail, Ordering::Relaxed);
    }
}

#[cfg(target_family = "unix")]
fn meminfo() -> Option<(u64, u64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = None;
    let mut avail = None;
    for line in text.lines() {
        if let Some(v) = line.strip_prefix("MemTotal:") {
            total = v.trim().split_whitespace().next()?.parse::<u64>().ok().map(|kb| kb * 1024);
        }
        if let Some(v) = line.strip_prefix("MemAvailable:") {
            avail = v.trim().split_whitespace().next()?.parse::<u64>().ok().map(|kb| kb * 1024);
        }
    }
    Some((total?, avail.unwrap_or(total? / 2)))
}

#[cfg(not(target_family = "unix"))]
fn meminfo() -> Option<(u64, u64)> {
    None
}

/// Total RAM for guardrail purposes: real MemTotal when known, else the
/// documented 16 GB assumption.
pub fn total_ram() -> u64 {
    let t = TOTAL_MEM.load(Ordering::Relaxed);
    if t > 0 {
        t
    } else {
        crate::guard::DEFAULT_TOTAL_BYTES
    }
}

pub fn usable_ram() -> u64 {
    total_ram()
        .saturating_sub(crate::guard::OS_RESERVE_BYTES)
        .saturating_sub(crate::guard::ENGINE_SELF_BYTES)
}

pub fn snapshot() -> SystemInfo {
    let (load1, load5) = loadavg();
    let soc = soc_temp();
    SystemInfo {
        total_mem: TOTAL_MEM.load(Ordering::Relaxed),
        avail_mem: AVAIL_MEM.load(Ordering::Relaxed),
        load1,
        load5,
        soc_temp_c: soc,
        throttling: soc.map(|t| t >= 80.0).unwrap_or(false),
        cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
        os_name: os_name(),
    }
}

#[cfg(target_family = "unix")]
fn loadavg() -> (f64, f64) {
    let mut l: [libc::c_double; 3] = [0.0; 3];
    let n = unsafe { libc::getloadavg(l.as_mut_ptr(), 3) };
    if n >= 2 {
        (l[0], l[1])
    } else {
        (0.0, 0.0)
    }
}

#[cfg(not(target_family = "unix"))]
fn loadavg() -> (f64, f64) {
    (0.0, 0.0)
}

#[cfg(target_family = "unix")]
fn soc_temp() -> Option<f64> {
    // Pi 5: cpu thermal zone; also accepts the legacy vc temperature.
    for zone in ["/sys/class/thermal/thermal_zone0/temp", "/sys/devices/virtual/thermal/thermal_zone0/temp"] {
        if let Ok(t) = std::fs::read_to_string(zone) {
            if let Ok(milli) = t.trim().parse::<f64>() {
                return Some(milli / 1000.0);
            }
        }
    }
    None
}

#[cfg(not(target_family = "unix"))]
fn soc_temp() -> Option<f64> {
    None
}

#[cfg(target_family = "unix")]
fn os_name() -> String {
    std::fs::read_to_string("/etc/os-release")
        .ok()
        .and_then(|t| {
            t.lines()
                .find_map(|l| l.strip_prefix("PRETTY_NAME=").map(|s| s.trim_matches('"').to_string()))
        })
        .unwrap_or_else(|| "unknown".into())
}

#[cfg(not(target_family = "unix"))]
fn os_name() -> String {
    "host-dev".into()
}
