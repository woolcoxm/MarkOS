//! Axera accelerator (M5Stack LLM-8850 / AX8850 M.2 card) support: hardware
//! detection and ggml-axcl environment bootstrap.
//!
//! The card enumerates on PCIe as vendor 0x1f4b (Axera) device 0x0650
//! (AX650-class EP, from the axcl_host driver's PCI table). Detection is a
//! plain sysfs scan — no AXCL linkage needed, so the web UI can report the
//! accelerator even in CPU-only builds.
//!
//! `configure_once` must run before the first llama backend init: the fork
//! reads its mode switches from the environment (values are cached in
//! statics on first use). It is idempotent and process-wide, which matches
//! the appliance model — one box, one card, whole-layer NPU for the first
//! matching model.

use std::path::Path;
use std::sync::OnceLock;

pub const AXERA_PCI_VENDOR: &str = "0x1f4b";
pub const AXERA_PCI_DEVICE: &str = "0x0650";

/// Default engines root on the appliance; overridable for tests/dev.
pub fn engines_root() -> std::path::PathBuf {
    std::env::var("MARKOS_AXCL_SETS")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/data/axcl/sets"))
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct AccelStatus {
    pub present: bool,
    /// PCI address (0000:01:00.0 style) when detected
    pub pci_address: Option<String>,
    pub engines_root: String,
    pub n_sets: usize,
    /// driver modules loaded (axcl_host & friends)
    pub driver_loaded: bool,
}

/// Scan /sys/bus/pci for the Axera EP.
pub fn detect() -> AccelStatus {
    let root = engines_root();
    let mut st = AccelStatus {
        present: false,
        pci_address: None,
        engines_root: root.display().to_string(),
        n_sets: crate::axsets::scan(&root).len(),
        driver_loaded: false,
    };
    if let Ok(rd) = std::fs::read_dir("/sys/bus/pci/devices") {
        for ent in rd.flatten() {
            let dir = ent.path();
            let read = |f: &str| std::fs::read_to_string(dir.join(f)).ok();
            if read("vendor").as_deref().map(str::trim) == Some(AXERA_PCI_VENDOR)
                && read("device").as_deref().map(str::trim) == Some(AXERA_PCI_DEVICE)
            {
                st.present = true;
                st.pci_address = ent.file_name().to_str().map(str::to_string);
                break;
            }
        }
    }
    // the driver stack creates /dev/axcl_host when loaded
    st.driver_loaded = Path::new("/dev/axcl_host").exists() || Path::new("/dev/msg_userdev").exists();
    st
}

static CONFIGURED: OnceLock<()> = OnceLock::new();

/// Set the ggml-axcl environment before the first llama.cpp use.
///
/// - `MARKOS_AXCL=0` disables everything (CPU-only boot).
/// - Sets GGML_AXCL_ENGINES_ROOT + GGML_AXCL_LAYER + GGML_AXCL_GGUF; the
///   fork itself selects the engine set matching the loaded model's graph
///   geometry, and retires whole-layer mode cleanly when nothing matches
///   (per-op matmul / CPU reference keeps serving ANY gguf).
pub fn configure_once(status: &AccelStatus) {
    CONFIGURED.get_or_init(|| {
        if std::env::var("MARKOS_AXCL").as_deref() == Ok("0") {
            eprintln!("markos-engine: accel disabled via MARKOS_AXCL=0");
            return;
        }
        if !status.present {
            if std::env::var("MARKOS_AXCL").as_deref() == Ok("1") {
                eprintln!("markos-engine: Axera card requested (MARKOS_AXCL=1) but NOT detected on PCIe — CPU only");
            }
            return;
        }
        std::env::set_var("GGML_AXCL_ENGINES_ROOT", &status.engines_root);
        std::env::set_var("GGML_AXCL_LAYER", "1");
        std::env::set_var("GGML_AXCL_GGUF", "1");
        // card-drop resilience (fork defaults: signal guard on, auto-reboot
        // on EP offline, 15s connect budget) — pin the values so future
        // fork defaults can't silently change appliance behavior
        std::env::set_var("GGML_AXCL_SIGNAL_GUARD", "1");
        std::env::set_var("GGML_AXCL_CONNECT_TIMEOUT", "15");
        eprintln!(
            "markos-engine: Axera NPU active ({} engine sets under {}), whole-layer GGUF patching on",
            status.n_sets, status.engines_root
        );
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_never_panics_without_sysfs() {
        // Windows CI: /sys/bus/pci doesn't exist — detection reports absent
        let st = detect();
        assert!(!st.present);
        assert_eq!(st.pci_address, None);
    }
}
