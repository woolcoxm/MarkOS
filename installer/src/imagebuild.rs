//! Image assembly: take the prebuilt base image (from the OS build
//! pipeline), inject the user's configuration into its boot partition, and
//! produce a ready-to-flash image file. Everything runs on in-memory copies
//! of only the boot partition region — multi-GB rootfs partitions are
//! streamed, not parsed.

use crate::config::InstallConfig;
use crate::fat;
use std::fs::File;
use std::io::{Seek, SeekFrom};
use std::path::{Path, PathBuf};

pub struct BuildOutcome {
    pub image: PathBuf,
    pub bytes_written: u64,
}

/// Produce `<stem>.configured.img` next to the base image (or `out` when
/// given). Idempotent: re-running replaces the previous output.
pub fn build_configured_image(
    base: &Path,
    cfg: &InstallConfig,
    out: Option<&Path>,
) -> Result<BuildOutcome, String> {
    let pwhash = crate::auth::auth_hash(&cfg.admin_password)?;
    let provision = cfg.provision_env(&pwhash);

    // Everything lands under /boot/markos/ (consumed by S03markos-net and
    // the engine's provisioner on first boot).
    let mut files: Vec<(String, Vec<u8>)> =
        vec![("markos/provision.env".to_string(), provision.into_bytes())];
    let net = cfg.net_conf().into_bytes();
    files.push(("markos/net.conf".to_string(), net));
    if let Some(wpa) = cfg.wpa_supplicant_conf() {
        files.push(("markos/wpa_supplicant.conf".to_string(), wpa.into_bytes()));
    }
    if !cfg.ssh.authorized_keys.is_empty() {
        let keys: String = cfg
            .ssh
            .authorized_keys
            .iter()
            .map(|k| k.trim())
            .filter(|k| !k.is_empty())
            .map(|k| format!("{k}\n"))
            .collect();
        files.push(("markos/authorized_keys".to_string(), keys.into_bytes()));
    }
    if !cfg.ssh.enabled {
        files.push(("markos/ssh-disabled".to_string(), Vec::new()));
    } else {
        files.push(("markos/ssh-enabled".to_string(), Vec::new()));
    }

    // Work on a copy so the pristine base image is reusable.
    let out_path = match out {
        Some(p) => p.to_path_buf(),
        None => {
            let stem = base.file_stem().unwrap_or_default().to_string_lossy();
            base.with_file_name(format!("{stem}.configured.img"))
        }
    };
    std::fs::copy(base, &out_path).map_err(|e| format!("copy base image: {e}"))?;

    let mut f = File::options()
        .read(true)
        .write(true)
        .open(&out_path)
        .map_err(|e| format!("open {}: {e}", out_path.display()))?;

    // The installer knows the MarkOS layout: boot FAT32 is partition 1.
    fat::inject_provision(&mut f, &files.iter().map(|(k, v)| (k.as_str(), v.clone())).collect::<Vec<_>>()).map_err(|e| {
        format!(
            "injecting config into {} (is this a MarkOS image with a FAT32 boot \
             partition as partition 1?): {e}",
            out_path.display()
        )
    })?;

    let bytes = f.seek(SeekFrom::End(0)).unwrap_or(0);
    Ok(BuildOutcome { image: out_path, bytes_written: bytes })
}

/// Select the right base image variant for the chosen boot media (design §4).
pub fn variant_for_media(media: crate::config::BootMedia) -> &'static str {
    media.image_variant()
}
