//! Installer configuration model: everything the installer collects from the
//! user, its validation rules, and the generated provision file written into
//! the image's boot partition. Single source of truth shared by the GUI and
//! the CLI (`--config install.toml`).

use serde::{Deserialize, Serialize};
use std::fmt;

fn default_locale() -> String {
    "en_US.UTF-8".into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BootMedia {
    /// Wear-leveling matters: read-only squashfs root, state on ext4.
    Sd,
    /// Writable ext4 root, model storage still isolated.
    #[serde(rename = "usb-ssd", alias = "ssd")]
    UsbSsd,
    /// Via the official PCIe HAT / M.2 HAT+; same layout as USB SSD.
    Nvme,
}

impl BootMedia {
    pub fn name(self) -> &'static str {
        match self {
            BootMedia::Sd => "SD card",
            BootMedia::UsbSsd => "USB SSD",
            BootMedia::Nvme => "NVMe (PCIe HAT)",
        }
    }
    /// Which prebuilt image variant this media needs (design doc §4).
    pub fn image_variant(self) -> &'static str {
        match self {
            BootMedia::Sd => "markos-sd.img",
            BootMedia::UsbSsd | BootMedia::Nvme => "markos-ssd.img",
        }
    }
    pub fn wear_note(self) -> &'static str {
        match self {
            BootMedia::Sd => "read-only root + overlay: SD wear minimized, power-loss safe",
            _ => "writable ext4 root; model storage isolated on its own partition",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct NetworkConfig {
    /// true = DHCP (default), false = static below.
    pub dhcp: bool,
    pub ip: String,
    pub prefix: u8,
    pub gateway: String,
    pub dns: String,
    /// None = Ethernet; Some = WiFi (WPA2/WPA3-SAE).
    pub wifi_ssid: Option<String>,
    pub wifi_psk: Option<String>,
    /// mDNS/Avahi hostname, e.g. "pi-inference" → pi-inference.local.
    pub mdns_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SshConfig {
    pub enabled: bool,
    /// OpenAI-style public keys (one per line) injected into
    /// /root/.ssh/authorized_keys. Password logins are never enabled.
    pub authorized_keys: Vec<String>,
    /// Enable the serial (UART) recovery console — recommended on headless.
    pub serial_console: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct ModelPreseed {
    /// HuggingFace repo id (e.g. "bartowski/Llama-3.2-1B-Instruct-GGUF").
    pub repo: Option<String>,
    /// Or a direct URL to a GGUF file.
    pub url: Option<String>,
    /// Quantization variant to fetch (Q4_K_M etc.).
    pub quant: Option<String>,
}

/// The full installer configuration (TOML-serializable, GUI-editable).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct InstallConfig {
    pub boot_media: Option<BootMedia>,
    pub hostname: String,
    pub timezone: String,
    /// Locale (e.g. "en_US.UTF-8"); exported system-wide on the appliance.
    #[serde(default = "default_locale")]
    pub locale: String,
    pub network: NetworkConfig,
    pub ssh: SshConfig,
    /// Initial web-UI admin account. Stored hashed in the provision file —
    /// the raw password never persists beyond the host machine's config.
    pub admin_user: String,
    pub admin_password: String,
    /// "stable" pre-fills the update manifest URL; "manual" leaves it empty.
    pub update_channel: String,
    pub update_manifest_url: Option<String>,
    pub model_preseed: Option<ModelPreseed>,
    /// User confirmed an active cooler for sustained load (warning only).
    pub active_cooling: bool,
}

pub struct ValidationError {
    pub field: String,
    pub message: String,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.field, self.message)
    }
}

impl InstallConfig {
    /// Validate everything BEFORE any disk is touched. Rules enforce the
    /// "no headless box without a way in" invariant (spec §Component 1).
    pub fn validate(&self) -> Result<(), Vec<ValidationError>> {
        let mut errs: Vec<ValidationError> = Vec::new();

        if self.boot_media.is_none() {
            errs.push(ValidationError {
                field: "boot_media".into(),
                message: "choose SD / USB SSD / NVMe — it decides the filesystem layout".into(),
            });
        }
        if self.hostname.trim().is_empty() {
            errs.push(ValidationError { field: "hostname".into(), message: "hostname is required".into() });
        } else if self.hostname.len() > 63
            || !self
                .hostname
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-')
        {
            errs.push(ValidationError {
                field: "hostname".into(),
                message: "use letters, digits and dashes only (max 63)".into(),
            });
        }

        // --- the critical rule: at least one way back in ---
        let has_ssh_key = self.ssh.enabled && self.ssh.authorized_keys.iter().any(|k| !k.trim().is_empty());
        let has_password = !self.admin_password.is_empty();
        let has_console = self.ssh.serial_console;
        if !has_ssh_key && !has_password && !has_console {
            errs.push(ValidationError {
                field: "admin_password".into(),
                message: "no way in: set an admin password, or enable SSH with a key, or enable the serial console".into(),
            });
        }
        if !self.admin_password.is_empty() && self.admin_password.len() < 8 {
            errs.push(ValidationError {
                field: "admin_password".into(),
                message: "password must be at least 8 characters".into(),
            });
        }
        if self.admin_user.trim().is_empty() {
            errs.push(ValidationError { field: "admin_user".into(), message: "admin username is required".into() });
        }

        // network
        if !self.network.dhcp {
            if self.network.ip.parse::<std::net::Ipv4Addr>().is_err() {
                errs.push(ValidationError { field: "network.ip".into(), message: "static mode needs a valid IPv4 address".into() });
            }
            if !self.network.gateway.is_empty()
                && self.network.gateway.parse::<std::net::Ipv4Addr>().is_err()
            {
                errs.push(ValidationError { field: "network.gateway".into(), message: "invalid gateway IPv4".into() });
            }
        }
        if let Some(ssid) = &self.network.wifi_ssid {
            if ssid.trim().is_empty() {
                errs.push(ValidationError { field: "network.wifi_ssid".into(), message: "SSID is empty".into() });
            }
            match &self.network.wifi_psk {
                Some(psk) if psk.len() >= 8 && psk.len() <= 63 => {}
                _ => errs.push(ValidationError {
                    field: "network.wifi_psk".into(),
                    message: "WPA passphrase must be 8–63 characters".into(),
                }),
            }
        }
        if self.network.mdns_name.trim().is_empty() {
            errs.push(ValidationError { field: "network.mdns_name".into(), message: "mDNS name is required".into() });
        }

        // update channel
        if !matches!(self.update_channel.as_str(), "stable" | "manual" | "") {
            errs.push(ValidationError { field: "update_channel".into(), message: "must be \"stable\" or \"manual\"".into() });
        }

        // model preseed
        if let Some(p) = &self.model_preseed {
            if p.repo.is_none() && p.url.is_none() {
                errs.push(ValidationError {
                    field: "model_preseed".into(),
                    message: "set either repo or url (or remove the preseed section)".into(),
                });
            }
        }

        if errs.is_empty() {
            Ok(())
        } else {
            Err(errs)
        }
    }

    /// Non-blocking warnings surfaced in the installer output/UI.
    pub fn warnings(&self) -> Vec<String> {
        let mut w = Vec::new();
        if !self.active_cooling {
            w.push(
                "sustained inference needs an ACTIVE COOLER (official Pi 5 cooler or \
                 equivalent) — thermal throttling will cut tokens/sec dramatically"
                    .into(),
            );
        }
        w.push("power: use a 27 W USB-C PD supply (5 V × 5 A) — underpowered boards \
                throttle or brown out under load"
            .into());
        if matches!(self.boot_media, Some(BootMedia::Nvme) | Some(BootMedia::UsbSsd)) {
            w.push("NVMe/USB boot requires the board EEPROM BOOT_ORDER to include NVMe/USB \
                    (one-time config on the Pi — see docs/build.md)"
                .into());
        }
        if self.ssh.enabled && !self.ssh.authorized_keys.iter().any(|k| !k.trim().is_empty()) {
            w.push("SSH enabled without any authorized key: only the web UI and serial \
                    console remain as ways in"
                .into());
        }
        w
    }

    /// The provision file injected into the boot partition. The admin
    /// password is written hashed (argon2id) — the engine never needs the
    /// plaintext after this point.
    pub fn provision_env(&self, pwhash: &str) -> String {
        let mut out = String::new();
        out.push_str("# MarkOS first-boot provision file (consumed once by markos-engine)\n");
        out.push_str(&format!("MARKOS_ADMIN_USER={}\n", self.admin_user.trim()));
        out.push_str(&format!("MARKOS_ADMIN_PASSWORD_HASH={pwhash}\n"));
        out.push_str(&format!("MARKOS_HOSTNAME={}\n", self.hostname.trim()));
        out.push_str(&format!("MARKOS_TZ={}\n", self.timezone.trim()));
        if let Some(p) = &self.model_preseed {
            if let Some(repo) = &p.repo {
                out.push_str(&format!("MARKOS_MODEL_PRESEED_REPO={repo}\n"));
            }
            if let Some(url) = &p.url {
                out.push_str(&format!("MARKOS_MODEL_PRESEED_URL={url}\n"));
            }
            if let Some(q) = &p.quant {
                out.push_str(&format!("MARKOS_MODEL_PRESEED_QUANT={q}\n"));
            }
        }
        // Network + ssh flags are applied by the OS provisioner from these.
        out.push_str(&format!(
            "MARKOS_NET_MODE={}\n",
            if self.network.dhcp { "dhcp" } else { "static" }
        ));
        out
    }

    /// The OS-side network config consumed by S03markos-net.
    pub fn net_conf(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "MODE={}\n",
            if self.network.dhcp { "dhcp" } else { "static" }
        ));
        if !self.network.dhcp {
            out.push_str(&format!("IP={}\n", self.network.ip));
            out.push_str(&format!("GW={}\n", self.network.gateway));
            out.push_str(&format!("DNS1={}\n", self.network.dns));
        }
        if self.network.wifi_ssid.is_some() {
            out.push_str("IFACE_SET=wlan0\n");
        }
        out
    }

    pub fn wpa_supplicant_conf(&self) -> Option<String> {
        let ssid = self.network.wifi_ssid.as_ref()?;
        let psk = self.network.wifi_psk.as_ref()?;
        Some(format!(
            "country=00\nctrl_interface=DIR=/var/run/wpa_supplicant GROUP=netdev\nupdate_config=1\n\
             network={{\n    ssid=\"{ssid}\"\n    psk=\"{psk}\"\n    key_mgmt=WPA-PSK WPA-PSK-SHA256 SAE\n}}\n"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn good() -> InstallConfig {
        InstallConfig {
            boot_media: Some(BootMedia::Sd),
            hostname: "markos-1".into(),
            timezone: "UTC".into(),
            admin_user: "admin".into(),
            admin_password: "long-enough-pass".into(),
            update_channel: "stable".into(),
            active_cooling: true,
            network: NetworkConfig {
                dhcp: true,
                mdns_name: "pi-inference".into(),
                ..Default::default()
            },
            ..Default::default()
        }
    }

    #[test]
    fn valid_config_passes() {
        assert!(good().validate().is_ok());
    }

    #[test]
    fn no_way_in_is_refused() {
        let mut c = good();
        c.admin_password.clear();
        c.ssh.enabled = false;
        c.ssh.serial_console = false;
        let errs = c.validate().unwrap_err();
        assert!(errs.iter().any(|e| e.field == "admin_password"));
    }

    #[test]
    fn ssh_key_alone_is_a_way_in() {
        let mut c = good();
        c.admin_password.clear();
        c.ssh.enabled = true;
        c.ssh.authorized_keys = vec!["ssh-ed25519 AAAA...".into()];
        assert!(c.validate().is_ok());
    }

    #[test]
    fn short_password_and_bad_static_ip_refused() {
        let mut c = good();
        c.admin_password = "short".into();
        c.network.dhcp = false;
        c.network.ip = "not-an-ip".into();
        let errs = c.validate().unwrap_err();
        assert!(errs.len() >= 2);
    }

    #[test]
    fn wifi_needs_valid_psk() {
        let mut c = good();
        c.network.wifi_ssid = Some("home".into());
        c.network.wifi_psk = Some("short".into());
        assert!(c.validate().is_err());
        c.network.wifi_psk = Some("a-longer-passphrase".into());
        assert!(c.validate().is_ok());
        let wpa = c.wpa_supplicant_conf().unwrap();
        assert!(wpa.contains("SAE"));
    }

    #[test]
    fn media_selects_variant() {
        assert_eq!(BootMedia::Sd.image_variant(), "markos-sd.img");
        assert_eq!(BootMedia::Nvme.image_variant(), "markos-ssd.img");
    }

    #[test]
    fn provision_hashes_password() {
        let c = good();
        let env = c.provision_env("$argon2id$fake$hash");
        assert!(env.contains("MARKOS_ADMIN_PASSWORD_HASH=$argon2id$fake$hash"));
        assert!(!env.contains("long-enough-pass"));
    }
}
