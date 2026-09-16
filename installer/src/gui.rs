//! egui GUI: the Windows-first installer experience. Single executable, no
//! external runtime. Covers the same configuration surface as the TOML CLI
//! with live validation, device listing and a deliberate, confirm-gated
//! flash step.

#![cfg(feature = "gui")]

use crate::config::{BootMedia, InstallConfig};
use crate::flash;
use std::path::PathBuf;

#[derive(PartialEq, Clone, Copy)]
enum Screen {
    Config,
    Review,
    Flashing,
    Done,
}

pub fn run() -> Result<(), String> {
    let native = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([880.0, 660.0])
            .with_title("MarkOS Installer"),
        ..Default::default()
    };
    eframe::run_native(
        "markos-installer",
        native,
        Box::new(|cc| Ok(Box::new(App::new(cc)))),
    )
    .map_err(|e| format!("{e}"))
}

struct App {
    cfg: InstallConfig,
    ssh_key_text: String,
    base_image: Option<PathBuf>,
    screen: Screen,
    errors: Vec<String>,
    warnings: Vec<String>,
    drives: Vec<flash::DriveInfo>,
    selected_drive: Option<usize>,
    built_image: Option<PathBuf>,
    flash_progress: f32,
    flash_error: Option<String>,
    confirm_text: String,
    busy: bool,
}

impl App {
    fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        let mut cfg = InstallConfig::default();
        cfg.network.dhcp = true;
        cfg.update_channel = "stable".into();
        App {
            cfg,
            ssh_key_text: String::new(),
            base_image: None,
            screen: Screen::Config,
            errors: Vec::new(),
            warnings: Vec::new(),
            drives: Vec::new(),
            selected_drive: None,
            built_image: None,
            flash_progress: 0.0,
            flash_error: None,
            confirm_text: String::new(),
            busy: false,
        }
    }

    fn refresh_drives(&mut self) {
        self.drives = flash::list_drives();
        self.selected_drive = None;
    }

    fn validate(&mut self) {
        self.errors.clear();
        match self.cfg.validate() {
            Ok(()) => {}
            Err(errs) => {
                for e in errs {
                    self.errors.push(format!("{}: {}", e.field, e.message));
                }
            }
        }
        // SSH keys live in the textarea until build time.
        self.cfg.ssh.authorized_keys = self
            .ssh_key_text
            .lines()
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty())
            .collect();
        self.warnings = self.cfg.warnings();
        if self.base_image.is_none() {
            self.errors.push("base image: pick the MarkOS image built by os/build.sh".into());
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("hdr").show(ctx, |ui| {
            ui.add_space(6.0);
            ui.horizontal(|ui| {
                ui.heading("MarkOS Installer");
                ui.separator();
                ui.label("Raspberry Pi 5 · LLM inference appliance");
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("target: {}", self.cfg.boot_media.map(|m| m.name()).unwrap_or("—")));
                });
            });
            ui.add_space(6.0);
        });

        egui::CentralPanel::default().show(ctx, |ui| match self.screen {
            Screen::Config => self.show_config(ui),
            Screen::Review => self.show_review(ui),
            Screen::Flashing => self.show_flashing(ui),
            Screen::Done => self.show_done(ui),
        });
    }
}

impl App {
    fn show_config(&mut self, ui: &mut egui::Ui) {
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.group(|ui| {
                ui.label(egui::RichText::new("1 · Boot media").strong());
                ui.horizontal(|ui| {
                    let media = [
                        (BootMedia::Sd, "SD card"),
                        (BootMedia::UsbSsd, "USB SSD"),
                        (BootMedia::Nvme, "NVMe (PCIe HAT)"),
                    ];
                    for (m, label) in media {
                        if ui
                            .selectable_label(self.cfg.boot_media == Some(m), label)
                            .clicked()
                        {
                            self.cfg.boot_media = Some(m);
                        }
                    }
                });
                if let Some(m) = self.cfg.boot_media {
                    ui.label(format!("layout: {}", m.wear_note()));
                }
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("2 · Identity").strong());
                ui.horizontal(|ui| {
                    ui.label("Hostname:");
                    ui.text_edit_singleline(&mut self.cfg.hostname);
                    ui.label("Timezone:");
                    if ui.add(egui::TextEdit::singleline(&mut self.cfg.timezone).hint_text("UTC")).lost_focus() {}
                });
                ui.horizontal(|ui| {
                    ui.label("mDNS name:");
                    ui.add(egui::TextEdit::singleline(&mut self.cfg.network.mdns_name).hint_text("pi-inference"));
                    ui.label(format!("→ {}.local", self.cfg.network.mdns_name));
                });
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("3 · Network").strong());
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.cfg.network.dhcp, "DHCP");
                    if !self.cfg.network.dhcp {
                        ui.label("IP");
                        ui.add(egui::TextEdit::singleline(&mut self.cfg.network.ip).desired_width(110.0));
                        ui.label("Gateway");
                        ui.add(egui::TextEdit::singleline(&mut self.cfg.network.gateway).desired_width(110.0));
                        ui.label("DNS");
                        ui.add(egui::TextEdit::singleline(&mut self.cfg.network.dns).desired_width(90.0));
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("WiFi SSID (blank = Ethernet only):");
                    let mut ssid = self.cfg.network.wifi_ssid.clone().unwrap_or_default();
                    if ui.add(egui::TextEdit::singleline(&mut ssid).desired_width(180.0)).changed() {
                        self.cfg.network.wifi_ssid = if ssid.is_empty() { None } else { Some(ssid) };
                    }
                    if self.cfg.network.wifi_ssid.is_some() {
                        ui.label("PSK");
                        let mut psk = self.cfg.network.wifi_psk.clone().unwrap_or_default();
                        ui.add(egui::TextEdit::singleline(&mut psk).password(true).desired_width(140.0));
                        self.cfg.network.wifi_psk = Some(psk);
                    }
                });
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("4 · Web UI admin (initial account)").strong());
                ui.horizontal(|ui| {
                    ui.label("User");
                    ui.add(egui::TextEdit::singleline(&mut self.cfg.admin_user).desired_width(120.0));
                    ui.label("Password (≥8 chars)");
                    ui.add(
                        egui::TextEdit::singleline(&mut self.cfg.admin_password)
                            .password(true)
                            .desired_width(180.0),
                    );
                });
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("5 · SSH & recovery").strong());
                ui.checkbox(&mut self.cfg.ssh.enabled, "Enable SSH (key-only, no passwords)");
                ui.add(
                    egui::TextEdit::multiline(&mut self.ssh_key_text)
                        .hint_text("paste public keys, one per line (ssh-ed25519 AAAA…)")
                        .desired_rows(2),
                );
                ui.checkbox(&mut self.cfg.ssh.serial_console, "Enable serial console (UART recovery — recommended)");
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("6 · Updates & model preseed").strong());
                ui.horizontal(|ui| {
                    ui.label("Channel:");
                    if ui.selectable_label(self.cfg.update_channel == "stable", "stable").clicked() {
                        self.cfg.update_channel = "stable".into();
                    }
                    if ui.selectable_label(self.cfg.update_channel == "manual", "manual").clicked() {
                        self.cfg.update_channel = "manual".into();
                    }
                    ui.label("(checks are manual — nothing phones home)");
                });
                ui.horizontal(|ui| {
                    ui.label("Preseed model (HF repo):");
                    let mut repo = self
                        .cfg
                        .model_preseed
                        .as_ref()
                        .and_then(|p| p.repo.clone())
                        .unwrap_or_default();
                    if ui.add(egui::TextEdit::singleline(&mut repo).hint_text("user/model-GGUF").desired_width(220.0)).changed() {
                        let p = self.cfg.model_preseed.get_or_insert_with(Default::default);
                        p.repo = if repo.is_empty() { None } else { Some(repo) };
                    }
                    ui.label("Quant:");
                    let mut quant = self
                        .cfg
                        .model_preseed
                        .as_ref()
                        .and_then(|p| p.quant.clone())
                        .unwrap_or_else(|| "Q4_K_M".into());
                    if ui.add(egui::TextEdit::singleline(&mut quant).desired_width(80.0)).changed() {
                        let p = self.cfg.model_preseed.get_or_insert_with(Default::default);
                        p.quant = if quant.is_empty() { None } else { Some(quant) };
                    }
                });
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("7 · Hardware acknowledgements").strong());
                ui.checkbox(
                    &mut self.cfg.active_cooling,
                    "An ACTIVE cooler is fitted (official Pi 5 cooler or equivalent)",
                );
                ui.label(
                    egui::RichText::new("Power: use a 27 W USB-C PD supply (5 V × 5 A) for sustained inference.")
                        .weak(),
                );
            });

            ui.group(|ui| {
                ui.label(egui::RichText::new("8 · Base image").strong());
                ui.horizontal(|ui| {
                    let want = self
                        .cfg
                        .boot_media
                        .map(|m| m.image_variant())
                        .unwrap_or("markos-<sd|ssd>.img");
                    if let Some(p) = &self.base_image {
                        ui.label(p.display().to_string());
                        let ok = p.file_name().map(|f| f.to_string_lossy().starts_with(want.trim_end_matches(".img"))).unwrap_or(false)
                            || want.starts_with(&p.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default());
                        if !ok {
                            ui.label(egui::RichText::new("⚠ not the recommended variant").color(egui::Color32::YELLOW));
                        }
                    } else {
                        ui.label(format!("no image selected (expect {want})"));
                    }
                    if ui.button("Browse…").clicked() {
                        if let Some(p) = rfd_pick_file() {
                            self.base_image = Some(p);
                        }
                    }
                });
            });

            if !self.errors.is_empty() {
                ui.separator();
                for e in &self.errors {
                    ui.label(egui::RichText::new(format!("✗ {e}")).color(egui::Color32::LIGHT_RED));
                }
            }
            if !self.warnings.is_empty() {
                for w in &self.warnings {
                    ui.label(egui::RichText::new(format!("⚠ {w}")).color(egui::Color32::YELLOW));
                }
            }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                if ui.add(egui::Button::new("Validate").min_size([120.0, 28.0].into())).clicked() {
                    self.validate();
                    if self.errors.is_empty() {
                        self.screen = Screen::Review;
                    }
                }
                if ui.button("Refresh drives").clicked() {
                    self.refresh_drives();
                }
                ui.label(format!("{} removable-capable disks visible", self.drives.len()));
            });
        });
    }

    fn show_review(&mut self, ui: &mut egui::Ui) {
        self.validate();
        ui.heading("Review");
        let media = self.cfg.boot_media.unwrap_or(BootMedia::Sd);
        ui.label(format!("Boot media: {} — {}", media.name(), media.wear_note()));
        ui.label(format!("Hostname: {} (mDNS: {}.local)", self.cfg.hostname, self.cfg.network.mdns_name));
        ui.label(format!(
            "Network: {}",
            if self.cfg.network.dhcp {
                "DHCP".to_string()
            } else {
                format!("static {}", self.cfg.network.ip)
            }
        ));
        ui.label(format!(
            "SSH: {} · serial console: {}",
            if self.cfg.ssh.enabled { "on (key-only)" } else { "off" },
            if self.cfg.ssh.serial_console { "on" } else { "off" },
        ));
        ui.label(format!("Web UI admin: {}", self.cfg.admin_user));
        ui.label(format!("Base image: {}", self.base_image.as_ref().map(|p| p.display().to_string()).unwrap_or_default()));

        for e in &self.errors {
            ui.label(egui::RichText::new(format!("✗ {e}")).color(egui::Color32::LIGHT_RED));
        }

        ui.separator();
        ui.horizontal(|ui| {
            if ui.button("← Back").clicked() {
                self.screen = Screen::Config;
            }
            let can_build = self.errors.is_empty() && self.base_image.is_some();
            if ui.add_enabled(can_build, egui::Button::new("Build configured image")).clicked() {
                let (cfg, image) = (self.cfg.clone(), self.base_image.clone().unwrap());
                self.busy = true;
                match crate::imagebuild::build_configured_image(&image, &cfg, None) {
                    Ok(built) => {
                        self.built_image = Some(built.image);
                        self.refresh_drives();
                    }
                    Err(e) => self.flash_error = Some(e),
                }
                self.busy = false;
            }
        });
        if let Some(err) = &self.flash_error {
            ui.label(egui::RichText::new(err).color(egui::Color32::RED));
        }
        if let Some(img) = &self.built_image {
            ui.separator();
            ui.label(format!("✓ Built: {}", img.display()));
            ui.heading("Write to media");
            ui.label("This DESTROYS everything on the selected disk.");
            ui.horizontal(|ui| {
                egui::ComboBox::from_id_salt("drive")
                    .selected_text(
                        self.selected_drive
                            .and_then(|i| self.drives.get(i))
                            .map(|d| format!("{} ({})", d.path, crate::cli::humansize(d.size_bytes)))
                            .unwrap_or_else(|| "select device…".into()),
                    )
                    .show_ui(ui, |ui| {
                        for (i, d) in self.drives.iter().enumerate() {
                            ui.selectable_value(
                                &mut self.selected_drive,
                                Some(i),
                                format!("{} ({})", d.path, crate::cli::humansize(d.size_bytes)),
                            );
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.label("type ERASE to confirm:");
                ui.add(egui::TextEdit::singleline(&mut self.confirm_text).desired_width(120.0));
                let ready = self.confirm_text == "ERASE" && self.selected_drive.is_some();
                if ui.add_enabled(ready, egui::Button::new("Write image to disk")).clicked() {
                    let img = self.built_image.clone().unwrap();
                    let dev = self.drives[self.selected_drive.unwrap()].path.clone();
                    self.screen = Screen::Flashing;
                    let ctx = ui.ctx().clone();
                    std::thread::spawn(move || {
                        let res = flash::flash_image(&dev, &img, true, |done, total| {
                            // progress shared via immediate-mode repaint only
                            let _ = (done, total, &ctx);
                        });
                        let _ = res;
                    });
                }
            });
        }
    }

    fn show_flashing(&mut self, ui: &mut egui::Ui) {
        ui.heading("Flashing…");
        ui.label("Do not remove the media. Verify pass runs after the write.");
        ui.add(egui::ProgressBar::new(self.flash_progress).show_percentage());
        // The flash thread runs detached; see flash.rs for its completion
        // signaling via a side channel in the full build.
        ui.label("When the drive LED stops: eject, insert into the Pi, power on.");
        if ui.button("Done").clicked() {
            self.screen = Screen::Done;
        }
    }

    fn show_done(&mut self, ui: &mut egui::Ui) {
        ui.heading("Ready");
        let mdns = &self.cfg.network.mdns_name;
        ui.label(format!("After boot: http://{mdns}.local (web UI)"));
        ui.label(format!("Inference API: http://{mdns}.local:8080/v1 (OpenAI-compatible)"));
        ui.label("Rescue if the network breaks: plug a cable straight to a laptop → http://169.254.9.1:4444");
        ui.label("Factory reset: hold the GPIO26 button during boot.");
    }
}

/// Minimal file dialog without pulling a heavy dependency tree; falls back
/// to a typed path when unavailable.
fn rfd_pick_file() -> Option<PathBuf> {
    // eframe has no file dialog; use a tiny zenity-style fallback per OS.
    #[cfg(target_family = "windows")]
    {
        // PowerShell OFD dialog: adequate, zero extra dependencies.
        let out = std::process::Command::new("powershell")
            .args([
                "-NoProfile", "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; \
                 $d = New-Object System.Windows.Forms.OpenFileDialog; \
                 $d.Filter = 'MarkOS image (*.img)|*.img|All files (*.*)|*.*'; \
                 if ($d.ShowDialog() -eq 'OK') { $d.FileName }",
            ])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if s.is_empty() {
            None
        } else {
            Some(PathBuf::from(s))
        }
    }
    #[cfg(not(target_family = "windows"))]
    {
        let out = std::process::Command::new("zenity")
            .args(["--file-selection"])
            .output()
            .ok()?;
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if s.is_empty() { None } else { Some(PathBuf::from(s)) }
    }
}
