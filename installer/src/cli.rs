//! Scriptable CLI — the automation/reproducibility path. Same validation and
//! image assembly as the GUI; driven entirely by flags or a TOML config
//! file so builds can run headless (CI, provisioning farms).

use crate::config::{BootMedia, InstallConfig};
use std::path::PathBuf;

pub struct Cli {
    pub config_file: Option<PathBuf>,
    pub image: Option<PathBuf>,
    pub out: Option<PathBuf>,
    pub write: Option<String>,
    pub verify: bool,
    pub list_drives: bool,
    pub dry_run: bool,
    /// Inline overrides (config-file fields win when both given? No — flags
    /// win, they are the last word for automation).
    pub overrides: Vec<(String, String)>,
}

fn usage() -> ! {
    println!(
        "markos-installer — build & flash configured MarkOS images

USAGE:
  markos-installer                          launch the GUI
  markos-installer --config install.toml --image markos-sd.img --target sd
                                            [--out out.img]
                                            [--write \\\\.\\PhysicalDriveN | /dev/sdX]
                                            [--verify] [--dry-run]
  markos-installer --list-drives

The TOML config mirrors the GUI (see install.toml.example). Flags:
  --target sd|ssd|nvme      overrides boot_media (selects image layout)
  --hostname, --timezone, --mdns
  --admin-user, --admin-password
  --dhcp | --static-ip IP --gateway GW --dns DNS
  --wifi SSID PSK           (WPA2/WPA3)
  --ssh-key PATH            repeatable; enables SSH key-only access
  --serial-console          enable the UART recovery console
  --no-cooling-warn         acknowledge active cooling
  --preseed REPO QUANT      model download on first boot
"
    );
    std::process::exit(0);
}

pub fn parse_args() -> Cli {
    let mut cli = Cli {
        config_file: None,
        image: None,
        out: None,
        write: None,
        verify: false,
        list_drives: false,
        dry_run: false,
        overrides: Vec::new(),
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "-h" | "--help" => usage(),
            "--list-drives" => cli.list_drives = true,
            "--config" => {
                i += 1;
                cli.config_file = Some(PathBuf::from(args.get(i).expect("--config needs a value")));
            }
            "--image" => {
                i += 1;
                cli.image = Some(PathBuf::from(args.get(i).expect("--image needs a value")));
            }
            "--out" => {
                i += 1;
                cli.out = Some(PathBuf::from(args.get(i).expect("--out needs a value")));
            }
            "--write" => {
                i += 1;
                cli.write = Some(args.get(i).expect("--write needs a device path").clone());
            }
            "--verify" => cli.verify = true,
            "--dry-run" => cli.dry_run = true,
            "--target" => {
                i += 1;
                let v = args.get(i).expect("--target needs sd|ssd|nvme").to_lowercase();
                let media = match v.as_str() {
                    "sd" => BootMedia::Sd,
                    "ssd" | "usb-ssd" => BootMedia::UsbSsd,
                    "nvme" => BootMedia::Nvme,
                    other => {
                        eprintln!("invalid --target {other}");
                        std::process::exit(2);
                    }
                };
                cli.overrides.push(("boot_media".into(), format!("{media:?}")));
            }
            "--hostname" => {
                i += 1;
                cli.overrides.push(("hostname".into(), args.get(i).unwrap().clone()));
            }
            "--timezone" => {
                i += 1;
                cli.overrides.push(("timezone".into(), args.get(i).unwrap().clone()));
            }
            "--locale" => {
                i += 1;
                cli.overrides.push(("locale".into(), args.get(i).unwrap().clone()));
            }
            "--mdns" => {
                i += 1;
                cli.overrides.push(("network.mdns_name".into(), args.get(i).unwrap().clone()));
            }
            "--admin-user" => {
                i += 1;
                cli.overrides.push(("admin_user".into(), args.get(i).unwrap().clone()));
            }
            "--admin-password" => {
                i += 1;
                cli.overrides.push(("admin_password".into(), args.get(i).unwrap().clone()));
            }
            "--static-ip" => {
                i += 1;
                cli.overrides.push(("network.ip".into(), args.get(i).unwrap().clone()));
                cli.overrides.push(("network.dhcp".into(), "false".into()));
            }
            "--gateway" => {
                i += 1;
                cli.overrides.push(("network.gateway".into(), args.get(i).unwrap().clone()));
            }
            "--dns" => {
                i += 1;
                cli.overrides.push(("network.dns".into(), args.get(i).unwrap().clone()));
            }
            "--wifi" => {
                i += 1;
                let ssid = args.get(i).expect("--wifi needs SSID PSK").clone();
                i += 1;
                let psk = args.get(i).expect("--wifi needs SSID PSK").clone();
                cli.overrides.push(("network.wifi_ssid".into(), ssid));
                cli.overrides.push(("network.wifi_psk".into(), psk));
            }
            "--ssh-key" => {
                i += 1;
                let path = PathBuf::from(args.get(i).expect("--ssh-key needs a path"));
                let key = std::fs::read_to_string(&path)
                    .unwrap_or_else(|e| panic!("read ssh key {path:?}: {e}"));
                cli.overrides.push(("ssh.authorized_keys+".into(), key));
            }
            "--serial-console" => {
                cli.overrides.push(("ssh.serial_console".into(), "true".into()));
            }
            "--no-cooling-warn" => {
                cli.overrides.push(("active_cooling".into(), "true".into()));
            }
            "--preseed" => {
                i += 1;
                let repo = args.get(i).expect("--preseed needs REPO QUANT").clone();
                i += 1;
                let quant = args.get(i).expect("--preseed needs REPO QUANT").clone();
                cli.overrides.push(("model_preseed.repo".into(), repo));
                cli.overrides.push(("model_preseed.quant".into(), quant));
            }
            // --- maintenance subcommands (CI/e2e) ---
            "--make-test-image" => {
                i += 1;
                let path = args.get(i).expect("--make-test-image needs a path");
                crate::testimg::write_mbr_fat32_image(&PathBuf::from(path))
                    .unwrap_or_else(|e| panic!("make-test-image: {e}"));
                println!("wrote synthetic MarkOS-style image: {path}");
                std::process::exit(0);
            }
            "--inspect" => {
                i += 1;
                let path = args.get(i).expect("--inspect needs an image path");
                crate::testimg::inspect(&PathBuf::from(path))
                    .unwrap_or_else(|e| panic!("inspect: {e}"));
                std::process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other} (see --help)");
                std::process::exit(2);
            }
        }
        i += 1;
    }
    cli
}

/// Load TOML then apply CLI overrides.
pub fn load_config(cli: &Cli) -> Result<InstallConfig, String> {
    let mut cfg: InstallConfig = match &cli.config_file {
        Some(p) => {
            let text = std::fs::read_to_string(p).map_err(|e| format!("read {p:?}: {e}"))?;
            toml::from_str(&text).map_err(|e| format!("parse {p:?}: {e}"))?
        }
        None => InstallConfig::default(),
    };
    for (k, v) in &cli.overrides {
        apply_override(&mut cfg, k, v)?;
    }
    Ok(cfg)
}

fn apply_override(cfg: &mut InstallConfig, key: &str, value: &str) -> Result<(), String> {
    match key {
        "boot_media" => {
            cfg.boot_media = Some(match value.as_bytes() {
                b"Sd" => BootMedia::Sd,
                b"UsbSsd" => BootMedia::UsbSsd,
                b"Nvme" => BootMedia::Nvme,
                _ => return Err(format!("bad boot_media override {value}")),
            });
        }
        "hostname" => cfg.hostname = value.to_string(),
        "timezone" => cfg.timezone = value.to_string(),
        "locale" => cfg.locale = value.to_string(),
        "admin_user" => cfg.admin_user = value.to_string(),
        "admin_password" => cfg.admin_password = value.to_string(),
        "active_cooling" => cfg.active_cooling = value == "true",
        "network.mdns_name" => cfg.network.mdns_name = value.to_string(),
        "network.dhcp" => cfg.network.dhcp = value == "true",
        "network.ip" => cfg.network.ip = value.to_string(),
        "network.gateway" => cfg.network.gateway = value.to_string(),
        "network.dns" => cfg.network.dns = value.to_string(),
        "network.wifi_ssid" => cfg.network.wifi_ssid = Some(value.to_string()),
        "network.wifi_psk" => cfg.network.wifi_psk = Some(value.to_string()),
        "ssh.serial_console" => cfg.ssh.serial_console = value == "true",
        "ssh.authorized_keys+" => cfg.ssh.authorized_keys.push(value.to_string()),
        "model_preseed.repo" => {
            cfg.model_preseed.get_or_insert_with(Default::default).repo = Some(value.to_string())
        }
        "model_preseed.url" => {
            cfg.model_preseed.get_or_insert_with(Default::default).url = Some(value.to_string())
        }
        "model_preseed.quant" => {
            cfg.model_preseed.get_or_insert_with(Default::default).quant = Some(value.to_string())
        }
        other => return Err(format!("unknown override key {other}")),
    }
    Ok(())
}

pub fn run(cli: &Cli) -> Result<(), String> {
    if cli.list_drives {
        for d in crate::flash::list_drives() {
            println!(
                "{}  {:>10}  {}",
                d.path,
                humansize(d.size_bytes),
                if d.removable_hint { "removable?" } else { "" }
            );
        }
        return Ok(());
    }

    let cfg = load_config(cli)?;

    match cfg.validate() {
        Ok(()) => {}
        Err(errs) => {
            eprintln!("configuration INVALID — nothing was written:");
            for e in errs {
                eprintln!("  - {e}");
            }
            std::process::exit(2);
        }
    }
    for w in cfg.warnings() {
        eprintln!("warning: {w}");
    }
    let media = cfg.boot_media.unwrap();
    println!("target: {} ({})", media.name(), media.wear_note());

    let Some(image) = &cli.image else {
        eprintln!("--image is required (path to the base MarkOS image)");
        std::process::exit(2);
    };
    if !image.exists() {
        eprintln!("base image not found: {}", image.display());
        std::process::exit(2);
    }

    if cli.dry_run {
        println!("dry-run: validation passed; would build {}", crate::imagebuild::variant_for_media(media));
        return Ok(());
    }

    println!("building configured image from {}...", image.display());
    let built = crate::imagebuild::build_configured_image(image, &cfg, cli.out.as_deref())?;
    println!(
        "wrote {} ({} bytes) — ready to flash",
        built.image.display(),
        built.bytes_written
    );

    if let Some(dev) = &cli.write {
        if media == BootMedia::Sd || matches!(media, BootMedia::UsbSsd | BootMedia::Nvme) {
            let size = std::fs::metadata(&built.image).map(|m| m.len()).unwrap_or(0);
            let drives = crate::flash::list_drives();
            let known = drives.iter().find(|d| d.path == *dev);
            if known.is_none() {
                eprintln!("refusing: {dev} is not a detected disk (run --list-drives)");
                std::process::exit(2);
            }
            println!(
                "WRITING to {dev} ({}) — this erases the device. Ctrl-C within 5 s to abort...",
                humansize(size)
            );
            for left in (1..=5).rev() {
                print!("{left} ");
                use std::io::Write;
                std::io::stdout().flush().ok();
                std::thread::sleep(std::time::Duration::from_secs(1));
            }
            println!();
            crate::flash::flash_image(dev, &built.image, cli.verify, |done, total| {
                let pct = done * 100 / total.max(1);
                print!("\rflashing: {pct}%  ");
                use std::io::Write;
                std::io::stdout().flush().ok();
            })
            .map_err(|e| format!("flash failed: {e}"))?;
            println!("\nflash complete{}.", if cli.verify { " and verified" } else { "" });
        }
    }

    println!("\nPost-flash checklist:");
    println!("  - insert media into the Pi 5 and power on (27 W USB-C PD supply)");
    if matches!(media, BootMedia::UsbSsd | BootMedia::Nvme) {
        println!("  - first boot from SD may be needed once to set EEPROM BOOT_ORDER for {} boot", media.name());
    }
    println!("  - web UI: http://{}.local  (rescue: http://169.254.9.1:4444)", cfg.network.mdns_name);
    println!("  - inference API: http://{}.local:8080/v1", cfg.network.mdns_name);
    Ok(())
}

pub fn humansize(b: u64) -> String {
    let units = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u < units.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{b} B")
    } else {
        format!("{v:.1} {}", units[u])
    }
}
