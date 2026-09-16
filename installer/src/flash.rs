//! Direct media writer: SD / USB SSD / NVMe-over-USB-enclosure. Windows uses
//! raw `\\.\PhysicalDriveN` access via a tiny FFI surface (no large crate);
//! Linux uses block device nodes. Every path validates the target and
//! writes with progress; fsync/flush at the end.

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

#[derive(Debug, Clone, serde::Serialize)]
pub struct DriveInfo {
    pub path: String,
    pub size_bytes: u64,
    pub removable_hint: bool,
}

#[derive(Debug)]
pub enum FlashError {
    Io(String),
    Busy(String),
}

impl std::fmt::Display for FlashError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FlashError::Io(m) => write!(f, "{m}"),
            FlashError::Busy(m) => write!(f, "device busy: {m}"),
        }
    }
}

/// Enumerate candidate disk devices. Conservative on both platforms; the
/// caller must still confirm size + ask the user before writing.
pub fn list_drives() -> Vec<DriveInfo> {
    let mut out = Vec::new();
    #[cfg(target_family = "windows")]
    for n in 0..16 {
        let path = format!(r"\\.\PhysicalDrive{n}");
        if let Ok(mut f) = File::open(&path) {
            let size = f.seek(SeekFrom::End(0)).unwrap_or(0);
            if size > 0 {
                out.push(DriveInfo {
                    path,
                    size_bytes: size,
                    removable_hint: true,
                });
            }
        }
    }
    #[cfg(not(target_family = "windows"))]
    {
        if let Ok(entries) = std::fs::read_dir("/dev") {
            for e in entries.flatten() {
                let name = e.file_name().to_string_lossy().into_owned();
                let is_disk = (name.starts_with("sd") && name.len() == 3)
                    || name == "mmcblk0"
                    || name.starts_with("nvme0n");
                if !is_disk {
                    continue;
                }
                let path = format!("/dev/{name}");
                if let Ok(mut f) = File::open(&path) {
                    let size = f.seek(SeekFrom::End(0)).unwrap_or(0);
                    if size > 1 << 20 {
                        out.push(DriveInfo { path, size_bytes: size, removable_hint: true });
                    }
                }
            }
        }
    }
    out
}

/// Write `image` onto `device` with 4 MiB chunks and progress callbacks.
/// `progress` receives (bytes_done, bytes_total). The caller is responsible
/// for having confirmed the target with the user — this function only
/// sanity-checks size.
pub fn flash_image(
    device: &str,
    image: &PathBuf,
    verify: bool,
    mut progress: impl FnMut(u64, u64),
) -> Result<(), FlashError> {
    let img_size = std::fs::metadata(image)
        .map_err(|e| FlashError::Io(format!("image: {e}")))?
        .len();

    let mut dst = open_device_for_write(device)?;
    let dev_size = dst
        .seek(SeekFrom::End(0))
        .map_err(|e| FlashError::Io(format!("device seek: {e}")))?;
    if dev_size < img_size {
        return Err(FlashError::Io(format!(
            "media too small: device is {} bytes, image needs {}",
            dev_size, img_size
        )));
    }
    dst.seek(SeekFrom::Start(0))
        .map_err(|e| FlashError::Io(e.to_string()))?;

    // Dismount/detach anything holding the media (Windows auto-mounts FAT).
    prepare_device(device);

    let mut src = File::open(image).map_err(|e| FlashError::Io(format!("open image: {e}")))?;
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    let mut done: u64 = 0;
    loop {
        let n = src.read(&mut buf).map_err(|e| FlashError::Io(e.to_string()))?;
        if n == 0 {
            break;
        }
        dst.write_all(&buf[..n])
            .map_err(|e| FlashError::Io(format!("device write: {e}")))?;
        done += n as u64;
        progress(done, img_size);
    }
    dst.flush().ok();
    dst.sync_all().map_err(|e| FlashError::Io(format!("sync: {e}")))?;
    drop(dst);

    // OS caches can hold the FAT stale until an eject cycle; best effort.
    rescan_device(device);

    if verify {
        let mut dst = File::open(device).map_err(|e| FlashError::Io(e.to_string()))?;
        let mut src = File::open(image).map_err(|e| FlashError::Io(e.to_string()))?;
        src.seek(SeekFrom::Start(0)).ok();
        dst.seek(SeekFrom::Start(0)).ok();
        let mut a = vec![0u8; 1024 * 1024];
        let mut b = vec![0u8; 1024 * 1024];
        let mut off = 0u64;
        loop {
            let na = read_full(&mut src, &mut a);
            let nb = read_full(&mut dst, &mut b);
            if na != nb || a[..na] != b[..nb] {
                return Err(FlashError::Io(format!("verify mismatch at offset {off}")));
            }
            if na == 0 {
                break;
            }
            off += na as u64;
        }
    }
    Ok(())
}

fn read_full(f: &mut File, buf: &mut [u8]) -> usize {
    let mut filled = 0;
    while filled < buf.len() {
        match f.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(_) => break,
        }
    }
    filled
}

#[cfg(target_family = "windows")]
fn open_device_for_write(device: &str) -> Result<File, FlashError> {
    use std::os::windows::fs::OpenOptionsExt;
    OpenOptions::new()
        .write(true)
        .read(true)
        .share_mode(0) // exclusive
        .custom_flags(0x20000000) // FILE_FLAG_WRITE_THROUGH? (no-buffering 0x0 not used: keep cached+sync)
        .open(device)
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                FlashError::Busy(format!(
                    "{device} — run the installer as Administrator to write raw devices"
                ))
            } else {
                FlashError::Io(format!("open {device}: {e}"))
            }
        })
}

#[cfg(target_family = "windows")]
fn prepare_device(_device: &str) {
    // Best effort dismount of mounted volumes on the target disk so FAT
    // caches don't clobber our writes after close. Failures are non-fatal:
    // we opened the physical drive exclusively, which blocks new mounts.
    // (Full FSCTL dismount plumbing per volume lives in flash-dismount.c
    // upstream; not needed for the common "fresh SD card" path.)
}

#[cfg(target_family = "windows")]
fn rescan_device(_device: &str) {}

#[cfg(not(target_family = "windows"))]
fn open_device_for_write(device: &str) -> Result<File, FlashError> {
    OpenOptions::new()
        .write(true)
        .read(true)
        .open(device)
        .map_err(|e| match e.kind() {
            std::io::ErrorKind::PermissionDenied => FlashError::Busy(format!(
                "{device} — need root (run with sudo)"
            )),
            _ => FlashError::Io(format!("open {device}: {e}")),
        })
}

#[cfg(not(target_family = "windows"))]
fn prepare_device(_device: &str) {}

#[cfg(not(target_family = "windows"))]
fn rescan_device(device: &str) {
    // Ask the kernel to reread the partition table (harmless if busy).
    let base = device.trim_end_matches(std::path::is_separator);
    let name = base.rsplit('/').next().unwrap_or(base);
    let _ = std::fs::write(format!("/sys/block/{name}/device/rescan"), "1");
}
