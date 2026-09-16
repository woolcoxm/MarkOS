//! Maintenance helpers for CI/e2e: synthesize a MarkOS-shaped image
//! (MBR + FAT32 boot partition + fake rootfs partitions) and inspect the
//! provision files inside a real image. These make the installer's
//! build-and-inject path testable end to end without buildroot.

use crate::fat;
use std::io::Write;
use std::path::Path;

/// Build a synthetic image with the same partition geometry as the real
/// pipeline (boot FAT32 at 1 MiB, then two ext4-ish blobs and a data blob).
pub fn write_mbr_fat32_image(path: &Path) -> Result<(), String> {
    let fat = make_fat32_volume()?;
    let mut img = vec![0u8; 1024 * 1024];
    let mut mbr = [0u8; 512];

    // p1: FAT32 LBA at 1 MiB
    let e1 = &mut mbr[446..462];
    e1[0] = 0x80;
    e1[4] = 0x0C;
    e1[8..12].copy_from_slice(&2048u32.to_le_bytes());
    e1[12..16].copy_from_slice(&((fat.len() / 512) as u32).to_le_bytes());
    // p2/p3/p4: fake partitions so the layout resembles the real thing
    let extra = 8 * 1024 * 1024u64;
    let total_after = (1024 * 1024 + fat.len() as u64 + 3 * extra) as u32;
    let lba2 = (1024 * 1024 + fat.len() as u64) as u32 / 512;
    let e2 = &mut mbr[462..478];
    e2[4] = 0x83;
    e2[8..12].copy_from_slice(&lba2.to_le_bytes());
    e2[12..16].copy_from_slice(&((extra / 512) as u32).to_le_bytes());
    let e3 = &mut mbr[478..494];
    e3[4] = 0x83;
    e3[8..12].copy_from_slice(&(lba2 + (extra / 512) as u32).to_le_bytes());
    e3[12..16].copy_from_slice(&((extra / 512) as u32).to_le_bytes());
    let e4 = &mut mbr[494..510];
    e4[4] = 0x83;
    e4[8..12].copy_from_slice(&(lba2 + 2 * (extra / 512) as u32).to_le_bytes());
    e4[12..16].copy_from_slice(&((extra / 512) as u32).to_le_bytes());
    mbr[510] = 0x55;
    mbr[511] = 0xAA;
    img[..512].copy_from_slice(&mbr);
    img.extend_from_slice(&fat);
    img.resize(total_after as usize, 0);

    let mut f = std::fs::File::create(path).map_err(|e| e.to_string())?;
    f.write_all(&img).map_err(|e| e.to_string())?;
    Ok(())
}

/// A valid (if small) FAT32 volume with a boot signature — same builder as
/// the unit tests, exposed for CI.
fn make_fat32_volume() -> Result<Vec<u8>, String> {
    let bps = 512u32;
    let spc = 8u32;
    let total_sectors = 16 * 1024 * 1024 / bps;
    let reserved = 32u32;
    let nfats = 2u32;
    let cluster_count = (total_sectors - reserved) / spc;
    let fat_sectors = (cluster_count * 4).div_ceil(bps);
    let mut img = vec![0u8; (total_sectors * bps) as usize];
    let mut bpb = [0u8; 512];
    bpb[0] = 0xEB;
    bpb[1] = 0x3C;
    bpb[2] = 0x90;
    bpb[3..11].copy_from_slice(b"MARKOS  ");
    bpb[11..13].copy_from_slice(&(bps as u16).to_le_bytes());
    bpb[13] = spc as u8;
    bpb[14..16].copy_from_slice(&(reserved as u16).to_le_bytes());
    bpb[16] = nfats as u8;
    bpb[32..36].copy_from_slice(&total_sectors.to_le_bytes());
    bpb[36..40].copy_from_slice(&fat_sectors.to_le_bytes());
    bpb[44..48].copy_from_slice(&2u32.to_le_bytes());
    bpb[82..90].copy_from_slice(b"FAT32   ");
    bpb[510] = 0x55;
    bpb[511] = 0xAA;
    img[..512].copy_from_slice(&bpb);
    let fat_off = reserved as usize * 512;
    for fat_copy in 0..nfats as usize {
        let off = fat_off + fat_copy * fat_sectors as usize * 512;
        img[off..off + 4].copy_from_slice(&0x0FFFFFF8u32.to_le_bytes());
        img[off + 4..off + 8].copy_from_slice(&0x0FFFFFFFu32.to_le_bytes());
        img[off + 8..off + 12].copy_from_slice(&0x0FFFFFFFu32.to_le_bytes());
    }
    Ok(img)
}

/// Print the provision files from an image's boot partition.
pub fn inspect(path: &Path) -> Result<(), String> {
    let mut f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut part = fat::open_boot_partition(&mut f).map_err(|e| e.to_string())?;
    for name in [
        "config.txt",
        "markos/provision.env",
        "markos/net.conf",
        "markos/wpa_supplicant.conf",
        "markos/authorized_keys",
        "markos/ssh-enabled",
        "markos/ssh-disabled",
    ] {
        match part.read_file(name).map_err(|e| e.to_string())? {
            Some(content) => {
                let shown = String::from_utf8_lossy(&content);
                println!("--- /{name} ({} bytes) ---", content.len());
                println!("{}", shown.trim_end());
            }
            None => println!("--- /{name}: absent ---"),
        }
    }
    Ok(())
}
