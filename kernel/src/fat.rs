//! Minimal read-only FAT32 (over any block device).
//!
//! Scope decisions (appliance, not general FS):
//! - MBR partition scan for the first FAT32 (type 0x0B/0x0C) partition.
//! - 512-byte sectors, FAT32 only (FAT16/LFN not implemented — the
//!   installer writes model files under fixed 8.3-safe names).
//! - Full-cluster reads via the block layer; FAT entry lookups re-read the
//!   FAT sector (correctness first — a FAT cache lands with the perf phase).
//!
//! owns: the mounted volume geometry (static, single volume).
//! invariants: `mount` runs once on the BSP; all buffers passed in by
//! callers are identity-mapped RAM.


use crate::uart;

use crate::virtio_blk;

const SECTOR: usize = 512;
const ATTR_LFN: u8 = 0x0F;
const ATTR_DIR: u8 = 0x10;
const EOC_MIN: u32 = 0x0FFF_FFF8;

/// Scratch for partial-cluster reads in `read_at` (max FAT32 cluster is
/// 32 MiB in theory; 32 KiB covers every mkfs.vfat default).
static mut CLUSTER_SCRATCH: [u8; 32 * 1024] = [0u8; 32 * 1024];
const CLUSTER_SCRATCH_LEN: usize = 32 * 1024;

pub struct FatVolume {
    part_lba: u64,
    sectors_per_cluster: u32,
    num_fats: u32,
    fat_start_lba: u64,
    fat_sectors: u32,
    root_cluster: u32,
}

pub struct File {
    pub first_cluster: u32,
    pub size: u32,
}

/// One-sector cache for FAT entry lookups: sequential cluster walks hit
/// the same sector 128 times in a row (512B / 4B entries).
/// Soundness: BSP-only volume state; mount/read happen on one core.
static mut FAT_CACHE_LBA: u64 = u64::MAX;
static mut FAT_CACHE: [u8; 512] = [0u8; 512];
const FAT_CACHE_LEN: usize = 512;

fn rd16(buf: &[u8], off: usize) -> u16 {
    u16::from_le_bytes([buf[off], buf[off + 1]])
}

fn rd32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Scan MBR partition entries for a FAT32 partition; parse its boot sector.
pub fn mount() -> Result<FatVolume, &'static str> {
    let mut mbr = [0u8; SECTOR];
    virtio_blk::read_sectors(0, 1, &mut mbr)?;
    if mbr[510] != 0x55 || mbr[511] != 0xAA {
        return Err("MBR signature missing");
    }

    let mut part_lba: Option<u64> = None;
    for p in 0..4 {
        let e = 446 + p * 16;
        let ptype = mbr[e + 4];
        if ptype == 0x0B || ptype == 0x0C {
            part_lba = Some(rd32(&mbr, e + 8) as u64);
            break;
        }
    }
    let part_lba = part_lba.ok_or("no FAT32 partition in MBR")?;

    let mut bs = [0u8; SECTOR];
    virtio_blk::read_sectors(part_lba, 1, &mut bs)?;
    if bs[510] != 0x55 || bs[511] != 0xAA {
        return Err("FAT boot sector signature missing");
    }
    let sectors_per_cluster = bs[13] as u32;
    let reserved_sectors = rd16(&bs, 14) as u32;
    let num_fats = bs[16] as u32;
    let fat_sectors = if rd16(&bs, 22) != 0 {
        rd16(&bs, 22) as u32
    } else {
        rd32(&bs, 36)
    };
    let root_cluster = rd32(&bs, 44);
    if sectors_per_cluster == 0 || num_fats == 0 || fat_sectors == 0 {
        // Debug: show the first 16 bytes of what the guest read.
        uart::locked_write(format_args!("fat: bs[0..16]={:02x?}\n", &bs[..16]));
        return Err("FAT BPB fields invalid");
    }

    Ok(FatVolume {
        part_lba,
        sectors_per_cluster,
        num_fats,
        fat_start_lba: part_lba + reserved_sectors as u64,
        fat_sectors,
        root_cluster,
    })
}

impl FatVolume {
    fn cluster_lba(&self, cluster: u32) -> u64 {
        let data_start = self.fat_start_lba + (self.fat_sectors as u64) * (self.num_fats as u64);
        data_start + ((cluster as u64) - 2) * self.sectors_per_cluster as u64
    }

    /// FAT entry for `cluster` (FAT32: 32-bit LE, top 4 bits reserved).
    fn fat_entry(&self, cluster: u32) -> Result<u32, &'static str> {
        let fat_off_bytes = cluster as usize * 4;
        let lba = self.fat_start_lba + (fat_off_bytes / SECTOR) as u64;
        let off = fat_off_bytes % SECTOR;
        // One-sector cache: sequential cluster walks hit the same FAT
        // sector up to 128 times in a row (512B / 4B entries). Without it,
        // every read_at re-reads each FAT sector once per cluster skip.
        // Soundness: BSP-only volume state; reads happen on one core.
        let cached = unsafe {
            let mut hit = false;
            if FAT_CACHE_LBA == lba {
                hit = true;
            }
            if !hit {
                virtio_blk::read_sectors(
                    lba,
                    1,
                    &mut *(core::slice::from_raw_parts_mut(
                        (&raw mut FAT_CACHE) as *mut u8,
                        FAT_CACHE_LEN,
                    )),
                )?;
                FAT_CACHE_LBA = lba;
            }
            let bytes = core::slice::from_raw_parts(
                (&raw const FAT_CACHE) as *const u8,
                FAT_CACHE_LEN,
            );
            rd32(bytes, off) & 0x0FFF_FFFF
        };
        Ok(cached)
    }

    /// Read the whole cluster `cluster` into `buf`.
    fn read_cluster(&self, cluster: u32, buf: &mut [u8]) -> Result<(), &'static str> {
        virtio_blk::read_sectors(
            self.cluster_lba(cluster),
            self.sectors_per_cluster as usize,
            buf,
        )
    }

    /// Follow the cluster chain of `first_cluster`, passing each cluster's
    /// bytes to `f`; stops when the chain ends or `f` returns false.
    fn for_each_cluster<F>(&self, first: u32, mut f: F) -> Result<(), &'static str>
    where
        F: FnMut(&[u8]) -> bool,
    {
        let csize = self.sectors_per_cluster as usize * SECTOR;
        let mut cbuf = [0u8; 8 * SECTOR];
        if csize > cbuf.len() {
            return Err("cluster larger than 4 KiB scratch buffer");
        }
        let mut cluster = Some(first);
        while let Some(c) = cluster {
            self.read_cluster(c, &mut cbuf[..csize])?;
            if !f(&cbuf[..csize]) {
                return Ok(());
            }
            cluster = match self.fat_entry(c)? {
                next if next >= EOC_MIN => None,
                0 => None,
                next => Some(next),
            };
        }
        Ok(())
    }

    /// Find `MODEL.BIN` (raw 8.3 short name) in the root directory.
    pub fn open_model(&self) -> Result<File, &'static str> {
        let mut want = [0x20u8; 11];
        want[..5].copy_from_slice(b"MODEL");
        want[8..11].copy_from_slice(b"BIN");
        self.open_short(&want)
            .map_err(|_| "MODEL.BIN not found in volume root")
    }

    /// Find a file by raw 8.3 short name (11 bytes, space padded) in the
    /// volume root directory.
    pub fn open_short(&self, want: &[u8; 11]) -> Result<File, &'static str> {
        let mut found: Option<File> = None;
        self.for_each_cluster(self.root_cluster, |chunk| {
            for e in 0..chunk.len() / 32 {
                let entry = &chunk[e * 32..(e + 1) * 32];
                if entry[0] == 0x00 {
                    return false; // end of directory
                }
                if entry[0] == 0xE5 || entry[11] == ATTR_LFN || entry[11] & ATTR_DIR != 0 {
                    continue; // deleted / long-name / subdirectory
                }
                let m = entry[..11]
                    .iter()
                    .zip(want.iter())
                    .all(|(a, b)| a.to_ascii_uppercase() == *b);
                if m {
                    found = Some(File {
                        first_cluster: (rd16(entry, 20) as u32) << 16 | rd16(entry, 26) as u32,
                        size: rd32(entry, 28),
                    });
                    return false;
                }
            }
            true
        })?;
        found.ok_or("entry not found in volume root")
    }

    /// Read the entire file into `buf`; returns the number of bytes read.
    pub fn read_file(&self, file: &File, buf: &mut [u8]) -> Result<usize, &'static str> {
        if (buf.len() as u64) < file.size as u64 {
            return Err("file larger than buffer");
        }
        let csize = self.sectors_per_cluster as usize * SECTOR;
        let mut written = 0usize;
        let mut cluster = Some(file.first_cluster);
        while let Some(c) = cluster {
            if written + csize > buf.len() {
                break;
            }
            self.read_cluster(c, &mut buf[written..written + csize])?;
            written += csize;
            if written >= file.size as usize {
                break;
            }
            cluster = match self.fat_entry(c)? {
                next if next >= EOC_MIN => None,
                0 => None,
                next => Some(next),
            };
        }
        Ok(written)
    }

    /// Random-access read: `buf.len()` bytes at byte `offset` inside the
    /// file (short read at EOF). Walks the cluster chain to reach the
    /// offset, so GB-scale files can be sampled without loading them.
    pub fn read_at(&self, file: &File, offset: u64, buf: &mut [u8]) -> Result<usize, &'static str> {
        let csize = self.sectors_per_cluster as usize * SECTOR;
        if csize > CLUSTER_SCRATCH_LEN {
            return Err("cluster larger than scratch");
        }
        let mut cluster = file.first_cluster;
        // Skip the whole clusters that lie entirely before `offset`.
        let skip = (offset / csize as u64) as u32;
        for _ in 0..skip {
            cluster = self.fat_entry(cluster)?;
            if cluster == 0 || cluster >= EOC_MIN {
                return Err("offset past end of file");
            }
        }
        // Soundness: CLUSTER_SCRATCH is boot/selftest-stage scratch on the
        // BSP; the device DMAs into it while nothing else runs.
        let scratch = unsafe {
            core::slice::from_raw_parts_mut(
                (&raw mut CLUSTER_SCRATCH) as *mut u8,
                CLUSTER_SCRATCH_LEN,
            )
        };
        let mut pos_in_cluster = (offset % csize as u64) as usize;
        let mut written = 0usize;
        while written < buf.len() {
            self.read_cluster(cluster, &mut scratch[..csize])?;
            let avail = &scratch[pos_in_cluster..csize];
            let n = avail.len().min(buf.len() - written);
            buf[written..written + n].copy_from_slice(&avail[..n]);
            written += n;
            pos_in_cluster = 0;
            if written >= buf.len() {
                break;
            }
            cluster = self.fat_entry(cluster)?;
            if cluster == 0 || cluster >= EOC_MIN {
                break; // EOF: short read
            }
        }
        Ok(written)
    }
}
