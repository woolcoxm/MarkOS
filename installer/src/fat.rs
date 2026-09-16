//! Custom FAT32 writer: injects provision files into the boot partition of
//! a MarkOS image without external tools (no mkfs/mtools on Windows).
//!
//! Scope is deliberately narrow and matches what the installer needs:
//! open an existing FAT32 filesystem inside a raw image file, walk/create
//! directories (LFN entries written, short entries generated), write files
//! (cluster chains allocated from the FAT free space, both FAT copies
//! updated). Formatting/repair is out of scope — the build pipeline emits a
//! fresh, correct boot.vfat.

use std::io::{Read, Seek, SeekFrom, Write};

/// Any FAT entry >= EOC_MIN terminates a chain (mkdosfs writes 0x0FFFFFF8
/// for end-of-file; 0x0FFFFFFF is the generic EOC).
const EOC_MIN: u32 = 0x0FFF_FFF8;
const EOC_MARK: u32 = 0x0FFF_FFFF;
const FREE_MARK: u32 = 0x0000_0000;
const ATTR_LONG_NAME: u8 = 0x0F;
const ATTR_DIRECTORY: u8 = 0x10;
const ATTR_ARCHIVE: u8 = 0x20;

pub struct FatPartition<F: Read + Write + Seek> {
    file: F,
    /// Byte offset of the partition (volume start) inside the image.
    base: u64,
    bytes_per_sector: u32,
    sectors_per_cluster: u32,
    reserved_sectors: u32,
    num_fats: u32,
    fat_start_sector: u32,
    fat_sectors: u32,
    root_cluster: u32,
    cluster_count: u32,
    fat: Vec<u8>, // FAT[0] contents in memory
}

#[derive(Debug)]
pub enum FatError {
    Io(std::io::Error),
    NotFat32(&'static str),
    Full(&'static str),
    NotFound(String),
}

impl std::fmt::Display for FatError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FatError::Io(e) => write!(f, "io: {e}"),
            FatError::NotFat32(m) => write!(f, "not a FAT32 boot partition: {m}"),
            FatError::Full(m) => write!(f, "boot partition full: {m}"),
            FatError::NotFound(p) => write!(f, "path not found: {p}"),
        }
    }
}

impl From<std::io::Error> for FatError {
    fn from(e: std::io::Error) -> Self {
        FatError::Io(e)
    }
}

/// Locate the first FAT32 (type 0x0B/0x0C) partition in an MBR image and
/// open it. Returns None when no FAT32 partition exists.
pub fn open_boot_partition<F: Read + Write + Seek>(mut file: F) -> Result<FatPartition<F>, FatError> {
    let mut mbr = [0u8; 512];
    file.seek(SeekFrom::Start(0))?;
    file.read_exact(&mut mbr)?;
    if mbr[510] != 0x55 || mbr[511] != 0xAA {
        return Err(FatError::NotFat32("missing MBR signature"));
    }
    let mut lba_start: Option<u64> = None;
    for i in 0..4 {
        let e = &mbr[446 + i * 16..446 + i * 16 + 16];
        let ptype = e[4];
        let lba = u32::from_le_bytes([e[8], e[9], e[10], e[11]]);
        let nsec = u32::from_le_bytes([e[12], e[13], e[14], e[15]]);
        if (ptype == 0x0B || ptype == 0x0C) && nsec > 0 {
            lba_start = Some(lba as u64 * 512);
            break;
        }
    }
    let Some(base) = lba_start else {
        return Err(FatError::NotFat32("no FAT32 partition in MBR"));
    };
    check_boot_sector(&mut file, base)?;
    load_partition(file, base)
}

fn check_boot_sector<F: Read + Write + Seek>(file: &mut F, base: u64) -> Result<(), FatError> {
    let mut bpb = [0u8; 512];
    file.seek(SeekFrom::Start(base))?;
    file.read_exact(&mut bpb)?;
    if bpb[510] != 0x55 || bpb[511] != 0xAA {
        return Err(FatError::NotFat32("partition lacks boot signature"));
    }
    let bps = u16::from_le_bytes([bpb[11], bpb[12]]) as u32;
    if !matches!(bps, 512 | 1024 | 2048 | 4096) {
        return Err(FatError::NotFat32("implausible bytes/sector"));
    }
    Ok(())
}

fn load_partition<F: Read + Write + Seek>(mut file: F, base: u64) -> Result<FatPartition<F>, FatError> {
    let mut bpb = [0u8; 512];
    file.seek(SeekFrom::Start(base))?;
    file.read_exact(&mut bpb)?;
    let bytes_per_sector = u16::from_le_bytes([bpb[11], bpb[12]]) as u32;
    let sectors_per_cluster = bpb[13] as u32;
    let reserved_sectors = u16::from_le_bytes([bpb[14], bpb[15]]) as u32;
    let num_fats = bpb[16] as u32;
    let fat_sectors = u32::from_le_bytes([bpb[36], bpb[37], bpb[38], bpb[39]]);
    let root_cluster = u32::from_le_bytes([bpb[44], bpb[45], bpb[46], bpb[47]]) & 0x0FFF_FFFF;
    let total_sectors = {
        let s16 = u16::from_le_bytes([bpb[19], bpb[20]]) as u32;
        let s32 = u32::from_le_bytes([bpb[32], bpb[33], bpb[34], bpb[35]]);
        if s16 != 0 {
            s16
        } else {
            s32
        }
    };
    if bytes_per_sector == 0 || sectors_per_cluster == 0 || num_fats == 0 || fat_sectors == 0 {
        return Err(FatError::NotFat32("invalid BPB"));
    }
    // FAT16 BPBs put different fields at 0x24; reject early with a clear
    // message instead of tripping arithmetic overflow later.
    if &bpb[82..90] == b"FAT16   " {
        return Err(FatError::NotFat32(
            "boot partition is FAT16 (MarkOS images are FAT32)",
        ));
    }
    let fat_bytes = (fat_sectors as u64).saturating_mul(bytes_per_sector as u64);
    if fat_bytes == 0 || fat_bytes > 256 * 1024 * 1024 {
        return Err(FatError::NotFat32("FAT table implausibly large"));
    }
    let cluster_count =
        (total_sectors as u64).saturating_sub(reserved_sectors as u64 + num_fats as u64 * fat_sectors as u64)
            / sectors_per_cluster as u64;
    let fat_start_sector = reserved_sectors;
    file.seek(SeekFrom::Start(base + fat_start_sector as u64 * bytes_per_sector as u64))?;
    let mut fat = vec![0u8; fat_bytes as usize];
    file.read_exact(&mut fat)?;
    file.read_exact(&mut fat)?;
    Ok(FatPartition {
        file,
        base,
        bytes_per_sector,
        sectors_per_cluster,
        reserved_sectors,
        num_fats,
        fat_start_sector,
        fat_sectors,
        root_cluster,
        cluster_count: cluster_count.min(u32::MAX as u64) as u32,
        fat,
    })
}

impl<F: Read + Write + Seek> FatPartition<F> {
    fn cluster_offset(&self, cluster: u32) -> u64 {
        let data_start = self.reserved_sectors + self.num_fats * self.fat_sectors;
        self.base
            + (data_start + (cluster - 2) * self.sectors_per_cluster) as u64 * self.bytes_per_sector as u64
    }

    fn fat_entry(&self, cluster: u32) -> u32 {
        let off = cluster as usize * 4;
        u32::from_le_bytes([self.fat[off], self.fat[off + 1], self.fat[off + 2], self.fat[off + 3]])
            & 0x0FFF_FFFF
    }

    fn set_fat_entry(&mut self, cluster: u32, value: u32) {
        let off = cluster as usize * 4;
        let v = value & 0x0FFF_FFFF;
        self.fat[off..off + 4].copy_from_slice(&v.to_le_bytes());
    }

    fn read_cluster(&mut self, cluster: u32, buf: &mut [u8]) -> Result<(), FatError> {
        self.file.seek(SeekFrom::Start(self.cluster_offset(cluster)))?;
        self.file.read_exact(&mut buf[..self.cluster_bytes()])?;
        Ok(())
    }

    fn write_cluster(&mut self, cluster: u32, buf: &[u8]) -> Result<(), FatError> {
        self.file.seek(SeekFrom::Start(self.cluster_offset(cluster)))?;
        self.file.write_all(&buf[..self.cluster_bytes()])?;
        Ok(())
    }

    fn cluster_bytes(&self) -> usize {
        (self.bytes_per_sector * self.sectors_per_cluster) as usize
    }

    /// Allocate one free cluster (linear scan with a hint cursor is fine at
    /// our partition sizes).
    fn alloc_cluster(&mut self, hint: &mut u32) -> Result<u32, FatError> {
        let start = (*hint).max(2);
        for c in (start..2 + self.cluster_count).chain(2..start) {
            if self.fat_entry(c) == FREE_MARK {
                self.set_fat_entry(c, EOC_MARK);
                *hint = c + 1;
                // zero the cluster so stale data never leaks into dir entries
                let zeros = vec![0u8; self.cluster_bytes()];
                self.write_cluster(c, &zeros)?;
                return Ok(c);
            }
        }
        Err(FatError::Full("no free clusters"))
    }

    fn flush_fat(&mut self) -> Result<(), FatError> {
        let fat_start = self.base + self.fat_start_sector as u64 * self.bytes_per_sector as u64;
        for i in 0..self.num_fats {
            self.file.seek(SeekFrom::Start(
                fat_start + i as u64 * self.fat_sectors as u64 * self.bytes_per_sector as u64,
            ))?;
            self.file.write_all(&self.fat)?;
        }
        Ok(())
    }

    /// Read a full chain of clusters into a Vec.
    fn read_chain(&mut self, start: u32) -> Result<Vec<u8>, FatError> {
        let mut out = Vec::new();
        let mut c = start;
        let mut guard = 0;
        let mut buf = vec![0u8; self.cluster_bytes()];
        while c >= 2 && c < EOC_MIN {
            self.read_cluster(c, &mut buf)?;
            out.extend_from_slice(&buf);
            c = self.fat_entry(c);
            guard += 1;
            if guard > 1_000_000 {
                return Err(FatError::NotFat32("cluster chain loop"));
            }
        }
        Ok(out)
    }

    /// Write `data` as a fresh chain starting at `start` (replacing any old
    /// chain — caller frees the old one or passes a new cluster).
    fn write_chain(&mut self, start: u32, data: &[u8], hint: &mut u32) -> Result<(), FatError> {
        let cb = self.cluster_bytes();
        let need = data.len().div_ceil(cb).max(1);
        let mut clusters = Vec::with_capacity(need);
        let mut c = start;
        // Reuse the existing chain, extending as needed.
        for i in 0..need {
            if i == 0 {
                clusters.push(c);
            } else {
                match self.fat_entry(c) {
                    e if e >= 2 && e < EOC_MIN => {
                        clusters.push(e);
                        c = e;
                    }
                    _ => {
                        let n = self.alloc_cluster(hint)?;
                        self.set_fat_entry(c, n);
                        clusters.push(n);
                        c = n;
                    }
                }
            }
        }
        // Terminate and free any leftover tail.
        let last = *clusters.last().unwrap();
        match self.fat_entry(last) {
            e if e >= 2 && e < EOC_MIN => self.free_chain(e),
            _ => {}
        }
        self.set_fat_entry(last, EOC_MARK);
        for (i, cl) in clusters.iter().enumerate() {
            let chunk = &data[i * cb..((i + 1) * cb).min(data.len())];
            let mut padded = vec![0u8; cb];
            padded[..chunk.len()].copy_from_slice(chunk);
            self.write_cluster(*cl, &padded)?;
        }
        Ok(())
    }

    fn free_chain(&mut self, start: u32) {
        let mut c = start;
        let mut guard = 0;
        while c >= 2 && c < EOC_MIN {
            let next = self.fat_entry(c);
            self.set_fat_entry(c, FREE_MARK);
            c = next;
            guard += 1;
            if guard > 1_000_000 {
                break;
            }
        }
    }

    /// Walk/create a directory path; returns the directory's first cluster.
    fn resolve_dir(&mut self, path: &str, hint: &mut u32) -> Result<u32, FatError> {
        let mut cur = self.root_cluster;
        for part in path.split('/').filter(|p| !p.is_empty()) {
            cur = self.child_dir(cur, part, hint)?;
        }
        Ok(cur)
    }

    fn child_dir(&mut self, dir_cluster: u32, name: &str, hint: &mut u32) -> Result<u32, FatError> {
        if let Some((start, _size, is_dir)) = self.lookup(dir_cluster, name)? {
            if !is_dir {
                return Err(FatError::NotFound(format!("{name} (not a directory)")));
            }
            return Ok(start);
        }
        // Create the directory: data cluster with . / .. entries.
        let new_cluster = self.alloc_cluster(hint)?;
        let cb = self.cluster_bytes();
        let mut entries = vec![0u8; cb];
        let mut dot = Self::short_entry(".");
        dot[0] = b'.';
        dot[11] |= ATTR_DIRECTORY;
        Self::set_entry_cluster(&mut dot, new_cluster);
        entries[..32].copy_from_slice(&dot);

        let mut dotdot = Self::short_entry("..");
        dotdot[0] = b'.';
        dotdot[1] = b'.';
        dotdot[11] |= ATTR_DIRECTORY;
        Self::set_entry_cluster(&mut dotdot, dir_cluster);
        entries[32..64].copy_from_slice(&dotdot);
        self.write_cluster(new_cluster, &entries)?;

        self.add_entry(dir_cluster, name, new_cluster, 0, true, hint)?;
        Ok(new_cluster)
    }

    /// FAT32 splits the cluster number: low 16 bits at 26, high 16 at 20.
    fn set_entry_cluster(e: &mut [u8; 32], cluster: u32) {
        let v = cluster & 0x0FFF_FFFF;
        e[26..28].copy_from_slice(&(v as u16).to_le_bytes());
        e[20..22].copy_from_slice(&((v >> 16) as u16).to_le_bytes());
    }

    fn lookup(&mut self, dir_cluster: u32, name: &str) -> Result<Option<(u32, u32, bool)>, FatError> {
        let data = self.read_chain(dir_cluster)?;
        let mut i = 0;
        while i + 32 <= data.len() {
            let e = &data[i..i + 32];
            if e[0] == 0x00 {
                break; // end of directory
            }
            if e[0] == 0xE5 || e[11] & ATTR_LONG_NAME == ATTR_LONG_NAME {
                i += 32;
                continue;
            }
            if let Some(short) = ShortName::decode(e) {
                if short.matches(name) {
                    let cl = u32::from_le_bytes([e[26], e[27], 0, 0])
                        | ((u16::from_le_bytes([e[20], e[21]]) as u32) << 16);
                    let size = u32::from_le_bytes([e[28], e[29], e[30], e[31]]);
                    return Ok(Some((cl, size, e[11] & ATTR_DIRECTORY != 0)));
                }
            }
            i += 32;
        }
        Ok(None)
    }

    fn short_entry(name: &str) -> [u8; 32] {
        let sn = ShortName::encode(name);
        let mut e = [0u8; 32];
        e[..11].copy_from_slice(&sn.raw);
        e[11] = ATTR_ARCHIVE;
        // Timestamps: fixed — the appliance doesn't care, and determinism
        // keeps image builds reproducible.
        e[13] = 0; // tenths
        e[14..16].copy_from_slice(&0x6000u16.to_le_bytes()); // 12:00:00
        e[16..18].copy_from_slice(&0x5A40u16.to_le_bytes()); // 2026-01-01
        e[18..20].copy_from_slice(&0x5A40u16.to_le_bytes());
        e[22..24].copy_from_slice(&0x6000u16.to_le_bytes());
        e[24..26].copy_from_slice(&0x5A40u16.to_le_bytes());
        e
    }

    /// Append a directory entry (LFN + short) for `name` into dir at
    /// `dir_cluster`, creating a new directory cluster if the current one
    /// is full. Returns the entry offset chain end.
    fn add_entry(
        &mut self,
        dir_cluster: u32,
        name: &str,
        start_cluster: u32,
        size: u32,
        is_dir: bool,
        hint: &mut u32,
    ) -> Result<(), FatError> {
        let sn = ShortName::encode(name);
        let lfn_entries = Self::lfn_entries(name, &sn.checksum);
        let total = (lfn_entries.len() + 1) * 32;

        let mut data = self.read_chain(dir_cluster)?;
        // Find the start of the clean zero tail (end-of-directory marker).
        // The MarkOS installer never deletes boot-partition files, so free
        // space is exactly that tail; if it can't hold the new entry set,
        // the directory grows by one cluster.
        let mut free_at: Option<usize> = None;
        let mut i = 0;
        while i + 32 <= data.len() {
            if data[i] == 0xE5 {
                i += 32;
                continue;
            }
            if data[i] == 0x00 {
                if data.len() - i >= total && data[i..].iter().all(|&b| b == 0) {
                    free_at = Some(i);
                }
                break;
            }
            i += 32;
        }
        let at = match free_at {
            Some(a) => a,
            None => {
                // extend the directory with a new cluster worth of slots
                let old_len = data.len();
                data.resize(old_len + self.cluster_bytes(), 0);
                // fix the chain
                let mut last = dir_cluster;
                let mut guard = 0;
                while let nx @ 2..=0x0FFF_FFF7 = self.fat_entry(last) {
                    last = nx;
                    guard += 1;
                    if guard > 1_000_000 {
                        return Err(FatError::NotFat32("dir chain loop"));
                    }
                }
                let mut hint_local = hint;
                let n = self.alloc_cluster(&mut hint_local)?;
                self.set_fat_entry(last, n);
                let at = old_len;
                self.write_chain(dir_cluster, &data, &mut hint_local)?;
                // write entry bytes below using the extended buffer
                let mut all = self.read_chain(dir_cluster)?;
                let mut off = at;
                for e in lfn_entries.iter().chain(std::iter::once(&{
                    let mut s = Self::short_entry(name);
                    s[11] = if is_dir { ATTR_DIRECTORY } else { ATTR_ARCHIVE };
                    Self::set_entry_cluster(&mut s, start_cluster);
                    s[28..32].copy_from_slice(&size.to_le_bytes());
                    s
                })) {
                    all[off..off + 32].copy_from_slice(e);
                    off += 32;
                }
                self.write_chain(dir_cluster, &all, &mut hint_local)?;
                return Ok(());
            }
        };
        let mut off = at;
        for e in lfn_entries.iter().chain(std::iter::once(&{
            let mut s = Self::short_entry(name);
            s[11] = if is_dir { ATTR_DIRECTORY } else { ATTR_ARCHIVE };
            Self::set_entry_cluster(&mut s, start_cluster);
            s[28..32].copy_from_slice(&size.to_le_bytes());
            s
        })) {
            data[off..off + 32].copy_from_slice(e);
            off += 32;
        }
        self.write_chain(dir_cluster, &data, hint)?;
        Ok(())
    }

    fn lfn_entries(name: &str, checksum: &u8) -> Vec<[u8; 32]> {
        // UTF-16 code units, NUL-terminated, padded with 0xFFFF; entries are
        // written in reverse order with sequence numbers 2..N then 1|0x40.
        let units: Vec<u16> = name.encode_utf16().collect();
        let mut padded: Vec<u16> = units.clone();
        padded.push(0x0000);
        while padded.len() % 13 != 0 {
            padded.push(0xFFFF);
        }
        let seq_total = padded.len().div_ceil(13) as u8;
        let mut out = Vec::new();
        for seq in (1..=seq_total).rev() {
            let mut e = [0xFFu8; 32];
            e[0] = if seq == seq_total { seq | 0x40 } else { seq };
            e[11] = ATTR_LONG_NAME;
            e[12] = 0;
            e[13] = *checksum;
            // Chars live in slots 1..10, 14..25, 28..31.
            let chunk: Vec<u16> = padded[(seq as usize - 1) * 13..seq as usize * 13].to_vec();
            for (idx, u) in chunk.iter().enumerate() {
                let b = u.to_le_bytes();
                let (off, term) = match idx {
                    0..=4 => (1 + idx * 2, false),
                    5..=10 => (14 + (idx - 5) * 2, false),
                    _ => (28 + (idx - 11) * 2, idx == 12),
                };
                let _ = term;
                e[off] = b[0];
                e[off + 1] = b[1];
            }
            out.push(e);
        }
        out
    }

    /// Write (create or replace) a file at `path` with `content`.
    pub fn write_file(&mut self, path: &str, content: &[u8]) -> Result<(), FatError> {
        let mut hint = 2u32;
        let (dir_path, fname) = match path.rsplit_once('/') {
            Some((d, f)) => (d.to_string(), f.to_string()),
            None => (String::new(), path.to_string()),
        };
        let dir_cluster = self.resolve_dir(&dir_path, &mut hint)?;
        // Replace if present (free old chain), else create.
        if let Some((old_start, _size, is_dir)) = self.lookup(dir_cluster, &fname)? {
            if is_dir {
                return Err(FatError::NotFound(format!("{path} is a directory")));
            }
            self.write_chain(old_start, content, &mut hint)?;
            self.update_entry_size(dir_cluster, &fname, content.len() as u32)?;
        } else {
            let first = self.alloc_cluster(&mut hint)?;
            self.write_chain(first, content, &mut hint)?;
            self.add_entry(dir_cluster, &fname, first, content.len() as u32, false, &mut hint)?;
        }
        self.flush_fat()?;
        self.file.flush()?;
        Ok(())
    }

    /// Patch the size field of an existing entry (after overwrite).
    fn update_entry_size(&mut self, dir_cluster: u32, name: &str, size: u32) -> Result<(), FatError> {
        let mut data = self.read_chain(dir_cluster)?;
        let mut i = 0;
        while i + 32 <= data.len() {
            let e = &data[i..i + 32];
            if e[0] == 0x00 {
                break;
            }
            if e[0] != 0xE5 && e[11] & ATTR_LONG_NAME != ATTR_LONG_NAME {
                if let Some(short) = ShortName::decode(e) {
                    if short.matches(name) {
                        data[i + 28..i + 32].copy_from_slice(&size.to_le_bytes());
                        let mut hint = 2;
                        self.write_chain(dir_cluster, &data, &mut hint)?;
                        return Ok(());
                    }
                }
            }
            i += 32;
        }
        Err(FatError::NotFound(name.to_string()))
    }

    /// Walk a directory path WITHOUT creating anything; None if absent.
    fn find_dir(&mut self, path: &str) -> Result<Option<u32>, FatError> {
        let mut cur = self.root_cluster;
        for part in path.split('/').filter(|p| !p.is_empty()) {
            match self.lookup(cur, part)? {
                Some((start, _s, true)) => cur = start,
                Some(_) => return Err(FatError::NotFound(format!("{part} is not a directory"))),
                None => return Ok(None),
            }
        }
        Ok(Some(cur))
    }

    /// Read a file's contents through the same lookup path the OS would use.
    /// Returns None when absent.
    pub fn read_file(&mut self, path: &str) -> Result<Option<Vec<u8>>, FatError> {
        let (dir_path, fname) = match path.rsplit_once('/') {
            Some((d, f)) => (d, f),
            None => ("", path),
        };
        let Some(dir_cluster) = self.find_dir(dir_path)? else {
            return Ok(None);
        };
        match self.lookup(dir_cluster, fname)? {
            Some((start, size, false)) => {
                let mut data = self.read_chain(start)?;
                data.truncate(size as usize);
                Ok(Some(data))
            }
            Some(_) => Err(FatError::NotFound(format!("{path} is a directory"))),
            None => Ok(None),
        }
    }

    /// Create a directory if missing (used for the snapshot dir).
    #[allow(dead_code)] // test/CI surface; write_file creates parents implicitly
    pub fn ensure_dir(&mut self, path: &str) -> Result<(), FatError> {
        let mut hint = 2u32;
        self.resolve_dir(path, &mut hint)?;
        self.flush_fat()?;
        self.file.flush()?;
        Ok(())
    }
}

/// 8.3 short name handling: generation for new entries + decode for lookups.
struct ShortName {
    raw: [u8; 11],
    checksum: u8,
}

impl ShortName {
    fn encode(name: &str) -> ShortName {
        let upper = name.to_ascii_uppercase();
        let (base, ext) = match upper.rsplit_once('.') {
            Some((b, e)) if !b.is_empty() || !e.is_empty() => (b, e),
            _ => (upper.as_str(), ""),
        };
        let mut raw = [b' '; 11];
        let mut prefix: Vec<u8> = base.bytes().filter(|b| *b != b' ').collect();
        let tail = 6; // NAME~1 style mangle when too long
        if prefix.len() > 8 {
            prefix.truncate(tail);
            prefix.extend_from_slice(b"~1");
        }
        raw[..prefix.len().min(8)].copy_from_slice(&prefix[..prefix.len().min(8)]);
        let ext_b: Vec<u8> = ext.bytes().take(3).collect();
        raw[8..8 + ext_b.len()].copy_from_slice(&ext_b);
        let checksum = Self::checksum(&raw);
        ShortName { raw, checksum }
    }

    fn decode(e: &[u8]) -> Option<ShortName> {
        if e[0] == 0x00 || e[0] == 0xE5 {
            return None;
        }
        let mut raw = [0u8; 11];
        raw.copy_from_slice(&e[..11]);
        if raw[0] == 0x05 {
            raw[0] = 0xE5;
        }
        Some(ShortName { raw, checksum: e[13] })
    }

    fn matches(&self, name: &str) -> bool {
        let candidate = ShortName::encode(name);
        candidate.raw == self.raw
            || self.to_string_lossy().eq_ignore_ascii_case(name)
    }

    fn to_string_lossy(&self) -> String {
        let base = String::from_utf8_lossy(&self.raw[..8]).trim_end().to_string();
        let ext = String::from_utf8_lossy(&self.raw[8..]).trim_end().to_string();
        if ext.is_empty() {
            base
        } else {
            format!("{base}.{ext}")
        }
    }

    fn checksum(raw: &[u8; 11]) -> u8 {
        let mut sum = 0u8;
        for b in raw {
            sum = sum.rotate_right(1).wrapping_add(*b);
        }
        sum
    }
}

/// Convenience: inject the standard provision file set into an image.
pub fn inject_provision<F: Read + Write + Seek>(
    file: &mut F,
    files: &[(&str, Vec<u8>)],
) -> Result<(), FatError> {
    let mut part = open_boot_partition(file)?;
    for (path, content) in files {
        part.write_file(path, content)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Build a minimal-but-valid FAT32 volume (used by the tests as a stand-in
    /// for genimage's boot.vfat): 32 MiB, 512 B sectors, 4 KiB clusters.
    fn make_fat32_image() -> Vec<u8> {
        let bps = 512u32;
        let spc = 8u32;
        let total_sectors = 32 * 1024 * 1024 / bps; // 32 MiB
        let reserved = 32u32;
        let nfats = 2u32;
        let cluster_count = (total_sectors - reserved) / spc;
        let fat_sectors = (cluster_count * 4).div_ceil(bps);
        let mut img = vec![0u8; (total_sectors * bps) as usize];
        let mut bpb = [0u8; 512];
        bpb[0] = 0xEB; // jmp
        bpb[1] = 0x3C;
        bpb[2] = 0x90;
        bpb[3..11].copy_from_slice(b"MARKOS  ");
        bpb[11..13].copy_from_slice(&(bps as u16).to_le_bytes());
        bpb[13] = spc as u8;
        bpb[14..16].copy_from_slice(&(reserved as u16).to_le_bytes());
        bpb[16] = nfats as u8;
        bpb[17..19].copy_from_slice(&0u16.to_le_bytes()); // root entries (fat32)
        bpb[19..21].copy_from_slice(&0u16.to_le_bytes());
        bpb[21] = 0xF8; // media
        bpb[22..24].copy_from_slice(&0u16.to_le_bytes());
        bpb[24..26].copy_from_slice(&(63u16).to_le_bytes()); // sectors/track
        bpb[26..28].copy_from_slice(&(255u16).to_le_bytes()); // heads
        bpb[28..32].copy_from_slice(&0u32.to_le_bytes()); // hidden
        bpb[32..36].copy_from_slice(&total_sectors.to_le_bytes());
        bpb[36..40].copy_from_slice(&fat_sectors.to_le_bytes());
        bpb[40..42].copy_from_slice(&0u16.to_le_bytes()); // ext flags
        bpb[42..44].copy_from_slice(&0u16.to_le_bytes()); // fs version
        bpb[44..48].copy_from_slice(&2u32.to_le_bytes()); // root cluster
        bpb[48..50].copy_from_slice(&1u16.to_le_bytes()); // fsinfo sector
        bpb[50..52].copy_from_slice(&6u16.to_le_bytes()); // backup boot sector
        bpb[82..90].copy_from_slice(b"FAT32   ");
        bpb[510] = 0x55;
        bpb[511] = 0xAA;
        img[..512].copy_from_slice(&bpb);
        // FSInfo (sector 1)
        let mut fsinfo = [0u8; 512];
        fsinfo[0..4].copy_from_slice(&0x41615252u32.to_le_bytes());
        fsinfo[484..488].copy_from_slice(&0x61417272u32.to_le_bytes());
        fsinfo[488..492].copy_from_slice(&(cluster_count as u32).to_le_bytes());
        fsinfo[492..496].copy_from_slice(&(1u32).to_le_bytes());
        let at = reserved as usize * 512;
        img[at..at + 512].copy_from_slice(&fsinfo);
        // FAT[0] = 0x0FFFFFF8, FAT[1] = EOC, FAT[2] (root) = EOC
        let fat_off = reserved as usize * 512;
        img[fat_off..fat_off + 4].copy_from_slice(&0x0FFFFFF8u32.to_le_bytes());
        img[fat_off + 4..fat_off + 8].copy_from_slice(&0x0FFFFFFFu32.to_le_bytes());
        img[fat_off + 8..fat_off + 12].copy_from_slice(&0x0FFFFFFFu32.to_le_bytes());
        let fat2_off = fat_off + fat_sectors as usize * 512;
        img[fat2_off..fat2_off + 4].copy_from_slice(&0x0FFFFFF8u32.to_le_bytes());
        img[fat2_off + 4..fat2_off + 8].copy_from_slice(&0x0FFFFFFFu32.to_le_bytes());
        img[fat2_off + 8..fat2_off + 12].copy_from_slice(&0x0FFFFFFFu32.to_le_bytes());
        img
    }

    /// Wrap the volume in an MBR with a FAT32 LBA partition at 1 MiB.
    fn mbr_wrap(fat_img: &[u8]) -> Vec<u8> {
        let mut img = vec![0u8; 1024 * 1024 + fat_img.len()];
        let mut mbr = [0u8; 512];
        let e = &mut mbr[446..462];
        e[0] = 0x80; // bootable
        e[4] = 0x0C; // FAT32 LBA
        e[8..12].copy_from_slice(&2048u32.to_le_bytes()); // 1 MiB
        e[12..16].copy_from_slice(&((fat_img.len() as u32) / 512).to_le_bytes());
        mbr[510] = 0x55;
        mbr[511] = 0xAA;
        img[..512].copy_from_slice(&mbr);
        img[1024 * 1024..].copy_from_slice(fat_img);
        img
    }

    fn cursor(img: Vec<u8>) -> Cursor<Vec<u8>> {
        Cursor::new(img)
    }

    #[test]
    fn write_read_roundtrip_with_lfn() {
        let img = mbr_wrap(&make_fat32_image());
        let mut c = cursor(img);
        {
            let mut part = open_boot_partition(&mut c).expect("open");
            part.write_file("config.txt", b"kernel=Image\n").unwrap();
            part.ensure_dir("markos").unwrap();
            part.write_file("markos/provision.env", b"MARKOS_ADMIN_USER=admin\n").unwrap();
            part.write_file("markos/authorized_keys", b"ssh-ed25519 AAAA test\n").unwrap();
            part.write_file("markos-snapshot/x.txt", b"snapshot").unwrap();
        }
        // Re-open and read back through the same code path (lookup walks).
        let mut part = open_boot_partition(&mut c).unwrap();
        let (cfg_start, _, _) = part.lookup(part.root_cluster, "config.txt").unwrap().unwrap();
        let data = part.read_chain(cfg_start).unwrap();
        assert_eq!(&data[..13], b"kernel=Image\n");
        let markos = part.child_dir(part.root_cluster, "markos", &mut 2).unwrap();
        let (start, size, is_dir) = part.lookup(markos, "provision.env").unwrap().unwrap();
        assert!(!is_dir);
        assert_eq!(size, 24);
        let content = part.read_chain(start).unwrap();
        assert_eq!(&content[..24], b"MARKOS_ADMIN_USER=admin\n");
    }

    #[test]
    fn overwrite_replaces_content() {
        let img = mbr_wrap(&make_fat32_image());
        let mut c = cursor(img);
        {
            let mut part = open_boot_partition(&mut c).unwrap();
            part.write_file("config.txt", b"v1\n").unwrap();
            part.write_file("config.txt", b"v2-longer-content\n").unwrap();
        }
        let mut part = open_boot_partition(&mut c).unwrap();
        let (start, size, _) = part.lookup(part.root_cluster, "config.txt").unwrap().unwrap();
        let content = part.read_chain(start).unwrap();
        assert_eq!(size, 18);
        assert_eq!(&content[..size as usize], b"v2-longer-content\n");
    }

    #[test]
    fn many_files_fit_and_are_found() {
        let img = mbr_wrap(&make_fat32_image());
        let mut c = cursor(img);
        {
            let mut part = open_boot_partition(&mut c).unwrap();
            for i in 0..200 {
                part.write_file(&format!("entry-{i}.txt"), format!("content {i}").as_bytes()).unwrap();
            }
        }
        let mut part = open_boot_partition(&mut c).unwrap();
        let (start, _, _) = part.lookup(part.root_cluster, "entry-199.txt").unwrap().unwrap();
        let content = part.read_chain(start).unwrap();
        assert_eq!(&content[..11], b"content 199");
    }
}

#[cfg(test)]
mod debug_tests {
    use super::*;
    use crate::testimg;

    #[test]
    fn cli_flow_repro() {
        let dir = std::env::temp_dir().join(format!("fatdbg-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let base = dir.join("base.img");
        testimg::write_mbr_fat32_image(&base).unwrap();
        let out = dir.join("configured.img");
        std::fs::copy(&base, &out).unwrap();
        let mut f = std::fs::OpenOptions::new().read(true).write(true).open(&out).unwrap();
        let files = vec![
            ("provision.env".to_string(), b"MARKOS_ADMIN_USER=admin\n".to_vec()),
            ("net.conf".to_string(), b"MODE=dhcp\n".to_vec()),
            ("ssh-enabled".to_string(), Vec::new()),
        ];
        // replicate imagebuild: write into markos/ dir
        let mut part = open_boot_partition(&mut f).unwrap();
        part.ensure_dir("markos").unwrap();
        for (k, v) in &files {
            part.write_file(&format!("markos/{k}"), v).unwrap();
        }
        drop(part);
        // re-open like inspect
        let mut f2 = std::fs::File::open(&out).unwrap();
        let mut part2 = open_boot_partition(&mut f2).unwrap();
        for (k, _) in &files {
            let r = part2.read_file(&format!("markos/{k}")).unwrap();
            println!("{k}: {} bytes", r.as_ref().map(|v| v.len()).unwrap_or(9999));
            assert!(r.is_some(), "{k} not found after re-open");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
