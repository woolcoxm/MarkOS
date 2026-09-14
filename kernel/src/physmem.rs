//! Bitmap physical frame allocator (4 KiB frames).
//!
//! owns: the exclusive right to hand out frames from the largest usable
//! memory-map region (a unikernel dedicated to one model needs exactly one
//! big region; smaller regions are ignored by design, not by accident).
//! invariants:
//! - `init` runs once, on the BSP, before any allocation is served.
//! - The allocator metadata lives in the managed region itself (start of it)
//!   and is reached through the Limine higher-half direct map (HHDM), so no
//!   page-table work is needed before this module works.
//! - Exclusive access is guaranteed by the `Mutex` around the single
//!   allocator instance; bitmaps are then plain (non-volatile) memory.
//! - `alloc` never panics (returns `None` on exhaustion); `dealloc` validates
//!   its argument and reports programmer errors instead of corrupting state.

// The public API below is exercised across the different selftest builds;
// some functions are unused in the default feature set.
#![allow(dead_code)]

use core::fmt::Write as _;
use core::hint::spin_loop;

use limine::memmap;
use spin::Mutex;
use x86_64::structures::paging::{PhysFrame, Size4KiB};
use x86_64::PhysAddr;

use crate::{serial, HHDM, MEMMAP};

const FRAME_SIZE: u64 = 4096;
const FRAME_SIZE_USIZE: usize = FRAME_SIZE as usize;

struct FrameAllocator {
    /// Physical address of the first managed frame (4 KiB aligned).
    region_base: u64,
    /// Total managed frames, including those backing the bitmap.
    frame_count: usize,
    /// HHDM virtual address of the bitmap (one bit per frame).
    bitmap: *mut u64,
    bitmap_words: usize,
    /// First possibly-free bitmap word; speeds up allocation after churn.
    hint: usize,
    /// Frames currently free (never counts bitmap-reserved frames).
    free: usize,
}
// Soundness: the raw bitmap pointer targets allocator-owned RAM inside the
// managed region; all access is funneled through `ALLOCATOR`'s Mutex, so
// moving instances between cores is safe.
unsafe impl Send for FrameAllocator {}

impl FrameAllocator {
    /// Build an allocator over `[base, base+len)`, bitmap placed at the very
    /// beginning of the region.
    unsafe fn from_region(base: u64, len: u64, hhdm: u64) -> Self {
        // Soundness: `base..base+len` is marked usable by the bootloader's
        // memory map; the HHDM maps all of physical memory read-write, so
        // `base + hhdm` is a valid virtual address for our bitmap. No other
        // code knows about this memory yet (Limine reserves its own structures
        // outside the regions it reports as usable to us).
        let base = base.div_ceil(FRAME_SIZE) * FRAME_SIZE;
        let end = (base + len) / FRAME_SIZE * FRAME_SIZE;
        assert!(end > base, "usable region smaller than one frame");
        let frame_count = ((end - base) / FRAME_SIZE) as usize;

        let bitmap_words = frame_count.div_ceil(64);
        let bitmap_frames = (bitmap_words * 8).div_ceil(FRAME_SIZE_USIZE);
        assert!(bitmap_frames <= frame_count, "region too small for bitmap");

        let bitmap = (base + hhdm) as *mut u64;

        // Start all-free, then reserve the frames the bitmap itself lives in.
        // Soundness: bitmap points at `bitmap_words * 8` bytes of usable RAM
        // (region start), exclusively owned by this instance.
        unsafe {
            core::ptr::write_bytes(bitmap, 0x00, bitmap_words);
            for i in 0..bitmap_frames {
                Self::set_bit(bitmap, i);
            }
        }

        Self {
            region_base: base,
            frame_count,
            bitmap,
            bitmap_words,
            hint: 0,
            free: frame_count - bitmap_frames,
        }
    }

    unsafe fn get_bit(bitmap: *mut u64, i: usize) -> bool {
        // Soundness: i < frame_count <= bitmap_words * 64, in-bounds by
        // construction; caller holds exclusive access via the Mutex.
        unsafe { (bitmap.add(i / 64).read_volatile() >> (i % 64)) & 1 == 1 }
    }

    unsafe fn set_bit(bitmap: *mut u64, i: usize) {
        // Soundness: as above; read-modify-write under exclusive access.
        unsafe {
            let w = bitmap.add(i / 64);
            w.write_volatile(w.read_volatile() | (1 << (i % 64)));
        }
    }

    unsafe fn clear_bit(bitmap: *mut u64, i: usize) {
        unsafe {
            let w = bitmap.add(i / 64);
            w.write_volatile(w.read_volatile() & !(1 << (i % 64)));
        }
    }

    fn alloc(&mut self) -> Option<PhysFrame<Size4KiB>> {
        if self.free == 0 {
            return None;
        }
        for skip in 0..self.bitmap_words {
            let w = (self.hint + skip) % self.bitmap_words;
            // Soundness: in-bounds by modulo; exclusive access held.
            let word = unsafe { self.bitmap.add(w).read_volatile() };
            if word != u64::MAX {
                let bit = word.trailing_ones() as usize;
                let index = w * 64 + bit;
                if index >= self.frame_count {
                    continue; // padding bits of the last word
                }
                unsafe { Self::set_bit(self.bitmap, index) };
                self.free -= 1;
                self.hint = w;
                let phys = self.region_base + (index as u64) * FRAME_SIZE;
                // Soundness: phys is within the usable region we own and was
                // marked occupied just above.
                return Some(unsafe { PhysFrame::from_start_address_unchecked(PhysAddr::new(phys)) });
            }
        }
        None // all remaining bits are padding
    }

    fn dealloc(&mut self, frame: PhysFrame<Size4KiB>) -> Result<(), &'static str> {
        let phys = frame.start_address().as_u64();
        if phys < self.region_base
            || phys >= self.region_base + (self.frame_count as u64) * FRAME_SIZE
        {
            return Err("frame outside managed region");
        }
        if phys % FRAME_SIZE != 0 {
            return Err("frame not 4 KiB aligned");
        }
        let index = ((phys - self.region_base) / FRAME_SIZE) as usize;
        if index < (self.bitmap_words * 8).div_ceil(FRAME_SIZE_USIZE) {
            return Err("cannot free allocator metadata");
        }
        // Soundness: in-bounds by the range check above.
        if unsafe { Self::get_bit(self.bitmap, index) } == false {
            return Err("double free");
        }
        unsafe { Self::clear_bit(self.bitmap, index) };
        self.free += 1;
        Ok(())
    }
}

static ALLOCATOR: Mutex<Option<FrameAllocator>> = Mutex::new(None);

/// Pick the largest usable region and start the allocator over it.
/// Returns `(managed_mib, free_frames)` for the boot log.
pub fn init() -> (u64, usize) {
    let hhdm = HHDM.response().expect("limine: no HHDM response").offset;

    let resp = MEMMAP.response().expect("limine: no memory map response");
    let entries: &[&memmap::Entry] = resp.entries();

    let mut best: Option<(u64, u64)> = None; // (base, len)
    for e in entries {
        if e.type_ == memmap::MEMMAP_USABLE {
            match best {
                Some((_, len)) if len >= e.length => {}
                _ => best = Some((e.base, e.length)),
            }
        }
    }
    let (base, len) = best.expect("limine memory map has no usable region");

    // Soundness: region comes straight from the bootloader memory map; the
    // allocator owns everything in it from here on.
    let alloc = unsafe { FrameAllocator::from_region(base, len, hhdm) };
    let mib = len / (1024 * 1024);
    let free = alloc.free;
    *ALLOCATOR.lock() = Some(alloc);
    (mib, free)
}

/// Allocate one 4 KiB physical frame. `None` on exhaustion — never panics.
pub fn alloc_frame() -> Option<PhysFrame<Size4KiB>> {
    let mut guard = ALLOCATOR.lock();
    let alloc = guard.as_mut()?;
    while alloc.free > 0 {
        let f = alloc.alloc();
        if f.is_some() {
            return f;
        }
        spin_loop();
    }
    None
}

/// Return a frame to the pool. Programmer errors are reported, not panics.
pub fn dealloc_frame(frame: PhysFrame<Size4KiB>) -> Result<(), &'static str> {
    let mut guard = ALLOCATOR.lock();
    match guard.as_mut() {
        Some(alloc) => alloc.dealloc(frame),
        None => Err("allocator not initialized"),
    }
}

/// Frames currently free (excludes bitmap metadata).
pub fn free_frames() -> usize {
    ALLOCATOR.lock().as_ref().map_or(0, |a| a.free)
}

/// Human-readable memory map entry type for the boot log.
pub fn entry_type_name(t: u64) -> &'static str {
    match t {
        memmap::MEMMAP_USABLE => "usable",
        memmap::MEMMAP_RESERVED => "reserved",
        memmap::MEMMAP_ACPI_RECLAIMABLE => "acpi-reclaimable",
        memmap::MEMMAP_ACPI_NVS => "acpi-nvs",
        memmap::MEMMAP_BAD_MEMORY => "bad-memory",
        memmap::MEMMAP_BOOTLOADER_RECLAIMABLE => "bootloader-reclaimable",
        memmap::MEMMAP_EXECUTABLE_AND_MODULES => "executable-and-modules",
        memmap::MEMMAP_FRAMEBUFFER => "framebuffer",
        memmap::MEMMAP_MAPPED_RESERVED => "mapped-reserved",
        _ => "unknown",
    }
}

/// Dump the bootloader memory map over serial (boot-time evidence).
pub fn dump_memory_map() {
    let Some(resp) = MEMMAP.response() else {
        serial::write_str("no memory map response\n");
        return;
    };
    let _ = write!(serial::Serial, "memory map ({} entries):\n", resp.entries().len());
    for e in resp.entries() {
        let _ = write!(
            serial::Serial,
            "  base={:#018x} len={:#014x} [{}]\n",
            e.base,
            e.length,
            entry_type_name(e.type_)
        );
    }
}
