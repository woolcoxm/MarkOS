//! Active page-table management for the kernel's virtual address space.
//!
//! The kernel runs higher-half; Limine establishes the initial mapping
//! (kernel image, HHDM of all RAM). This module adds `map`/`unmap`/
//! `translate` on the *active* address space, walking the tables through
//! the HHDM — no recursive-mapping tricks needed, since every table frame
//! is reachable at `phys + HHDM_OFFSET`.
//!
//! owns: two virtual regions reserved for the kernel:
//!   0xffffffff80000000 ..  kernel image (linked + mapped by Limine)
//!   0xffffffffc0000000 ..  kernel heap (mapped by `heap::init`)
//! invariants:
//! - `init` (EFER.NXE + HHDM offset) runs once on the BSP before any map.
//! - Mapping happens with interrupts disabled and, until Phase 4, on a
//!   single core, so the plain-Mutex critical section is sufficient.
//! - Newly created intermediate tables are PRESENT|WRITABLE|NO_EXECUTE.

use spin::Mutex;
use x86_64::registers::control::Cr3;
use x86_64::registers::model_specific::Msr;
use x86_64::structures::paging::page_table::{PageTable, PageTableEntry, PageTableFlags};
use x86_64::structures::paging::{Page, PageSize, PhysFrame, Size1GiB, Size2MiB, Size4KiB};
use x86_64::{PhysAddr, VirtAddr};

/// MSR IA32_EFER; bit 11 = NXE (enables the PTE no-execute bit).
const IA32_EFER: u32 = 0xc000_0080;
const IA32_EFER_NXE: u64 = 1 << 11;

/// Base of the kernel heap virtual region (topmost 2 GiB, above the image).
pub const HEAP_BASE: u64 = 0xffff_ffff_c000_0000;

static HHDM_OFFSET: Mutex<u64> = Mutex::new(0);

/// Enable NX page permissions and latch the HHDM offset (idempotent).
pub fn init() {
    // Soundness: EFER is kernel-owned MSR state; NXE must be set before any
    // PTE may carry the NX bit. Limine's own PTEs already assume it is set.
    let efer = unsafe { Msr::new(IA32_EFER).read() };
    if efer & IA32_EFER_NXE == 0 {
        unsafe { Msr::new(IA32_EFER).write(efer | IA32_EFER_NXE) };
    }
    *HHDM_OFFSET.lock() = crate::HHDM
        .response()
        .expect("limine HHDM response missing")
        .offset;
}

/// Physical → virtual through the higher-half direct map.
pub fn phys_to_virt(phys: PhysAddr) -> VirtAddr {
    VirtAddr::new(phys.as_u64() + *HHDM_OFFSET.lock())
}

/// The active PML4, reached through the HHDM.
fn root_table() -> &'static mut PageTable {
    let (frame, _) = Cr3::read();
    // Soundness: the HHDM maps all physical memory read-write, so the root
    // table frame is addressable; the active tables live for the life of
    // the kernel and are never freed.
    unsafe { &mut *(phys_to_virt(frame.start_address()).as_u64() as *mut PageTable) }
}

/// Step one level down along an existing (present, non-huge) entry.
fn next_table(entry: &PageTableEntry) -> Option<&'static mut PageTable> {
    if !entry.flags().contains(PageTableFlags::PRESENT)
        || entry.flags().contains(PageTableFlags::HUGE_PAGE)
    {
        return None;
    }
    let frame = entry.frame().ok()?;
    // Soundness: same HHDM reasoning as `root_table`; intermediate tables
    // we created are zero-initialized and exclusively owned by the walk.
    unsafe { Some(&mut *(phys_to_virt(frame.start_address()).as_u64() as *mut PageTable)) }
}

/// Descend one level, allocating and linking a fresh table if absent.
fn next_table_or_create(entry: &mut PageTableEntry) -> Result<&'static mut PageTable, &'static str> {
    if entry.flags().contains(PageTableFlags::HUGE_PAGE) {
        return Err("huge page blocks table walk");
    }
    if !entry.flags().contains(PageTableFlags::PRESENT) {
        let frame = crate::physmem::alloc_frame().ok_or("out of frames while mapping")?;
        let table_virt = phys_to_virt(frame.start_address()).as_u64() as *mut PageTable;
        // Soundness: freshly allocated frame is exclusively ours until the
        // entry that publishes it is written below.
        unsafe {
            core::ptr::write_bytes(table_virt as *mut u8, 0, size_of::<PageTable>());
        }
        let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE;
        entry.set_frame(frame, flags);
    }
    // Soundness: entry is PRESENT and non-huge after the branch above.
    Ok(next_table(entry).expect("entry present but unframeable"))
}

/// Map `page` → `frame` with `flags` in the active address space.
/// Refuses to clobber an existing mapping (reports, never panics).
pub fn map(
    page: Page,
    frame: PhysFrame<Size4KiB>,
    flags: PageTableFlags,
) -> Result<(), &'static str> {
    if flags.contains(PageTableFlags::HUGE_PAGE) {
        return Err("4 KiB map called with HUGE_PAGE flag");
    }
    let pml4 = root_table();
    let p3 = next_table_or_create(&mut pml4[page.p4_index()])?;
    let p2 = next_table_or_create(&mut p3[page.p3_index()])?;
    let p1 = next_table_or_create(&mut p2[page.p2_index()])?;

    let entry = &mut p1[page.p1_index()];
    if entry.flags().contains(PageTableFlags::PRESENT) {
        return Err("page already mapped");
    }
    entry.set_frame(frame, flags);
    x86_64::instructions::tlb::flush(page.start_address());
    Ok(())
}

/// Remove the mapping for `page`, returning the frame it pointed at.
pub fn unmap(page: Page) -> Result<PhysFrame<Size4KiB>, &'static str> {
    let pml4 = root_table();
    let Some(p3) = next_table(&pml4[page.p4_index()]) else {
        return Err("no mapping (pml4)");
    };
    let Some(p2) = next_table(&p3[page.p3_index()]) else {
        return Err("no mapping (pdpt)");
    };
    let Some(p1) = next_table(&p2[page.p2_index()]) else {
        return Err("no mapping (pd)");
    };
    let entry = &mut p1[page.p1_index()];
    if !entry.flags().contains(PageTableFlags::PRESENT) {
        return Err("page not mapped (pt)");
    }
    let frame = entry.frame().map_err(|_| "entry had no frame")?;
    entry.set_unused();
    x86_64::instructions::tlb::flush(page.start_address());
    Ok(frame)
}

/// Translate a virtual address to its physical address, if mapped.
/// Follows 1 GiB / 2 MiB huge entries; returns `None` when not present.
pub fn translate(virt: VirtAddr) -> Option<PhysAddr> {
    let pml4 = root_table();
    let p3 = next_table(&pml4[virt.p4_index()])?;
    let p3e = &p3[virt.p3_index()];
    if !p3e.flags().contains(PageTableFlags::PRESENT) {
        return None;
    }
    if p3e.flags().contains(PageTableFlags::HUGE_PAGE) {
        // 1 GiB page: phys = entry frame base + offset within GiB.
        let frame = p3e.frame().ok()?;
        let off = virt.as_u64() & (Size1GiB::SIZE - 1);
        return Some(PhysAddr::new(frame.start_address().as_u64() + off));
    }
    let p2 = next_table(p3e)?;
    let p2e = &p2[virt.p2_index()];
    if !p2e.flags().contains(PageTableFlags::PRESENT) {
        return None;
    }
    if p2e.flags().contains(PageTableFlags::HUGE_PAGE) {
        // 2 MiB page.
        let frame = p2e.frame().ok()?;
        let off = virt.as_u64() & (Size2MiB::SIZE - 1);
        return Some(PhysAddr::new(frame.start_address().as_u64() + off));
    }
    let p1 = next_table(p2e)?;
    let p1e = &p1[virt.p1_index()];
    if !p1e.flags().contains(PageTableFlags::PRESENT) {
        return None;
    }
    let frame = p1e.frame().ok()?;
    Some(PhysAddr::new(
        frame.start_address().as_u64() + u64::from(virt.page_offset()),
    ))
}
