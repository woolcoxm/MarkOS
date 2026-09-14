//! Kernel heap: `alloc::Box`/`Vec`/etc. for the whole OS.
//!
//! Layout: a fixed 64 MiB virtual window at `paging::HEAP_BASE`, backed by
//! frames from the physical allocator (each heap page maps an arbitrary
//! frame — contiguity is virtual, never physical). The window never grows:
//! an out-of-heap condition is a hard error by design (the unikernel refuses
//! work it cannot hold rather than degrading).
//!
//! owns: the heap virtual region and every frame mapped into it.
//! invariants: `init` runs once, on the BSP, before any allocation; after
//! `init` the global allocator must never be left returning null for a
//! satisfiable request (exhaustion => null => Rust aborts the allocation).

use core::alloc::{GlobalAlloc, Layout};
use core::fmt::Write as _;
use core::ptr::NonNull;

use linked_list_allocator::Heap;
use spin::Mutex;
use x86_64::structures::paging::page_table::PageTableFlags;
use x86_64::structures::paging::{Page, PageSize, Size4KiB};
use x86_64::VirtAddr;

use crate::paging;
use crate::serial;

/// 64 MiB — generous for kernel metadata; model weights never come from here
/// (they get their own huge-page regions in a later phase).
const HEAP_SIZE: u64 = 64 * 1024 * 1024;
const HEAP_PAGES: u64 = HEAP_SIZE / Size4KiB::SIZE;

struct KernelHeap {
    inner: Mutex<Option<Heap>>,
}

/// Until `init`, allocations fail loudly (null) — nothing allocates before
/// that point by construction.
#[global_allocator]
static HEAP: KernelHeap = KernelHeap {
    inner: Mutex::new(None),
};

impl KernelHeap {
    fn with_heap<R>(&self, f: impl FnOnce(&mut Heap) -> Option<R>) -> Option<R> {
        let mut guard = self.inner.lock();
        let heap = guard.as_mut()?;
        f(heap)
    }
}

unsafe impl GlobalAlloc for KernelHeap {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        self.with_heap(|heap| heap.allocate_first_fit(layout).ok().map(|p| p.as_ptr()))
            .unwrap_or(core::ptr::null_mut())
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let Some(ptr) = NonNull::new(ptr) else {
            return;
        };
        // Soundness: `ptr` came from `alloc` with this `layout` (the
        // `GlobalAlloc` contract); freeing with a matching layout keeps the
        // underlying linked list consistent.
        unsafe {
            self.with_heap(|heap| Some(heap.deallocate(ptr, layout)));
        }
    }
}

/// Map the heap window and hand it to the allocator. Idempotence is not
/// provided: call exactly once.
pub fn init() -> Result<(), &'static str> {
    let flags = PageTableFlags::PRESENT | PageTableFlags::WRITABLE | PageTableFlags::NO_EXECUTE;
    for i in 0..HEAP_PAGES {
        let page = Page::containing_address(VirtAddr::new(paging::HEAP_BASE + i * Size4KiB::SIZE));
        let Some(frame) = crate::physmem::alloc_frame() else {
            return Err("ran out of physical frames while sizing the heap");
        };
        paging::map(page, frame, flags)?;
    }

    let mut guard = HEAP.inner.lock();
    debug_assert!(guard.is_none(), "heap initialized twice");
    let mut heap = Heap::empty();
    // Soundness: the region is fully mapped read-write by the loop above and
    // reserved for the allocator from now on.
    unsafe {
        heap.init(paging::HEAP_BASE as *mut u8, HEAP_SIZE as usize);
    }
    *guard = Some(heap);
    Ok(())
}

/// (used_bytes, free_bytes) for the boot log and later observability.
pub fn stats() -> (usize, usize) {
    HEAP.with_heap(|heap| Some((heap.used(), heap.free())))
        .unwrap_or((0, 0))
}

/// One-line health log for the boot sequence.
pub fn log_stats() {
    let (used, free) = stats();
    let _ = write!(
        serial::Serial,
        "heap: used={}KiB free={}KiB\n",
        used / 1024,
        free / 1024
    );
}
