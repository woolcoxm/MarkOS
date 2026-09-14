//! Freestanding C memory routines (`memset` etc.).
//!
//! LLVM is free to emit calls to these from ordinary Rust code (e.g. zeroing
//! a large array), so a `no_std` kernel must provide them. They use the exact
//! libc signatures; bodies are implemented with `rep` string instructions
//! rather than loops so LLVM's loop-idiom recognition cannot turn the bodies
//! back into calls to themselves (infinite recursion).
//!
//! invariants: EFLAGS.DF must be clear (true at boot, and we never `std`) —
//! this is the ABI contract the `rep` instructions rely on.

use core::ffi::c_void;

/// Fill `count` bytes at `dest` with the low byte of `val`. Returns `dest`.
#[unsafe(no_mangle)]
unsafe extern "C" fn memset(dest: *mut c_void, val: i32, count: usize) -> *mut c_void {
    // Soundness: `rep stosb` writes exactly `count` bytes starting at `dest`;
    // the caller guarantees `dest` is valid for `count` writes (C contract).
    unsafe {
        core::arch::asm!(
            "rep stosb",
            in("rdi") dest as *mut u8,
            in("al") val as u8,
            in("rcx") count,
            lateout("rdi") _,
            lateout("rcx") _,
        );
    }
    dest
}

/// Copy `count` bytes from `src` to `dest` (must not overlap). Returns `dest`.
#[unsafe(no_mangle)]
unsafe extern "C" fn memcpy(dest: *mut c_void, src: *const c_void, count: usize) -> *mut c_void {
    // Soundness: `rep movsb` copies exactly `count` bytes; non-overlap is the
    // caller's obligation (C contract) — overlapping copies go to `memmove`.
    unsafe {
        core::arch::asm!(
            "rep movsb",
            in("rdi") dest as *mut u8,
            in("rsi") src as *const u8,
            in("rcx") count,
            lateout("rdi") _,
            lateout("rsi") _,
            lateout("rcx") _,
        );
    }
    dest
}

/// Copy `count` bytes from `src` to `dest`, overlap-safe. Returns `dest`.
#[unsafe(no_mangle)]
unsafe extern "C" fn memmove(dest: *mut c_void, src: *const c_void, count: usize) -> *mut c_void {
    let dest = dest as *mut u8;
    let src = src as *const u8;
    if (dest as usize) < (src as usize) || (dest as usize) >= (src as usize + count) {
        // Forward copy is safe (or ranges don't overlap at all).
        // Soundness: both ranges valid for `count` bytes per C contract.
        unsafe {
            memcpy(dest as _, src as _, count);
        }
    } else {
        // `dest` lies inside the source range: copy back to front.
        // Soundness: volatile byte moves stay within [src, src+count) which
        // the caller guarantees valid; volatiles also block LLVM's loop-idiom
        // recognition from rewriting this into a `memmove` call (recursion).
        unsafe {
            for i in 0..count {
                let s = src.add(count - 1 - i).read_volatile();
                dest.add(count - 1 - i).write_volatile(s);
            }
        }
    }
    dest as *mut c_void
}

/// Compare `count` bytes; returns negative/zero/positive like C `memcmp`.
#[unsafe(no_mangle)]
unsafe extern "C" fn memcmp(a: *const c_void, b: *const c_void, count: usize) -> i32 {
    let (a, b) = (a as *const u8, b as *const u8);
    // Soundness: reads exactly `count` bytes from each pointer; the caller
    // guarantees both ranges are valid (C contract). Volatiles keep LLVM from
    // idioming the loop into a `memcmp` call (recursion).
    unsafe {
        for i in 0..count {
            let x = a.add(i).read_volatile();
            let y = b.add(i).read_volatile();
            if x != y {
                return x as i32 - y as i32;
            }
        }
    }
    0
}
