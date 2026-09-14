//! MARKOS control protocol (v0): newline-terminated ASCII commands over the
//! TCP transport (port 8080). One request line -> one response line. This is
//! the appliance's remote-control surface; install-time configuration (Pi-6)
//! is the only other place the appliance accepts non-LLM input.
//!
//! Commands:
//!   HELLO | MARKOS HELLO -> banner + hardware capabilities
//!   STATUS               -> load state, uptime, commands served
//!   LOAD                 -> mount SD (FAT32), read MODEL.BIN, validate GGUF
//!   RUN                  -> UDOT matmul over mm.a x mm.b across all cores
//!   PING / ECHO <text>   -> liveness / debug
//!
//! owns: the loaded model image and load/run state.
//! invariants: BSP-only (runs inside the TCP input path); no heap —
//! responses are formatted into a caller-provided fixed buffer. Commands
//! must arrive in one TCP segment (Pi-8 hardening: reassembly).

use crate::{board, cache, config, cpu, fat, gguf, matmul, pool, timer, uart, virtio_blk};
use core::fmt;

const MODEL_CAP: usize = 65536;

static mut MODEL_BUF: [u8; MODEL_CAP] = [0u8; MODEL_CAP];
static mut MODEL_BYTES: usize = 0;
static mut MODEL_TENSORS: u64 = 0;
static mut DATA_START: u64 = 0;
static mut BLK_UP: bool = false;
static mut SERVED: u64 = 0;
static mut AUTHED: bool = false;

/// Boot-time configuration (Pi-6): read MARKOS.CFG from the SD root. Must
/// run before net::init — the appliance IP/port/token take effect from it.
pub fn boot() {
    match boot_config() {
        Ok(()) => {
            let mut buf = [0u8; 96];
            let n = config::describe(&mut buf);
            let s = core::str::from_utf8(&buf[..n]).unwrap_or("cfg: ?\n");
            uart::write_str(s);
        }
        Err(e) => {
            uart::locked_write(format_args!("cfg: defaults ({e})\n"));
        }
    }
}

fn boot_config() -> Result<(), &'static str> {
    ensure_blk()?;
    config::load_from_sd()
}

/// One-time virtio-blk bring-up; the device stays configured afterwards.
fn ensure_blk() -> Result<(), &'static str> {
    if !unsafe { BLK_UP } {
        virtio_blk::init()?;
        unsafe { BLK_UP = true };
    }
    Ok(())
}

/// Format adapter: append into a fixed byte buffer, truncating at capacity.
struct BufW<'a> {
    buf: &'a mut [u8],
    len: usize,
}

impl<'a> fmt::Write for BufW<'a> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        let bytes = s.as_bytes();
        let room = self.buf.len() - self.len;
        let n = bytes.len().min(room);
        self.buf[self.len..self.len + n].copy_from_slice(&bytes[..n]);
        self.len += n;
        Ok(())
    }
}

/// Handle one control payload; write the response and return its length.
/// Returns 0 for payloads that are not control commands (the caller then
/// applies its transport-level fallback).
pub fn dispatch(payload: &[u8], out: &mut [u8]) -> usize {
    // Trim trailing line terminators.
    let mut line = payload;
    while let Some((&b, rest)) = line.split_last() {
        if b == b'\n' || b == b'\r' {
            line = rest;
        } else {
            break;
        }
    }
    let (verb, arg) = match line.iter().position(|&b| b == b' ') {
        Some(i) => (&line[..i], &line[i + 1..]),
        None => (line, &[] as &[u8]),
    };
    unsafe { SERVED += 1 };
    let authed = unsafe { AUTHED };

    let mut w = BufW { buf: out, len: 0 };
    match verb {
        b"HELLO" => {
            if token_ok(arg) {
                unsafe { AUTHED = true };
                hello(&mut w);
            } else {
                let _ = fmt::write(&mut w, format_args!("ERR auth"));
            }
        }
        // Every command except HELLO requires a prior authenticated HELLO.
        _ if !authed => {
            let _ = fmt::write(&mut w, format_args!("ERR auth"));
        }
        b"STATUS" => status(&mut w),
        b"LOAD" => load(&mut w),
        b"RUN" => run(&mut w),
        b"PING" => {
            let _ = fmt::write(&mut w, format_args!("PONG"));
        }
        b"ECHO" => {
            let text = core::str::from_utf8(arg).unwrap_or("\u{FFFD}");
            let _ = fmt::write(&mut w, format_args!("ECHO {text}"));
        }
        _ => return 0,
    }
    w.len
}

/// HELLO token check: the installer bakes the admin token into MARKOS.CFG;
/// an unset token accepts any HELLO (development convenience only).
fn token_ok(arg: &[u8]) -> bool {
    let tok = config::token();
    tok.is_empty() || arg == tok
}

fn hello(w: &mut BufW) {
    let _ = fmt::write(
        w,
        format_args!(
            "MARKOS/1 READY cores={} cpu={} dotprod={}",
            board::CORE_COUNT,
            cpu::core_name(),
            cpu::has_dotprod() as u8
        ),
    );
}

fn status(w: &mut BufW) {
    let (bytes, tensors, served) = unsafe { (MODEL_BYTES, MODEL_TENSORS, SERVED) };
    let state = if bytes == 0 { "idle" } else { "loaded" };
    let _ = fmt::write(
        w,
        format_args!(
            "OK state={state} bytes={bytes} tensors={tensors} uptime_ms={} served={served}",
            timer::uptime_ms()
        ),
    );
}

fn load(w: &mut BufW) {
    if let Err(e) = ensure_blk() {
        let _ = fmt::write(w, format_args!("ERR blk {e}"));
        return;
    }
    let vol = match fat::mount() {
        Ok(v) => v,
        Err(e) => {
            let _ = fmt::write(w, format_args!("ERR fat {e}"));
            return;
        }
    };
    let file = match vol.open_model() {
        Ok(f) => f,
        Err(e) => {
            let _ = fmt::write(w, format_args!("ERR model {e}"));
            return;
        }
    };
    let n = {
        // Soundness: MODEL_BUF is control-plane scratch on the BSP; the
        // device DMAs into it while the poll loop is the only runner.
        let buf = unsafe {
            core::slice::from_raw_parts_mut((&raw mut MODEL_BUF) as *mut u8, MODEL_CAP)
        };
        match vol.read_file(&file, buf) {
            Ok(n) => n,
            Err(e) => {
                let _ = fmt::write(w, format_args!("ERR read {e}"));
                return;
            }
        }
    };
    let model =
        unsafe { core::slice::from_raw_parts((&raw const MODEL_BUF) as *const u8, n) };
    let info = match gguf::parse_and_dump(model) {
        Ok(i) => i,
        Err(e) => {
            unsafe { MODEL_BYTES = 0 };
            let _ = fmt::write(w, format_args!("ERR gguf {e}"));
            return;
        }
    };
    unsafe {
        MODEL_BYTES = n;
        MODEL_TENSORS = info.tensor_count;
        DATA_START = info.data_start;
    }
    let _ = fmt::write(
        w,
        format_args!(
            "OK loaded bytes={} gguf=v{} tensors={}",
            n, info.version, info.tensor_count
        ),
    );
}

fn run(w: &mut BufW) {
    if !cpu::has_dotprod() {
        let _ = fmt::write(w, format_args!("ERR dotprod unavailable"));
        return;
    }
    let (n, data_start) = unsafe { (MODEL_BYTES, DATA_START) };
    if n == 0 {
        let _ = fmt::write(w, format_args!("ERR no model"));
        return;
    }
    let Some((_, _, _, a_off)) = gguf::find_tensor(b"mm.a") else {
        let _ = fmt::write(w, format_args!("ERR mm.a missing"));
        return;
    };
    let Some((_, _, _, b_off)) = gguf::find_tensor(b"mm.b") else {
        let _ = fmt::write(w, format_args!("ERR mm.b missing"));
        return;
    };
    let a_len = matmul::M * matmul::K;
    let b_len = matmul::N * matmul::K;

    let model =
        unsafe { core::slice::from_raw_parts((&raw const MODEL_BUF) as *const u8, n) };
    // Soundness: both tensors lie fully inside the loaded file (GGUF offsets
    // were bounds-validated by the parser).
    unsafe {
        core::ptr::copy_nonoverlapping(
            model.as_ptr().add(data_start as usize + a_off as usize),
            &raw mut matmul::A_BUF as *mut u8,
            a_len,
        );
        core::ptr::copy_nonoverlapping(
            model.as_ptr().add(data_start as usize + b_off as usize),
            &raw mut matmul::B_BUF as *mut u8,
            b_len,
        );
    }
    // The APs run with their MMU off (uncached): clean inputs, then
    // invalidate the stale cached C lines after they finish.
    cache::clean_range(
        (&raw const matmul::A_BUF) as usize,
        core::mem::size_of::<[u8; matmul::M * matmul::K]>(),
    );
    cache::clean_range(
        (&raw const matmul::B_BUF) as usize,
        core::mem::size_of::<[u8; matmul::N * matmul::K]>(),
    );
    pool::run_on_all(matmul::pool_matmul_job, 0, board::CORE_COUNT);
    cache::invalidate_range(
        (&raw const matmul::C_BUF) as usize,
        core::mem::size_of::<[i32; matmul::M * matmul::N]>(),
    );

    // Scalar reference on the stack (M*N = 256 i32s), compare exactly.
    let a = unsafe {
        core::slice::from_raw_parts((&raw const matmul::A_BUF) as *const u8, a_len)
    };
    let b = unsafe {
        core::slice::from_raw_parts((&raw const matmul::B_BUF) as *const u8, b_len)
    };
    let mut c_ref = [0i32; matmul::M * matmul::N];
    matmul::matmul_scalar(a, b, &mut c_ref);
    let c = unsafe {
        core::slice::from_raw_parts((&raw const matmul::C_BUF) as *const i32, c_ref.len())
    };
    let mismatches = c_ref.iter().zip(c.iter()).filter(|(r, g)| r != g).count();
    let sum: i64 = c.iter().fold(0i64, |acc, &v| acc + v as i64);
    let _ = fmt::write(
        w,
        format_args!(
            "OK run c00={} clast={} sum={} exact={}",
            c[0],
            c[c.len() - 1],
            sum,
            (mismatches == 0) as u8
        ),
    );
}
