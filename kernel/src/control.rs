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

use crate::{board, cache, config, cpu, engine, fat, gguf, matmul, pool, tcp, timer, uart, virtio_blk};
use core::fmt;

const MODEL_CAP: usize = 65536;
/// Metadata window for real GB-scale GGUFs (Qwen3-0.6B needs ~6 MB).
const META_CAP: usize = 8 * 1024 * 1024;

static mut MODEL_BUF: [u8; MODEL_CAP] = [0u8; MODEL_CAP];
/// Metadata section of a meta-mode (big) model: kv pairs + tensor table +
/// tokenizer arrays. Weights stay on the volume and stream per-matvec.
static mut META_BUF: [u8; META_CAP] = [0u8; META_CAP];
static mut META_LEN: usize = 0;
static mut META_OK: bool = false;
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
        b"STATS" => stats(&mut w),
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

/// Observability (brief Phase 10): live health counters for a monitoring
/// client — uptime, command count, model state, timer tick rate, cores.
fn stats(w: &mut BufW) {
    let (bytes, served) = unsafe { (MODEL_BYTES, SERVED) };
    let _ = fmt::write(
        w,
        format_args!(
            "OK stats uptime_ms={} tick_hz={} served={served} model_bytes={bytes} cores={}",
            timer::uptime_ms(),
            timer::frequency(),
            board::CORE_COUNT
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
    // Two load modes: files that fit MODEL_BUF are read fully (the
    // synthetic mm.a/mm.b matmul path); real GB-scale GGUFs load
    // metadata-only — the engine streams their weights from the volume
    // during GEN.
    if file.size as usize > MODEL_CAP {
        let n = {
            // Soundness: META_BUF is control-plane scratch on the BSP.
            let buf = unsafe {
                core::slice::from_raw_parts_mut((&raw mut META_BUF) as *mut u8, META_CAP)
            };
            match vol.read_at(&file, 0, buf) {
                Ok(n) => n,
                Err(e) => {
                    let _ = fmt::write(w, format_args!("ERR read {e}"));
                    return;
                }
            }
        };
        let meta = unsafe {
            core::slice::from_raw_parts((&raw const META_BUF) as *const u8, n)
        };
        let info = match gguf::parse_and_dump(meta) {
            Ok(i) => i,
            Err(e) => {
                let _ = fmt::write(w, format_args!("ERR gguf {e}"));
                return;
            }
        };
        unsafe {
            META_LEN = n;
            META_OK = true;
            MODEL_BYTES = file.size as u64 as usize;
            MODEL_TENSORS = info.tensor_count;
            DATA_START = info.data_start;
        }
        let _ = fmt::write(
            w,
            format_args!(
                "OK loaded bytes={} gguf=v{} tensors={} meta=1",
                file.size, info.version, info.tensor_count
            ),
        );
        return;
    }
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

// ===== GEN: streaming generation over the control transport (Phase 9) =====

/// GEN emits replies while it decodes (one TOK line per generated token),
/// so tcp.rs routes it here instead of the single-response dispatch.
/// Requires an authenticated connection.
pub fn is_stream_command(payload: &[u8]) -> bool {
    let authed = unsafe { AUTHED };
    authed
        && (payload == b"GEN" || (payload.len() > 4 && &payload[..4] == b"GEN "))
}

fn decimal(bytes: &[u8]) -> Option<u32> {
    if bytes.is_empty() || bytes.len() > 2 {
        return None;
    }
    bytes.iter().try_fold(0u32, |acc, &b| {
        if b.is_ascii_digit() {
            Some(acc * 10 + (b - b'0') as u32)
        } else {
            None
        }
    })
}

/// GEN <steps> <prompt>: tokenize, prefill through every layer, then
/// stream "TOK g=<g> id=<id> text=<bytes>" per greedy step and a final
/// "GEN_END ids=.. n=..". Text is included only when the decoded token is
/// printable ASCII (newlines etc. would break the line protocol).
/// Runs on the BSP; the NIC is not polled for the duration — the client
/// must read continuously (single connection, documented limitation).
pub fn gen_stream(payload: &[u8]) {
    if !unsafe { META_OK } {
        tcp::stream(b"ERR no meta model (LOAD a large GGUF first)\n");
        return;
    }
    // The dispatch path trims line terminators; the stream path must too,
    // or the prompt gains a newline token and generation shifts a step.
    let payload = {
        let mut p = payload;
        while let Some((&b, rest)) = p.split_last() {
            if b == b'\n' || b == b'\r' {
                p = rest;
            } else {
                break;
            }
        }
        p
    };
    let sp0 = match payload.iter().position(|&b| b == b' ') {
        Some(i) => i,
        None => {
            tcp::stream(b"ERR usage: GEN <steps 1-8> <prompt>\n");
            return;
        }
    };
    let rest = &payload[sp0 + 1..];
    let sp1 = match rest.iter().position(|&b| b == b' ') {
        Some(i) => i,
        None => {
            tcp::stream(b"ERR usage: GEN <steps 1-8> <prompt>\n");
            return;
        }
    };
    let steps = match decimal(&rest[..sp1]) {
        Some(s) if (1..=8).contains(&s) => s as usize,
        _ => {
            tcp::stream(b"ERR steps must be 1-8\n");
            return;
        }
    };
    let prompt = &rest[sp1 + 1..];
    if prompt.is_empty() {
        tcp::stream(b"ERR usage: GEN <steps 1-8> <prompt>\n");
        return;
    }

    let meta_len = unsafe { META_LEN };
    let meta = unsafe {
        core::slice::from_raw_parts((&raw const META_BUF) as *const u8, meta_len)
    };
    let ds = unsafe { DATA_START };

    let geo = match engine::geometry() {
        Ok(g) => g,
        Err(_) => {
            tcp::stream(b"ERR geometry\n");
            return;
        }
    };
    let vol = match fat::mount() {
        Ok(v) => v,
        Err(_) => {
            tcp::stream(b"ERR fat\n");
            return;
        }
    };
    let file = match vol.open_model() {
        Ok(f) => f,
        Err(_) => {
            tcp::stream(b"ERR model\n");
            return;
        }
    };
    let Some((_, emb_dims, _, emb_off)) = gguf::find_tensor(b"token_embd.weight") else {
        tcp::stream(b"ERR token_embd missing\n");
        return;
    };
    let Some((_, _, _, fin_off)) = gguf::find_tensor(b"output_norm.weight") else {
        tcp::stream(b"ERR output_norm missing\n");
        return;
    };
    let row_elems = emb_dims[0] as usize;
    let vocab = emb_dims[1] as u64;
    let emb_abs = ds + emb_off;
    let fin_abs = ds + fin_off;

    let mut ids = [0u32; engine::MAX_TOKENS];
    let n_tok = match engine::tokenize(meta, prompt, &mut ids) {
        Ok(n) if n > 0 => n,
        _ => {
            tcp::stream(b"ERR tokenize\n");
            return;
        }
    };
    if n_tok + steps > engine::MAX_POS {
        tcp::stream(b"ERR prompt+steps exceed position cache\n");
        return;
    }

    let act = engine::activations_mut();

    // Prefill: prompt positions 0..n_tok through all layers.
    for pos in 0..n_tok {
        let row = emb_abs + (ids[pos] as u64) * (row_elems / 32 * 34) as u64;
        if engine::dequant_q8_0_row(&vol, &file, row, row_elems, &mut act.x[..geo.n_embd])
            .is_err()
        {
            tcp::stream(b"ERR embed\n");
            return;
        }
        for l in 0..geo.n_layers as usize {
            if engine::layer_forward(&vol, &file, ds, l, &geo, pos, act).is_err() {
                tcp::stream(b"ERR layer\n");
                return;
            }
        }
    }

    // Greedy steps: norm -> argmax -> stream -> feed back.
    let mut gen_ids = [0u32; 8];
    for g in 0..steps {
        if engine::rmsnorm_with_weight(
            &vol, &file, fin_abs, &act.x[..geo.n_embd], &mut act.n1[..geo.n_embd], geo.eps,
        )
        .is_err()
        {
            tcp::stream(b"ERR final norm\n");
            return;
        }
        let Ok((tok, logit)) =
            engine::argmax_q8_0(&vol, &file, emb_abs, row_elems, vocab, &act.n1[..geo.n_embd])
        else {
            tcp::stream(b"ERR lm_head\n");
            return;
        };
        gen_ids[g] = tok;
        let mut line = [0u8; 96];
        let mut w = BufW { buf: &mut line, len: 0 };
        let _ = fmt::write(
            &mut w,
            format_args!("TOK g={g} id={tok} logit={logit:.6e}"),
        );
        // Detokenized text when fully printable ASCII.
        let mut txt = [0u8; 32];
        if let Some(n) = engine::detok(meta, tok, &mut txt) {
            if txt[..n].iter().all(|&b| (0x20..0x7F).contains(&b)) {
                let _ = fmt::write(&mut w, format_args!(" text="));
                w.buf[w.len..w.len + n].copy_from_slice(&txt[..n]);
                w.len += n;
            }
        }
        let _ = fmt::write(&mut w, format_args!("\n"));
        let l = w.len;
        tcp::stream(&line[..l]);
        uart::locked_write(format_args!("gen: step {g} tok {tok}\n"));

        if g + 1 < steps {
            let row = emb_abs + (tok as u64) * (row_elems / 32 * 34) as u64;
            if engine::dequant_q8_0_row(&vol, &file, row, row_elems, &mut act.x[..geo.n_embd])
                .is_err()
            {
                tcp::stream(b"ERR embed\n");
                return;
            }
            for l in 0..geo.n_layers as usize {
                if engine::layer_forward(&vol, &file, ds, l, &geo, n_tok + g, act).is_err() {
                    tcp::stream(b"ERR layer\n");
                    return;
                }
            }
        }
    }

    let mut end = [0u8; 96];
    let mut w = BufW { buf: &mut end, len: 0 };
    let _ = fmt::write(&mut w, format_args!("GEN_END ids="));
    for (i, t) in gen_ids.iter().take(steps).enumerate() {
        if i > 0 {
            let _ = fmt::write(&mut w, format_args!(","));
        }
        let _ = fmt::write(&mut w, format_args!("{t}"));
    }
    let _ = fmt::write(&mut w, format_args!(" n={steps}\n"));
    let l = w.len;
    tcp::stream(&end[..l]);
}
