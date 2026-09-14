//! Install-time configuration (Pi-6): the installer bakes MARKOS.CFG into
//! the SD image's boot partition; the kernel reads it once at boot. This is
//! the appliance's only non-LLM configuration surface — after boot, control
//! happens exclusively through the network protocol.
//!
//! MARKOS.CFG format (lines, `#` comments):
//!   ip=10.0.2.15      appliance IPv4
//!   port=8080         TCP control port
//!   token=...         admin token required by control HELLO
//!
//! owns: the applied network identity and admin token.
//! invariants: written once during boot on the BSP, read-only afterwards.

use crate::fat;

const DEFAULT_IP: [u8; 4] = [10, 0, 2, 15];
const DEFAULT_PORT: u16 = 8080;
const MAX_TOKEN: usize = 32;

static mut CFG_IP: [u8; 4] = DEFAULT_IP;
static mut CFG_PORT: u16 = DEFAULT_PORT;
static mut CFG_TOKEN: [u8; MAX_TOKEN] = [0; MAX_TOKEN];
static mut CFG_TOKEN_LEN: usize = 0;

pub fn ip() -> [u8; 4] {
    unsafe { CFG_IP }
}

pub fn port() -> u16 {
    unsafe { CFG_PORT }
}

/// The configured admin token (empty slice = no token required).
pub fn token() -> &'static [u8] {
    unsafe { core::slice::from_raw_parts((&raw const CFG_TOKEN) as *const u8, CFG_TOKEN_LEN) }
}

/// Read MARKOS.CFG from the SD root and apply it. A missing or invalid
/// file keeps the defaults; the boot log reports what was applied.
pub fn load_from_sd() -> Result<(), &'static str> {
    let vol = fat::mount()?;
    let file = vol.open_short(b"MARKOS  CFG")?;
    let mut buf = [0u8; 512];
    let n = vol.read_file(&file, &mut buf)?;
    apply(&buf[..n]);
    Ok(())
}

fn apply(data: &[u8]) {
    for raw in data.split(|&b| b == b'\n') {
        let line: &[u8] = {
            let mut l = raw;
            while let Some((&b, rest)) = l.split_first() {
                if b == b' ' || b == b'\t' || b == b'\r' {
                    l = rest;
                } else {
                    break;
                }
            }
            while let Some((&b, rest)) = l.split_last() {
                if b == b' ' || b == b'\t' || b == b'\r' {
                    l = rest;
                } else {
                    break;
                }
            }
            l
        };
        if line.is_empty() || line[0] == b'#' {
            continue;
        }
        if let Some(v) = strip_prefix(line, b"ip=") {
            if let Some(ip) = parse_ipv4(v) {
                unsafe { CFG_IP = ip };
            }
        } else if let Some(v) = strip_prefix(line, b"port=") {
            if let Some(port) = parse_u16(v) {
                unsafe { CFG_PORT = port };
            }
        } else if let Some(v) = strip_prefix(line, b"token=") {
            let n = v.len().min(MAX_TOKEN);
            unsafe {
                CFG_TOKEN[..n].copy_from_slice(&v[..n]);
                CFG_TOKEN_LEN = n;
            }
        }
    }
}

fn strip_prefix<'a>(data: &'a [u8], prefix: &[u8]) -> Option<&'a [u8]> {
    if data.len() >= prefix.len() && &data[..prefix.len()] == prefix {
        Some(&data[prefix.len()..])
    } else {
        None
    }
}

fn parse_ipv4(s: &[u8]) -> Option<[u8; 4]> {
    let mut out = [0u8; 4];
    let mut octets = s.split(|&b| b == b'.');
    for slot in out.iter_mut() {
        let o = octets.next()?;
        let v = parse_u16(o)?;
        if v > 255 {
            return None;
        }
        *slot = v as u8;
    }
    if octets.next().is_some() {
        return None;
    }
    Some(out)
}

fn parse_u16(s: &[u8]) -> Option<u16> {
    if s.is_empty() || s.len() > 5 {
        return None;
    }
    s.iter().try_fold(0u16, |acc, &b| {
        if b.is_ascii_digit() {
            Some(acc * 10 + (b - b'0') as u16)
        } else {
            None
        }
    })
}

/// Boot log line describing the applied configuration.
pub fn describe(buf: &mut [u8]) -> usize {
    let ip = ip();
    let tok = token();
    let text = format_args!(
        "cfg: ip={}.{}.{}.{} port={} token={}\n",
        ip[0], ip[1], ip[2], ip[3],
        port(),
        if tok.is_empty() { "none" } else { "set" }
    );
    let mut len = 0;
    let _ = core::fmt::write(&mut Writer { buf: &mut buf[..], len: &mut len }, text);
    len
}

struct Writer<'a> {
    buf: &'a mut [u8],
    len: &'a mut usize,
}

impl<'a> core::fmt::Write for Writer<'a> {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let room = self.buf.len().saturating_sub(*self.len);
        let n = bytes.len().min(room);
        self.buf[*self.len..*self.len + n].copy_from_slice(&bytes[..n]);
        *self.len += n;
        Ok(())
    }
}
