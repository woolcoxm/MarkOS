//! Authentication: session-cookie login for the web UI (argon2id), roles,
//! and optional bearer API key for the inference API. No default credentials
//! exist — the store must be provisioned by the installer or first-boot
//! provisioner before login succeeds.

use crate::config::{Role, User, UserDb};
use argon2::password_hash::{PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

pub fn hash_password(pw: &str) -> Result<String, String> {
    let mut seed = [0u8; 16];
    random_bytes(&mut seed);
    let salt = SaltString::encode_b64(&seed).map_err(|e| format!("salt: {e}"))?;
    Argon2::default()
        .hash_password(pw.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| format!("hash: {e}"))
}

pub fn verify_password(pw: &str, phc: &str) -> bool {
    match PasswordHash::new(phc) {
        Ok(parsed) => Argon2::default()
            .verify_password(pw.as_bytes(), &parsed)
            .is_ok(),
        Err(_) => false,
    }
}

pub fn sha256_hex(data: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(data);
    hex::encode(h.finalize())
}

#[derive(Debug, Clone)]
pub struct Session {
    pub username: String,
    pub role: Role,
    pub created: Instant,
}

pub const SESSION_TTL: Duration = Duration::from_secs(12 * 3600);

pub struct Sessions {
    map: Mutex<HashMap<String, Session>>,
}

impl Sessions {
    pub fn new() -> Self {
        Sessions { map: Mutex::new(HashMap::new()) }
    }

    /// Create a session; returns the token. Tokens come from the OS CSPRNG
    /// (getrandom); failure is effectively impossible on the appliance.
    pub fn create(&self, username: &str, role: Role) -> String {
        let mut seed = [0u8; 32];
        random_bytes(&mut seed);
        let tok = sha256_hex(&seed);
        self.map.lock().unwrap().insert(
            tok.clone(),
            Session { username: username.into(), role, created: Instant::now() },
        );
        tok
    }

    pub fn get(&self, token: &str) -> Option<Session> {
        let mut map = self.map.lock().unwrap();
        let s = map.get(token)?.clone();
        if s.created.elapsed() > SESSION_TTL {
            map.remove(token);
            return None;
        }
        Some(s)
    }

    pub fn remove(&self, token: &str) {
        self.map.lock().unwrap().remove(token);
    }
}

impl Default for Sessions {
    fn default() -> Self {
        Self::new()
    }
}

fn random_bytes(buf: &mut [u8]) {
    getrandom::getrandom(buf).expect("OS CSPRNG unavailable");
}

/// Check a bearer token against the configured API key hash.
pub fn api_key_ok(server: &crate::config::ServerConfig, bearer: Option<&str>) -> bool {
    match (&server.api_key_required, &server.api_key_hash) {
        (false, _) => true,
        (true, None) => false, // required but unset: deny everything
        (true, Some(hash)) => bearer
            .map(|k| {
                let v = sha256_hex(k.trim().as_bytes());
                constant_time_eq(v.as_bytes(), hash.as_bytes())
            })
            .unwrap_or(false),
    }
}

fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Ensure a user exists with the given password (used by provisioning).
pub fn ensure_user(db: &mut UserDb, username: &str, password: &str, role: Role) -> Result<(), String> {
    if username.is_empty() || password.len() < 8 {
        return Err("username must be non-empty and password >= 8 chars".into());
    }
    let hash = hash_password(password)?;
    if let Some(u) = db.users.iter_mut().find(|u| u.username == username) {
        u.pwhash = hash;
        u.role = role;
    } else {
        db.users.push(User { username: username.into(), pwhash: hash, role });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn password_hash_verify() {
        let h = hash_password("hunter2hunter2").unwrap();
        assert!(h.starts_with("$argon2id$"));
        assert!(verify_password("hunter2hunter2", &h));
        assert!(!verify_password("wrong", &h));
    }

    #[test]
    fn sessions_roundtrip() {
        let s = Sessions::new();
        let t = s.create("admin", Role::Admin);
        assert_eq!(s.get(&t).unwrap().username, "admin");
        s.remove(&t);
        assert!(s.get(&t).is_none());
    }

    #[test]
    fn api_key_gate() {
        use crate::config::ServerConfig;
        let mut sc = ServerConfig::default();
        assert!(api_key_ok(&sc, None));
        sc.api_key_required = true;
        assert!(!api_key_ok(&sc, None));
        sc.api_key_hash = Some(sha256_hex(b"sk-test"));
        assert!(api_key_ok(&sc, Some("sk-test")));
        assert!(!api_key_ok(&sc, Some("sk-bad")));
    }
}
