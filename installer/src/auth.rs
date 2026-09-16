//! argon2id hashing for the provision file (same scheme as the engine).

use argon2::password_hash::{PasswordHasher, SaltString};
use argon2::Argon2;

pub fn auth_hash(password: &str) -> Result<String, String> {
    let mut seed = [0u8; 16];
    getrandom::getrandom(&mut seed).map_err(|e| format!("rng: {e}"))?;
    let salt = SaltString::encode_b64(&seed).map_err(|e| format!("salt: {e}"))?;
    Argon2::default()
        .hash_password(password.as_bytes(), &salt)
        .map(|h| h.to_string())
        .map_err(|e| format!("hash: {e}"))
}
