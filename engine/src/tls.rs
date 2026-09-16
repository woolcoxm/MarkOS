//! TLS for the web UI / API (feature `tls`). Self-signed certificate is
//! generated on first boot for LAN use; operator can replace cert/key via
//! the server settings.

#![cfg(feature = "tls")]

use crate::state::EngineCtx;
use rustls::pki_types::{CertificateDer, PrivateKeyDer};
use std::sync::Arc;

pub struct RustlsWrap(pub Arc<rustls::ServerConfig>);

impl crate::http::StreamWrap for RustlsWrap {
    fn wrap(&self, sock: std::net::TcpStream) -> std::io::Result<crate::http::BoxedStream> {
        let conn = rustls::ServerConnection::new(self.0.clone())
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;
        Ok(Box::new(rustls::StreamOwned::new(conn, sock)))
    }
}

fn ensure_self_signed(ctx: &EngineCtx) -> Result<(Vec<u8>, Vec<u8>), String> {
    let dir = ctx.data_dir.join("tls");
    let cert_path = dir.join("cert.pem");
    let key_path = dir.join("key.pem");
    if let (Ok(c), Ok(k)) = (std::fs::read(&cert_path), std::fs::read(&key_path)) {
        return Ok((c, k));
    }
    std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
    let host = {
        let s = ctx.store.read().unwrap();
        s.engine.server.mdns_name.clone()
    };
    // SANs: browsers are reached via mDNS name, hostname, or static IP —
    // cover all of them so only the self-signed warning ever shows.
    let mut sans = vec![host.clone(), "localhost".to_string()];
    if let Ok(h) = std::fs::read_to_string(ctx.data_dir.join("hostname")) {
        let h = h.trim().to_string();
        if !h.is_empty() && !sans.contains(&h) {
            sans.push(h);
        }
    }
    if let Ok(net) = std::fs::read_to_string(ctx.data_dir.join("net.conf")) {
        for line in net.lines() {
            if let Some(ip) = line.strip_prefix("IP=") {
                if let Ok(ip) = ip.trim().parse::<std::net::IpAddr>() {
                    sans.push(ip.to_string());
                }
            }
        }
    }
    sans.push("127.0.0.1".to_string());
    let dns_names: Vec<_> = sans.iter().filter(|s| s.parse::<std::net::IpAddr>().is_err()).cloned().collect();
    let ips: Vec<_> = sans.iter().filter_map(|s| s.parse::<std::net::IpAddr>().ok()).collect();
    let mut params = rcgen::CertificateParams::new(dns_names)
        .map_err(|e| e.to_string())?;
    for ip in ips {
        params.subject_alt_names.push(rcgen::SanType::IpAddress(ip));
    }
    params.distinguished_name.push(rcgen::DnType::CommonName, host.clone());
    let key_pair = rcgen::KeyPair::generate().map_err(|e| e.to_string())?;
    let cert = params.self_signed(&key_pair).map_err(|e| e.to_string())?;
    let cert_pem = cert.pem().into_bytes();
    let key_pem = key_pair.serialize_pem().into_bytes();
    std::fs::write(&cert_path, &cert_pem).map_err(|e| e.to_string())?;
    std::fs::write(&key_path, &key_pem).map_err(|e| e.to_string())?;
    Ok((cert_pem, key_pem))
}

/// Build the TLS wrapper when enabled in config, else None.
pub fn wrap_if_enabled(ctx: &EngineCtx) -> Result<Option<Arc<RustlsWrap>>, String> {
    if !ctx.engine_config().server.tls_enabled {
        return Ok(None);
    }
    let (cert_pem, key_pem) = ensure_self_signed(ctx)?;
    let certs: Vec<CertificateDer> = rustls_pemfile::certs(&mut cert_pem.as_slice())
        .collect::<Result<_, _>>()
        .map_err(|e| format!("bad cert pem: {e}"))?;
    let key: PrivateKeyDer = rustls_pemfile::private_key(&mut key_pem.as_slice())
        .map_err(|e| format!("bad key pem: {e}"))?
        .ok_or("no private key found")?;
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = rustls::ServerConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| e.to_string())?
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| format!("tls config: {e}"))?;
    Ok(Some(Arc::new(RustlsWrap(Arc::new(config)))))
}
