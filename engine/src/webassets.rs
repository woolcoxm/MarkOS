//! Embedded web UI assets — hand-written vanilla HTML/JS/CSS, no framework,
//! no CDN, compiled into the engine binary. The appliance works airgapped.

pub fn get(name: &str) -> Option<(&'static str, &'static [u8])> {
    match name {
        "index.html" => Some(("text/html; charset=utf-8", include_bytes!("../web/index.html"))),
        "app.js" => Some(("application/javascript; charset=utf-8", include_bytes!("../web/app.js"))),
        "style.css" => Some(("text/css; charset=utf-8", include_bytes!("../web/style.css"))),
        "favicon.svg" => Some(("image/svg+xml", include_bytes!("../web/favicon.svg"))),
        _ => None,
    }
}
