//! markos-installer: GUI (egui, default) or scriptable CLI. One binary,
//! both frontends; the GUI writes the same config TOML the CLI consumes.

mod auth;
mod cli;
mod config;
mod fat;
mod flash;
mod imagebuild;
mod testimg;

#[cfg(feature = "gui")]
mod gui;

fn main() {
    // No args at all → GUI (the primary Windows experience); any argument
    // selects the CLI for scripting.
    let mut any_args = std::env::args().skip(1).peekable();
    if any_args.peek().is_none() {
        #[cfg(feature = "gui")]
        {
            gui::run().unwrap_or_else(|e| {
                eprintln!("installer GUI failed: {e}");
                std::process::exit(1);
            });
            return;
        }
        #[cfg(not(feature = "gui"))]
        {
            cli::parse_args();
            return;
        }
    }
    let parsed = cli::parse_args();
    if let Err(e) = cli::run(&parsed) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
