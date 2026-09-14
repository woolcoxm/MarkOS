fn main() {
    // The kernel links with its own script (flat physical layout at the
    // board's kernel load address). Absolute path so the -T argument is
    // valid regardless of the directory rustc runs from.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let script = if std::env::var("CARGO_FEATURE_BOARD_VIRT").is_ok() {
        "aarch64-virt.ld"
    } else {
        "aarch64.ld"
    };
    println!("cargo:rerun-if-changed={script}");
    println!("cargo:rustc-link-arg=-T{manifest_dir}/{script}");
}
