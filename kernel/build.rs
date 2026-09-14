fn main() {
    // The kernel links with its own script (flat physical layout at the
    // Pi's 64-bit kernel load address). Absolute path so -T is valid
    // regardless of the directory rustc runs from.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rerun-if-changed=aarch64.ld");
    println!("cargo:rustc-link-arg=-T{manifest_dir}/aarch64.ld");
}
