fn main() {
    // The kernel is linked with our own linker script (higher-half layout,
    // Limine requests in a dedicated PHDR). Absolute path so the -T argument
    // is valid regardless of the directory rustc is invoked from.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    println!("cargo:rerun-if-changed=linker.ld");
    println!("cargo:rustc-link-arg=-T{manifest_dir}/linker.ld");
}
