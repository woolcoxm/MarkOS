//! Build the Axera-GGUF llama.cpp fork (ggml-axcl NPU backend) and the C shim
//! that gives markos-engine a plain-C, no-struct-FFI surface.
//!
//! Source resolution: `MARKOS_LLAMA_CPP_DIR` must point at a checkout of the
//! woolcoxm/llama.cpp Axera fork (branch `axera-any-gguf`, or the PoC branch).
//! Dev builds set it to the local clone; the Buildroot package sets it to the
//! pinned tarball it downloaded into BR2_DL_DIR.
//!
//! AXCL SDK: headers+libs come from the axclhost package (/usr/include/axcl,
//! /usr/lib/axcl on the target; the Buildroot staging dir via
//! MARKOS_AXCL_ROOT when cross-compiling).

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=MARKOS_LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=MARKOS_AXCL_ROOT");
    println!("cargo:rerun-if-env-changed=MARKOS_LLAMA_SKIP_BUILD");

    if !cfg!(feature = "axcl") {
        return; // stub build: host unit tests never link llama
    }
    if env::var_os("MARKOS_LLAMA_SKIP_BUILD").is_some() {
        // link-check / lint runs: emit the same flags without building
        emit_link_flags(PathBuf::from("/nonexistent"));
        return;
    }

    let src = env::var("MARKOS_LLAMA_CPP_DIR").unwrap_or_else(|_| {
        panic!(
            "markos-llama-sys (axcl): MARKOS_LLAMA_CPP_DIR must point at the \
             Axera llama.cpp fork checkout (see docs/axera.md)"
        )
    });
    let src = PathBuf::from(src);
    assert!(
        src.join("ggml/src/ggml-axcl").is_dir(),
        "MARKOS_LLAMA_CPP_DIR={}: not the Axera fork (ggml/src/ggml-axcl missing)",
        src.display()
    );

    let mut cfg = cmake::Config::new(&src);
    cfg.define("BUILD_SHARED_LIBS", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        // the fork's app/ target assumes common/'s include dir
        // unconditionally (arg.h) and the appliance ships no CLI app
        .define("LLAMA_BUILD_APP", "OFF")
        .define("LLAMA_CURL", "OFF")
        .define("GGML_AXCL", "ON")
        // portable code; Pi 5 codegen flags arrive via CFLAGS/CXXFLAGS from
        // the Buildroot toolchain env
        .define("GGML_NATIVE", "OFF")
        .define("CMAKE_BUILD_TYPE", "Release")
        // out-of-source into OUT_DIR so incremental cargo builds reuse it
        .out_dir(env::var("OUT_DIR").unwrap());

    if let Ok(axcl_root) = env::var("MARKOS_AXCL_ROOT") {
        cfg.define("AXCL_INSTALL_DIR", axcl_root);
    }

    let build_dir = cfg.build();

    // the shim: one C file against the fork's public llama.h, compiled with
    // the same CC/CFLAGS environment (Buildroot cross toolchain or host cc)
    let shim = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shim/markos_llama_shim.c");
    println!("cargo:rerun-if-changed={}", shim.display());
    let mut cc = cc::Build::new();
    cc.file(&shim)
        .flag_if_supported("-std=c11")
        .include(src.join("include"))
        .include(src.join("ggml/include"))
        .warnings_into_errors(false)
        .compile("markos_llama_shim");

    emit_link_flags(build_dir);
}

fn emit_link_flags(build_dir: PathBuf) {
    // static archives, dependents before dependencies
    println!("cargo:rustc-link-lib=static=llama");
    println!("cargo:rustc-link-lib=static=ggml-axcl");
    println!("cargo:rustc-link-lib=static=ggml-base");
    println!("cargo:rustc-link-lib=static=ggml-cpu");
    // the cmake install tree (out/lib) is flat; multi-config generators
    // nest the build tree under Release/
    println!("cargo:rustc-link-search=native={}", build_dir.join("lib").display());
    for sub in ["", "Release/"] {
        println!("cargo:rustc-link-search=native={}", build_dir.join("build").join(sub).display());
    }
    // AXCL runtime libs (target /usr/lib/axcl; staging via sysroot when cross)
    if let Some(sysroot) = env::var_os("TARGET_SYSROOT") {
        // Buildroot staging: the axclhost package lays out <sysroot>/usr/axcl/lib
        let p = PathBuf::from(sysroot).join("usr/axcl/lib");
        println!("cargo:rustc-link-search=native={}", p.display());
    } else if cfg!(target_os = "linux") {
        // dev/host linux builds: the deb's runtime layout
        println!("cargo:rustc-link-search=native=/usr/lib/axcl");
        println!("cargo:rustc-link-search=native=/usr/axcl/lib");
    }
    println!("cargo:rustc-link-lib=dylib=axcl_rt");
    println!("cargo:rustc-link-lib=dylib=axcl_npu");
    println!("cargo:rustc-link-lib=dylib=axcl_sys");
    println!("cargo:rustc-link-lib=dylib=axcl_pcie_msg");
    println!("cargo:rustc-link-lib=dylib=axcl_pcie_dma");
}
