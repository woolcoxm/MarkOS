#!/bin/bash
# Cross-compile the MarkOS engine for the Pi (aarch64) via the Buildroot
# toolchain, without rebuilding the whole image.
#
# Hard-won notes (2026-09-16):
# - The buildroot local-site package re-extracts from the CANONICAL tree
#   (/root/markos-os/engine) on every -rebuild: syncing only the package
#   build dir is useless.
# - /mnt/c (DrvFs) mtimes make plain rsync skip changed files: use
#   checksum mode (-c) and touch everything after the sync.
# - Appending symbols to .config + olddefconfig can silently drop
#   BR2_PACKAGE_MARKOS_ENGINE (→ the tls feature compiles out and the
#   binary looks mysteriously stale). Regenerate with
#   `make markos_pi5_defconfig` if the sanity grep below ever fails.
# Usage: wsl -- bash -c "tr -d '\r' < this-file > /tmp/x.sh && bash /tmp/x.sh"
set -e
export PATH="/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SRC=/mnt/c/Users/Mark/Desktop/Projects/MarkOS
CANON=/root/markos-os/engine
B=/root/markos-os/os/build-sd/build/markos-engine-0.1.0

rsync -a -c --exclude target "$SRC/engine/" "$CANON/"
cp "$SRC/Cargo.lock" "$CANON/../Cargo.lock"
find "$CANON/src" "$CANON/web" -type f -exec touch {} +

rm -rf "$B/src" "$B/web" "$B"/.stamp_*
cd /root/markos-os
ROOTFS_VARIANT=sd make -C os/buildroot O=/root/markos-os/os/build-sd \
	BR2_EXTERNAL=/root/markos-os/os markos-engine-rebuild 2>&1 | grep -E "Compiling markos|Finished" | head -3

BIN="$B/target/aarch64-unknown-linux-gnu/release/markos-engine"
ls -l --block-size=M "$BIN"
# Sanity: the tls feature must be compiled in (0 hits = the config lost
# BR2_PACKAGE_MARKOS_ENGINE — regenerate the defconfig, see header).
SANITY=$(strings "$BIN" | grep -c "TLS init failed" || true)
[ "$SANITY" -gt 0 ] || { echo "ENGINE BUILD MISSING TLS FEATURE — aborting"; exit 1; }
cp -v "$BIN" "$SRC/os/output/markos-engine.pi"
echo "ENGINE-BUILD-OK"
