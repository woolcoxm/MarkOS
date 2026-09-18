#!/bin/bash
# Rebuild markos-llama (new pin) + markos-engine through Buildroot, then copy
# the binary out to os/output/markos-engine.pi.
set -e
export PATH="/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SRC=/mnt/c/Users/Mark/Desktop/Projects/MarkOS
CANON=/root/markos-os/engine
B=/root/markos-os/os/build-sd/build/markos-engine-0.1.0

rsync -a -c --exclude target "$SRC/engine/" "$CANON/"
cp "$SRC/Cargo.lock" "$CANON/../Cargo.lock"
find "$CANON/src" "$CANON/web" -type f -exec touch {} +

cd /root/markos-os
# legal-dep rebuild picks up the bumped markos-llama pin (downloads the new
# tarball, re-extracts, no-op install) then rebuilds the engine against it.
ROOTFS_VARIANT=sd make -C os/buildroot O=/root/markos-os/os/build-sd \
  BR2_EXTERNAL=/root/markos-os/os markos-llama-rebuild markos-engine-rebuild 2>&1 | tail -25

BIN="$B/target/aarch64-unknown-linux-gnu/release/markos-engine"
ls -l --block-size=M "$BIN"
SANITY=$(strings "$BIN" | grep -c "TLS init failed" || true)
[ "$SANITY" -gt 0 ] || { echo "ENGINE BUILD MISSING TLS FEATURE — aborting"; exit 1; }
cp -v "$BIN" "$SRC/os/output/markos-engine.pi"
echo "ENGINE-BUILD-OK"
