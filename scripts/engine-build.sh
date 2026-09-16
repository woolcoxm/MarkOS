#!/bin/bash
# Cross-compile the MarkOS engine for the Pi (aarch64) via the Buildroot
# toolchain, without rebuilding the whole image. Changed sources are synced
# from the Windows checkout; the binary lands on the Windows side for scp.
# Usage: wsl -- bash -c "tr -d '\r' < this-file > /tmp/x.sh && bash /tmp/x.sh"
set -e
export PATH="/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
SRC=/mnt/c/Users/Mark/Desktop/Projects/MarkOS
DST=/root/markos-os
B=/root/markos-os/os/build-sd/build/markos-engine-0.1.0

# Sync the full engine workspace (canonical tree + package build copy).
mkdir -p "$DST/engine"
rsync -a --delete --exclude target "$SRC/engine/" "$DST/engine/"
rsync -a --delete --exclude target "$SRC/engine/" "$B/" 2>/dev/null || true
cp "$SRC/Cargo.toml" "$DST/Cargo.toml" 2>/dev/null || true
cp "$SRC/Cargo.lock" "$DST/Cargo.lock" 2>/dev/null || true
cp "$SRC/Cargo.lock" "$B/Cargo.lock" 2>/dev/null || true

# Force the package to rebuild (local-site stamps don't see our sync).
rm -f "$B"/.stamp_built "$B"/.stamp_target_installed "$B"/.stamp_installed

cd /root/markos-os
ROOTFS_VARIANT=sd make -C os/buildroot O=/root/markos-os/os/build-sd \
	BR2_EXTERNAL=/root/markos-os markos-engine-rebuild 2>&1 | tail -5

BIN="$B/target/aarch64-unknown-linux-gnu/release/markos-engine"
ls -l --block-size=M "$BIN"
cp -v "$BIN" "$SRC/os/output/markos-engine.pi"
echo "ENGINE-BUILD-OK"
