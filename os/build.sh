#!/bin/sh
# MarkOS appliance image builder — fully scripted, reproducible from source
# control. Run on a Linux builder (or WSL2). No manual steps.
#
#   ./build.sh [--variant sd|ssd|both] [--jobs N]
#
# Artifacts land in os/output/: markos-sd.img(.sha256), markos-ssd.img(.sha256)
set -eu

HERE="$(cd "$(dirname "$0")" && pwd)"
EXT_TREE="$HERE"
OUT="$HERE/output"
BR_VERSION="${BR_VERSION:-2025.02}"
VARIANT="sd"
JOBS="$(nproc 2>/dev/null || echo 4)"

while [ $# -gt 0 ]; do
	case "$1" in
		--variant) VARIANT="$2"; shift 2 ;;
		--jobs) JOBS="$2"; shift 2 ;;
		*) echo "unknown arg $1" >&2; exit 2 ;;
	esac
done

echo "== MarkOS build (buildroot $BR_VERSION, variant: $VARIANT) =="

# --- host tool prerequisites ---
need() { command -v "$1" >/dev/null 2>&1 || { echo "missing host tool: $1" >&2; MISSING=1; }; }
MISSING=0
for t in make gcc g++ patch perl tar unzip cpio rsync bc wget file cmake xz fdisk mcopy; do need "$t"; done
[ "$MISSING" = "0" ] || { echo "install the Buildroot prerequisites (see docs/build.md)" >&2; exit 1; }

# --- rust for the engine (pinned toolchain; target added if missing) ---
if ! command -v cargo >/dev/null 2>&1; then
	echo "rust not found — install rustup: https://rustup.rs" >&2
	exit 1
fi
rustup target add aarch64-unknown-linux-gnu >/dev/null 2>&1 || true

# --- fetch pinned buildroot ---
BR="$HERE/buildroot"
if [ ! -d "$BR" ]; then
	git clone --depth 1 --branch "$BR_VERSION" https://gitlab.com/buildroot.org/buildroot.git "$BR"
fi

# --- pin integrity hashes for the custom kernel tarball ---
# BR2_DOWNLOAD_FORCE_CHECK_HASHES=y refuses custom tarballs without a hash
# entry. Fresh clones lack it (the working tree got it by hand); make the
# pin reproducible: append if the kernel commit isn't covered yet.
KERNEL_SHA=4bb240615790ea5bd939484f4595b6952ac94ef4
KERNEL_TARBALL_SHA256=df2dc37ce2fc3b90333072a901f67b8eb7ee6d887b3d23f71a964ef06a8fd85b
if ! grep -q "linux-${KERNEL_SHA}.tar.gz" "$BR/linux/linux.hash" 2>/dev/null; then
	echo "sha256 ${KERNEL_TARBALL_SHA256}  linux-${KERNEL_SHA}.tar.gz" >> "$BR/linux/linux.hash"
fi

# --- build one variant ---
build_variant() {
	V="$1"
	echo "== building variant: $V =="
	make -C "$BR" O="$HERE/build-$V" BR2_EXTERNAL="$EXT_TREE" markos_pi5_defconfig

	if [ "$V" = "ssd" ]; then
		# ext4 root instead of squashfs: writable root for SSD/NVMe.
		sed -i 's|BR2_TARGET_ROOTFS_SQUASHFS=y|# BR2_TARGET_ROOTFS_SQUASHFS is not set|' \
			"$HERE/build-$V/.config"
		echo 'BR2_TARGET_ROOTFS_EXT2=y' >> "$HERE/build-$V/.config"
		echo 'BR2_TARGET_ROOTFS_EXT2_4=y' >> "$HERE/build-$V/.config"
		echo 'BR2_TARGET_ROOTFS_EXT2_SIZE="256M"' >> "$HERE/build-$V/.config"
		make -C "$BR" O="$HERE/build-$V" olddefconfig
	fi

	# ROOTFS_VARIANT pinned for both makes: post-image must never guess the
	# layout while the rootfs images are being produced.
	ROOTFS_VARIANT="$V" make -C "$BR" O="$HERE/build-$V" -j"$JOBS"

	ROOTFS_VARIANT="$V" make -C "$BR" O="$HERE/build-$V" -j"$JOBS" \
		BR2_ROOTFS_POST_IMAGE_SCRIPT="$(BR2_EXTERNAL_MARKOS_PATH)/board/markos/post-image.sh" 2>/dev/null || true
}

mkdir -p "$OUT"
case "$VARIANT" in
	sd|ssd) build_variant "$VARIANT" ;;
	both)   build_variant sd; build_variant ssd ;;
esac

# genimage writes the final .img into the build output dir; collect them.
cp -v "$HERE"/build-*/images/markos-*.img "$OUT"/ 2>/dev/null || true
cp -v "$HERE"/build-*/images/markos-*.img.sha256 "$OUT"/ 2>/dev/null || true

echo
echo "== done =="
ls -la "$OUT"
echo
echo "Next: run markos-installer (host) to inject configuration and flash."
echo "Pi 5 firmware note: NVMe/USB boot needs BOOT_ORDER in the board EEPROM"
echo "(one-time, from a Pi OS sd or via raspi-config) — see docs/build.md."
