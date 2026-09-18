#!/bin/sh
# MarkOS post-image: produce markos-sd.img (squashfs RO root) and
# markos-ssd.img (ext4 RW root) from the single Buildroot run, plus sha256s.
#
# The two variants share one rootfs tree; build.sh runs `make` twice with the
# rootfs-type override, so this script runs once per variant and images the
# matching layout. Args: $1 = BINARIES_DIR is implicit via env; $2 = board dir.
set -e
BOARD_DIR="$2"
BIN_DIR="${BINARIES_DIR}"
BUILD_DIR="${BUILD_DIR}"
IMAGES_DIR="${BIN_DIR}"
ROOTFS_VARIANT="${ROOTFS_VARIANT:-sd}"

GENIMAGE_TMP="${BUILD_DIR}/genimage.tmp"

# Kernel/dtb/firmware land in BINARIES_DIR; boot files must sit flat in the
# FAT partition.
cp "${BOARD_DIR}/config_5.txt"  "${BIN_DIR}/config.txt"
cp "${BOARD_DIR}/cmdline.txt"   "${BIN_DIR}/cmdline.txt"

# Boot firmware is VENDORED in this tree (os/board/markos/bootfw/):
# hardware-proven on the 2026-production d0-stepping board (verified
# 2026-09-16 — it booted exactly these bytes). Neither the buildroot
# rpi-firmware package's start4.elf NOR the 2026-09-15 Raspberry Pi OS
# release's works: both wedge this board in the bootloader with the
# 2-long-2-short "failed to read from partition" LED code, before the
# kernel ever runs. Bump the vendored files only behind a hardware boot
# gate; do not "freshen" them from a current download.
PIOS_BOOT="${BOARD_DIR}/bootfw"
if [ ! -f "${PIOS_BOOT}/start4.elf" ]; then
	echo "post-image: vendored boot firmware missing (${PIOS_BOOT}/start4.elf)" >&2
	exit 1
fi

# Firmware files staged flat for genimage + the overlays tree (injected
# after genimage — its vfat node syntax has no recursive directories).
rm -rf "${BIN_DIR}/firmware-flat"
mkdir -p "${BIN_DIR}/firmware-flat"
for f in start4.elf fixup4.dat bootcode.bin; do
	[ -f "${PIOS_BOOT}/$f" ] && cp "${PIOS_BOOT}/$f" "${BIN_DIR}/firmware-flat/"
done
[ -d "${PIOS_BOOT}/overlays" ] && cp -r "${PIOS_BOOT}/overlays" "${BIN_DIR}/firmware-flat/"

# Rootfs variant decides the genimage layout. Explicit env wins; otherwise
# auto-detect from what Buildroot produced (the ssd variant's post-image
# runs inside the main make, before build.sh exports ROOTFS_VARIANT).
if [ -z "${ROOTFS_VARIANT:-}" ]; then
	if [ -f "${BIN_DIR}/rootfs.squashfs" ]; then
		ROOTFS_VARIANT=sd
	elif [ -f "${BIN_DIR}/rootfs.ext4" ]; then
		ROOTFS_VARIANT=ssd
	else
		echo "no rootfs.squashfs or rootfs.ext4 in ${BIN_DIR}" >&2
		exit 1
	fi
fi

# RootfsB = exact copy of rootfsA (inactive A/B slot; updated by markos-update).
case "${ROOTFS_VARIANT:-sd}" in
	sd)
		GENIMAGE_CFG="${BOARD_DIR}/genimage-sd.cfg"
		cp "${BIN_DIR}/rootfs.squashfs" "${BIN_DIR}/rootfsB.squashfs"
		ROOTFS_A="rootfs.squashfs"
		ROOTFS_B="rootfsB.squashfs"
		ROOTFS_TYPE="squashfs"
		;;
	ssd)
		GENIMAGE_CFG="${BOARD_DIR}/genimage-ssd.cfg"
		cp "${BIN_DIR}/rootfs.ext4" "${BIN_DIR}/rootfsB.ext4"
		ROOTFS_A="rootfs.ext4"
		ROOTFS_B="rootfsB.ext4"
		ROOTFS_TYPE="ext4"
		;;
	*)
		echo "unknown ROOTFS_VARIANT ${ROOTFS_VARIANT}" >&2
		exit 1
		;;
esac
export ROOTFS_A ROOTFS_B ROOTFS_TYPE

rm -rf "${GENIMAGE_TMP}"

# Data filesystem pre-created with GDT growth reserves: genimage's default
# mkfs leaves far too few reserved-gdt blocks, and first-boot resize2fs
# dies with "Not enough reserved gdt blocks" when growing 512M to ~1T
# (observed on hardware). -E resize=<2T> pre-reserves descriptor space so
# the online grow works. The genimage configs reference this file directly.
rm -f "${BIN_DIR}/data.ext4"
truncate -s 512M "${BIN_DIR}/data.ext4"
mke2fs -q -t ext4 -F -L data -E resize=2000398934016 "${BIN_DIR}/data.ext4"

genimage \
	--rootpath "${TARGET_DIR}" \
	--tmppath "${GENIMAGE_TMP}" \
	--inputpath "${BIN_DIR}" \
	--outputpath "${IMAGES_DIR}" \
	--config "${GENIMAGE_CFG}"

# Provision snapshot template: the factory-reset path restores the boot
# partition's provision files; ship an empty marker dir in the image.
mkdir -p "${IMAGES_DIR}/markos-provision-template"

IMG="markos-${ROOTFS_VARIANT}.img"
# genimage's vfat node cannot embed a directory tree: inject overlays into
# the finished image's boot FAT directly. config.txt carries
# dtoverlay=pciex1-compat-pi5 (Axera M.2 card: MSI IRQ allocation) — a boot
# FAT without that overlay means the card never enumerates, so a failed
# injection is a LOUD warning, not a silent `|| true`.
for IMG in markos-sd.img markos-ssd.img; do
	[ -f "${IMAGES_DIR}/${IMG}" ] || continue
	if [ -d "${BIN_DIR}/firmware-flat/overlays" ]; then
		if ! mcopy -s -o -i "${IMAGES_DIR}/${IMG}@@1048576" "${BIN_DIR}/firmware-flat/overlays" ::/; then
			echo "post-image: WARNING — overlay injection into ${IMG} failed; dtoverlay lines will not resolve" >&2
		fi
	else
		echo "post-image: WARNING — no overlays staged; dtoverlay lines in config.txt will not resolve" >&2
	fi
done
IMG="markos-${ROOTFS_VARIANT}.img"

(
	cd "${IMAGES_DIR}"
	sha256sum "${IMG}" > "${IMG}.sha256"
	ls -la "${IMG}"
)
echo "post-image: wrote ${IMAGES_DIR}/${IMG} (${ROOTFS_TYPE} root, A/B slots, data partition)"
