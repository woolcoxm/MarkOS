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

# Firmware files staged by the rpi-firmware package live under rpi-firmware/.
rm -rf "${BIN_DIR}/firmware-flat"
mkdir -p "${BIN_DIR}/firmware-flat"
for f in start4.elf fixup4.dat; do
	[ -f "${BIN_DIR}/rpi-firmware/$f" ] && cp "${BIN_DIR}/rpi-firmware/$f" "${BIN_DIR}/firmware-flat/"
done
if [ -d "${BIN_DIR}/rpi-firmware/overlays" ]; then
	cp -r "${BIN_DIR}/rpi-firmware/overlays" "${BIN_DIR}/firmware-flat/"
fi

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
(
	cd "${IMAGES_DIR}"
	sha256sum "${IMG}" > "${IMG}.sha256"
	ls -la "${IMG}"
)
echo "post-image: wrote ${IMAGES_DIR}/${IMG} (${ROOTFS_TYPE} root, A/B slots, data partition)"
