#!/bin/sh
# MarkOS rootfs post-build: users/permissions, service enablement, /etc/inittab
# is provided by the overlay. Args: $1 = TARGET_DIR, $2.. = board dir (post
# script args from BR2_ROOTFS_POST_SCRIPT_ARGS).
set -e
TARGET_DIR="$1"
BOARD_DIR="$2"

# State/log dirs (bind targets; real storage is the `data` partition), and
# the /boot mountpoint for partition 1 (firmware + installer provision files).
mkdir -p "${TARGET_DIR}/data/state" "${TARGET_DIR}/data/models" "${TARGET_DIR}/data/log"
mkdir -p "${TARGET_DIR}/var/lib/markos" "${TARGET_DIR}/var/log/markos"
mkdir -p "${TARGET_DIR}/boot"

# Supervised services (runsvdir watches /etc/service).
mkdir -p "${TARGET_DIR}/etc/service/markos-engine" "${TARGET_DIR}/etc/service/avahi"
# dropbear is enabled by the installer via provision.env (see S03markos-net).

# Lock root's password outright (key-only dropbear; no default credentials).
sed -i 's/^root:[^:]*:/root:!:/' "${TARGET_DIR}/etc/shadow" || true

# Exec bits: Windows checkouts lose file modes, and s6/init refuse to run
# non-executable run scripts. Force them regardless of checkout platform.
chmod 755 "${TARGET_DIR}"/etc/init.d/S0*markos* 2>/dev/null || true
find "${TARGET_DIR}/etc/service" -name run -type f -exec chmod 755 {} \; 2>/dev/null || true
chmod 755 "${TARGET_DIR}/usr/bin/markos-update" "${TARGET_DIR}/usr/bin/markos-factory-reset" \n	"${TARGET_DIR}/usr/bin/markos-serial-getty" 2>/dev/null || true

# No stray shells: nologin for any generated non-root users.
chmod 1777 "${TARGET_DIR}/tmp" || true

# Busybox watchdog applet must exist for watchdogd; sanity check.
if [ ! -e "${TARGET_DIR}/bin/busybox" ]; then
	echo "post-build: busybox missing" >&2
	exit 1
fi

echo "post-build: MarkOS rootfs prepared (engine: $(ls -l "${TARGET_DIR}/usr/bin/markos-engine" 2>/dev/null || echo MISSING))"
