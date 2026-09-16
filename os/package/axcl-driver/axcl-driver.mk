################################################################################
# axcl-driver — Axera AXCL PCIe host kernel modules
#
# Source: the AXERA-TECH axcl_host driver deb (V3.6.2) from HuggingFace.
# The deb contains a DKMS-style tree at usr/src/axcl; we extract it and
# drive its kbuild ("make host=arm64") with the Buildroot kernel + cross
# toolchain. Modules: ax_pcie_host_dev ax_pcie_msg ax_pcie_mmb axcl_host
# ax_pcie_p2p_rc (the net/rc-net module is not needed for NPU inference).
################################################################################

AXCL_DRIVER_VERSION = V3.6.2_20250603154858_20250731064200
AXCL_DRIVER_SITE = https://huggingface.co/AXERA-TECH/AXCL/resolve/main
AXCL_DRIVER_SOURCE = axcl_host_aarch64_$(AXCL_DRIVER_VERSION).deb
AXCL_DRIVER_DEPENDENCIES = linux host-zstd host-kmod

AXCL_DRIVER_MODULES = ax_pcie_host_dev ax_pcie_msg ax_pcie_mmb axcl_host \
	ax_pcie_p2p_rc

# Buildroot does not know the .deb format natively: ar-pull + zstd tar,
# then keep just the usr/src/axcl source tree as $(@D)/axcl
define AXCL_DRIVER_EXTRACT_CMDS
	mkdir -p $(@D)/deb $(@D)/root
	cp $(AXCL_DRIVER_DL_DIR)/$(AXCL_DRIVER_SOURCE) $(@D)/deb/
	cd $(@D)/deb && $(AR) x $(AXCL_DRIVER_SOURCE) && \
		tar --zstd -xf data.tar.zst -C $(@D)/root
	mv $(@D)/root/usr/src/axcl $(@D)/axcl
	rm -rf $(@D)/deb $(@D)/root
endef

# Their krules pass KERNEL_BUILD ?= / CROSS := — command-line overrides
# win over both. Driving their make tree keeps the inter-module
# KBUILD_EXTRA_SYMBOLS ordering (axcl_host links against host_dev symbols).
define AXCL_DRIVER_BUILD_CMDS
	$(TARGET_MAKE_ENV) $(MAKE) -C "$(@D)/axcl/drv" host=arm64 \
		KERNEL_DIR=$(LINUX_DIR) KERNEL_BUILD=$(LINUX_DIR) \
		CROSS=$(TARGET_CROSS) ARCH=arm64
endef

define AXCL_DRIVER_INSTALL_TARGET_CMDS
	$(INSTALL) -d $(TARGET_DIR)/lib/modules/$(LINUX_VERSION)/extra
	cd $(@D) && for m in $(AXCL_DRIVER_MODULES); do \
		f=$$(find . -name $$m.ko | head -1); \
		[ -n "$$f" ] || { echo "axcl-driver: $$m.ko not built" >&2; exit 1; }; \
		$(INSTALL) -m 0644 "$$f" $(TARGET_DIR)/lib/modules/$(LINUX_VERSION)/extra/; \
	done
	$(HOST_DIR)/sbin/depmod -b $(TARGET_DIR) $(LINUX_VERSION)
endef

$(eval $(generic-package))
