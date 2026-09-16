################################################################################
# axclhost — Axera AXCL userspace runtime (M5Stack LLM-8850 card)
#
# Vendored from the M5Stack axclhost_3.6.5-m5stack1_arm64.deb (unpack the
# deb and place its root tree at os/vendor/axclhost-root/ — build.sh seeds
# this from Projects/Axera-refs when available). The deb ships runtime
# libs (usr/lib/axcl), card firmware (lib/firmware/axcl/ax650_card.pac),
# udev rules, ld.so.conf and tools (axcl-smi). No kernel modules here —
# those come from the axcl-driver package, built against our kernel.
################################################################################

AXCLHOST_VERSION = 3.6.5-m5stack1
AXCLHOST_SITE = $(BR2_EXTERNAL_MARKOS_PATH)/vendor/axclhost-root
AXCLHOST_SITE_METHOD = local
AXCLHOST_LICENSE = PROPRIETARY (Axera/M5Stack runtime redistribution)

AXCLHOST_INSTALL_STAGING = YES

# what we ship on the appliance; ffmpeg transcode libs and the sample/demo
# binaries are deliberately dropped (the engine needs only the NPU path)
AXCLHOST_RUNTIME_LIBS = axcl_rt axcl_npu axcl_sys axcl_comm axcl_pcie_msg \
	axcl_pcie_dma axcl_pkg axcl_lite axcl_skel axcl_token axcl_ppl \
	axcl_pcie_mmb axcl_native axcl_dmadim

define AXCLHOST_INSTALL_STAGING_CMDS
	# flat layout for the fork's cmake (AXCL_INSTALL_DIR=<root> expects
	# <root>/include/*.h and <root>/lib/lib*.so); markos-engine passes
	# MARKOS_AXCL_ROOT=$(STAGING_DIR)/usr/axcl
	$(INSTALL) -d $(STAGING_DIR)/usr/axcl/include $(STAGING_DIR)/usr/axcl/lib
	$(INSTALL) -m 0644 $(AXCLHOST_SITE)/usr/include/axcl/*.h $(STAGING_DIR)/usr/axcl/include/
	for l in $(AXCLHOST_RUNTIME_LIBS); do \
		for f in $(AXCLHOST_SITE)/usr/lib/axcl/lib$$l.so.*; do \
			$(INSTALL) -m 0755 $$f $(STAGING_DIR)/usr/axcl/lib/; \
		done; \
	done
	for f in $(STAGING_DIR)/usr/axcl/lib/*.so.*; do \
		b="$$(basename $$f)"; ln -sf "$$b" $(STAGING_DIR)/usr/axcl/lib/"$$(echo "$$b" | sed 's/\.so\..*//').so"; \
	done
endef

define AXCLHOST_INSTALL_TARGET_CMDS
	$(INSTALL) -d $(TARGET_DIR)/usr/lib/axcl
	for l in $(AXCLHOST_RUNTIME_LIBS); do \
		$(INSTALL) -m 0755 $(AXCLHOST_SITE)/usr/lib/axcl/lib$$l.so.* $(TARGET_DIR)/usr/lib/axcl/ 2>/dev/null || true; \
		$(INSTALL) -m 0755 $(AXCLHOST_SITE)/usr/lib/axcl/lib$$l.so $(TARGET_DIR)/usr/lib/axcl/ 2>/dev/null || true; \
	done
	# symlink farm so unversioned/solo-versioned builds both resolve
	for f in $(TARGET_DIR)/usr/lib/axcl/*.so.*; do \
		b="$$(basename $$f)"; \
		case "$$b" in *.so) continue;; esac; \
		major="$$(echo "$$b" | sed 's/\.so\..*//').so$$(echo "$$b" | sed -n 's/.*\.so\.\([0-9]*\).*/\1/p')"; \
		ln -sf "$$b" $(TARGET_DIR)/usr/lib/axcl/"$$major"; \
		ln -sf "$$b" $(TARGET_DIR)/usr/lib/axcl/"$$(echo "$$b" | sed 's/\.so\..*//').so"; \
	done
	# spdlog is a hard DT_NEEDED of several axcl libs
	$(INSTALL) -m 0755 $(AXCLHOST_SITE)/usr/lib/axcl/libspdlog.so* $(TARGET_DIR)/usr/lib/axcl/ 2>/dev/null || true
	# loader config + udev + modules-load + firmware
	$(INSTALL) -d $(TARGET_DIR)/etc/ld.so.conf.d $(TARGET_DIR)/etc/udev/rules.d \
		$(TARGET_DIR)/etc/modules-load.d $(TARGET_DIR)/lib/firmware/axcl
	printf '/usr/lib/axcl\n' > $(TARGET_DIR)/etc/ld.so.conf.d/axcl.conf
	printf 'ax_pcie_host_dev\nax_pcie_msg\nax_pcie_mmb\naxcl_host\nax_pcie_p2p_rc\n' \
		> $(TARGET_DIR)/etc/modules-load.d/axcl.conf
	# the deb's rules use GROUP="<users>" (a literal placeholder) — the
	# appliance has no users group; the engine runs as root
	printf 'KERNEL=="msg_userdev", MODE="0666"\nKERNEL=="ax_mmb_dev", MODE="0666"\nKERNEL=="axcl_host", MODE="0666"\nKERNEL=="p2p", MODE="0666"\n' \
		> $(TARGET_DIR)/etc/udev/rules.d/axcl_host.rules
	$(INSTALL) -m 0644 $(AXCLHOST_SITE)/lib/firmware/axcl/ax650_card.pac $(TARGET_DIR)/lib/firmware/axcl/
	# axcl-smi (diagnostics; axcl.json is its config)
	$(INSTALL) -d $(TARGET_DIR)/usr/bin
	$(INSTALL) -m 0755 $(AXCLHOST_SITE)/usr/bin/axcl/axcl-smi $(TARGET_DIR)/usr/bin/ 2>/dev/null || true
	$(INSTALL) -m 0644 $(AXCLHOST_SITE)/usr/bin/axcl/axcl.json $(TARGET_DIR)/usr/bin/ 2>/dev/null || true
	# engine-set + matmul engine locations: lives on the data partition so
	# sets can be installed without reflashing; the fork's compiled-in
	# defaults point under /usr/local/share/ggml-axcl
	$(INSTALL) -d $(TARGET_DIR)/usr/local/share/ggml-axcl
	ln -sfn /data/axcl/sets $(TARGET_DIR)/usr/local/share/ggml-axcl/sets
	ln -sfn /data/axcl/matmul $(TARGET_DIR)/usr/local/share/ggml-axcl/matmul
endef

$(eval $(generic-package))
