################################################################################
# markos-engine — built from the local workspace with the Buildroot cross
# toolchain. The cargo tensor-backend feature compiles llama.cpp (ggml)
# through cmake using CC/CXX from Buildroot; `-C target-cpu=cortex-a76`
# codegen is set via RUSTFLAGS so both the serving layer and ggml NEON
# kernels target the Pi 5.
#
# Backends (see Config.in choice):
#   llama — upstream llama.cpp via llama-cpp-2 (CPU/NEON only)
#   axcl  — the woolcoxm/llama.cpp fork's ggml-axcl NPU backend (M5Stack
#           LLM-8850 card): source from the markos-llama pin, AXCL SDK
#           from the axclhost staging layout
################################################################################

MARKOS_ENGINE_VERSION = 0.1.0
MARKOS_ENGINE_SITE = $(BR2_EXTERNAL_MARKOS_PATH)/../engine
MARKOS_ENGINE_SITE_METHOD = local
MARKOS_ENGINE_LICENSE = MIT

ifeq ($(BR2_PACKAGE_MARKOS_ENGINE_LLAMA),y)
MARKOS_ENGINE_BACKEND_FEATURE = llama
endif
ifeq ($(BR2_PACKAGE_MARKOS_ENGINE_AXCL),y)
MARKOS_ENGINE_BACKEND_FEATURE = axcl
MARKOS_ENGINE_DEPENDENCIES += axclhost markos-llama
endif

MARKOS_ENGINE_DEPENDENCIES += host-cmake host-pkgconf
MARKOS_ENGINE_CARGO_TARGET = aarch64-unknown-linux-gnu

ifeq ($(BR2_PACKAGE_MARKOS_ENGINE_TLS),y)
MARKOS_ENGINE_CARGO_FEATURES += tls
endif

# Cross build via host rustup toolchain; Buildroot provides CC/CXX/ar and
# sysroot through the standard environment variables.
define MARKOS_ENGINE_BUILD_CMDS
	# Pin crate versions to the dev workspace's lockfile (standalone build
	# would otherwise resolve fresh versions).
	cp $(BR2_EXTERNAL_MARKOS_PATH)/../Cargo.lock $(@D)/Cargo.lock
	cd $(@D) && \
	$(TARGET_MAKE_ENV) $(TARGET_CONFIGURE_OPTS) \
	CFLAGS="$(CFLAGS) -mcpu=cortex-a76+dotprod+fp16" \
	CXXFLAGS="$(CXXFLAGS) -mcpu=cortex-a76+dotprod+fp16" \
	CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=$(TARGET_CC) \
	BINDGEN_EXTRA_CLANG_ARGS="--sysroot=$(STAGING_DIR) -I$(STAGING_DIR)/usr/include" \
	BINDGEN_EXTRA_CLANG_ARGS_aarch64_unknown_linux_gnu="--sysroot=$(STAGING_DIR) -I$(STAGING_DIR)/usr/include" \
	MARKOS_LLAMA_CPP_DIR=$(MARKOS_LLAMA_MARKOS_LLAMA_CPP_DIR) \
	MARKOS_AXCL_ROOT=$(STAGING_DIR)/usr/axcl \
	TARGET_SYSROOT=$(STAGING_DIR) \
	CARGO_TARGET_DIR=$(@D)/target \
	cargo build --release \
		--target $(MARKOS_ENGINE_CARGO_TARGET) \
		--no-default-features --features $(MARKOS_ENGINE_BACKEND_FEATURE)$(comma)$(MARKOS_ENGINE_CARGO_FEATURES) && \
	$(TARGET_STRIP) $(@D)/target/$(MARKOS_ENGINE_CARGO_TARGET)/release/markos-engine
endef

define MARKOS_ENGINE_INSTALL_TARGET_CMDS
	$(INSTALL) -D -m 0755 $(@D)/target/$(MARKOS_ENGINE_CARGO_TARGET)/release/markos-engine \
		$(TARGET_DIR)/usr/bin/markos-engine
endef

$(eval $(generic-package))
