################################################################################
# markos-llama — pinned source of the Axera llama.cpp fork
#
# Source-only: no build/install steps. markos-engine (feature `axcl`)
# consumes the checkout through MARKOS_LLAMA_CPP_DIR. The commit below is
# the head of the axera-any-gguf branch (runtime geometry + engine-set
# manifests + OPT-IN device registration); it must exist on GitHub before a
# clean-machine build.
#
# 1bddded ("opt-in device registration — only register when GGML_AXCL_LAYER=1")
# is REQUIRED for correct + fast CPU-tier serving: without it the axcl backend
# registers unconditionally whenever the card is up, shares the CPU buffer
# type, and llama.cpp's scheduler routes every graph through the axcl
# graph_compute — host ops then run single-threaded and each request pays
# NPU/PCIe probing (hardware-measured 2026-09-17: 2.4 t/s decode on a 0.5B
# Q4_K_M vs 16-23 t/s from the same fork's raw CPU stack).
################################################################################

MARKOS_LLAMA_VERSION = 1bdddede8b0d41d7418b642e10873d8ee487956d
MARKOS_LLAMA_SITE = $(call github,woolcoxm,llama.cpp,$(MARKOS_LLAMA_VERSION))
MARKOS_LLAMA_LICENSE = MIT
MARKOS_LLAMA_DEPENDENCIES = host-cmake

MARKOS_LLAMA_MARKOS_LLAMA_CPP_DIR = $(BUILD_DIR)/markos-llama-$(MARKOS_LLAMA_VERSION)

$(eval $(generic-package))
