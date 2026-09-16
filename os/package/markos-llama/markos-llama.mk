################################################################################
# markos-llama — pinned source of the Axera llama.cpp fork
#
# Source-only: no build/install steps. markos-engine (feature `axcl`)
# consumes the checkout through MARKOS_LLAMA_CPP_DIR. The commit below is
# the head of the axera-any-gguf branch (runtime geometry + engine-set
# manifests); it must exist on GitHub before a clean-machine build.
################################################################################

MARKOS_LLAMA_VERSION = c8d226b4dc94b4eaa7638e5313c2165029dcc17e
MARKOS_LLAMA_SITE = $(call github,woolcoxm,llama.cpp,$(MARKOS_LLAMA_VERSION))
MARKOS_LLAMA_LICENSE = MIT
MARKOS_LLAMA_DEPENDENCIES = host-cmake

MARKOS_LLAMA_MARKOS_LLAMA_CPP_DIR = $(BUILD_DIR)/markos-llama-$(MARKOS_LLAMA_VERSION)

$(eval $(generic-package))
