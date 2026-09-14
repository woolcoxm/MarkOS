# MarkOS build orchestration (AArch64 / Raspberry Pi).
# All commands run inside WSL (Ubuntu 24.04): the host Windows side only
# stores the sources. Requires: rustup nightly + aarch64-unknown-none target,
# objcopy, qemu-system-aarch64 (dev loop), mtools/xorriso for SD images.

SHELL := /bin/bash

# Keep build artifacts off the slow 9p mount (/mnt/c) and out of the repo.
export CARGO_TARGET_DIR ?= $(HOME)/.markos-target

TARGET      := aarch64-unknown-none
KERNEL_ELF  := $(CARGO_TARGET_DIR)/aarch64-unknown-none/release/kernel
IMAGE       := kernel8.img
SD_DIR      := sd

# Ubuntu's objcopy lacks AArch64 support; rustup's LLVM tooling has it.
LLVM_OBJCOPY := $(shell find $(HOME)/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu -name llvm-objcopy | head -1)

QEMU        := qemu-system-aarch64
# raspi3b = automated dev machine (PL011 @ 0x3F201000, 4x A53).
# Real Pi 4/5 validation runs the same image from SD (see README).
QEMUFLAGS   := -M raspi3b -serial stdio -display none -no-reboot
TIMEOUT     := 25

.PHONY: all kernel image run run-log sd clean distclean

all: image

kernel:
	cargo build --release --target $(TARGET)

## Raw kernel image the Pi firmware (or QEMU -kernel) loads at 0x80000.
image: $(IMAGE)

$(IMAGE): kernel
	$(LLVM_OBJCOPY) -O binary $(KERNEL_ELF) $(IMAGE)

## Headless dev run: 25 s budget, capture PL011 to serial.log, print it.
run-log: image
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(QEMUFLAGS) \
		-kernel $(IMAGE) > serial.log 2>&1 || true
	@echo "--- serial.log ---"
	@cat serial.log

## Interactive dev run: serial console on stdout, Ctrl-C to quit.
run: image
	$(QEMU) $(QEMUFLAGS) -kernel $(IMAGE)

clean:
	cargo clean || true
	rm -f $(IMAGE) serial.log

distclean: clean
	rm -rf $(HOME)/.markos-target
