# MarkOS build orchestration (AArch64 / Raspberry Pi).
# All commands run inside WSL (Ubuntu 24.04): the host Windows side only
# stores the sources. Requires: rustup nightly + aarch64-unknown-none target,
# llvm-objcopy (from rustup's llvm-tools), qemu-system-aarch64 (dev loop),
# mtools/xorriso for SD images (installer phase).

SHELL := /bin/bash

# Keep build artifacts off the slow 9p mount (/mnt/c) and out of the repo.
export CARGO_TARGET_DIR ?= $(HOME)/.markos-target

TARGET       := aarch64-unknown-none
KERNEL_ELF   := $(CARGO_TARGET_DIR)/aarch64-unknown-none/release/kernel
IMAGE        := kernel8.img           # Pi 3/4 firmware name (baseline codegen)
PI5_IMAGE    := kernel_2712.img       # Pi 5 firmware name (A76-optimized codegen)
VIRT_IMAGE   := virt.img              # QEMU virt test image (PSCI/GIC board)
SD_DIR       := sd

# Ubuntu's objcopy lacks AArch64 support; rustup's LLVM tooling has it.
LLVM_OBJCOPY := $(shell find $(HOME)/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu -name llvm-objcopy | head -1)

QEMU         := qemu-system-aarch64
# raspi3b = Pi-shaped dev machine (QEMU quirk: secondaries parked in an
# AArch32 stub, so SMP acceptance runs on the virt board instead; real Pi
# 4/5 boards are the SMP validation target). virt = 4-core automated test
# machine (PSCI release, GIC-400, PL011 @ 0x09000000).
QEMUFLAGS    := -M raspi3b -serial stdio -display none -no-reboot
VIRTFLAGS    := -M virt -cpu cortex-a53 -smp 4 -serial stdio -display none -no-reboot
TIMEOUT      := 25

# Optional cargo features for acceptance-test builds.
KERNEL_FEATURES ?=
FEATURES_ARG := $(if $(KERNEL_FEATURES),--features $(KERNEL_FEATURES),)

.PHONY: all kernel image image-pi5 kernel-virt image-virt run run-log \
        test-exceptions test-smp clean distclean

all: image

kernel:
	cargo build --release --target $(TARGET) $(FEATURES_ARG)

## Raw kernel image the Pi firmware (or QEMU -kernel) loads at 0x80000.
image: $(IMAGE)

$(IMAGE): kernel
	$(LLVM_OBJCOPY) -O binary $(KERNEL_ELF) $(IMAGE)

## Pi 5 deployment image: Cortex-A76 codegen (UDOT/SDOT int8, FP16), Pi 5
## firmware filename. Run on real Pi 5 hardware.
image-pi5:
	RUSTFLAGS="-C target-cpu=cortex-a76" cargo build --release --target $(TARGET)
	$(LLVM_OBJCOPY) -O binary $(KERNEL_ELF) $(PI5_IMAGE)

## QEMU virt test image: board-virt (PSCI release, PL011 @ 0x09000000),
## linked at the virt kernel load address 0x40080000.
kernel-virt:
	cargo build --release --target $(TARGET) --no-default-features --features board-virt

image-virt: kernel-virt
	$(LLVM_OBJCOPY) -O binary $(KERNEL_ELF) $(VIRT_IMAGE)

## Headless dev run: 25 s budget, capture PL011 to serial.log, print it.
run-log: image
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(QEMUFLAGS) \
		-kernel $(IMAGE) > serial.log 2>&1 || true
	@echo "--- serial.log ---"
	@cat serial.log

## Interactive dev run: serial console on stdout, Ctrl-C to quit.
run: image
	$(QEMU) $(QEMUFLAGS) -kernel $(IMAGE)

## Pi-1 acceptance: brk caught+resumed, data abort caught+logged, MMU/timer logs.
test-exceptions:
	$(MAKE) image KERNEL_FEATURES=selftest-exceptions
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(QEMUFLAGS) \
		-kernel $(IMAGE) > serial-exc.log 2>&1 || true
	@echo "--- serial-exc.log ---"; cat serial-exc.log
	@grep -q "CAUGHT exception: brk" serial-exc.log \
		&& grep -q "PASS: brk caught" serial-exc.log \
		&& grep -q "CAUGHT exception: data_abort" serial-exc.log \
		&& echo "PASS: exception handling (brk resumed, data abort caught)" \
		|| { echo "FAIL: exception handling"; exit 1; }

## Pi-2 acceptance (qemu-virt, PSCI): 4 cores online, exact shared counter.
test-smp: image-virt
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(VIRTFLAGS) \
		-kernel $(VIRT_IMAGE) > serial-smp.log 2>&1 || true
	@echo "--- serial-smp.log ---"; cat serial-smp.log
	@grep -q "smp: .*PASS" serial-smp.log \
		&& echo "PASS: SMP bring-up" \
		|| { echo "FAIL: SMP bring-up"; exit 1; }

clean:
	cargo clean || true
	rm -f $(IMAGE) $(PI5_IMAGE) $(VIRT_IMAGE) serial.log serial-exc.log serial-smp.log

distclean: clean
	rm -rf $(HOME)/.markos-target
