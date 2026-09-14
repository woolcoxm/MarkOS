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
# Image names: Pi 3/4 firmware loads kernel8.img, Pi 5 loads kernel_2712.img.
# (No trailing comments on these lines — trailing whitespace becomes part of
# the value and word-splits QEMU arguments downstream.)
IMAGE        := kernel8.img
PI5_IMAGE    := kernel_2712.img
VIRT_IMAGE   := virt.img
TEST_IMG     := tests/test.img
SD_DIR       := sd

# Ubuntu's objcopy lacks AArch64 support; rustup's LLVM tooling has it.
LLVM_OBJCOPY := $(shell find $(HOME)/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu -name llvm-objcopy | head -1)

QEMU         := qemu-system-aarch64
# raspi3b = Pi-shaped dev machine (QEMU quirk: secondaries parked in an
# AArch32 stub, so SMP acceptance runs on the virt board instead; real Pi
# 4/5 boards are the SMP validation target). virt = 4-core automated test
# machine (PSCI release, GIC-400, PL011 @ 0x09000000).
QEMUFLAGS    := -M raspi3b -serial stdio -display none -no-reboot
# force-legacy=false: the virtio-mmio transport speaks the modern (v2)
# register interface our driver implements.
VIRTFLAGS    := -M virt -cpu cortex-a53 -smp 4 -global virtio-mmio.force-legacy=false -serial stdio -display none -no-reboot
TIMEOUT      := 25

# Optional cargo features for acceptance-test builds.
KERNEL_FEATURES ?=
FEATURES_ARG := $(if $(KERNEL_FEATURES),--features $(KERNEL_FEATURES),)
comma := ,
VIRT_FEATURES := board-virt$(if $(KERNEL_FEATURES),$(comma)$(KERNEL_FEATURES),)

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
	cargo build --release --target $(TARGET) --no-default-features --features "$(VIRT_FEATURES)"

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

## Test disk for block-device acceptance (MBR signature + marker string).
$(TEST_IMG):
	mkdir -p $(dir $(TEST_IMG))
	dd if=/dev/zero of=$(TEST_IMG) bs=1M count=1 status=none
	printf 'MARKOS-BLK-TEST!' | dd of=$(TEST_IMG) bs=1 seek=0 conv=notrunc status=none
	printf '\125\252' | dd of=$(TEST_IMG) bs=1 seek=510 conv=notrunc status=none

## Pi-3a acceptance (qemu-virt): virtio-blk bring-up + LBA0 read.
test-block: image-virt $(TEST_IMG)
	$(MAKE) image-virt KERNEL_FEATURES=selftest-block
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(VIRTFLAGS) -kernel $(VIRT_IMAGE) -drive file=$(TEST_IMG),format=raw,if=none,id=blk0 -device virtio-blk-device,drive=blk0 > serial-blk.log 2>&1 || true
	@echo "--- serial-blk.log ---"; cat serial-blk.log
	@grep -q "PASS: block device verified" serial-blk.log 		&& echo "PASS: block device" 		|| { echo "FAIL: block device"; exit 1; }

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
	rm -f $(IMAGE) $(PI5_IMAGE) $(VIRT_IMAGE) serial.log serial-exc.log serial-smp.log serial-blk.log
	rm -f $(TEST_IMG)

distclean: clean
	rm -rf $(HOME)/.markos-target
