# MarkOS build orchestration.
# All commands run inside WSL (Ubuntu 24.04): the host Windows side only
# stores the sources. Requires: rustup nightly (rust-src, llvm-tools),
# cargo, make, gcc (for the Limine host tool), xorriso, qemu-system-x86_64.

SHELL := /bin/bash

# Cargo must find target/x86_64-unikernel.json by name.
export RUST_TARGET_PATH := $(CURDIR)/target
# Keep build artifacts off the slow 9p mount (/mnt/c) and out of the repo.
export CARGO_TARGET_DIR ?= $(HOME)/.markos-target

KERNEL_JSON  := target/x86_64-unikernel.json
KERNEL_ELF   := $(CARGO_TARGET_DIR)/x86_64-unikernel/release/kernel
LIMINE_DIR   := third_party/limine
LIMINE_TOOL  := $(LIMINE_DIR)/limine
LIMINE_VER   := v12.9.0
IMAGE        := markos.iso

QEMU     := qemu-system-x86_64
QEMUFLAGS := -M q35 -m 2G -smp 1 -display none -no-reboot
TIMEOUT  := 25

# Optional cargo features for acceptance-test builds (e.g. selftest-div).
KERNEL_FEATURES ?=
FEATURES_ARG := $(if $(KERNEL_FEATURES),--features $(KERNEL_FEATURES),)

.PHONY: all kernel deps iso run run-log debug test-div test-pf test-phase1 clean distclean

all: iso

## Fetch + build the pinned Limine boot binaries and host tool.
deps: $(LIMINE_TOOL)

$(LIMINE_TOOL):
	rm -rf $(LIMINE_DIR) && mkdir -p $(LIMINE_DIR)
	curl -fL https://github.com/limine-bootloader/limine/releases/download/$(LIMINE_VER)/limine-binary.tar.gz \
		| tar -xz -C $(LIMINE_DIR) --strip-components=1
	$(MAKE) -C $(LIMINE_DIR) CC=cc
	chmod +x $(LIMINE_TOOL)

kernel:
	cargo build -Zunstable-options -Zjson-target-spec --release --target $(KERNEL_JSON) $(FEATURES_ARG)

## Build the bootable ISO (BIOS + UEFI capable) and install the Limine stages.
iso: kernel $(LIMINE_TOOL)
	rm -rf iso_root
	mkdir -p iso_root/boot/limine iso_root/EFI/BOOT
	cp $(KERNEL_ELF) iso_root/boot/kernel
	cp boot/limine.conf iso_root/boot/limine/
	cp $(LIMINE_DIR)/limine-bios.sys $(LIMINE_DIR)/limine-bios-cd.bin \
		$(LIMINE_DIR)/limine-uefi-cd.bin iso_root/boot/limine/
	cp $(LIMINE_DIR)/BOOTX64.EFI $(LIMINE_DIR)/BOOTIA32.EFI iso_root/EFI/BOOT/
	xorriso -as mkisofs -R -r -J \
		-b boot/limine/limine-bios-cd.bin -no-emul-boot -boot-load-size 4 -boot-info-table \
		-hfsplus -apm-block-size 2048 \
		--efi-boot boot/limine/limine-uefi-cd.bin \
		-efi-boot-part --efi-boot-image --protective-msdos-label \
		iso_root -o $(IMAGE)
	$(LIMINE_TOOL) bios-install $(IMAGE)
	rm -rf iso_root

## Interactive run: serial console on stdout, Ctrl-C to quit.
run: iso
	$(QEMU) $(QEMUFLAGS) -cdrom $(IMAGE) -boot d -serial stdio

## Headless acceptance run: 25 s budget, capture COM1 to serial.log, print it.
run-log: iso
	timeout --preserve-status 25 $(QEMU) $(QEMUFLAGS) -cdrom $(IMAGE) -boot d \
		-serial stdio > serial.log 2>&1 || true
	@echo "--- serial.log ---"
	@cat serial.log

## Run with a GDB stub on tcp::1234 (use `make debug` in one shell, then
## gdb in another: target remote :1234).
debug: iso
	$(QEMU) $(QEMUFLAGS) -cdrom $(IMAGE) -boot d -serial stdio -s -S

## Phase 1 acceptance, part 1: deliberate divide-by-zero is caught and logged.
test-div:
	$(MAKE) iso KERNEL_FEATURES=selftest-div
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(QEMUFLAGS) -cdrom $(IMAGE) -boot d \
		-serial stdio > serial-div.log 2>&1 || true
	@echo "--- serial-div.log ---"; cat serial-div.log
	@grep -q "CAUGHT exception: divide_error" serial-div.log \
		&& echo "PASS: divide_error caught and logged (no triple fault)" \
		|| { echo "FAIL: divide_error not caught"; exit 1; }

## Phase 1 acceptance, part 2: deliberate page fault is caught and logged.
test-pf:
	$(MAKE) iso KERNEL_FEATURES=selftest-pf
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(QEMUFLAGS) -cdrom $(IMAGE) -boot d \
		-serial stdio > serial-pf.log 2>&1 || true
	@echo "--- serial-pf.log ---"; cat serial-pf.log
	@grep -q "CAUGHT exception: page_fault" serial-pf.log \
		&& echo "PASS: page_fault caught and logged (no triple fault)" \
		|| { echo "FAIL: page_fault not caught"; exit 1; }

## Full Phase 1 acceptance: both deliberate faults, two boots.
test-phase1: test-div test-pf

## Phase 2 acceptance: 10,240 alloc/dealloc frame pairs, leak check.
test-frames:
	$(MAKE) iso KERNEL_FEATURES=selftest-frames
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(QEMUFLAGS) -cdrom $(IMAGE) -boot d \
		-serial stdio > serial-frames.log 2>&1 || true
	@echo "--- serial-frames.log ---"; cat serial-frames.log
	@grep -q "PASS: frame stress" serial-frames.log \
		&& echo "PASS: frame allocator stress test" \
		|| { echo "FAIL: frame allocator stress test"; exit 1; }

test-phase2: test-frames

## Phase 3 acceptance: heap churn, fragmentation recovery, leak check.
test-heap:
	$(MAKE) iso KERNEL_FEATURES=selftest-heap
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(QEMUFLAGS) -cdrom $(IMAGE) -boot d \
		-serial stdio > serial-heap.log 2>&1 || true
	@echo "--- serial-heap.log ---"; cat serial-heap.log
	@grep -q "PASS: heap churn" serial-heap.log \
		&& echo "PASS: kernel heap acceptance" \
		|| { echo "FAIL: kernel heap acceptance"; exit 1; }

test-phase3: test-heap

clean:
	cargo clean || true
	rm -f $(IMAGE) serial.log

distclean: clean
	rm -rf $(LIMINE_DIR) $(HOME)/.markos-target
