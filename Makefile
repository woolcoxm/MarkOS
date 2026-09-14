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
FAT_TEST_IMG := tests/fat.img
TEST_IMG     := tests/test.img
INSTALL_IMG  := tests/install.img
# Real-model artifacts live outside the repo (in $HOME, off the slow 9p
# mount): a 640 MB GGUF is one-time download; the image is one-time build.
MODEL_FILE ?= $(HOME)/markos-models/Qwen3-0.6B-Q8_0.gguf
REAL_IMG   := $(HOME)/.markos-tests/real.img
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

## FAT32 test disk: MBR + one FAT32 partition holding MODEL.BIN (64 KiB
## deterministic pattern: byte[i] == i %% 251).
$(FAT_TEST_IMG):
	mkdir -p $(dir $(FAT_TEST_IMG))
	dd if=/dev/zero of=$@ bs=1M count=17 status=none
	echo "2048,30720,0x0c" | sfdisk $@ >/dev/null
	dd if=/dev/zero of=tests/fatpart.img bs=512 count=30720 status=none
	mkfs.vfat -F32 tests/fatpart.img >/dev/null
	python3 scripts/make_gguf_test.py tests/MODEL.BIN
	mcopy -i tests/fatpart.img tests/MODEL.BIN ::/MODEL.BIN
	dd if=tests/fatpart.img of=$@ bs=512 seek=2048 conv=notrunc status=none

## Pi-3b acceptance (qemu-virt): FAT32 mount + MODEL.BIN read + pattern check.
test-fat: image-virt $(FAT_TEST_IMG)
	$(MAKE) image-virt KERNEL_FEATURES=selftest-fat
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(VIRTFLAGS) -kernel $(VIRT_IMAGE) -drive file=$(FAT_TEST_IMG),format=raw,if=none,id=blk0 -device virtio-blk-device,drive=blk0 > serial-fat.log 2>&1 || true
	@echo "--- serial-fat.log ---"; cat serial-fat.log
	@grep -q "PASS: FAT32 file read" serial-fat.log 		&& echo "PASS: FAT32 acceptance" 		|| { echo "FAIL: FAT32 acceptance"; exit 1; }

## Pi-4 acceptance (qemu-virt): parallel sum across all cores via the pool.
test-pool: image-virt
	$(MAKE) image-virt KERNEL_FEATURES=selftest-pool
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(VIRTFLAGS) -kernel $(VIRT_IMAGE) > serial-pool.log 2>&1 || true
	@echo "--- serial-pool.log ---"; cat serial-pool.log
	@grep -q "pool: PASS" serial-pool.log 		&& echo "PASS: execution pool" 		|| { echo "FAIL: execution pool"; exit 1; }

## Pi-4b acceptance (qemu-virt, cortex-a76): NEON UDOT int8 matmul across
## all cores over the loaded GGUF model tensors.
test-matmul: export RUSTFLAGS = -C target-feature=+dotprod
test-matmul: image-virt $(FAT_TEST_IMG)
	$(MAKE) image-virt KERNEL_FEATURES=selftest-matmul
	timeout --preserve-status $(TIMEOUT) qemu-system-aarch64 -M virt -cpu cortex-a76 -smp 4 -global virtio-mmio.force-legacy=false -serial stdio -display none -no-reboot -kernel $(VIRT_IMAGE) -drive file=$(FAT_TEST_IMG),format=raw,if=none,id=blk0 -device virtio-blk-device,drive=blk0 > serial-mm.log 2>&1 || true
	@echo "--- serial-mm.log ---"; cat serial-mm.log
	@grep -q "PASS: matmul" serial-mm.log 		&& echo "PASS: matmul acceptance" 		|| { echo "FAIL: matmul acceptance"; exit 1; }

## Pi-5 acceptance (qemu-virt): TCP MARKOS-PING -> MARKOS-PONG end to end.
## Leftover QEMUs from earlier runs hold fat.img/port 8080 and poison the
## gate, so clear them first and kill our own QEMU before the verdict.
test-net: image-virt $(FAT_TEST_IMG)
	$(MAKE) image-virt KERNEL_FEATURES=selftest-net
	bash scripts/kill_qemu.sh; sleep 1; \
	timeout 25 $(QEMU) -M virt -cpu cortex-a53 -smp 4 -global virtio-mmio.force-legacy=false -serial stdio -display none -no-reboot -kernel $(VIRT_IMAGE) -drive file=$(FAT_TEST_IMG),format=raw,if=none,id=blk0 -device virtio-blk-device,drive=blk0 -device virtio-net-device,netdev=n0 -netdev user,id=n0,hostfwd=tcp:127.0.0.1:8080-:8080 > serial-net.log 2>&1 & qpid=$$!; \
	sleep 4; \
	python3 scripts/net_client.py 127.0.0.1 8080 || true; \
	sleep 2; \
	kill $$qpid 2>/dev/null; \
	grep -aq "net: PASS" serial-net.log && echo "PASS: network ping/pong" || { echo "FAIL: network ping/pong"; exit 1; }

## Pi-5b acceptance (qemu-virt, cortex-a76): control protocol over TCP —
## HELLO/STATUS/LOAD/RUN; RUN computes the UDOT matmul on all cores, so the
## dotprod target feature is required (as in test-matmul).
test-control: export RUSTFLAGS = -C target-feature=+dotprod
test-control: image-virt $(FAT_TEST_IMG)
	$(MAKE) image-virt KERNEL_FEATURES=selftest-net
	bash scripts/kill_qemu.sh; sleep 1; \
	timeout 30 $(QEMU) -M virt -cpu cortex-a76 -smp 4 -global virtio-mmio.force-legacy=false -serial stdio -display none -no-reboot -kernel $(VIRT_IMAGE) -drive file=$(FAT_TEST_IMG),format=raw,if=none,id=blk0 -device virtio-blk-device,drive=blk0 -device virtio-net-device,netdev=n0 -netdev user,id=n0,hostfwd=tcp:127.0.0.1:8080-:8080 > serial-ctl.log 2>&1 & qpid=$$!; \
	sleep 4; \
	python3 scripts/control_client.py 127.0.0.1 8080 > client-ctl.log 2>&1; rc=$$?; cat client-ctl.log; \
	kill $$qpid 2>/dev/null; \
	[ $$rc -eq 0 ] && echo "PASS: control protocol" || { echo "FAIL: control protocol"; exit 1; }

## Installer-baked appliance image (Pi-6): kernel + MODEL.BIN + MARKOS.CFG.
## The config here (port 8081, token) is what the kernel must pick up at
## boot for the test-install gate to pass.
$(INSTALL_IMG): image-virt tests/MODEL.BIN scripts/install.py
	python3 scripts/install.py --image $(INSTALL_IMG) --kernel $(VIRT_IMAGE) \
		--model tests/MODEL.BIN --ip 10.0.2.15 --port 8081 --token SEKRIT-TOKEN-1

## Pi-6 acceptance (qemu-virt, cortex-a76): boot with installer-baked
## config — control comes up on the configured port and the token is
## enforced; wrong-token HELLOs are rejected, then the full flow runs.
test-install: export RUSTFLAGS = -C target-feature=+dotprod
test-install: $(INSTALL_IMG)
	$(MAKE) image-virt KERNEL_FEATURES=selftest-net
	bash scripts/kill_qemu.sh; sleep 1; \
	timeout 30 $(QEMU) -M virt -cpu cortex-a76 -smp 4 -global virtio-mmio.force-legacy=false -serial stdio -display none -no-reboot -kernel $(VIRT_IMAGE) -drive file=$(INSTALL_IMG),format=raw,if=none,id=blk0 -device virtio-blk-device,drive=blk0 -device virtio-net-device,netdev=n0 -netdev user,id=n0,hostfwd=tcp:127.0.0.1:8081-:8081 > serial-inst.log 2>&1 & qpid=$$!; \
	sleep 4; \
	python3 scripts/control_client.py 127.0.0.1 8081 SEKRIT-TOKEN-1 > client-inst.log 2>&1; rc=$$?; cat client-inst.log; \
	kill $$qpid 2>/dev/null; \
	[ $$rc -eq 0 ] && echo "PASS: install config + auth" || { echo "FAIL: install config + auth"; exit 1; }

## Pi-7a acceptance (qemu-virt): PCIe ECAM enumeration — the root port
## (Red Hat 1b36:000c) must be found by the bare-metal config-space walk.
test-pcie: image-virt
	$(MAKE) image-virt KERNEL_FEATURES=selftest-pcie
	timeout --preserve-status $(TIMEOUT) $(QEMU) $(VIRTFLAGS) \
		-kernel $(VIRT_IMAGE) -device pcie-root-port > serial-pcie.log 2>&1 || true
	@echo "--- serial-pcie.log ---"; cat serial-pcie.log
	@grep -q "PASS: pcie enumerated" serial-pcie.log \
		&& echo "PASS: pcie enumeration" \
		|| { echo "FAIL: pcie enumeration"; exit 1; }

## Pi-8 acceptance (qemu-virt, cortex-a76): soak — dozens of control
## transactions (remote RUN + ECHO per round) over one TCP connection.
## RX-ring wraparound and connection-state bugs die here.
test-soak: export RUSTFLAGS = -C target-feature=+dotprod
test-soak: image-virt $(FAT_TEST_IMG)
	$(MAKE) image-virt KERNEL_FEATURES=selftest-net
	bash scripts/kill_qemu.sh; sleep 1; \
	timeout 60 $(QEMU) -M virt -cpu cortex-a76 -smp 4 -global virtio-mmio.force-legacy=false -serial stdio -display none -no-reboot -kernel $(VIRT_IMAGE) -drive file=$(FAT_TEST_IMG),format=raw,if=none,id=blk0 -device virtio-blk-device,drive=blk0 -device virtio-net-device,netdev=n0 -netdev user,id=n0,hostfwd=tcp:127.0.0.1:8080-:8080 > serial-soak.log 2>&1 & qpid=$$!; \
	sleep 4; \
	python3 scripts/soak_client.py 127.0.0.1 8080 40 > client-soak.log 2>&1; rc=$$?; tail -3 client-soak.log; \
	kill $$qpid 2>/dev/null; \
	[ $$rc -eq 0 ] && echo "PASS: soak" || { echo "FAIL: soak"; exit 1; }

## Real-model test image: MBR + one FAT32 partition holding the full GGUF.
$(REAL_IMG): $(MODEL_FILE)
	mkdir -p $(HOME)/.markos-tests
	rm -f $(REAL_IMG) $(HOME)/.markos-tests/realpart.img
	dd if=/dev/zero of=$(REAL_IMG) bs=1M count=700 status=none
	echo "2048,,0x0c" | sfdisk $(REAL_IMG) >/dev/null
	dd if=/dev/zero of=$(HOME)/.markos-tests/realpart.img bs=1024 count=715776 status=none
	mkfs.vfat -F32 $(HOME)/.markos-tests/realpart.img >/dev/null
	mcopy -i $(HOME)/.markos-tests/realpart.img $(MODEL_FILE) ::/MODEL.BIN
	dd if=$(HOME)/.markos-tests/realpart.img of=$(REAL_IMG) bs=512 seek=2048 conv=notrunc status=none
	@echo "real image ready: $(REAL_IMG)"

## Phase 7 acceptance: load the REAL GGUF from the disk image and verify
## the full tensor table + payload CRCs against the host reference
## (scripts/gguf_ref.py) — the kernel's serial output must byte-match it.
test-model: image-virt $(REAL_IMG)
	$(MAKE) image-virt KERNEL_FEATURES=selftest-model
	python3 scripts/gguf_ref.py $(MODEL_FILE) $(HOME)/.markos-tests/expected.txt
	bash scripts/kill_qemu.sh; sleep 1; \
	timeout 180 $(QEMU) $(VIRTFLAGS) -kernel $(VIRT_IMAGE) -drive file=$(REAL_IMG),format=raw,if=none,id=blk0 -device virtio-blk-device,drive=blk0 > serial-model.log 2>&1 || true; \
	tr -d "\r" < serial-model.log | grep -E "^(model:|MT |MV )" > $(HOME)/.markos-tests/got.txt || true; \
	tr -d "\r" < serial-model.log | grep -q "^PASS: model load" || { echo "FAIL: model load (no PASS marker)"; tail -5 serial-model.log; exit 1; }; \
	diff $(HOME)/.markos-tests/expected.txt $(HOME)/.markos-tests/got.txt > $(HOME)/.markos-tests/model.diff && echo "PASS: model load" || { echo "FAIL: model load (expected vs got):"; head -10 $(HOME)/.markos-tests/model.diff; exit 1; }

## Phase 8a acceptance: single-layer forward pass over the real GGUF —
## in-kernel BPE tokenizer, embedding, RMSNorm, q8_0 matmuls, RoPE, GQA
## attention, SwiGLU FFN — validated against the numpy reference
## (scripts/forward_ref.py) with scripts/forward_check.py.
test-forward: image-virt $(REAL_IMG)
	$(MAKE) image-virt KERNEL_FEATURES=selftest-forward
	python3 scripts/forward_ref.py $(MODEL_FILE) $(HOME)/.markos-tests/fwd-expected.txt
	bash scripts/kill_qemu.sh; sleep 1; \
	timeout 300 $(QEMU) $(VIRTFLAGS) -kernel $(VIRT_IMAGE) -drive file=$(REAL_IMG),format=raw,if=none,id=blk0 -device virtio-blk-device,drive=blk0 > serial-fwd.log 2>&1 || true; \
	tr -d "\r" < serial-fwd.log | grep -E "^(TOKS|EMB|NRM|QK|ATT|MID|HID|PASS: forward)" > $(HOME)/.markos-tests/fwd-got.txt || true; \
	tr -d "\r" < serial-fwd.log | grep -q "^PASS: forward" || { echo "FAIL: forward (no PASS marker)"; tail -5 serial-fwd.log; exit 1; }; \
	python3 scripts/forward_check.py $(HOME)/.markos-tests/fwd-expected.txt $(HOME)/.markos-tests/fwd-got.txt && echo "PASS: forward pass" || { echo "FAIL: forward pass"; exit 1; }

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
	rm -f $(TEST_IMG) $(FAT_TEST_IMG) tests/fatpart.img tests/MODEL.BIN serial-pool.log serial-mm.log serial-net.log client.log serial-ctl.log client-ctl.log serial-soak.log client-soak.log serial-inst.log client-inst.log net.pcap

distclean: clean
	rm -rf $(HOME)/.markos-target
