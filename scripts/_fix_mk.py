#!/usr/bin/env python3
"""One-shot: replace the corrupted test-gen block (lines 256..261) in the
Makefile with a correct recipe."""

lines = open("Makefile").readlines()
# Locate the corrupted block boundaries.
start = next(i for i, l in enumerate(lines) if l.startswith("## Phase 8b acceptance:"))
end = next(i for i, l in enumerate(lines) if i > start and l.startswith("## Pi-2 acceptance:"))

new = """## Phase 8b acceptance: FULL decode — 28-layer prefill + greedy steps —
## validated token-for-token against the numpy reference (--gen).
test-gen: image-virt $(REAL_IMG)
\t$(MAKE) image-virt KERNEL_FEATURES=selftest-gen
\tpython3 scripts/forward_ref.py $(MODEL_FILE) $(HOME)/.markos-tests/gen-expected.txt --gen 2
\tbash scripts/kill_qemu.sh; sleep 1; \\
\ttimeout 540 $(QEMU) $(VIRTFLAGS) -kernel $(VIRT_IMAGE) -drive file=$(REAL_IMG),format=raw,if=none,id=blk0 -device virtio-blk-device,drive=blk0 > serial-gen.log 2>&1 || true; \\
\ttr -d "\\r" < serial-gen.log | grep -E "^(TOKS|STEP|GEN|PASS: gen)" > $(HOME)/.markos-tests/gen-got.txt || true; \\
\ttr -d "\\r" < serial-gen.log | grep -q "^PASS: gen" || { echo "FAIL: gen (no PASS marker)"; tail -5 serial-gen.log; exit 1; }; \\
\tpython3 scripts/forward_check.py $(HOME)/.markos-tests/gen-expected.txt $(HOME)/.markos-tests/gen-got.txt && echo "PASS: generation" || { echo "FAIL: generation"; exit 1; }

"""
lines[start:end] = [new]
open("Makefile", "w").write("".join(lines))
print("test-gen block rewritten")
