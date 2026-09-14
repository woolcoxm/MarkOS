#!/usr/bin/env bash
# Diagnostic: after the pcie selftest's abort, inspect the page tables and
# the ECAM window from the monitor.
set -u
cd "$(dirname "$0")/.."
rm -f /tmp/qmon-ecam
(timeout 15 qemu-system-aarch64 -M virt -cpu cortex-a53 -smp 1 \
  -global virtio-mmio.force-legacy=false -display none -no-reboot \
  -kernel virt.img -device pcie-root-port \
  -serial null \
  -monitor unix:/tmp/qmon-ecam,server=on,wait=off > /dev/null 2>&1 &)
sleep 4
python3 - <<'PY'
import socket, time
s = socket.socket(socket.AF_UNIX)
s.connect("/tmp/qmon-ecam")
time.sleep(0.3)
try:
    s.setblocking(False)
    s.recv(1 << 20)
except Exception:
    pass
s.setblocking(True)
for cmd in ("xp /2wx 0x400cd000", "xp /2wx 0x400ce000", "xp /2wx 0x400cdff8",
            "xp /2wx 0x4010000000", "xp /2wx 0x400cf000"):
    s.sendall((cmd + "\n").encode())
    time.sleep(0.5)
time.sleep(0.5)
s.setblocking(False)
out = s.recv(1 << 20).decode(errors="replace")
for line in out.splitlines():
    if "xp " not in line and line.strip():
        print(line)
PY
