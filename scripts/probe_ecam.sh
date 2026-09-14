#!/usr/bin/env bash
# Diagnostic: probe QEMU virt's PCIe ECAM window from the monitor to find
# the real config-space address before trusting the kernel mapping.
set -u
cd "$(dirname "$0")/.."
rm -f /tmp/qmon-ecam
(timeout 15 qemu-system-aarch64 -M virt -cpu cortex-a53 -smp 1 \
  -global virtio-mmio.force-legacy=false -display none -no-reboot \
  -kernel virt.img -device pcie-root-port \
  -serial null \
  -monitor unix:/tmp/qmon-ecam,server=on,wait=off >/dev/null 2>&1 &)
sleep 3
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
for cmd in ("info pci", "info mtree"):
    s.sendall((cmd + chr(10)).encode())
    time.sleep(0.6)
time.sleep(0.6)
s.setblocking(False)
out = s.recv(16 << 20).decode(errors="replace")
lines = out.splitlines()
for l in lines:
    low = l.lower()
    if "ecam" in low or "pci" in low and "mtree" not in low or "info pci" in low or "gpex" in low or "3f00" in low:
        print(l)
PY
