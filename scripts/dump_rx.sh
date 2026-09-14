#!/bin/bash
# Diagnostic: run QEMU, generate traffic, dump RX frame buffer + used ring.
cd /mnt/c/Users/Mark/Desktop/Projects/MarkOS
pkill -f qemu-system 2>/dev/null
rm -f serial-net.log
(timeout 20 qemu-system-aarch64 -M virt -cpu cortex-a53 -smp 4 \
  -global virtio-mmio.force-legacy=false -display none -no-reboot \
  -kernel virt.img \
  -drive file=tests/fat.img,format=raw,if=none,id=blk0 \
  -device virtio-blk-device,drive=blk0 \
  -netdev user,id=n0,hostfwd=tcp:127.0.0.1:8080-:8080 \
  -device virtio-net-device,netdev=n0 \
  -serial file:serial-net.log > /dev/null 2>&1 &)
sleep 4
python3 scripts/net_client.py 127.0.0.1 8080 || true
sleep 3
python3 - <<'PY'
import socket, time
s = socket.socket(socket.AF_UNIX)
s.connect("/tmp/qmon6")
time.sleep(0.3)
try:
    s.setblocking(False)
    s.recv(1 << 20)
except Exception:
    pass
s.setblocking(True)
s.sendall(b"xp /16wx 0x400d4000\n")
s.sendall(b"xp /8wx 0x400d2248\n")
time.sleep(0.6)
s.setblocking(False)
out = s.recv(1 << 20).decode(errors="replace")
print(out)
PY
