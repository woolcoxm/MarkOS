#!/usr/bin/env bash
# Wire-level probe of the HELLO exchange: capture and dump the pcap.
set -u
cd "$(dirname "$0")/.."
pkill -f qemu-system 2>/dev/null
rm -f net.pcap serial-probe.log
(timeout 20 qemu-system-aarch64 -M virt -cpu cortex-a76 -smp 4 \
  -global virtio-mmio.force-legacy=false -display none -no-reboot \
  -kernel virt.img \
  -drive file=tests/fat.img,format=raw,if=none,id=blk0 \
  -device virtio-blk-device,drive=blk0 \
  -netdev user,id=n0,hostfwd=tcp:127.0.0.1:8080-:8080 \
  -device virtio-net-device,netdev=n0 \
  -object filter-dump,id=f0,netdev=n0,file=net.pcap \
  -serial file:serial-probe.log >/dev/null 2>&1 &)
sleep 4
python3 - <<'PY'
import socket
s = socket.create_connection(("127.0.0.1", 8080), timeout=6)
s.sendall(b"HELLO\n")
try:
    print("reply:", s.recv(256))
except Exception as e:
    print("no reply:", e)
s.close()
PY
sleep 1
python3 scripts/probe_hello.py net.pcap
grep -aE "control|PANIC|established|closed" serial-probe.log
