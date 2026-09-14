#!/bin/bash
# Capture what QEMU's virtio-net netdev sees during the net test.
cd /mnt/c/Users/Mark/Desktop/Projects/MarkOS
pkill -f qemu-system 2>/dev/null
rm -f net.pcap serial-net.log
(timeout 20 qemu-system-aarch64 -M virt -cpu cortex-a53 -smp 4 \
  -global virtio-mmio.force-legacy=false -display none -no-reboot \
  -kernel virt.img \
  -drive file=tests/fat.img,format=raw,if=none,id=blk0 \
  -device virtio-blk-device,drive=blk0 \
  -netdev user,id=n0,hostfwd=tcp:127.0.0.1:8080-:8080 \
  -device virtio-net-device,netdev=n0 \
  -object filter-dump,id=f0,netdev=n0,file=net.pcap \
  -serial file:serial-net.log > /dev/null 2>&1 &)
sleep 4
python3 scripts/net_client.py 127.0.0.1 8080 || true
sleep 3
ls -la net.pcap
python3 - <<'PY'
import struct
data = open('net.pcap','rb').read()
off = 24
n = 0
while off + 16 <= len(data):
    ts, tu, incl, orig = struct.unpack('<IIII', data[off:off+16])
    off += 16
    pkt = data[off:off+incl]
    off += incl
    n += 1
    if len(pkt) >= 14:
        etype = int.from_bytes(pkt[12:14], 'big')
        print(f"pkt {n}: len={incl} ethertype={etype:#06x} dst={pkt[0:6].hex()} src={pkt[6:12].hex()}")
    else:
        print(f"pkt {n}: short ({len(pkt)})")
print("total packets:", n)
PY
