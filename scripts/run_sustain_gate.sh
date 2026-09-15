#!/bin/bash
# Detached runner for the sustained-generation gate (test-sustain): boots
# QEMU with the real-model image, runs the acceptance client, kills QEMU,
# and records the client's exit code in /tmp/sustain_rc. Launched via
# setsid so it survives the launching shell.
cd /mnt/c/Users/Mark/Desktop/Projects/MarkOS || exit 9
rm -f /tmp/sustain_rc client-sustain-dbg.log serial-sustain.log
timeout 2700 qemu-system-aarch64 -M virt -cpu cortex-a76 -smp 4 \
    -global virtio-mmio.force-legacy=false -serial stdio -display none \
    -no-reboot -m 2048 -kernel virt.img \
    -drive file="$HOME/.markos-tests/real.img",format=raw,if=none,id=blk0 \
    -device virtio-blk-device,drive=blk0 \
    -device virtio-net-device,netdev=n0 \
    -netdev user,id=n0,hostfwd=tcp:127.0.0.1:8080-:8080 \
    > serial-sustain.log 2>&1 &
qp=$!
sleep 8
python3 -u scripts/sustain_client.py 127.0.0.1 8080 > client-sustain-dbg.log 2>&1
rc=$?
kill "$qp" 2>/dev/null
pkill -f "[q]emu-system-aarch64" 2>/dev/null
echo "$rc" > /tmp/sustain_rc
