#!/usr/bin/env bash
# Launch the Phase 11 soak detached from this shell so it survives the
# caller; verdict lands in /tmp/gensoak.verdict.
set -u
cd /mnt/c/Users/Mark/Desktop/Projects/MarkOS
pkill -f "^qemu-system-aarch64" 2>/dev/null
sleep 1
rm -f /tmp/gensoak.verdict
setsid nohup bash -c '
  make test-gensoak > /tmp/gensoak.log 2>&1
  rc=$?
  echo "$rc" > /tmp/gensoak.verdict
  tail -3 /tmp/gensoak.log
' >/dev/null 2>&1 &
echo "soak launched"
