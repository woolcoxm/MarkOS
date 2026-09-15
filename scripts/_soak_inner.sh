#!/usr/bin/env bash
# Soak launcher inner script: runs detached (via Windows `start /b`) and
# therefore survives the tool-call session that spawned it. Non-login
# shell, so the rustup paths are set explicitly.
set -u
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/c/Users/Mark/Desktop/Projects/MarkOS
pkill -f "^qemu-system-aarch64" 2>/dev/null
sleep 1
rm -f /tmp/gensoak.verdict
make test-gensoak > /tmp/gensoak.log 2>&1
echo "$?" > /tmp/gensoak.verdict
