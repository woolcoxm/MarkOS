#!/usr/bin/env bash
# Kill leftover MarkOS test QEMUs. The pattern is anchored (^) so only
# processes whose command line STARTS with qemu-system-aarch64 match —
# an unanchored match also hits calling shells whose command string
# merely CONTAINS that text (make recipe lines), killing our own build.
pkill -f "^qemu-system-aarch64" 2>/dev/null
exit 0
