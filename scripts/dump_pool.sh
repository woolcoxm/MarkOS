#!/bin/bash
# Diagnostic: dump literal pool region + symbols (run inside WSL).
OBJDUMP=$(find "$HOME/.rustup" -name llvm-objdump | head -1)
KERNEL=/home/mark/.markos-target/aarch64-unknown-none/release/kernel
echo "=== symbols around entry:"
"$OBJDUMP" -t "$KERNEL" | grep -E "_start|el_ready|from_el|vector|parked|__bss|__stack" | sort
echo "=== bytes 0x808e0-0x80920 (literal pools):"
"$OBJDUMP" -s --start-address=0x808e0 --stop-address=0x80920 "$KERNEL"
echo "=== full section headers:"
"$OBJDUMP" -h "$KERNEL"
