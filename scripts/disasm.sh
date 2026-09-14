#!/bin/bash
# Diagnostic: disassemble the kernel entry stub (run inside WSL).
OBJDUMP=$(find "$HOME/.rustup" -name llvm-objdump | head -1)
"$OBJDUMP" -d --start-address=0x80000 --stop-address=0x80078 \
    /home/mark/.markos-target/aarch64-unknown-none/release/kernel | tail -30
