#!/bin/bash
# Diagnostic: disassemble kernel range (arg1=start hex, arg2=end hex).
OBJDUMP=$(find "$HOME/.rustup" -name llvm-objdump | head -1)
"$OBJDUMP" -d --start-address="0x$1" --stop-address="0x$2" \
    /home/mark/.markos-target/aarch64-unknown-none/release/kernel | tail -30
