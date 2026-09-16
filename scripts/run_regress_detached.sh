#!/bin/bash
# Detached runner for the full acceptance sweep (scripts/regress.sh).
export PATH="$HOME/.cargo/bin:$PATH"
cd /mnt/c/Users/Mark/Desktop/Projects/MarkOS || exit 9
rm -f /tmp/regress_rc
bash scripts/regress.sh > /tmp/regress.log 2>&1
echo "$?" > /tmp/regress_rc
