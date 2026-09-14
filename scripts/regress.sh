#!/usr/bin/env bash
# Full acceptance sweep: runs every gate target; each Makefile target
# already greps its own serial log and exits nonzero on failure, so the
# exit code is the verdict.
#
# Usage: bash scripts/regress.sh [make-target ...]   (default: all gates)
set -u
cd "$(dirname "$0")/.."

# /mnt/c has coarse mtime granularity — touch sources so cargo never skips
# a rebuild that feature-gated edits require.
find kernel/src -name '*.rs' -exec touch {} +

# Stray QEMUs from interrupted runs hold tests/fat.img and port 8080.
pkill -f "^qemu-system-[a]arch64" 2>/dev/null
sleep 1

gates=(test-exceptions test-smp test-block test-fat test-pool test-matmul test-net test-control test-install)
[ $# -gt 0 ] && gates=("$@")

fails=0
for t in "${gates[@]}"; do
    log="/tmp/markos-$t.log"
    echo "=== $t ==="
    if make "$t" > "$log" 2>&1; then
        grep -a '^PASS:' "$log" | tail -1
    else
        rc=$?
        echo "FAIL  $t (exit $rc) — log: $log"
        grep -aE 'FAIL|PANIC' "$log" | tail -3
        fails=$((fails+1))
    fi
done

if [ "$fails" -eq 0 ]; then
    echo "regress: ALL GATES PASS (${#gates[@]} targets)"
else
    echo "regress: $fails gate(s) FAILED"
fi
exit "$fails"
