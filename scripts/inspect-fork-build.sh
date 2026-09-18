#!/bin/bash
# Inspect the fork cmake build state inside the markos-engine cargo build dir.
O=/root/markos-os/os/build-sd/build/markos-engine-0.1.0/target/aarch64-unknown-linux-gnu/release/build
echo "=== markos-llama-sys out dirs ==="
ls -d $O/markos-llama-sys-*/out 2>/dev/null
D=$(ls -d $O/markos-llama-sys-*/out/build 2>/dev/null | head -1)
echo "=== build dir: $D ==="
ls -la "$D" | head -15
echo "=== CMakeCache flags ==="
grep -a -E 'GGML_OPENMP|GGML_AXCL|CMAKE_BUILD_TYPE|GGML_NATIVE|CMAKE_C_FLAGS:|CMAKE_CXX_FLAGS:|BUILD_SHARED_LIBS' "$D/CMakeCache.txt" 2>/dev/null
echo "=== static libs ==="
find "$D" -name '*.a' -printf '%s %T@ %p\n' 2>/dev/null | sort -rn | head -10
echo "=== libggml-cpu arch flags: check for OpenMP symbols ==="
CPU=$(find "$D" -name 'libggml-cpu.a' | head -1)
echo "libggml-cpu: $CPU"
if [ -n "$CPU" ]; then
  nm "$CPU" 2>/dev/null | grep -c -i 'omp_' || echo "0 omp symbols"
  nm "$CPU" 2>/dev/null | grep -c -i 'pthread_create\|ggml_threadpool' || true
  strings "$CPU" | grep -m3 -i 'omp\|GOMP'
fi
echo "=== ggml build flags actually used (flags.make) ==="
find "$D/ggml" -name flags.make 2>/dev/null | head -3
for f in $(find "$D/ggml/src" -name flags.make 2>/dev/null | head -2); do echo "--- $f"; grep -E 'C_FLAGS|C_DEFINES' "$f" | head -4; done
