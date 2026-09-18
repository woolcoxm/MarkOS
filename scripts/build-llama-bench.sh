#!/bin/bash
# Cross-build llama-bench (CPU-only ggml, same toolchain/flags as the engine)
# from the pinned fork, to bench the raw stack on the Pi.
set -e
export PATH="/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
F=/root/markos-os/os/build-sd/build/markos-llama-c8d226b4dc94b4eaa7638e5313c2165029dcc17e
TC=/root/markos-os/os/build-sd/host/bin
B=/tmp/bench-build
rm -rf $B && mkdir -p $B
export CC=$TC/aarch64-buildroot-linux-gnu-gcc
export CXX=$TC/aarch64-buildroot-linux-gnu-g++
cmake -S $F -B $B \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_C_COMPILER=$CC -DCMAKE_CXX_COMPILER=$CXX \
  -DCMAKE_FIND_ROOT_PATH=/root/markos-os/os/build-sd/host/aarch64-buildroot-linux-gnu/sysroot \
  -DCMAKE_C_FLAGS="-mcpu=cortex-a76+dotprod+fp16" \
  -DCMAKE_CXX_FLAGS="-mcpu=cortex-a76+dotprod+fp16" \
  -DBUILD_SHARED_LIBS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_CURL=OFF \
  -DGGML_AXCL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF 2>&1 | tail -5
cmake --build $B --target llama-bench -j4 2>&1 | tail -3
ls -la $B/bin/llama-bench
cp $B/bin/llama-bench /mnt/c/Users/Mark/Desktop/Projects/MarkOS/os/output/llama-bench.pi
echo BENCH-BUILD-OK
