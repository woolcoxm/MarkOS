#!/bin/bash
F=/root/markos-os/os/build-sd/build/markos-llama-c8d226b4dc94b4eaa7638e5313c2165029dcc17e
echo "=== llama.cpp version ==="
grep -m1 -r "LLAMA_BUILD_VERSION\|project(" $F/CMakeLists.txt | head -3
grep -rn "set(LLAMA_BUILD_NUMBER\|LLAMA_BUILD_COMMIT" $F/CMakeLists.txt 2>/dev/null | head -3
echo "=== set_n_threads / threadpool wiring in src/llama-context.cpp ==="
grep -n "n_threads\|set_n_threads\|threadpool" $F/src/llama-context.cpp 2>/dev/null | head -40
echo "=== ggml-cpu threadpool create conditions ==="
grep -n "ggml_threadpool_new\|n_threads_cur\|ggml_backend_cpu_set_n_threads" $F/ggml/src/ggml-cpu/ggml-cpu.cpp 2>/dev/null | head -20
echo "=== cpu backend graph_compute n_threads default ==="
grep -n "set_n_threads\|n_threads" $F/ggml/src/ggml-backend.cpp 2>/dev/null | head -10
