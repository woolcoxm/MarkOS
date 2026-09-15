#!/usr/bin/env bash
# Phase 8 host reference: run llama.cpp (llama-simple) greedily on the
# same prompt with the same GGUF, then tokenize the generated text back
# into ids. Emits "LLREF prompt_ids=... gen_ids=..." for the gate.
set -euo pipefail
MODEL="${1:?model path required}"
OUT="${2:?output path required}"
SIMPLE="${LLAMA_BIN:-$HOME/llama.cpp-host/build/bin/llama-simple}"
TOK="${LLAMA_TOK:-$HOME/llama.cpp-host/build/bin/llama-tokenize}"

# Greedy continuation, 3 tokens ("hello world" -> \n hello world ...).
FULL_TEXT="$("$SIMPLE" -m "$MODEL" -n 3 "hello world" 2>/dev/null)"

# Tokenize prompt and full output with the model's own vocab; the prompt
# ids are a prefix of the full-output ids, the rest are the continuation.
PROMPT_IDS="$("$TOK" --model "$MODEL" --prompt "hello world" 2>/dev/null | awk '{print $1}' | paste -sd, -)"
ALL_IDS="$("$TOK" --model "$MODEL" --stdin <<< "$FULL_TEXT" 2>/dev/null | awk '{print $1}' | paste -sd, -)"
NPROMPT=$("$TOK" --model "$MODEL" --prompt "hello world" 2>/dev/null | wc -l)
GEN_IDS="$(echo "$ALL_IDS" | awk -v n="$NPROMPT" 'BEGIN{FS=","; s=""} {for(i=n+1;i<=NF;i++) s=s (s==""?"":",") $i} END{print s}')"

echo "LLREF prompt_ids=${PROMPT_IDS}"  > "$OUT"
echo "LLREF all_ids=${ALL_IDS}"     >> "$OUT"
echo "LLREF gen_ids=${GEN_IDS}"     >> "$OUT"
cat "$OUT"
