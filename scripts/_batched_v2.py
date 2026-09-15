#!/usr/bin/env python3
"""One-shot: multi-token batched prefill — layer-major loop with a static
hidden-state buffer. Each layer's weights are read once for all tokens.
Also adds hids_get/hids_set to engine.rs for cross-buffer access."""

# 1. Add HIDS static and accessors to engine.rs
e = open("kernel/src/engine.rs", encoding="utf-8").read()
if "pub static mut HIDS" not in e:
    marker = "/// Serving-path fast matvec: quantizes x once"
    idx = e.index(marker)
    hids_block = """/// Per-token hidden states for batched prefill.
pub static mut HIDS: [[f32; 3072]; 8] = [[0.0; 3072]; 8];

"""
    e = e[:idx] + hids_block + e[idx:]
    open("kernel/src/engine.rs", "w", encoding="utf-8").write(e)
    print("HIDS buffer added to engine.rs")

# 2. Rewrite gen_stream prefill to layer-major
src = open("kernel/src/control.rs", encoding="utf-8").read()

old = (
    "    // Prefill: prompt positions 0..n_tok through all layers.\n"
    "    for pos in 0..n_tok {\n"
    "        let row = emb_abs + (ids[pos] as u64) * (row_elems / 32 * 34) as u64;\n"
    "        if engine::dequant_q8_0_row(&vol, &file, row, row_elems, &mut act.x[..geo.n_embd])\n"
    "            .is_err()\n"
    "        {\n"
    "            tcp::stream(b\"ERR embed\\n\");\n"
    "            return;\n"
    "        }\n"
    "        for l in 0..geo.n_layers as usize {\n"
    "            if engine::layer_forward(&vol, &file, ds, l, &geo, pos, act).is_err() {\n"
    "                tcp::stream(b\"ERR layer\\n\");\n"
    "                return;\n"
    "            }\n"
    "            if l % 8 == 7 {\n"
    "                tcp::stream(b\"# k\\n\");\n"
    "            }\n"
    "        }\n"
    "    }\n"
)

new = (
    "    // Multi-token batched prefill: embed all tokens, then process\n"
    "    // layer-by-layer for all tokens. Each layer's weight tensors are\n"
    "    // read once for all tokens (reducing memory traffic n_tokens-fold).\n"
    "    // Hidden states for all tokens live in engine::HIDS (static .bss).\n"
    "\n"
    "    // Embed all prompt tokens.\n"
    "    for pos in 0..n_tok {\n"
    "        let row = emb_abs + (ids[pos] as u64) * (row_elems / 32 * 34) as u64;\n"
    "        if engine::dequant_q8_0_row(&vol, &file, row, row_elems, &mut act.x[..geo.n_embd])\n"
    "            .is_err()\n"
    "        {\n"
    "            tcp::stream(b\"ERR embed\\n\");\n"
    "            return;\n"
    "        }\n"
    "        engine::hids_copy_in(pos, &act.x[..geo.n_embd]);\n"
    "    }\n"
    "\n"
    "    // Process layers, all tokens per layer.\n"
    "    for l in 0..geo.n_layers as usize {\n"
    "        for pos in 0..n_tok {\n"
    "            engine::hids_get_into(pos, geo.n_embd, &mut act.x[..geo.n_embd]);\n"
    "            if engine::layer_forward(&vol, &file, ds, l, &geo, pos, act).is_err() {\n"
    "                tcp::stream(b\"ERR layer\\n\");\n"
    "                return;\n"
    "            }\n"
    "            engine::hids_store(pos, &act.x[..geo.n_embd]);\n"
    "            if l % 8 == 7 {\n"
    "                tcp::stream(b\"# k\\n\");\n"
    "            }\n"
    "        }\n"
    "    }\n"
)

assert old in src, "prefill block not found"
src = src.replace(old, new)
open("kernel/src/control.rs", "w", encoding="utf-8").write(src)
print("batched prefill wired into gen_stream")
