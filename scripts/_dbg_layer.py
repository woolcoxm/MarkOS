#!/usr/bin/env python3
"""One-shot: temp per-layer digest print in selftest_gen prefill (pos 0)."""

src = open("kernel/src/main.rs").read()
old = """        for l in 0..geo.n_layers as usize {
            if engine::layer_forward(&vol, &file, ds, l, &geo, pos, act).is_err() {
                uart::write_str("FAIL: layer\\n");
                park()
            }
        }
    }

    // Greedy generation from the last prompt position."""
new = """        for l in 0..geo.n_layers as usize {
            if engine::layer_forward(&vol, &file, ds, l, &geo, pos, act).is_err() {
                uart::write_str("FAIL: layer\\n");
                park()
            }
            if pos == 0 {
                let mut s = 0f64;
                for i in 0..geo.n_embd {
                    s += act.x[i] as f64;
                }
                uart::locked_write(format_args!("LD {l} xsum={s:.6e}\\n"));
            }
        }
    }

    // Greedy generation from the last prompt position."""
assert old in src, "prefill loop not found"
src = src.replace(old, new)
open("kernel/src/main.rs", "w").write(src)
print("debug print added")
