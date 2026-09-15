#!/usr/bin/env python3
"""One-shot: GEN keepalive lines during long silent stretches.

The kernel does not poll the NIC while decoding; a GEN decode can stay
silent for minutes and slirp then tears the idle hostfwd connection down
(client sees EOF). Emit a '#' comment line every 8 layers (client-side
these lines are ignored) to keep the connection and give the client
progress."""

src = open("kernel/src/control.rs").read()

old1 = """        for l in 0..geo.n_layers as usize {
            if engine::layer_forward(&vol, &file, ds, l, &geo, pos, act).is_err() {
                tcp::stream(b"ERR layer\\n");
                return;
            }
        }
    }

    // Greedy steps: norm -> argmax -> stream -> feed back."""
new1 = """        for l in 0..geo.n_layers as usize {
            if engine::layer_forward(&vol, &file, ds, l, &geo, pos, act).is_err() {
                tcp::stream(b"ERR layer\\n");
                return;
            }
            if l % 8 == 7 {
                tcp::stream(b"# k\\n");
            }
        }
    }

    // Greedy steps: norm -> argmax -> stream -> feed back."""
assert old1 in src, "prefill loop anchor missing"
src = src.replace(old1, new1)

old2 = """            for l in 0..geo.n_layers as usize {
                if engine::layer_forward(&vol, &file, ds, l, &geo, n_tok + g, act).is_err() {
                    tcp::stream(b"ERR layer\\n");
                    return;
                }
            }"""
new2 = """            for l in 0..geo.n_layers as usize {
                if engine::layer_forward(&vol, &file, ds, l, &geo, n_tok + g, act).is_err() {
                    tcp::stream(b"ERR layer\\n");
                    return;
                }
                if l % 8 == 7 {
                    tcp::stream(b"# k\\n");
                }
            }"""
assert old2 in src, "feed loop anchor missing"
src = src.replace(old2, new2)
open("kernel/src/control.rs", "w").write(src)
print("keepalives added")

c = open("scripts/gen_client.py").read()
oldc = """        line = line.strip().decode()
        if line.startswith("TOK "):"""
newc = """        line = line.strip().decode()
        if line.startswith("#"):
            continue  # keepalive comment
        if line.startswith("TOK "):"""
assert oldc in c
open("scripts/gen_client.py", "w").write(c.replace(oldc, newc))
print("client skips keepalives")
